
import torch
from torch import nn
import torch.nn.functional as F
import torch_sparse
from torch_geometric.utils import softmax

from NSD.models.sheaf_base import SheafDiffusion
from NSD.models import laplacian_builders as lb
from NSD.models.sheaf_models import LocalConcatSheafLearner, EdgeWeightLearner, LocalConcatSheafLearnerVariant, SheafLearner


def partial_spectrum(L, size, k=20):
    """
    L: torch sparse laplacian (indices, values)
    """
    L_sparse = torch.sparse_coo_tensor(L[0], L[1], (size, size))

    eigvals, eigvecs = torch.lobpcg(L_sparse, k=k, largest=False)
    return eigvals, eigvecs


def select_unfair_modes(eigvals, eigvecs, s, select=3, mode="topk", weighted=False, return_eigvecs=False):
    s_col = s.view(-1, 1)
    corr = torch.abs(torch.matmul(eigvecs.T, s_col)).squeeze()
    if weighted:
        corr = corr / (eigvals + 1e-5)
    if mode == "topk":
        k = min(select, corr.size(0))
        idx = torch.topk(corr, k).indices
    elif mode == "threshold":
        idx = torch.where(corr > select)[0]

    if return_eigvecs:
        return eigvals[idx], eigvecs[:,idx]
    else:
        return eigvals[idx]
    

def spectral_filter(nodes, targets, gamma=10.0, sigma=0.05, max_eig=2.0, epsilon=0.):
    lambdas = (nodes + 1) * (max_eig / 2.0)
    g = lambdas.clone() + epsilon
    for l0 in targets:
        g += gamma * torch.exp(-(lambdas - l0) ** 2 / (2 * sigma ** 2))
    return g

def compute_chebyshev_coeffs(targets, K, gamma=10.0, sigma=0.05, max_eig=2.0, n_points=10000, epsilon=0.):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    k_idx = torch.arange(1, n_points + 1, device=device).float()
    nodes_cheb = torch.cos((2 * k_idx - 1) * torch.pi / (2 * n_points))
    
    y_nodes = spectral_filter(nodes_cheb, targets, gamma=gamma, sigma=sigma, max_eig=max_eig, epsilon=epsilon)
    
    factor = 2.0 / n_points
    term_angle = (2 * k_idx - 1) * torch.pi / (2 * n_points)
    
    coeffs = []
    for j in range(K + 1):
        T_j = torch.cos(j * term_angle)
        c_j = factor * torch.sum(y_nodes * T_j)
        coeffs.append(c_j)
        
    coeffs = torch.stack(coeffs)
    coeffs[0] /= 2.0
    
    return coeffs


def apply_filter(L, x, filter, device=None, max_eig=2.0):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    coeffs = filter["coeffs"]
    K = filter["K"]

    size = x.size(0)
    L_sp = torch.sparse_coo_tensor(L[0], L[1], (size, size), device=device)
    I_idx = torch.arange(size, device=device).repeat(2, 1)
    I_val = torch.ones(size, device=device)
    I_sp = torch.sparse_coo_tensor(I_idx, I_val, (size, size), device=device)

    L_tilde = (2/max_eig)*L_sp - I_sp

    T0 = x
    out = coeffs[0] * T0

    if K > 0:
        T1 = torch.sparse.mm(L_tilde, x)
        out += coeffs[1] * T1

    for k in range(2, K + 1):
        Tk = 2 * torch.sparse.mm(L_tilde, T1) - T0
        out += coeffs[k] * Tk
        T0, T1 = T1, Tk

    return out

#===============================================
#           Spectrum filter
#===============================================

class PolyFilterDiscreteDiagSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(PolyFilterDiscreteDiagSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 0

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.d,), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.d,), sheaf_act=self.sheaf_act))
        self.laplacian_builder = lb.DiagLaplacianBuilder(self.graph_size, edge_index, d=self.d,
                                                        normalised=self.normalised,
                                                        deg_normalised=self.deg_normalised,
                                                        add_hp=self.add_hp, add_lp=self.add_lp)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)
        
        #==========================
        if self.nonlinear:
            self.filter = {layer: None for layer in range(self.layers)}
            self._laplacians = {layer: None for layer in range(self.layers)}
        else:
            self.filter = {0: None}
            self._laplacians = {0: None}
        #==========================

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0 = x
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
                L, trans_maps = self.laplacian_builder(maps)
                self._laplacians[layer] = L
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.graph_size * self.final_d).t()

            if self.right_weights:
                x = self.lin_right_weights[layer](x)

            filter_layer = layer if self.nonlinear else 0
            if self.filter[filter_layer] is None:
                x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)
            else:
                x = apply_filter(L, x, self.filter[filter_layer], max_eig=self.max_eig)

            if self.use_act:
                x = F.elu(x)

            coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1))
            x0 = coeff * x0 - x
            x = x0

        x = x.reshape(self.graph_size, -1)
        return self.lin2(x)
    
    @torch.no_grad()
    def set_filter(self, vector,
                   k_eig=20, select=3, K_cheb=10, gamma=10.0, sigma=0.05,
                   n_points=1000, mode="topk", weighted=False, nonzero=True, epsilon = 0.01
                   ):
        size = self.graph_size * self.final_d
        for layer in self.filter.keys():
            L = self._laplacians[layer]

            # Compute relevant 
            eigvals, eigvecs = partial_spectrum(L, size, k=k_eig)
            if nonzero:
                eigvals, eigvecs = eigvals[eigvals > 1e-5], eigvecs[:, eigvals > 1e-5]
            s = vector.to(eigvecs.device)
            self.max_eig, _ = torch.lobpcg(torch.sparse_coo_tensor(L[0], L[1], (size, size)), k=1, largest=True)

            # Fit Chebyshev polynomial on the filter
            target_lambdas = select_unfair_modes(eigvals, eigvecs, s, select=select, weighted=weighted, mode=mode)
            coeffs = compute_chebyshev_coeffs(
                targets=target_lambdas,
                K=K_cheb,
                gamma=gamma,
                sigma=sigma,
                max_eig=self.max_eig,
                n_points=n_points,
                epsilon=epsilon
                )

            # Save filter
            self.filter[layer] = {"coeffs": coeffs, "K": K_cheb}

    @torch.no_grad()
    def give_filter_matrix(self, layer, smdp=True):
        coeffs = self.filter[layer]["coeffs"]
        K = self.filter[layer]["K"]
        L = self._laplacians[layer]
        device = L[0].device

        size = self.graph_size * self.final_d
        
        L_sp = torch.sparse_coo_tensor(L[0], L[1], (size, size), device=device)
        
        I_idx = torch.arange(size, device=device).unsqueeze(0).repeat(2, 1)
        I_val = torch.ones(size, device=device)
        I_sp = torch.sparse_coo_tensor(I_idx, I_val, (size, size), device=device)

        L_tilde = L_sp - I_sp

        T0 = I_sp
        out = coeffs[0] * T0

        if K > 0:
            T1 = L_tilde
            out = out + (coeffs[1] * T1)

        for k in range(2, K + 1):
            Tk = 2 * torch.sparse.mm(L_tilde, T1.to_dense()).to_sparse() - T0
            out = out + (coeffs[k] * Tk)
            T0, T1 = T1, Tk

        return out


class PolyFilterDiscreteBundleSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(PolyFilterDiscreteBundleSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 1
        assert not self.deg_normalised

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))
            
            if self.use_edge_weights:
                self.weight_learners.append(EdgeWeightLearner(self.hidden_dim, edge_index))
        self.laplacian_builder = lb.NormConnectionLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_hp=self.add_hp,
            add_lp=self.add_lp, orth_map=self.orth_trans)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        if self.nonlinear:
            self.filter = {layer: None for layer in range(self.layers)}
            self._laplacians = {layer: None for layer in range(self.layers)}
        else:
            self.filter = {0: None}
            self._laplacians = {0: None}

    def get_param_size(self):
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        for weight_learner in self.weight_learners:
            weight_learner.update_edge_index(edge_index)

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                x_maps = x_maps.reshape(self.graph_size, -1)
                maps = self.sheaf_learners[layer](x_maps, self.edge_index)
                edge_weights = self.weight_learners[layer](x_maps, self.edge_index) if self.use_edge_weights else None
                L, trans_maps = self.laplacian_builder(maps, edge_weights)
                self._laplacians[layer] = L
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            left_w = self.lin_left_weights[layer] if self.left_weights else None
            right_w = self.lin_right_weights[layer] if self.right_weights else None

            x = self.left_right_linear(x, left_w, right_w)

            # Use the adjacency matrix rather than the diagonal
            filter_layer = layer if self.nonlinear else 0
            if self.filter[filter_layer] is None:
                x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)
            else:
                x = apply_filter(L, x, self.filter[filter_layer], max_eig=self.max_eig)

            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        x = x.reshape(self.graph_size, -1)
        return self.lin2(x)
    
    @torch.no_grad()
    def set_filter(self, vector, k_eig=20, select=3, K_cheb=10, gamma=10.0, sigma=0.05, mode="topk", n_points=1000, weighted=False, nonzero = True, epsilon=0.):
        size = self.graph_size * self.final_d
        for layer in self.filter.keys():
            L = self._laplacians[layer]
            # Compute relevant 
            eigvals, eigvecs = partial_spectrum(L, size, k=k_eig)
            s = vector.to(eigvecs.device)
            self.max_eig, _ = torch.lobpcg(torch.sparse_coo_tensor(L[0], L[1], (size, size)), k=1, largest=True)
            if nonzero:
                eigvals, eigvecs = eigvals[eigvals > 1e-5], eigvecs[:, eigvals > 1e-5]
            # Fit Chebyshev polynomial on the filter
            target_lambdas = select_unfair_modes(eigvals, eigvecs, s, select=select, mode=mode, weighted=weighted)
            coeffs = compute_chebyshev_coeffs(
                targets=target_lambdas,
                K=K_cheb,
                gamma=gamma,
                sigma=sigma,
                max_eig=self.max_eig,
                n_points=n_points,
                epsilon=epsilon
                )

            # Save filter
            self.filter[layer] = {"coeffs": coeffs, "K": K_cheb}

    @torch.no_grad()
    def give_filter_matrix(self, layer, smdp=True):
        coeffs = self.filter[layer]["coeffs"]
        K = self.filter[layer]["K"]
        L = self._laplacians[layer]
        device = L[0].device

        size = self.graph_size * self.final_d
        
        L_sp = torch.sparse_coo_tensor(L[0], L[1], (size, size), device=device)
        
        I_idx = torch.arange(size, device=device).unsqueeze(0).repeat(2, 1)
        I_val = torch.ones(size, device=device)
        I_sp = torch.sparse_coo_tensor(I_idx, I_val, (size, size), device=device)

        L_tilde = L_sp - I_sp

        T0 = I_sp
        out = coeffs[0] * T0

        if K > 0:
            T1 = L_tilde
            out = out + (coeffs[1] * T1)

        for k in range(2, K + 1):
            Tk = 2 * torch.sparse.mm(L_tilde, T1.to_dense()).to_sparse() - T0
            out = out + (coeffs[k] * Tk)
            T0, T1 = T1, Tk

        return out


class PolyFilterDiscreteGeneralSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(PolyFilterDiscreteGeneralSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 1

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act))
        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_lp=self.add_lp, add_hp=self.add_hp,
            normalised=self.normalised, deg_normalised=self.deg_normalised)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        if self.nonlinear:
            self.filter = {layer: None for layer in range(self.layers)}
            self._laplacians = {layer: None for layer in range(self.layers)}
        else:
            self.filter = {0: None}
            self._laplacians = {0: None}

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
                L, trans_maps = self.laplacian_builder(maps)
                self._laplacians[layer] = L
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            left_w = self.lin_left_weights[layer] if self.left_weights else None
            right_w = self.lin_right_weights[layer] if self.right_weights else None

            x = self.left_right_linear(x, left_w, right_w)

            # Use the adjacency matrix rather than the diagonal
            filter_layer = layer if self.nonlinear else 0
            if self.filter[filter_layer] is None:
                x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)
            else:
                x = apply_filter(L, x, self.filter[filter_layer], max_eig=self.max_eig)

            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(x))

        x = x.reshape(self.graph_size, -1)
        return self.lin2(x)
    
    @torch.no_grad()
    def set_filter(self, vector, k_eig=20, select=3, K_cheb=10, gamma=10.0, sigma=0.05, mode="topk", n_points=1000, weighted=False, nonzero=True, epsilon=0.):
        size = self.graph_size * self.final_d
        for layer in self.filter.keys():
            L = self._laplacians[layer]
            # Compute relevant 
            eigvals, eigvecs = partial_spectrum(L, size, k=k_eig)
            s = vector.to(eigvecs.device)
            self.max_eig, _ = torch.lobpcg(torch.sparse_coo_tensor(L[0], L[1], (size, size)), k=1, largest=True)
            if nonzero:
                eigvals, eigvecs = eigvals[eigvals > 1e-5], eigvecs[:, eigvals > 1e-5]
            # Fit Chebyshev polynomial on the filter
            target_lambdas = select_unfair_modes(eigvals, eigvecs, s, select=select, mode=mode, weighted=weighted)
            coeffs = compute_chebyshev_coeffs(
                targets=target_lambdas,
                K=K_cheb,
                gamma=gamma,
                sigma=sigma,
                max_eig=self.max_eig,
                n_points=n_points,
                epsilon=epsilon
                )

            # Save filter
            self.filter[layer] = {"coeffs": coeffs, "K": K_cheb}

    @torch.no_grad()
    def give_filter_matrix(self, layer, smdp=True):
        coeffs = self.filter[layer]["coeffs"]
        K = self.filter[layer]["K"]
        L = self._laplacians[layer]
        device = L[0].device

        size = self.graph_size * self.final_d
        
        L_sp = torch.sparse_coo_tensor(L[0], L[1], (size, size), device=device)
        
        I_idx = torch.arange(size, device=device).unsqueeze(0).repeat(2, 1)
        I_val = torch.ones(size, device=device)
        I_sp = torch.sparse_coo_tensor(I_idx, I_val, (size, size), device=device)

        L_tilde = L_sp - I_sp

        T0 = I_sp
        out = coeffs[0] * T0

        if K > 0:
            T1 = L_tilde
            out = out + (coeffs[1] * T1)

        for k in range(2, K + 1):
            Tk = 2 * torch.sparse.mm(L_tilde, T1.to_dense()).to_sparse() - T0
            out = out + (coeffs[k] * Tk)
            T0, T1 = T1, Tk

        return out



#===============================================
#           Spectrum filter
#===============================================



class SpectrumFilterDiscreteDiagSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(SpectrumFilterDiscreteDiagSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 0

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.d,), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.d,), sheaf_act=self.sheaf_act))
        self.laplacian_builder = lb.DiagLaplacianBuilder(self.graph_size, edge_index, d=self.d,
                                                        normalised=self.normalised,
                                                        deg_normalised=self.deg_normalised,
                                                        add_hp=self.add_hp, add_lp=self.add_lp)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)
        
        #==========================
        if self.nonlinear:
            self.filter = {layer: None for layer in range(self.layers)}
            self._laplacians = {layer: None for layer in range(self.layers)}
        else:
            self.filter = {0: None}
            self._laplacians = {0: None}
        #==========================

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0 = x
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
                L, trans_maps = self.laplacian_builder(maps)
                self._laplacians[layer] = L
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.graph_size * self.final_d).t()

            if self.right_weights:
                x = self.lin_right_weights[layer](x)

            filter_layer = layer if self.nonlinear else 0
            x_L = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)
            if self.filter[filter_layer] is not None:
                x_proj = x
                vectors = self.filter[filter_layer]["vectors"]
                values = self.filter[filter_layer]["values"]

                dots = vectors.T @ x_proj
                dots = dots * values.unsqueeze(1)
                x = x_L + vectors @ dots
            else:
                x = x_L

            if self.use_act:
                x = F.elu(x)

            coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1))
            x0 = coeff * x0 - x
            x = x0

        x = x.reshape(self.graph_size, -1)
        return self.lin2(x)
    
    @torch.no_grad()
    def set_filter(self, vector, f, k_eig=20, select=3, mode="topk", weighted=True):
        size = self.graph_size * self.final_d
        for layer in self.filter.keys():
            L = self._laplacians[layer]

            # Compute relevant 
            eigvals, eigvecs = partial_spectrum(L, size, k=k_eig)
            s = vector.to(eigvecs.device)

            # Fit Chebyshev polynomial on the filter
            target_eigvals, target_eigvecs = select_unfair_modes(
                eigvals,
                eigvecs,
                s,
                select=select,
                mode=mode,
                return_eigvecs=True,
                weighted=weighted
                )

            # Save filter
            self.filter[layer] = {"values": f*torch.ones_like(target_eigvals), "vectors": target_eigvecs}


class SpectrumFilterDiscreteBundleSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(SpectrumFilterDiscreteBundleSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 1
        assert not self.deg_normalised

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))
            
            if self.use_edge_weights:
                self.weight_learners.append(EdgeWeightLearner(self.hidden_dim, edge_index))
        self.laplacian_builder = lb.NormConnectionLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_hp=self.add_hp,
            add_lp=self.add_lp, orth_map=self.orth_trans)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        if self.nonlinear:
            self.filter = {layer: None for layer in range(self.layers)}
            self._laplacians = {layer: None for layer in range(self.layers)}
        else:
            self.filter = {0: None}
            self._laplacians = {0: None}

    def get_param_size(self):
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        for weight_learner in self.weight_learners:
            weight_learner.update_edge_index(edge_index)

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                x_maps = x_maps.reshape(self.graph_size, -1)
                maps = self.sheaf_learners[layer](x_maps, self.edge_index)
                edge_weights = self.weight_learners[layer](x_maps, self.edge_index) if self.use_edge_weights else None
                L, trans_maps = self.laplacian_builder(maps, edge_weights)
                self._laplacians[layer] = L
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            left_w = self.lin_left_weights[layer] if self.left_weights else None
            right_w = self.lin_right_weights[layer] if self.right_weights else None

            x = self.left_right_linear(x, left_w, right_w)

            # Use the adjacency matrix rather than the diagonal
            filter_layer = layer if self.nonlinear else 0
            x_L = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)
            if self.filter[filter_layer] is not None:
                x_proj = x
                vectors = self.filter[filter_layer]["vectors"]
                values = self.filter[filter_layer]["values"]

                dots = vectors.T @ x_proj
                dots = dots * values.unsqueeze(1)
                x = x_L + vectors @ dots
            else:
                x = x_L


            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        x = x.reshape(self.graph_size, -1)
        return self.lin2(x)
    
    @torch.no_grad()
    def set_filter(self, vector, f, k_eig=20, select=3, mode="topk", weighted=True):
        size = self.graph_size * self.final_d
        for layer in self.filter.keys():
            L = self._laplacians[layer]

            # Compute relevant 
            eigvals, eigvecs = partial_spectrum(L, size, k=k_eig)
            s = vector.to(eigvecs.device)

            # Find biased spectrum
            if not (eigvals > 1e-15).any():
                pass

            target_eigvals, target_eigvecs = select_unfair_modes(
                eigvals,
                eigvecs,
                s,
                select=select,
                mode=mode,
                return_eigvecs=True,
                weighted=weighted
                )


            # Save filter
            self.filter[layer] = {"values": f*torch.ones_like(target_eigvals), "vectors": target_eigvecs}


class SpectrumFilterDiscreteGeneralSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(SpectrumFilterDiscreteGeneralSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 1

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act))
        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_lp=self.add_lp, add_hp=self.add_hp,
            normalised=self.normalised, deg_normalised=self.deg_normalised)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        if self.nonlinear:
            self.filter = {layer: None for layer in range(self.layers)}
            self._laplacians = {layer: None for layer in range(self.layers)}
        else:
            self.filter = {0: None}
            self._laplacians = {0: None}

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
                L, trans_maps = self.laplacian_builder(maps)
                self._laplacians[layer] = L
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            left_w = self.lin_left_weights[layer] if self.left_weights else None
            right_w = self.lin_right_weights[layer] if self.right_weights else None

            x = self.left_right_linear(x, left_w, right_w)

            # Use the adjacency matrix rather than the diagonal
            filter_layer = layer if self.nonlinear else 0
            x_L = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)
            if self.filter[filter_layer] is not None:
                x_proj = x
                vectors = self.filter[filter_layer]["vectors"]
                values = self.filter[filter_layer]["values"]

                dots = vectors.T @ x_proj
                dots = dots * values.unsqueeze(1)
                x = x_L + vectors @ dots
            else:
                x = x_L

            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(x))

        x = x.reshape(self.graph_size, -1)
        return self.lin2(x)
    
    @torch.no_grad()
    def set_filter(self, vector, f, k_eig=20, select=3, mode="topk", weighted=True):
        size = self.graph_size * self.final_d
        for layer in self.filter.keys():
            L = self._laplacians[layer]

            # Compute relevant 
            eigvals, eigvecs = partial_spectrum(L, size, k=k_eig)
            s = vector.to(eigvecs.device)

            # Fit Chebyshev polynomial on the filter
            target_eigvals, target_eigvecs = select_unfair_modes(
                eigvals,
                eigvecs,
                s,
                select=select,
                mode=mode,
                return_eigvecs=True,
                weighted=weighted
                )

            # Save filter
            self.filter[layer] = {"values": f*torch.ones_like(target_eigvals), "vectors": target_eigvecs}


#===============================================
#           Vector filter
#===============================================


class VectorFilterDiscreteDiagSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(VectorFilterDiscreteDiagSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 0

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.d,), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.d,), sheaf_act=self.sheaf_act))
        self.laplacian_builder = lb.DiagLaplacianBuilder(self.graph_size, edge_index, d=self.d,
                                                        normalised=self.normalised,
                                                        deg_normalised=self.deg_normalised,
                                                        add_hp=self.add_hp, add_lp=self.add_lp)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.vectors = args["vectors"].repeat_interleave(self.hidden_dim, dim=0)
        if self.vectors.ndim == 1:
            self.vectors = self.vectors.unsqueeze(1)
        self.vectors = self.vectors / (self.vectors.norm(dim=0, keepdim=True) + 1e-12)
        self.reg_strength = args["reg_strength"]
        self.filter = False

    def forward(self, x):
        V = self.vectors.to(x.device)
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0 = x
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.graph_size * self.final_d).t()

            if self.right_weights:
                x = self.lin_right_weights[layer](x)

            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)  
            if self.filter:        
                x_flat = x.view(-1)                   
                dots = V.T @ x_flat
                dots = dots * self.reg_strength
                x_new = x_flat + V @ dots
                x = x_new.view_as(x)

            if self.use_act:
                x = F.elu(x)

            coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1))
            x0 = coeff * x0 - x
            x = x0

        x = x.reshape(self.graph_size, -1)
        return self.lin2(x)
    
    def set_filter(self, val = True):
        self.filter = val


class VectorFilterDiscreteBundleSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(VectorFilterDiscreteBundleSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 1
        assert not self.deg_normalised

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.get_param_size(),), sheaf_act=self.sheaf_act))
            
            if self.use_edge_weights:
                self.weight_learners.append(EdgeWeightLearner(self.hidden_dim, edge_index))
        self.laplacian_builder = lb.NormConnectionLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_hp=self.add_hp,
            add_lp=self.add_lp, orth_map=self.orth_trans)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.vectors = args["vectors"].repeat_interleave(self.hidden_dim, dim=0)
        if self.vectors.ndim == 1:
            self.vectors = self.vectors.unsqueeze(1)
        self.vectors = self.vectors / (self.vectors.norm(dim=0, keepdim=True) + 1e-12)
        self.reg_strength = args["reg_strength"]
        self.filter = False

    def get_param_size(self):
        if self.orth_trans in ['matrix_exp', 'cayley']:
            return self.d * (self.d + 1) // 2
        else:
            return self.d * (self.d - 1) // 2

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x

    def update_edge_index(self, edge_index):
        super().update_edge_index(edge_index)
        for weight_learner in self.weight_learners:
            weight_learner.update_edge_index(edge_index)

    def forward(self, x):
        V = self.vectors.to(x.device)
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                x_maps = x_maps.reshape(self.graph_size, -1)
                maps = self.sheaf_learners[layer](x_maps, self.edge_index)
                edge_weights = self.weight_learners[layer](x_maps, self.edge_index) if self.use_edge_weights else None
                L, trans_maps = self.laplacian_builder(maps, edge_weights)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            left_w = self.lin_left_weights[layer] if self.left_weights else None
            right_w = self.lin_right_weights[layer] if self.right_weights else None

            x = self.left_right_linear(x, left_w, right_w)

            # Use the adjacency matrix rather than the diagonal
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)    
            if self.filter:      
                x_flat = x.view(-1)                   
                dots = V.T @ x_flat
                dots = dots * self.reg_strength
                x_new = x_flat + V @ dots
                x = x_new.view_as(x)

            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        x = x.reshape(self.graph_size, -1)
        return self.lin2(x)

    def set_filter(self, val = True):
        self.filter = val


class VectorFilterDiscreteGeneralSheafDiffusion(SheafDiffusion):

    def __init__(self, edge_index, args):
        super(VectorFilterDiscreteGeneralSheafDiffusion, self).__init__(edge_index, args)
        assert args['d'] > 1

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers if self.nonlinear else 1)
        for i in range(num_sheaf_learners):
            if self.sparse_learner:
                self.sheaf_learners.append(LocalConcatSheafLearnerVariant(self.final_d,
                    self.hidden_channels, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act))
            else:
                self.sheaf_learners.append(LocalConcatSheafLearner(
                    self.hidden_dim, out_shape=(self.d, self.d), sheaf_act=self.sheaf_act))
        self.laplacian_builder = lb.GeneralLaplacianBuilder(
            self.graph_size, edge_index, d=self.d, add_lp=self.add_lp, add_hp=self.add_hp,
            normalised=self.normalised, deg_normalised=self.deg_normalised)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        self.vectors = args["vectors"].repeat_interleave(self.hidden_dim, dim=0)
        if self.vectors.ndim == 1:
            self.vectors = self.vectors.unsqueeze(1)
        self.vectors = self.vectors / (self.vectors.norm(dim=0, keepdim=True) + 1e-12)
        self.reg_strength = args["reg_strength"]
        self.filter = False

    def left_right_linear(self, x, left, right):
        if self.left_weights:
            x = x.t().reshape(-1, self.final_d)
            x = left(x)
            x = x.reshape(-1, self.graph_size * self.final_d).t()

        if self.right_weights:
            x = right(x)

        return x

    def forward(self, x):
        V = self.vectors.to(x.device)
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0, L = x, None
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            left_w = self.lin_left_weights[layer] if self.left_weights else None
            right_w = self.lin_right_weights[layer] if self.right_weights else None

            x = self.left_right_linear(x, left_w, right_w)

            # Use the adjacency matrix rather than the diagonal
            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)          
            if self.filter:
                x_flat = x.view(-1)                   
                dots = V.T @ x_flat
                dots = dots * self.reg_strength
                x_new = x_flat + V @ dots
                x = x_new.view_as(x)

            if self.use_act:
                x = F.elu(x)

            x0 = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1)) * x0 - x
            x = x0

        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(x))

        x = x.reshape(self.graph_size, -1)
        return self.lin2(x)
    
    def set_filter(self, val = True):
        self.filter = val




#===============================================
#           GCN
#===============================================

class TrivialSheafLearner(SheafLearner):
    def __init__(self, output_dim):
        super(TrivialSheafLearner, self).__init__()
        self.output_dim = output_dim

    def forward(self, x, edge_index):
        return torch.ones(edge_index.shape[1], self.output_dim, device=x.device)
    
    def update_edge_index(self, edge_index):
        pass

class PolyFilterGCN(SheafDiffusion):
    def __init__(self, edge_index, args):
        args['d'] = 1
        super(PolyFilterGCN, self).__init__(edge_index, args)

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        for i in range(self.layers):
            self.sheaf_learners.append(TrivialSheafLearner(output_dim=self.d))

        self.laplacian_builder = lb.DiagLaplacianBuilder(self.graph_size, edge_index, d=self.d,
                                                        normalised=self.normalised,
                                                        deg_normalised=self.deg_normalised,
                                                        add_hp=self.add_hp, add_lp=self.add_lp)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)
        
        self.max_eig = 2.0

        if self.nonlinear:
            self.filter = {layer: None for layer in range(self.layers)}
            self._laplacians = {layer: None for layer in range(self.layers)}
        else:
            self.filter = {0: None}
            self._laplacians = {0: None}

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0 = x
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
                L, trans_maps = self.laplacian_builder(maps)
                self._laplacians[layer] = L
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.graph_size * self.final_d).t()

            if self.right_weights:
                x = self.lin_right_weights[layer](x)

            filter_layer = layer if self.nonlinear else 0
            
            if self.filter[filter_layer] is None:
                x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)
            else:
                x = apply_filter(L, x, self.filter[filter_layer], max_eig=self.max_eig)

            if self.use_act:
                x = F.elu(x)

            coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1))
            x0 = coeff * x0 - x
            x = x0

        x = x.reshape(self.graph_size, -1)
        return self.lin2(x)

    @torch.no_grad()
    def set_filter(self, vector,
                   k_eig=20, select=3, K_cheb=10, gamma=10.0, sigma=0.05,
                   n_points=1000, mode="topk", weighted=False, nonzero=True, epsilon = 0.01
                   ):
        size = self.graph_size * self.final_d
        for layer in self.filter.keys():
            L = self._laplacians[layer]
            if L is None: continue

            eigvals, eigvecs = partial_spectrum(L, size, k=k_eig)
            if nonzero:
                mask = eigvals > 1e-5
                eigvals, eigvecs = eigvals[mask], eigvecs[:, mask]
            
            s = vector.to(eigvecs.device)
            self.max_eig, _ = torch.lobpcg(torch.sparse_coo_tensor(L[0], L[1], (size, size)), k=1, largest=True)

            target_lambdas = select_unfair_modes(eigvals, eigvecs, s, select=select, weighted=weighted, mode=mode)
            coeffs = compute_chebyshev_coeffs(
                targets=target_lambdas,
                K=K_cheb,
                gamma=gamma,
                sigma=sigma,
                max_eig=self.max_eig,
                n_points=n_points,
                epsilon=epsilon
            )

            self.filter[layer] = {"coeffs": coeffs, "K": K_cheb}

    @torch.no_grad()
    def give_filter_matrix(self, layer):
        if self.filter[layer] is None: return None
        coeffs = self.filter[layer]["coeffs"]
        K = self.filter[layer]["K"]
        L = self._laplacians[layer]
        device = L[0].device

        size = self.graph_size * self.final_d
        
        L_sp = torch.sparse_coo_tensor(L[0], L[1], (size, size), device=device)
        
        I_idx = torch.arange(size, device=device).unsqueeze(0).repeat(2, 1)
        I_val = torch.ones(size, device=device)
        I_sp = torch.sparse_coo_tensor(I_idx, I_val, (size, size), device=device)

        L_tilde = L_sp - I_sp

        T0 = I_sp
        out = coeffs[0] * T0

        if K > 0:
            T1 = L_tilde
            out = out + (coeffs[1] * T1)

        for k in range(2, K + 1):
            Tk = 2 * torch.sparse.mm(L_tilde, T1.to_dense()).to_sparse() - T0
            out = out + (coeffs[k] * Tk)
            T0, T1 = T1, Tk

        return out


class SpectrumFilterGCN(SheafDiffusion):
    def __init__(self, edge_index, args):
        args['d'] = 1
        super(SpectrumFilterGCN, self).__init__(edge_index, args)

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        for i in range(self.layers):
            self.sheaf_learners.append(TrivialSheafLearner(output_dim=self.d))

        self.laplacian_builder = lb.DiagLaplacianBuilder(self.graph_size, edge_index, d=self.d,
                                                        normalised=self.normalised,
                                                        deg_normalised=self.deg_normalised,
                                                        add_hp=self.add_hp, add_lp=self.add_lp)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        if self.nonlinear:
            self.filter = {layer: None for layer in range(self.layers)}
            self._laplacians = {layer: None for layer in range(self.layers)}
        else:
            self.filter = {0: None}
            self._laplacians = {0: None}

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0 = x
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
                L, trans_maps = self.laplacian_builder(maps)
                self._laplacians[layer] = L
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.graph_size * self.final_d).t()

            if self.right_weights:
                x = self.lin_right_weights[layer](x)

            filter_layer = layer if self.nonlinear else 0
            x_L = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)
            
            if self.filter[filter_layer] is not None:
                x_proj = x
                vectors = self.filter[filter_layer]["vectors"]
                values = self.filter[filter_layer]["values"]

                dots = vectors.T @ x_proj
                dots = dots * values.unsqueeze(1)
                x = x_L + vectors @ dots
            else:
                x = x_L

            if self.use_act:
                x = F.elu(x)

            coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1))
            x0 = coeff * x0 - x
            x = x0

        x = x.reshape(self.graph_size, -1)
        return self.lin2(x)

    @torch.no_grad()
    def set_filter(self, vector, f, k_eig=20, select=3, mode="topk", weighted=True):
        size = self.graph_size * self.final_d
        for layer in self.filter.keys():
            L = self._laplacians[layer]
            if L is None: continue

            eigvals, eigvecs = partial_spectrum(L, size, k=k_eig)
            s = vector.to(eigvecs.device)

            target_eigvals, target_eigvecs = select_unfair_modes(
                eigvals,
                eigvecs,
                s,
                select=select,
                mode=mode,
                return_eigvecs=True,
                weighted=weighted
            )

            self.filter[layer] = {"values": f*torch.ones_like(target_eigvals), "vectors": target_eigvecs}


class VectorFilterGCN(SheafDiffusion):
    def __init__(self, edge_index, args):
        args['d'] = 1
        super(VectorFilterGCN, self).__init__(edge_index, args)

        self.vectors = args["vectors"].repeat_interleave(self.hidden_dim, dim=0)
        if self.vectors.ndim == 1:
            self.vectors = self.vectors.unsqueeze(1)
        self.vectors = self.vectors / (self.vectors.norm(dim=0, keepdim=True) + 1e-12)
        self.reg_strength = args["reg_strength"]
        self.filter = False

        self.sheaf_learners = nn.ModuleList()
        for i in range(self.layers):
            self.sheaf_learners.append(TrivialSheafLearner(output_dim=self.d))

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.laplacian_builder = lb.DiagLaplacianBuilder(self.graph_size, edge_index, d=self.d,
                                                        normalised=self.normalised,
                                                        deg_normalised=self.deg_normalised,
                                                        add_hp=self.add_hp, add_lp=self.add_lp)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def set_filter(self, val=True):
        self.filter = val

    def forward(self, x):
        V = self.vectors.to(x.device)

        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0 = x
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.graph_size * self.final_d).t()

            if self.right_weights:
                x = self.lin_right_weights[layer](x)

            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            if self.filter:
                x_flat = x.view(-1)
                dots = V.T @ x_flat
                dots = dots * self.reg_strength
                x_new = x_flat + V @ dots
                x = x_new.view_as(x)

            if self.use_act:
                x = F.elu(x)

            coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1))
            x0 = coeff * x0 - x
            x = x0

        x = x.reshape(self.graph_size, -1)
        return self.lin2(x)


class CustomGCN(SheafDiffusion):
    def __init__(self, edge_index, args):
        args['d'] = 1
        super(CustomGCN, self).__init__(edge_index, args)

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        for i in range(self.layers):
            self.sheaf_learners.append(TrivialSheafLearner(output_dim=self.d))

        self.laplacian_builder = lb.DiagLaplacianBuilder(self.graph_size, edge_index, d=self.d,
                                                        normalised=self.normalised,
                                                        deg_normalised=self.deg_normalised,
                                                        add_hp=self.add_hp, add_lp=self.add_lp)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0 = x
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.graph_size * self.final_d).t()

            if self.right_weights:
                x = self.lin_right_weights[layer](x)

            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1))
            x0 = coeff * x0 - x
            x = x0

        x = x.reshape(self.graph_size, -1)
        return self.lin2(x)
    


#===============================================
#           GAT
#===============================================


class AttentionSheafLearner(SheafLearner):
    def __init__(self, in_channels, d):
        super(AttentionSheafLearner, self).__init__()
        self.d = d
        self.lin = nn.Linear(in_channels * 2, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index):
        row, col = edge_index
        h = torch.cat([x[row], x[col]], dim=1)
        
        alpha = self.lin(h)
        alpha = self.leaky_relu(alpha)
        
        alpha = softmax(alpha, row, num_nodes=x.size(0))
        return alpha.view(-1, self.d)
    

class PolyFilterGAT(SheafDiffusion):
    def __init__(self, edge_index, args):
        args['d'] = 1
        super(PolyFilterGAT, self).__init__(edge_index, args)

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        for i in range(self.layers):
            self.sheaf_learners.append(AttentionSheafLearner(self.hidden_channels, self.d))

        self.laplacian_builder = lb.DiagLaplacianBuilder(self.graph_size, edge_index, d=self.d,
                                                        normalised=self.normalised,
                                                        deg_normalised=self.deg_normalised,
                                                        add_hp=self.add_hp, add_lp=self.add_lp)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)
        
        self.max_eig = 2.0

        if self.nonlinear:
            self.filter = {layer: None for layer in range(self.layers)}
            self._laplacians = {layer: None for layer in range(self.layers)}
        else:
            self.filter = {0: None}
            self._laplacians = {0: None}

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0 = x
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
                L, trans_maps = self.laplacian_builder(maps)
                self._laplacians[layer] = L
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.graph_size * self.final_d).t()

            if self.right_weights:
                x = self.lin_right_weights[layer](x)

            filter_layer = layer if self.nonlinear else 0
            
            if self.filter[filter_layer] is None:
                x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)
            else:
                x = apply_filter(L, x, self.filter[filter_layer], max_eig=self.max_eig)

            if self.use_act:
                x = F.elu(x)

            coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1))
            x0 = coeff * x0 - x
            x = x0

        x = x.reshape(self.graph_size, -1)
        return self.lin2(x)

    @torch.no_grad()
    def set_filter(self, vector,
                   k_eig=20, select=3, K_cheb=10, gamma=10.0, sigma=0.05,
                   n_points=1000, mode="topk", weighted=False, nonzero=True, epsilon = 0.01
                   ):
        size = self.graph_size * self.final_d
        for layer in self.filter.keys():
            L = self._laplacians[layer]
            if L is None: continue

            eigvals, eigvecs = partial_spectrum(L, size, k=k_eig)
            if nonzero:
                mask = eigvals > 1e-5
                eigvals, eigvecs = eigvals[mask], eigvecs[:, mask]
            
            s = vector.to(eigvecs.device)
            self.max_eig, _ = torch.lobpcg(torch.sparse_coo_tensor(L[0], L[1], (size, size)), k=1, largest=True)

            target_lambdas = select_unfair_modes(eigvals, eigvecs, s, select=select, weighted=weighted, mode=mode)
            coeffs = compute_chebyshev_coeffs(
                targets=target_lambdas,
                K=K_cheb,
                gamma=gamma,
                sigma=sigma,
                max_eig=self.max_eig,
                n_points=n_points,
                epsilon=epsilon
            )

            self.filter[layer] = {"coeffs": coeffs, "K": K_cheb}

    @torch.no_grad()
    def give_filter_matrix(self, layer):
        if self.filter[layer] is None: return None
        coeffs = self.filter[layer]["coeffs"]
        K = self.filter[layer]["K"]
        L = self._laplacians[layer]
        device = L[0].device

        size = self.graph_size * self.final_d
        
        L_sp = torch.sparse_coo_tensor(L[0], L[1], (size, size), device=device)
        
        I_idx = torch.arange(size, device=device).unsqueeze(0).repeat(2, 1)
        I_val = torch.ones(size, device=device)
        I_sp = torch.sparse_coo_tensor(I_idx, I_val, (size, size), device=device)

        L_tilde = L_sp - I_sp

        T0 = I_sp
        out = coeffs[0] * T0

        if K > 0:
            T1 = L_tilde
            out = out + (coeffs[1] * T1)

        for k in range(2, K + 1):
            Tk = 2 * torch.sparse.mm(L_tilde, T1.to_dense()).to_sparse() - T0
            out = out + (coeffs[k] * Tk)
            T0, T1 = T1, Tk

        return out

class SpectrumFilterGAT(SheafDiffusion):
    def __init__(self, edge_index, args):
        args['d'] = 1
        super(SpectrumFilterGAT, self).__init__(edge_index, args)

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        for i in range(self.layers):
            self.sheaf_learners.append(AttentionSheafLearner(self.hidden_channels, self.d))

        self.laplacian_builder = lb.DiagLaplacianBuilder(self.graph_size, edge_index, d=self.d,
                                                        normalised=self.normalised,
                                                        deg_normalised=self.deg_normalised,
                                                        add_hp=self.add_hp, add_lp=self.add_lp)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

        if self.nonlinear:
            self.filter = {layer: None for layer in range(self.layers)}
            self._laplacians = {layer: None for layer in range(self.layers)}
        else:
            self.filter = {0: None}
            self._laplacians = {0: None}

    def forward(self, x):
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0 = x
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
                L, trans_maps = self.laplacian_builder(maps)
                self._laplacians[layer] = L
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.graph_size * self.final_d).t()

            if self.right_weights:
                x = self.lin_right_weights[layer](x)

            filter_layer = layer if self.nonlinear else 0
            x_L = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)
            
            if self.filter[filter_layer] is not None:
                x_proj = x
                vectors = self.filter[filter_layer]["vectors"]
                values = self.filter[filter_layer]["values"]

                dots = vectors.T @ x_proj
                dots = dots * values.unsqueeze(1)
                x = x_L + vectors @ dots
            else:
                x = x_L

            if self.use_act:
                x = F.elu(x)

            coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1))
            x0 = coeff * x0 - x
            x = x0

        x = x.reshape(self.graph_size, -1)
        return self.lin2(x)

    @torch.no_grad()
    def set_filter(self, vector, f, k_eig=20, select=3, mode="topk", weighted=True):
        size = self.graph_size * self.final_d
        for layer in self.filter.keys():
            L = self._laplacians[layer]
            if L is None: continue

            eigvals, eigvecs = partial_spectrum(L, size, k=k_eig)
            s = vector.to(eigvecs.device)

            target_eigvals, target_eigvecs = select_unfair_modes(
                eigvals,
                eigvecs,
                s,
                select=select,
                mode=mode,
                return_eigvecs=True,
                weighted=weighted
            )

            self.filter[layer] = {"values": f*torch.ones_like(target_eigvals), "vectors": target_eigvecs}

class VectorFilterGAT(SheafDiffusion):
    def __init__(self, edge_index, args):
        args['d'] = 1
        super(VectorFilterGAT, self).__init__(edge_index, args)

        self.vectors = args["vectors"].repeat_interleave(self.hidden_dim, dim=0)
        if self.vectors.ndim == 1:
            self.vectors = self.vectors.unsqueeze(1)
        self.vectors = self.vectors / (self.vectors.norm(dim=0, keepdim=True) + 1e-12)
        self.reg_strength = args["reg_strength"]
        self.filter = False

        self.sheaf_learners = nn.ModuleList()
        for i in range(self.layers):
            self.sheaf_learners.append(AttentionSheafLearner(self.hidden_channels, self.d))

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)
        
        self.laplacian_builder = lb.DiagLaplacianBuilder(self.graph_size, edge_index, d=self.d,
                                                        normalised=self.normalised,
                                                        deg_normalised=self.deg_normalised,
                                                        add_hp=self.add_hp, add_lp=self.add_lp)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def set_filter(self, val=True):
        self.filter = val

    def forward(self, x):
        V = self.vectors.to(x.device)

        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0 = x
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.graph_size * self.final_d).t()

            if self.right_weights:
                x = self.lin_right_weights[layer](x)

            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            if self.filter:
                x_flat = x.view(-1)
                dots = V.T @ x_flat
                dots = dots * self.reg_strength
                x_new = x_flat + V @ dots
                x = x_new.view_as(x)

            if self.use_act:
                x = F.elu(x)

            coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1))
            x0 = coeff * x0 - x
            x = x0

        x = x.reshape(self.graph_size, -1)
        return self.lin2(x)
    


class CustomGAT(SheafDiffusion):
    def __init__(self, edge_index, args):
        args['d'] = 1
        super(CustomGAT, self).__init__(edge_index, args)

        self.sheaf_learners = nn.ModuleList()
        for i in range(self.layers):
            self.sheaf_learners.append(AttentionSheafLearner(self.hidden_channels, self.d))

        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        if self.right_weights:
            for i in range(self.layers):
                self.lin_right_weights.append(nn.Linear(self.hidden_channels, self.hidden_channels, bias=False))
                nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        if self.left_weights:
            for i in range(self.layers):
                self.lin_left_weights.append(nn.Linear(self.final_d, self.final_d, bias=False))
                nn.init.eye_(self.lin_left_weights[-1].weight.data)
        
        self.laplacian_builder = lb.DiagLaplacianBuilder(self.graph_size, edge_index, d=self.d,
                                                        normalised=self.normalised,
                                                        deg_normalised=self.deg_normalised,
                                                        add_hp=self.add_hp, add_lp=self.add_lp)

        self.epsilons = nn.ParameterList()
        for i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.final_d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        if self.second_linear:
            self.lin12 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):

        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        if self.use_act:
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.second_linear:
            x = self.lin12(x)
        x = x.view(self.graph_size * self.final_d, -1)

        x0 = x
        for layer in range(self.layers):
            if layer == 0 or self.nonlinear:
                x_maps = F.dropout(x, p=self.dropout if layer > 0 else 0., training=self.training)
                maps = self.sheaf_learners[layer](x_maps.reshape(self.graph_size, -1), self.edge_index)
                L, trans_maps = self.laplacian_builder(maps)
                self.sheaf_learners[layer].set_L(trans_maps)

            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.left_weights:
                x = x.t().reshape(-1, self.final_d)
                x = self.lin_left_weights[layer](x)
                x = x.reshape(-1, self.graph_size * self.final_d).t()

            if self.right_weights:
                x = self.lin_right_weights[layer](x)

            x = torch_sparse.spmm(L[0], L[1], x.size(0), x.size(0), x)

            if self.use_act:
                x = F.elu(x)

            coeff = (1 + torch.tanh(self.epsilons[layer]).tile(self.graph_size, 1))
            x0 = coeff * x0 - x
            x = x0

        x = x.reshape(self.graph_size, -1)
        return self.lin2(x)