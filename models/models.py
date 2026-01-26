import torch
import torch.nn.functional as F
from torch.nn import (
    ModuleList, Linear, Sequential, ReLU, Parameter,
    BCEWithLogitsLoss, MSELoss, CrossEntropyLoss
)
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import dropout_edge
import numpy as np

def get_activation(name):
    activations = {
        'relu': F.relu,
        'elu': F.elu,
        'gelu': F.gelu,
        'leaky_relu': F.leaky_relu,
        'tanh': torch.tanh,
        'sigmoid': torch.sigmoid
    }
    return activations.get(name, F.relu)

#====================================================
#                   GRAPH MODELS
#====================================================

class GCN(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layers = ModuleList()
        self.dropout = args['dropout']
        self.act = get_activation(args['activation'])
        num_layers = args['num_layers']
        hidden_dim = args['hidden_dim']

        self.layers.append(GCNConv(args['num_features'], hidden_dim))

        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))

        if args['num_classes'] == 2:
            self.layers.append(GCNConv(hidden_dim, 1, aggr='mean'))
        else:
            self.layers.append(GCNConv(hidden_dim, args['num_classes'], aggr='mean'))

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class SAGE(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layers = ModuleList()
        self.dropout = args['dropout']
        self.act = get_activation(args['activation'])
        num_layers = args['num_layers']
        hidden_dim = args['hidden_dim']

        self.layers.append(SAGEConv(args['num_features'], hidden_dim, aggr='mean'))

        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim, aggr='mean'))

        if args['num_classes'] == 2:
            self.layers.append(SAGEConv(hidden_dim, 1, aggr='mean'))
        else:
            self.layers.append(SAGEConv(hidden_dim, args['num_classes'], aggr='mean'))

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class GAT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layers = ModuleList()
        self.dropout = args['dropout']
        self.act = get_activation(args['activation'])
        num_layers = args['num_layers']
        hidden_dim = args['hidden_dim']
        heads = args['heads']

        self.layers.append(GATConv(args['num_features'], hidden_dim, heads=heads, dropout=self.dropout))

        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=self.dropout))

        if args['num_classes'] == 2:
            self.layers.append(GATConv(hidden_dim * heads, 1, heads=1, concat=False, dropout=self.dropout))
        else:
            self.layers.append(GATConv(hidden_dim * heads, args['num_classes'], heads=1, concat=False, dropout=self.dropout))

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class H2GCN(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = args['dropout']
        self.act = get_activation(args['activation'])
        self.k = ['num_layers']
        self.hidden_dim = args['hidden_dim']
        
        self.embed_layer = Linear(args['num_features'], self.hidden_dim)
        self.prop = GCNConv(self.hidden_dim, self.hidden_dim, cached=True, normalize=True)

        if args['num_classes'] == 2:
            self.final_layer = Linear(self.hidden_dim * (self.k + 1), 1)
        else:
            self.final_layer = Linear(self.hidden_dim * (self.k + 1), args['num_classes'])

    def forward(self, x, edge_index):
        x = self.embed_layer(x)
        x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        rs = [x]
        
        edge_index_norm, edge_weight_norm = self.prop.gcn_norm(edge_index, num_nodes=x.size(0))

        for _ in range(self.k):
            last_r = rs[-1]
            next_r = self.prop.propagate(edge_index_norm, x=last_r, edge_weight=edge_weight_norm, size=None)
            rs.append(next_r)

        combined = torch.cat(rs, dim=1)
        out = self.final_layer(combined)
        
        return out
    

#====================================================
#                   FAIR MODELS
#====================================================

#----------------------------------------------------
#                   Pre-processors
#----------------------------------------------------


def get_undersampled_mask(y, z, train_mask):
    if isinstance(y, torch.Tensor): y = y.cpu().numpy()
    if isinstance(z, torch.Tensor): z = z.cpu().numpy()
    if isinstance(train_mask, torch.Tensor): train_mask = train_mask.cpu().numpy()

    groups = [(0, 0), (0, 1), (1, 0), (1, 1)]
    min_count = float('inf')
    
    indices_by_group = {}
    
    for (label, sens) in groups:
        idxs = np.where((y == label) & (z == sens) & train_mask)[0]
        indices_by_group[(label, sens)] = idxs
        if len(idxs) > 0 and len(idxs) < min_count:
            min_count = len(idxs)
            
    final_indices = []
    for group_idxs in indices_by_group.values():
        if len(group_idxs) > 0:
            chosen = np.random.choice(group_idxs, min_count, replace=False)
            final_indices.extend(chosen)
            
    new_mask = torch.zeros_like(torch.tensor(train_mask), dtype=torch.bool)
    new_mask[final_indices] = True
    return new_mask


class FairDrop(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.p = args['drop_edge_rate']
        self.target_homophily = args['target_homophily']

    def forward(self, edge_index, z):
        if not self.training:
            return edge_index

        row, col = edge_index
        z_source = z[row]
        z_target = z[col]
        
        if self.target_homophily:
            mask = z_source == z_target
        else:
            mask = z_source != z_target
            
        edge_index_masked = edge_index[:, ~mask]
        edge_index_candidates = edge_index[:, mask]
        
        edge_index_kept, _ = dropout_edge(edge_index_candidates, p=self.p)
        
        return torch.cat([edge_index_masked, edge_index_kept], dim=1)


#----------------------------------------------------
#                   In-processors
#----------------------------------------------------


class FairGNN(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = args['dropout']
        self.cov_coeff = args.get('cov_coeff', 1.0)
        hidden_dim = args['hidden_dim']
        num_layers = args['num_layers']

        self.feat_mlp = Sequential(
            Linear(args['num_features'], hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU()
        )

        if args['type'] == 'gcn':
            layer = GCNConv
        elif args['type'] == 'gat':
            layer = GATConv
        elif args['type'] == 'sage':
            layer = SAGEConv

        self.gnn_layers = ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(layer(hidden_dim, hidden_dim))

        if args['num_classes'] == 2:
            self.classifier = Linear(hidden_dim, 1)
        else:
            self.classifier = Linear(hidden_dim, args['num_classes'])

        self.adversary = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, 1)
        )
        
        self.losses = {
            'encoder': BCEWithLogitsLoss() if args['num_classes'] == 2 else CrossEntropyLoss(),
            'adversary': BCEWithLogitsLoss()
        }

        self.coeffs = {
            'cov': args['cov_coeff'],
            'adv': args['adv_coeff']
        }

    def get_modules(self):
        return ['adversary', 'encoder']

    def get_params(self, name):
        if name == 'encoder':
            return list(self.feat_mlp.parameters()) + \
                   list(self.gnn_layers.parameters()) + \
                   list(self.classifier.parameters())
        elif name == 'adversary':
            return list(self.adversary.parameters())
        else:
            raise ValueError(f"Unknown parameter group: {name}")

    def covariance_loss(self, y_pred, sensitive_attr):
        probs = torch.sigmoid(y_pred)
        s = sensitive_attr.float()
        mean_prob = torch.mean(probs)
        mean_s = torch.mean(s)
        cov = torch.abs(torch.mean((probs - mean_prob) * (s - mean_s)))
        return cov * self.coeffs['cov']

    def forward(self, data, name=None):
        x = data.x
        edge_index = data.edge_index
        
        x = self.feat_mlp(x)
        
        for layer in self.gnn_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        if name == 'encoder':
            return self.classifier(x)
        elif name == 'adversary':
            return self.adversary(x)
        else:
            return self.classifier(x), self.adversary(x)


def get_nifty_perturbation(data, sensitive_idx=0, p=0.1):
    x_aug = data.x.clone()
    x_aug[:, sensitive_idx] = 1 - x_aug[:, sensitive_idx]
    
    edge_index_aug, _ = dropout_edge(data.edge_index, p=p)
    
    return x_aug, edge_index_aug


class NIFTY(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder_layers = ModuleList()
        self.dropout = args['dropout']
        self.sim_coeff = args['coeff']
        
        num_layers = args['num_layers']
        hidden_dim = args['hidden_dim']

        if args['type'] == 'gcn':
            layer = GCNConv
        elif args['type'] == 'gat':
            layer = GATConv
        elif args['type'] == 'sage':
            layer = SAGEConv

        self.encoder_layers.append(layer(args['num_features'], hidden_dim))
        for _ in range(num_layers - 1):
            self.encoder_layers.append(layer(hidden_dim, hidden_dim))
        
        if args['num_classes'] == 2:
            self.classifier = Linear(hidden_dim, 1)
        else:
            self.classifier = Linear(hidden_dim, args['num_classes'])
        
        self.projector = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )

    def forward_encoder(self, x, edge_index):
        for layer in self.encoder_layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def cosine_similarity_loss(self, h1, h2):
        p1 = self.projector(h1)
        p2 = self.projector(h2)
        return -0.5 * (F.cosine_similarity(p1, h2.detach()).mean() + F.cosine_similarity(p2, h1.detach()).mean())

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index

        x_aug = None
        edge_index_aug = None

        if self.training:
            x_aug = data.x_aug
            edge_index_aug = data.edge_index_aug

        h = self.forward_encoder(x, edge_index)
        y_pred = self.classifier(h)
        
        aux_loss = torch.tensor(0.0, device=x.device)
        if x_aug is not None and edge_index_aug is not None:
            h_aug = self.forward_encoder(x_aug, edge_index_aug)
            sim_loss = self.cosine_similarity_loss(h, h_aug)
            aux_loss = self.sim_coeff * sim_loss
            
        return y_pred, aux_loss


class FairSIN(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = args['dropout']
        hidden_dim = args['hidden_dim']
        num_layers = args['num_layers']
        
        self.feat_mlp = Sequential(
            Linear(args['num_features'], hidden_dim),
            ReLU(),
            Linear(hidden_dim, args['num_features']),
            ReLU()
        )

        if args['type'] == 'gcn':
            layer = GCNConv
        elif args['type'] == 'gat':
            layer = GATConv
        elif args['type'] == 'sage':
            layer = SAGEConv

        self.gnn_layers = ModuleList()
        self.gnn_layers.append(layer(args['num_features'], hidden_dim))
        for _ in range(num_layers-1):
            self.gnn_layers.append(layer(hidden_dim, hidden_dim))

        if args['num_classes'] == 2:
            self.classifier = Linear(hidden_dim, 1)
        else:
            self.classifier = Linear(hidden_dim, args['num_classes'])

        self.adversary = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, 1)
        )
        
        self.losses = {
            'mlp': MSELoss(),
            'adversary': BCEWithLogitsLoss(),
            'encoder': BCEWithLogitsLoss() if args['num_classes'] == 2 else CrossEntropyLoss()
        }

        self.coeffs = {
            'adv': args['adv_coeff']
        }

    def get_modules(self):
        return ['mlp', 'adversary', 'encoder']

    def get_params(self, name):
        if name == 'mlp':
            return list(self.feat_mlp.parameters())
        elif name == 'encoder':
            return list(self.gnn_layers.parameters()) + list(self.classifier.parameters())
        elif name == 'adversary':
            return list(self.adversary.parameters())

    def forward(self, data, name=None):
        x = data.x
        edge_index = data.edge_index
        
        x_emb = self.feat_mlp(x)
        
        if name == 'mlp':
            return x_emb

        x_gnn = x_emb
        for layer in self.gnn_layers:
            x_gnn = layer(x_gnn, edge_index)
            x_gnn = F.relu(x_gnn)
            x_gnn = F.dropout(x_gnn, p=self.dropout, training=self.training)
            
        if name == 'encoder':
            return self.classifier(x_gnn)
        elif name == 'adversary':
            return self.adversary(x_gnn)
        else:
            return self.classifier(x_gnn), self.adversary(x_gnn)