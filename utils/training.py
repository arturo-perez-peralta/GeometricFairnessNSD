"""Training utils.
This module stores functions which compute all fairness metrics, both group and individual metrics. It also includes two methods that implement the training loop
for both the sheaf models and the sklearn models (xgboost, mlp and logistic regression). Finally, the SheafTrainer class allows the user for an easy way of training and evaluating 
fair sheaf models.

Methods
-------
Metric related:
    1. compute_acc (Accuracy).
    2. compute_idp (Independence).
    3. compute_sep (Separation).
    4. compute_suf (Sufficiency).
    5. compute_con (Consistency).
    6. compute_lip (Lipschitz constant).
    7. compute_ent (Generalized entropy with exponent 2).  

Training loops:
    1. train_model (train loop for sheaf model).
    2. train_sklearn (train loop for sklearn model).

Classes
-------
    1. SheafTrainer.
"""

#===========================================
#                   IMPORTS
#===========================================

import copy

# data and ML
import numpy as np
import pandas as pd
import torch
from aif360.datasets import BinaryLabelDataset

# Metrics
from torchmetrics import MetricCollection, MeanSquaredError
from torchmetrics.classification import (
    BinaryRecall, 
    BinaryF1Score, 
    BinaryAUROC, 
)

# knn for consistency
from scipy.spatial.distance import cdist        
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from models.models import get_nifty_perturbation

# Typing
from typing import Optional, Tuple, Dict, List
from sklearn.base import BaseEstimator
from torch_geometric.data import Data
from torch.nn import Module
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss as Loss


#===========================================
#                   METRICS
#===========================================


#-------------------------------------------------
#                   GROUP METRICS
#-------------------------------------------------


def compute_acc(eval_pred: torch.Tensor, eval_labels: torch.Tensor) -> float:
    """Compute accuracy rate from predictions and labels.
    """
    return (eval_pred == eval_labels).float().mean().item()


def compute_balacc(eval_pred: torch.Tensor, eval_labels: torch.Tensor, group_0: torch.Tensor, group_1: torch.Tensor) -> float:
    """Compute balanced accuracy rate from predictions and labels.
    """
    matches = (eval_pred == eval_labels).float()
    bal_acc = (matches[group_0].mean() + matches[group_1].mean())/2
    return bal_acc.item()


def compute_idp(eval_pred: torch.Tensor, group_0: torch.Tensor, group_1: torch.Tensor) -> float:
    """Compute independence rate from predictions and indices of privileged and unprivileged groups.
    """
    p_y1_a0 = (eval_pred[group_0] == 1).float().mean()  # Group proportions.
    p_y1_a1 = (eval_pred[group_1] == 1).float().mean()
    dpd = (p_y1_a1 - p_y1_a0).abs().item()              # Absolute difference of group proportions.
    return dpd


def compute_suf(eval_pred: torch.Tensor, group_00: torch.Tensor, group_01: torch.Tensor, group_10: torch.Tensor, group_11: torch.Tensor) -> float:
    """Compute sufficiency rate from predictions and indices of groups by privilege and true label value.
    """
    p_a0y0 = (eval_pred[group_00] == 1).float().mean()                  # Elements of confusion matrix.
    p_a1y0 = (eval_pred[group_10] == 1).float().mean()
    p_a0y1 = (eval_pred[group_01] == 0).float().mean()
    p_a1y1 = (eval_pred[group_11] == 0).float().mean()
    suf = 0.5*abs((p_a0y1 - p_a1y1).item() + (p_a0y0 - p_a1y0).item())  # Absolute value of the average of difference of group proportions by label.
    return suf


def compute_sep(eval_pred: torch.Tensor, eval_labels: torch.Tensor, eval_sensitive: torch.Tensor) -> float:
    """Compute separation from predictions, labels and sensitive variable.
    """

    group_01 = (eval_sensitive == 0) & (eval_pred == 1)         # Group labels.
    group_11 = (eval_sensitive == 1) & (eval_pred == 1)

    if group_01.any():
        p_a0y1 = (eval_labels[group_01] == 1).float().mean()    # Group proportions.
    else:
        p_a0y1 = torch.tensor(0.0, device=eval_labels.device)

    if group_11.any():
        p_a1y1 = (eval_labels[group_11] == 1).float().mean()
    else:
        p_a1y1 = torch.tensor(0.0, device=eval_labels.device)

    sep = (p_a0y1 - p_a1y1).abs().item()                        # Absolute difference of group proportions.
    return sep


def compute_group_metrics(
        eval_pred: torch.Tensor, eval_labels: torch.Tensor, eval_sensitive: torch.Tensor,
        group_0: torch.Tensor, group_1: torch.Tensor,
        group_00: torch.Tensor, group_01: torch.Tensor, group_10: torch.Tensor, group_11: torch.Tensor
        ) -> Tuple[float, float, float, float]:
    """Compute group metrics (accuracy, independence, sufficiency and separation) from predictions, labels, sensitive variables and group indices.
    """
    acc = compute_acc(eval_pred.flatten(), eval_labels.flatten())
    balacc = compute_balacc(eval_pred.flatten(), eval_labels.flatten(), group_0.flatten(), group_1.flatten())
    idp = compute_idp(eval_pred.flatten(), group_0.flatten(), group_1.flatten())
    suf = compute_suf(eval_pred.flatten(), group_00.flatten(), group_01.flatten(), group_10.flatten(), group_11.flatten())
    sep = compute_sep(eval_pred.flatten(), eval_labels.flatten(), eval_sensitive.flatten())
    return acc, balacc, idp, suf, sep


#-------------------------------------------------
#                   INDIVIDUAL METRICS
#-------------------------------------------------

def compute_con(pred: torch.Tensor, eval_pred: torch.Tensor, eval_indices: torch.Tensor, k: int) -> float:
    """Compute consistency.

    Parameters
    ----------
    pred : torch.Tensor
        Tensor with predictions for the whole dataset
    eval_pred : torch.Tensor
        Tensor with the predictions of the evaluation dataset
    eval_indices : torch.Tensor
        Tensor with the indices of the k-nearest neighbors of the evaluation observations
    k : int
        Value of k for the consistency
    """
    neighbor_preds = pred[eval_indices[:, 1:k+1]]                   # Obtain knn of eval_preds.
    avg_neighbor_preds = neighbor_preds.double().mean(dim = 1)      # Obtain average of the knn of each eval pred.
    con = 1 - (eval_pred - avg_neighbor_preds).abs().mean().item()  # Compute consistency.
    return con


def compute_lip(eval_pred: torch.Tensor, distance_inputs: torch.Tensor, epsilon: float = 1e-5) -> float:
    """Compute Lipschitz constant.

    Parameters
    ----------
    eval_pred : torch.Tensor
        Tensor with predictions for the whole dataset
    distance_inputs : torch.Tensor
        Tensor with the distances of the features.
    epsilon : float
        Value used to input zero distances on the feature distance.
    """
    distance_preds = torch.abs(eval_pred.unsqueeze(1) - eval_pred)                                                          # Compute distance between predictions.
    safe_distance_inputs = torch.where(distance_inputs != 0, distance_inputs, torch.full_like(distance_inputs, epsilon))    # Write epsilon on zero entries.
    ratios = (distance_preds/safe_distance_inputs).flatten()                                                                # Compute lipschitz constant.
    # lipscthiz = ratios.max().item()
    # lipschitz = torch.quantile(ratios.cpu(), 0.99).item()
    k = int(0.99 * ratios.numel())
    value, _ = torch.kthvalue(ratios, k)
    lipschitz = value.item()
    return lipschitz


def compute_ent(eval_pred: torch.Tensor, eval_labels: torch.Tensor) -> float:
    """Compute generalized entropy with α=2 from the predictions and labels of the evaluation set.
    """
    benefit = (eval_pred - eval_labels).abs()
    mean_benefit = benefit.double().mean().item()
    entropy = 0.5*((benefit/mean_benefit).pow(2) - 1).mean().item()
    return entropy


def compute_individual_metrics(
        pred: torch.Tensor, eval_pred: torch.Tensor, eval_labels: torch.Tensor, eval_indices: torch.Tensor, k: int, distance_inputs: torch.Tensor
        ) -> Tuple[float, float, float]:
    """Compute the previous individual metrics with their respective parameters.
    """
    con = compute_con(pred, eval_pred, eval_indices, k)
    lip = compute_lip(eval_pred, distance_inputs)
    ent = compute_ent(eval_pred, eval_labels)
    return con, lip, ent



def np_to_aif(features, labels, sens):
    df = pd.DataFrame(features, columns=[f'f{i}' for i in range(features.shape[1])])
    df['label'] = labels
    df['sensitive'] = sens
    return BinaryLabelDataset(
        favorable_label=1, unfavorable_label=0,
        df=df, label_names=['label'], protected_attribute_names=['sensitive']
    )


def train(
        data: Data,
        model: Module,
        optimizer: Optimizer,
        criterion: Loss,
        train_mask: torch.Tensor,
        eval_sensitive: torch.Tensor,
        eval_labels: torch.Tensor,
        eval_indices: torch.Tensor,
        eval_mask: torch.Tensor,
        k: int,
        device: torch.device = torch.device("cuda"),
        metric: str = "euclidean",
        full_trace: bool = True,
        final_trace: bool = True,
        metrics: bool = False,
        early_stop: bool = False,
        stop_epochs: int = 5,
        epochs: int = 10,
        mask: bool = False,
        distances: np.ndarray = None,
        fairdrop: Optional[Module] = None,
        uses_edge_index: bool = False,
        additional_loss: bool = False,
        nifty_flag: bool = False
        ) -> Dict[str, float]:
    """Function that runs the training loop on a given model.
    """

    # Variables storing metrics.
    acc_vec = []
    balacc_vec = []
    idp_vec = []
    suf_vec = []
    sep_vec = []
    con_vec = []
    lip_vec = []
    ent_vec = []
    f1_vec = []
    recall_vec = []
    auc_vec = []
    brier_vec = []

    best_val_loss = float('inf')
    patience_counter = 0

    # Compute distance matrix.
    if distances is None:
        dist = cdist(
            data.x[eval_mask].cpu().numpy(), data.x[eval_mask].cpu().numpy(), metric=metric
            )
    else:
        dist = distances[eval_mask.cpu().numpy(), :][:, eval_mask.cpu().numpy()]
    distance_inputs = torch.from_numpy(dist).to(device)

    # Send every tensor to the device.
    eval_sensitive = eval_sensitive.to(device)
    eval_labels = eval_labels.to(device)
    eval_indices = eval_indices.to(device)
    eval_labels = eval_labels.to(device)

    if mask:
        eval_obs = data.x[eval_mask].detach().clone()
        data.x[eval_mask] = torch.zeros_like(data.x[eval_mask])

    if isinstance(eval_indices, np.ndarray):
        eval_indices = torch.Tensor(eval_indices).long()

    data = data.to(device)
    data.y = data.y.float().squeeze()
    model = model.to(device)

    # Compute group indices.
    group_0 = (eval_sensitive == 0)                         
    group_1 = (eval_sensitive == 1)
    group_00 = (eval_sensitive == 0)*(eval_labels == 0)
    group_10 = (eval_sensitive == 1)*(eval_labels == 0)
    group_01 = (eval_sensitive == 0)*(eval_labels == 1)
    group_11 = (eval_sensitive == 1)*(eval_labels == 1)

    # Additional metrics.
    additional_metrics = MetricCollection({
        'recall': BinaryRecall(),
        'f1':     BinaryF1Score(),
        'auc':    BinaryAUROC(),
        'brier':  MeanSquaredError()
    }).to(device)

    # Training loop.
    model.train()
    for epoch in range(10*epochs+1):
        optimizer.zero_grad()

        if nifty_flag:
            data.x_aug, data.edge_index_aug = get_nifty_perturbation(data, sensitive_idx=data.A)
        if fairdrop is not None:
            fd_edge_index = fairdrop(data.edge_index, data.sensitive_attr)

        if uses_edge_index and not additional_loss:
            out = model(data.x, data.edge_index).squeeze()
        elif uses_edge_index and not additional_loss and fairdrop is not None:
            out = model(data.x, fd_edge_index).squeeze()
        elif uses_edge_index and additional_loss:
            out, aux_loss = model(data)
            out = out.squeeze()
            aux_loss = aux_loss.squeeze()
        else:
            out = model(data.x).squeeze()

        loss = criterion(out[train_mask], data.y[train_mask])
        if additional_loss:
            loss += aux_loss

        loss.backward()
        optimizer.step()

        # Evaluation step.
        with torch.no_grad():
            
            if mask:
                data.x[eval_mask] = eval_obs

            if uses_edge_index and not additional_loss:
                out_eval = model(data.x, data.edge_index).squeeze()
            elif uses_edge_index and additional_loss:
                out_eval = model(data)[0]
                out_eval = out_eval.squeeze()
            else:
                out_eval = model(data.x).squeeze()

            probs = out_eval.sigmoid()
            pred = (probs > 0.5).long()
            eval_pred = pred[eval_mask]
                
            if mask:
                data.x[eval_mask] = torch.zeros_like(data.x[eval_mask]) 
            
            eval_probs = probs[eval_mask]
            eval_pred = pred[eval_mask]

            # Compute the metrics we are interested in.
            if metrics:
                acc, balacc, idp, suf, sep = compute_group_metrics(
                    eval_pred, eval_labels, eval_sensitive, group_0, group_1, group_00, group_01, group_10, group_11
                    )
                con, lip, ent = compute_individual_metrics(
                    pred, eval_pred, eval_labels, eval_indices, k, distance_inputs
                    )
                additional = additional_metrics(eval_probs, eval_labels.float())
                
                acc_vec.append(acc)
                balacc_vec.append(balacc)
                idp_vec.append(idp)
                suf_vec.append(suf)
                sep_vec.append(sep)    
                con_vec.append(con)
                lip_vec.append(lip)
                ent_vec.append(ent)
                f1_vec.append(additional['f1'].item())    
                recall_vec.append(additional['recall'].item())
                brier_vec.append(additional['brier'].item())
                auc_vec.append(additional['auc'].item())

                # Trace.
                if epoch % 10 == 0 and full_trace:
                    print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Acc: {acc:.4f} | Bal. acc: {balacc:.4f} | Idp: {idp:.4f} | Suf: {suf:.4f} | Sep: {sep:.4f} | Con: {con:.4f} | Lip: {lip:.4f} | Ent: {ent:.4f}")

            else:
                # Trace.
                acc = compute_acc(eval_pred, eval_labels)
                if epoch % 10 == 0 and full_trace:
                    print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

            # Early stop mechanism.
            if early_stop:
                loss_val = criterion(out_eval[eval_mask], data.y[eval_mask])
    
                if loss_val < best_val_loss:
                    best_val_loss = loss_val
                    patience_counter = 0
                    best_model_weights = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                    
                    if patience_counter >= stop_epochs:
                        print(f"Early stopping triggered at epoch {epoch}")
                        if best_model_weights is not None:
                            model.load_state_dict(best_model_weights)
                        break

    # Restore best model
    if early_stop and best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    # Final evaluation.
    if mask:
        data.x[eval_mask] = eval_obs

    model.eval()
    probs = out.sigmoid()
    pred = (probs > 0.5).long()

    eval_probs = probs[eval_mask]
    eval_pred = pred[eval_mask]


    acc, balacc, idp, suf, sep = compute_group_metrics(
        eval_pred, eval_labels, eval_sensitive, group_0, group_1, group_00, group_01, group_10, group_11
        )
    con, lip, ent = compute_individual_metrics(
        pred, eval_pred, eval_labels, eval_indices, k, distance_inputs
        )
    
    additional = additional_metrics(eval_probs, eval_labels.float())
    f1 = additional['f1'].item()
    recall = additional['recall'].item()
    brier = additional['brier'].item()
    auc = additional['auc'].item()
    
    # Trace with the final results.
    if full_trace or final_trace:
        print(f"Acc: {acc:.4f} | Bal. acc: {balacc:.4f} | Idp: {idp:.4f} | Suf: {suf:.4f} | Sep: {sep:.4f} | Con: {con:.4f} | Lip: {lip:.4f} | Ent: {ent:.4f}")
    results_dict = {
        "acc": acc, "balacc": balacc, "ind": idp, "suf": suf, "sep": sep, "con": con, "lip": lip, "ent": ent, "f1": f1, "recall": recall, "brier": brier, "auc": auc,
    }

    # Return results
    if not metrics:
        return results_dict
    else:
        vec_dic = {
            "acc_vec": acc_vec, "balacc_vec": balacc_vec, "ind_vec": idp_vec, "suf_vec": suf_vec, "sep_vec": sep_vec, "con_vec": con_vec, "lip_vec": lip_vec, "ent_vec": ent_vec, "f1_vec": f1_vec, "recall_vec": recall_vec, "brier_vec": brier_vec, "auc_vec": auc_vec,
        }
        results_dict = {**results_dict, **vec_dic}
        return results_dict



def train_fair_modular(
        data,
        model,
        optimizer,
        criterion,
        train_mask,
        eval_sensitive,
        eval_labels,
        eval_indices,
        eval_mask,
        k,
        device=torch.device("cuda"),
        metric="euclidean",
        full_trace=True,
        final_trace=True,
        metrics=False,
        early_stop=False,
        stop_epochs=5,
        epochs=10,
        inner_epochs={'mlp': 1, 'adversary': 5, 'encoder': 1},
        mask=False,
        distances=None
        ):

    optim_class = type(optimizer)
    config = {k: v for k, v in optimizer.param_groups[0].items() if k != 'params'}
    
    opts = {}
    for name in model.get_modules():
        params = model.get_params(name)
        if params:
            opts[name] = optim_class(params, **config)

    acc_vec = []
    balacc_vec = []
    idp_vec = []
    suf_vec = []
    sep_vec = []
    con_vec = []
    lip_vec = []
    ent_vec = []
    f1_vec = []
    recall_vec = []
    auc_vec = []
    brier_vec = []

    patience_counter = 0
    best_val_loss = float('inf')
    best_model_weights = None

    if distances is None:
        dist = cdist(
            data.x[eval_mask].cpu().numpy(), data.x[eval_mask].cpu().numpy(), metric=metric
            )
    else:
        dist = distances[eval_mask.cpu().numpy(), :][:, eval_mask.cpu().numpy()]
    distance_inputs = torch.from_numpy(dist).to(device)

    eval_sensitive = eval_sensitive.to(device)
    eval_labels = eval_labels.to(device)
    eval_indices = eval_indices.to(device)

    if mask:
        eval_obs = data.x[eval_mask].detach().clone()
        data.x[eval_mask] = torch.zeros_like(data.x[eval_mask])

    if isinstance(eval_indices, np.ndarray):
        eval_indices = torch.Tensor(eval_indices).long()

    data = data.to(device)
    data.y = data.y.float().squeeze()
    model = model.to(device)

    group_0 = (eval_sensitive == 0)                         
    group_1 = (eval_sensitive == 1)
    group_00 = (eval_sensitive == 0)*(eval_labels == 0)
    group_10 = (eval_sensitive == 1)*(eval_labels == 0)
    group_01 = (eval_sensitive == 0)*(eval_labels == 1)
    group_11 = (eval_sensitive == 1)*(eval_labels == 1)

    # Additional metrics.
    additional_metrics = MetricCollection({
        'recall': BinaryRecall(),
        'f1':     BinaryF1Score(),
        'auc':    BinaryAUROC(),
        'brier':  MeanSquaredError()
    }).to(device)

    model.train()
    
    for epoch in range(10*epochs+1):
        
        for _ in range(inner_epochs.get('adversary', 1)):
            opts['adversary'].zero_grad()
            z_pred = model(data, name='adversary').squeeze()
            loss_adv = model.losses['adversary'](z_pred, data.sensitive_attr.float())
            loss_adv.backward()
            opts['adversary'].step()

        if 'mlp' in opts:
            for _ in range(inner_epochs.get('mlp', 1)):
                opts['mlp'].zero_grad()
                h_real = model(data, name='mlp')

                if not hasattr(data, 'x_diff'):
                    row, col = data.edge_index
                    data.x_diff = torch.zeros_like(data.x)
                    data.x_diff.index_add_(0, row, data.x[col])
                    deg = torch.zeros(data.x.size(0), device=device)
                    deg.index_add_(0, row, torch.ones_like(row, dtype=torch.float))
                    deg = deg.clamp(min=1).unsqueeze(1)
                    data.x_diff = data.x_diff / deg
                
                loss_mlp = model.losses['mlp'](h_real, data.x_diff)
                loss_mlp.backward()
                opts['mlp'].step()

        for _ in range(inner_epochs.get('encoder', 1)):
            opts['encoder'].zero_grad()
            
            y_pred, z_pred = model(data)
            out = y_pred.squeeze()
            
            loss_task = criterion(out[train_mask], data.y[train_mask])
            loss_adv_penalty = model.losses['adversary'](z_pred.squeeze(), data.sensitive_attr.float())
            
            total_loss = loss_task - (model.coeffs['adv'] * loss_adv_penalty)

            if hasattr(model, 'covariance_loss') and hasattr(model, 'cov_coeff'):
                cov_loss = model.covariance_loss(y_pred[train_mask], data.sensitive_attr[train_mask])
                total_loss += model.cov_coeff * cov_loss

            total_loss.backward()
            opts['encoder'].step()

        with torch.no_grad():
            if mask:
                data.x[eval_mask] = eval_obs
                out_eval = model(data, name='encoder').squeeze()
                data.x[eval_mask] = torch.zeros_like(data.x[eval_mask])
            else:
                out_eval = out.detach()
            prob = out_eval.sigmoid()
            pred = (prob > 0.5).long()
            eval_pred = pred[eval_mask]
            eval_probs = prob[eval_mask]

            if metrics:
                acc, balacc, idp, suf, sep = compute_group_metrics(
                    eval_pred, eval_labels, eval_sensitive, group_0, group_1, group_00, group_01, group_10, group_11
                    )
                con, lip, ent = compute_individual_metrics(
                    pred, eval_pred, eval_labels, eval_indices, k, distance_inputs
                    )
                additional = additional_metrics(eval_probs, eval_labels.float())

                acc_vec.append(acc)
                balacc_vec.append(balacc)
                idp_vec.append(idp)
                suf_vec.append(suf)
                sep_vec.append(sep)    
                con_vec.append(con)
                lip_vec.append(lip)
                ent_vec.append(ent)
                f1_vec.append(additional['f1'].item())
                recall_vec.append(additional['recall'].item())
                brier_vec.append(additional['brier'].item())
                auc_vec.append(additional['auc'].item())
                auc_vec.append(additional['auc'].item())

                if epoch % 10 == 0 and full_trace:
                    print(f"Epoch {epoch:03d} | Task: {loss_task.item():.4f} | Adv: {loss_adv.item():.4f} | Acc: {acc:.4f} | Bal. acc: {balacc:.4f} | Idp: {idp:.4f} | Suf: {suf:.4f} | Sep: {sep:.4f} | Con: {con:.4f} | Lip: {lip:.4f} | Ent: {ent:.4f}")

            else:
                acc = compute_acc(eval_pred, eval_labels)
                if epoch % 10 == 0 and full_trace:
                    print(f"Epoch {epoch:03d} | Loss: {total_loss.item():.4f} | Acc: {acc:.4f}")

            if early_stop:
                val_loss = criterion(out_eval[eval_mask], data.y[eval_mask])
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_weights = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= stop_epochs:
                        if full_trace:
                            print(f"Early stopping triggered at epoch {epoch}")
                        if best_model_weights is not None:
                            model.load_state_dict(best_model_weights)
                        break

    if early_stop and best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    if mask:
        data.x[eval_mask] = eval_obs

    model.eval()
    out = model(data, name='encoder').squeeze()
    probs = out.sigmoid()
    pred = (probs > 0.5).long()

    eval_probs = probs[eval_mask]
    eval_pred = pred[eval_mask]


    acc, balacc, idp, suf, sep = compute_group_metrics(
        eval_pred, eval_labels, eval_sensitive, group_0, group_1, group_00, group_01, group_10, group_11
        )
    con, lip, ent = compute_individual_metrics(
        pred, eval_pred, eval_labels, eval_indices, k, distance_inputs
        )
    
    additional = additional_metrics(eval_probs, eval_labels.float())
    f1 = additional['f1'].item()
    recall = additional['recall'].item()
    brier = additional['brier'].item()
    auc = additional['auc'].item()
    
    # Trace with the final results.
    if full_trace or final_trace:
        print(f"Acc: {acc:.4f} | Bal. acc: {balacc:.4f} | Idp: {idp:.4f} | Suf: {suf:.4f} | Sep: {sep:.4f} | Con: {con:.4f} | Lip: {lip:.4f} | Ent: {ent:.4f}")
    results_dict = {
        "acc": acc, "balacc": balacc, "ind": idp, "suf": suf, "sep": sep, "con": con, "lip": lip, "ent": ent, "f1": f1, "recall": recall, "brier": brier, "auc": auc,
    }

    # Return results
    if not metrics:
        return results_dict
    else:
        vec_dic = {
            "acc_vec": acc_vec, "balacc_vec": balacc_vec, "ind_vec": idp_vec, "suf_vec": suf_vec, "sep_vec": sep_vec, "con_vec": con_vec, "lip_vec": lip_vec, "ent_vec": ent_vec, "f1_vec": f1_vec, "recall_vec": recall_vec, "brier_vec": brier_vec, "auc_vec": auc_vec,
        }
        results_dict = {**results_dict, **vec_dic}
        return results_dict



def train_sklearn(
        clf: BaseEstimator,
        data: Data,
        sensitive: torch.Tensor,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        test_mask: torch.Tensor,
        k: int,
        indices: Optional[torch.Tensor] = None,
        metric: str = "euclidean",
        trace: bool = False,
        numeric_cols: Optional[List[int]] = None,
        adversarial: bool = False,
        Reweighing: Optional[BaseEstimator] = None,
        RejectOption: Optional[BaseEstimator] = None,
        ) -> Tuple[Dict[str, float], Dict[str, float], torch.Tensor]:
    """Function that runs the training loop on a sklearn model (In our case, XGBoost or MLP).
    """

    # Additional metrics.
    additional_metrics = MetricCollection({
        'recall': BinaryRecall(),
        'f1':     BinaryF1Score(),
        'auc':    BinaryAUROC(),
        'brier':  MeanSquaredError()
    }).to(data.x.device)
    
    # Pass tensors to numpy
    x = data.x
    val_sensitive = sensitive[val_mask]
    test_sensitive = sensitive[test_mask]

    test_x = data.x[test_mask]
    test_y = data.y[test_mask]

    val_x = data.x[val_mask]
    val_y = data.y[val_mask]

    train_x = data.x[train_mask]
    train_y = data.y[train_mask]

    x_np = x.numpy() if not x.is_cuda else x.cpu().numpy()
    train_x_np = train_x.numpy() if not train_x.is_cuda else train_x.cpu().numpy()
    val_x_np = val_x.numpy() if not val_x.is_cuda else val_x.cpu().numpy()
    test_x_np = test_x.numpy() if not test_x.is_cuda else test_x.cpu().numpy()
    train_y_np = train_y.numpy() if not train_y.is_cuda else train_y.cpu().numpy()
    val_y_np = val_y.numpy() if not val_y.is_cuda else val_y.cpu().numpy()

    if numeric_cols is not None:
        scaler = MinMaxScaler()
        # fit on training subset only
        scaler.fit(train_x_np[:, numeric_cols])

        train_x_np[:, numeric_cols] = scaler.transform(train_x_np[:, numeric_cols])
        val_x_np[:, numeric_cols]   = scaler.transform(val_x_np[:, numeric_cols])
        test_x_np[:, numeric_cols]  = scaler.transform(test_x_np[:, numeric_cols])
        x_np[:, numeric_cols]       = scaler.transform(x_np[:, numeric_cols])

    # Groups according to label and sensitive variable

    group_0_val = (val_sensitive == 0)
    group_1_val = (val_sensitive == 1)

    group_00_val = (val_sensitive == 0)*(val_y == 0)
    group_10_val = (val_sensitive == 1)*(val_y == 0)
    group_01_val = (val_sensitive == 0)*(val_y == 1)
    group_11_val = (val_sensitive == 1)*(val_y == 1)

    group_0_test = (test_sensitive == 0)
    group_1_test = (test_sensitive == 1)

    group_00_test = (test_sensitive == 0)*(test_y == 0)
    group_10_test = (test_sensitive == 1)*(test_y == 0)
    group_01_test = (test_sensitive == 0)*(test_y == 1)
    group_11_test = (test_sensitive == 1)*(test_y == 1)

    # knn
    if indices is None:
        nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric)  # +1 node itself
        nbrs.fit(x_np)
        _, indices = nbrs.kneighbors(x_np)
        indices = torch.from_numpy(indices).to(data.x.device)
    val_indices = indices[val_mask]
    test_indices = indices[test_mask]

    differences_val = data.x[val_mask].unsqueeze(0) - data.x[val_mask].unsqueeze(1)
    distance_inputs_val = torch.norm(differences_val, p=2, dim=-1)

    differences_test = data.x[test_mask].unsqueeze(0) - data.x[test_mask].unsqueeze(1)
    distance_inputs_test = torch.norm(differences_test, p=2, dim=-1)

    fit_params = {}

    if Reweighing is not None:
        sens_train = sensitive[train_mask].cpu().numpy()
        ds_train = np_to_aif(train_x_np, train_y_np, sens_train)
        
        Reweighing.fit(ds_train)
        ds_transf = Reweighing.transform(ds_train)
        fit_params['sample_weight'] = ds_transf.instance_weights.ravel()

    # Fit, predict, evaluate
    if adversarial:
        sens_train = sensitive[train_mask].cpu().numpy()
        ds_train = np_to_aif(train_x_np, train_y_np, sens_train)
        clf.fit(ds_train)
    else:
        clf.fit(train_x_np, train_y_np, **fit_params)

    if RejectOption is not None:
        sens_val = sensitive[val_mask].cpu().numpy()
        ds_val_gt = np_to_aif(val_x_np, val_y_np, sens_val)
        val_probs = clf.predict_proba(val_x_np)[:, 1]

        ds_val_pred = ds_val_gt.copy()
        ds_val_pred.scores = val_probs.reshape(-1, 1)
        ds_val_pred.labels = (val_probs > 0.5).astype(float).reshape(-1, 1)

        RejectOption = RejectOption.fit(ds_val_gt, ds_val_pred)

        full_probs = clf.predict_proba(x_np)[:, 1]

        ds_full = np_to_aif(x_np, np.zeros(len(x_np)), sensitive.cpu().numpy())
        ds_full.scores = full_probs.reshape(-1, 1)
        ds_full.labels = (full_probs > 0.5).astype(float).reshape(-1, 1)
        
        ds_full_corr = RejectOption.predict(ds_full)
        full_preds = torch.Tensor(ds_full_corr.labels.ravel()).to(data.x.device)
        full_probs = torch.Tensor(ds_full.scores.ravel()).to(data.x.device)
        
        val_pred = full_preds[val_mask]
        test_pred = full_preds[test_mask]
        val_prob = full_probs[val_mask]
        test_prob = full_probs[test_mask]
        pred = full_preds

    else:
        if adversarial:
            ds_full = np_to_aif(x_np, np.zeros(len(x_np)), sensitive.cpu().numpy())
            
            ds_full_pred = clf.predict(ds_full)
            
            pred = torch.Tensor(ds_full_pred.labels.ravel()).to(data.x.device)
            prob = torch.Tensor(ds_full_pred.scores.ravel()).to(data.x.device)

            val_pred = pred[val_mask]
            test_pred = pred[test_mask]

            val_prob = prob[val_mask]
            test_prob = prob[test_mask]
        else:
            pred = torch.Tensor(clf.predict(x_np)).to(data.x.device)
            val_pred = torch.Tensor(clf.predict(val_x_np)).to(data.x.device)
            test_pred = torch.Tensor(clf.predict(test_x_np)).to(data.x.device)
            val_prob = torch.Tensor(clf.predict_proba(val_x_np)[:, 1]).to(data.x.device)
            test_prob = torch.Tensor(clf.predict_proba(test_x_np)[:, 1]).to(data.x.device)

    acc, balacc, idp, suf, sep = compute_group_metrics(
        val_pred, val_y, val_sensitive, group_0_val, group_1_val, group_00_val, group_01_val, group_10_val, group_11_val
        )
    
    con, lip, ent = compute_individual_metrics(
        pred, val_pred, val_y, val_indices, k, distance_inputs_val
        )
    

    additional = additional_metrics(val_prob, val_y.float())
    f1 = additional['f1'].item()
    recall = additional['recall'].item()
    brier = additional['brier'].item()
    auc = additional['auc'].item()
    
    # Trace with the final results.
    val_dict = {
        "acc": acc, "balacc": balacc, "ind": idp, "suf": suf, "sep": sep, "con": con, "lip": lip, "ent": ent, "f1": f1, "recall": recall, "brier": brier, "auc": auc,
    }
    if trace:
        print("Results for validation set")
        print(f"Acc: {acc:.4f} | Bal. acc: {balacc:.4f} | Idp: {idp:.4f} | Suf: {suf:.4f} | Sep: {sep:.4f} | Con: {con:.4f} | Lip: {lip:.4f} | Ent: {ent:.4f}")
    

    acc, balacc, idp, suf, sep = compute_group_metrics(
        test_pred, test_y, test_sensitive, group_0_test, group_1_test, group_00_test, group_01_test, group_10_test, group_11_test
        )
    
    con, lip, ent = compute_individual_metrics(
        pred, test_pred, test_y, test_indices, k, distance_inputs_test
        )
    
    additional = additional_metrics(test_prob, test_y.float())
    f1 = additional['f1'].item()
    recall = additional['recall'].item()
    brier = additional['brier'].item()
    auc = additional['auc'].item()
    
    if numeric_cols is not None:
        x_np[:, numeric_cols] = scaler.inverse_transform(x_np[:, numeric_cols])
        data.x = torch.from_numpy(x_np).to(data.x.device)

    test_dict = {
        "acc": acc, "balacc": balacc, "ind": idp, "suf": suf, "sep": sep, "con": con, "lip": lip, "ent": ent, "f1": f1, "recall": recall, "brier": brier, "auc": auc,
        }
    if trace:
        print("Results for test set")
        print(f"Acc: {acc:.4f} | Bal. acc: {balacc:.4f} | Idp: {idp:.4f} | Suf: {suf:.4f} | Sep: {sep:.4f} | Con: {con:.4f} | Lip: {lip:.4f} | Ent: {ent:.4f}")

    return val_dict, test_dict, indices




class SheafTrainer:
    """Class used to train the sheaf models.
    """

    def __init__(
            self,
            data: Data,
            model: Module,
            optimizer: Optimizer,
            criterion: Loss,
            sensitive: torch.Tensor,
            k: int = 5,
            device: torch.device = torch.device("cuda"),
            metric: str = 'euclidean',
            numeric: np.ndarray = None,
            precomputed: bool = False,
            distances: np.ndarray = None,
            uses_edge_index: bool = False,
            additional_loss: bool = False,
            nifty_flag: bool = False,
            modular_flag: bool = False,
            ):
        """Store parameters.
        """
        self.data = copy.deepcopy(data)
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.metric = metric
        self.k = k
        self.sensitive = sensitive
        self.numeric = numeric
        self.precomputed = precomputed
        self.distances = distances
        self.uses_edge_index = uses_edge_index
        self.additional_loss = additional_loss
        self.nifty_flag = nifty_flag
        self.modular_flag = modular_flag
    

    def _get_train_utils(self):
        """Auxiliary private method that computes the knn indices matrix and stores the indices of the val, test, train sets, their masks and their labels.
        """
        x_np = self.data.x.cpu()

        if self.precomputed:
            nbrs = NearestNeighbors(n_neighbors=self.k+1, metric="precomputed")
            nbrs.fit(self.distances)
            _, indices = nbrs.kneighbors(self.distances)
        else:
            nbrs = NearestNeighbors(n_neighbors=self.k+1, metric=self.metric)  # +1 node itself
            nbrs.fit(x_np) 
            _, indices = nbrs.kneighbors(x_np)

        indices = torch.tensor(indices, device = self.device)
        self.trn_mask = self.data.train_mask
        self.test_mask = self.data.test_mask
        self.val_mask = self.data.val_mask

        self.test_indices = indices[self.test_mask]
        self.val_indices = indices[self.val_mask]

        self.privileged_indices = torch.where(self.sensitive == 1)[0]
        self.sensitive_indices = torch.where(self.sensitive == 0)[0]
        self.test_sensitive = self.sensitive[self.test_mask]
        self.val_sensitive = self.sensitive[self.val_mask]

        self.test_labels = self.data.y[self.test_mask]
        self.val_labels = self.data.y[self.val_mask]

        if self.numeric is not None:
            x_numeric = self.data.x[:, self.numeric]
            self.min_vals, _ = torch.min(x_numeric[self.trn_mask], dim=0)
            self.max_vals, _ = torch.max(x_numeric[self.trn_mask], dim=0)
            self.range_vals = self.max_vals - self.min_vals
            self.range_vals[self.range_vals == 0] = 1.0
            self.data.x[:, self.numeric] = (x_numeric - self.min_vals[None, :]) / self.range_vals[None, :]

        return self
    
    def train_model(
        self,
        eval: str = 'test',
        train_mask: Optional[torch.Tensor] = None,
        val_mask: Optional[torch.Tensor] = None,
        test_mask: Optional[torch.Tensor] = None,
        full_trace: bool = True,
        final_trace: bool = True,
        metrics: bool = False,
        early_stop: bool = False,
        stop_epochs: int = 5,
        epochs: int = 10,
        fairdrop: Optional[Module] = None,
        ) -> Dict[str, float]:
        """Runs the train function on the model.
        """
        if train_mask is not None:
            self.data.train_mask = train_mask
        if val_mask is not None:
            self.data.val_mask = val_mask
        if test_mask is not None:
            self.data.test_mask = test_mask

        self._get_train_utils()

        if eval.lower().startswith('test'):
            eval_sensitive = self.test_sensitive
            eval_labels = self.test_labels
            eval_indices = self.test_indices
            eval_mask = self.test_mask
        elif eval.lower().startswith('val'):
            eval_sensitive = self.val_sensitive
            eval_labels = self.val_labels
            eval_indices = self.val_indices
            eval_mask = self.val_mask
        else:
            raise AttributeError

        if not self.modular_flag:
            return train(
                self.data, self.model, self.optimizer(self.model.parameters()), self.criterion, self.data.train_mask, eval_sensitive, eval_labels, eval_indices, eval_mask,
                k=self.k, metric=self.metric, device=self.device, full_trace=full_trace, metrics=metrics, early_stop=early_stop, stop_epochs=stop_epochs,
                final_trace=final_trace, epochs=epochs, distances=self.distances, uses_edge_index=self.uses_edge_index, additional_loss=self.additional_loss,
                nifty_flag=self.nifty_flag, fairdrop = fairdrop
            )
        else:
            return train_fair_modular(
                self.data, self.model, self.optimizer(self.model.parameters()), self.criterion, self.data.train_mask, eval_sensitive, eval_labels, eval_indices, eval_mask,
                k=self.k, metric=self.metric, device=self.device, full_trace=full_trace, metrics=metrics, early_stop=early_stop, stop_epochs=stop_epochs,
                final_trace=final_trace, epochs=epochs, distances=self.distances, mask=True,
                inner_epochs = {'mlp': 1, 'adversary': 5, 'encoder': 1}
            )

    def evaluate(self, test = True) -> Dict[str, float]:
        """Evaluate the trained model.
        """

        self.model.eval()
        # Additional metrics.
        additional_metrics = MetricCollection({
            'recall': BinaryRecall(),
            'f1':     BinaryF1Score(),
            'auc':    BinaryAUROC(),
            'brier':  MeanSquaredError()
        }).to(self.device)

        # Send tensors to the device.
        if test:
            eval_sensitive = self.test_sensitive.to(self.device)
            eval_labels   = self.test_labels.to(self.device)
            eval_indices  = self.test_indices.to(self.device)
            eval_mask     = self.test_mask
        else:
            eval_sensitive = self.val_sensitive.to(self.device)
            eval_labels   = self.val_labels.to(self.device)
            eval_indices  = self.val_indices.to(self.device)
            eval_mask     = self.val_mask

        # Obtain predictions.
        with torch.no_grad():
            if self.uses_edge_index and not (self.additional_loss or self.modular_flag):
                out = self.model(self.data.x, self.data.edge_index).squeeze()
            elif self.uses_edge_index and (self.additional_loss or self.modular_flag):
                out = self.model(self.data)[0]
                out = out.squeeze()
            else:
                out = self.model(self.data.x).squeeze()
            prob = out.sigmoid()
            pred = (prob > 0.5).long()
            eval_pred = pred[eval_mask].squeeze()
            eval_prob = prob[eval_mask].squeeze()

        # Compute distance matrix.
        if self.distances is None:
            dist = cdist(self.data.x[eval_mask].cpu().numpy(), self.data.x[eval_mask].cpu().numpy(), metric=self.metric)
        else:
            dist = self.distances[eval_mask.cpu().numpy(), :][:, eval_mask.cpu().numpy()]
        distance_inputs = torch.from_numpy(dist).to(self.device)

        # Group indices.
        group_0 = (eval_sensitive == 0)
        group_1 = (eval_sensitive == 1)
        group_00 = (eval_sensitive == 0) & (eval_labels == 0)
        group_10 = (eval_sensitive == 1) & (eval_labels == 0)
        group_01 = (eval_sensitive == 0) & (eval_labels == 1)
        group_11 = (eval_sensitive == 1) & (eval_labels == 1)

        # Compute metrics.
        acc, balacc, idp, suf, sep = compute_group_metrics(
            eval_pred, eval_labels, eval_sensitive, group_0, group_1, group_00, group_01, group_10, group_11
            )
        con, lip, ent = compute_individual_metrics(
            pred, eval_pred, eval_labels, eval_indices, self.k, distance_inputs
            )
        
        additional = additional_metrics(eval_prob, eval_labels.float())
        f1 = additional['f1'].item()
        recall = additional['recall'].item()
        brier = additional['brier'].item()
        auc = additional['auc'].item()
        
        if self.numeric is not None:
            self.data.x[:, self.numeric] = self.data.x[:, self.numeric] * self.range_vals + self.min_vals
    
        # Print them.
        results_dict = {
            "acc": acc, "balacc": balacc, "ind": idp, "suf": suf, "sep": sep, "con": con, "lip": lip, "ent": ent, "f1": f1, "recall": recall, "brier": brier, "auc": auc,
        }

        print(f"Acc: {acc:.4f} | Bal. acc: {balacc:.4f} | Idp: {idp:.4f} | Suf: {suf:.4f} | Sep: {sep:.4f} | Con: {con:.4f} | Lip: {lip:.4f} | Ent: {ent:.4f}")

        return results_dict