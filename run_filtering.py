#============================================================================
#                           IMPORTS AND LIBRARIES
#============================================================================

import random
from typing import Dict

# Machine learning and data science
import numpy as np
import pandas as pd

# Pytorch
import torch
import torch.optim as optim
from torch_geometric.utils import remove_self_loops, to_undirected

# Custom imports
from utils.data_processing import remove_duplicate_edges, data_loader
from utils.constants import (
    additional_loss, sklearn_models, filters, graph_datasets, graph_models, modular
)
from utils.training import SheafTrainer, train_sklearn

PATIENCE = 50
REWEIGHTING = True

def choose_criterion(rw, data, train_mask, device):
    if not rw:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        num_pos = data.y[train_mask].sum()
        num_neg = (data.y[train_mask] == 0).sum()
        weight_value = num_neg / (num_pos + 1e-6)
        pos_weight = torch.tensor([weight_value], device=device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return criterion

#============================================================================
#                           EXPERIMENT FUNCTION
#============================================================================

def run_experiment(
        model_str: str,
        model_args: Dict,
        params: Dict,
        compare: bool = False,
    ):

    #============================================================================
    #                           CONSTANTS
    #============================================================================

    seed = params['seed']
    optimizer = lambda x: optim.AdamW(
        x,
        lr=params['lr'],
        betas=(params['beta1'], params['beta2']),
        weight_decay=params['weight_decay']
        )

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    ablation_flag = model_str in ['custgat', 'custgcn', 'vectgat', 'vectgcn', 'specgat', 'specgcn', 'polygat', 'polygcn']

    sklearn_flag = model_str in sklearn_models
    graph_model_flag = model_str in graph_models and not ablation_flag
    additional_loss_flag = model_str in additional_loss
    nifty_flag = model_str == 'nifty'
    adversarial_flag = model_str == 'adversarial'
    fairdrop_flag = 'fd' in model_str
    undersampling_flag = 'us' in model_str
    graph_dataset_flag = params['name'] in graph_datasets
    modular_flag = model_str in modular

    if sklearn_flag:
        device = torch.device("cpu")
    else:
        device = torch.device(params['device'])
    processor = data_loader(params['name'], device)

    #============================================================================
    #                           DATA PROCESSING
    #============================================================================

    df = processor.df
    nobs, nfeat = df.values.shape
    nfeat -= 1
    sensitive_var = processor.sensitive
    sensitive = df[[sensitive_var]]

    if not params['kfold']:
        data, edges = (processor
                    .create_graph(unit_ball=params['distance'], k_neigh=params['knn'], sub=params['subset'], star=params['star'], **params['kwargs'])
                    .split(train_ratio=params['train_ratio'], val_ratio=params['val_ratio'], test_ratio=params['test_ratio'])
                    .build())
    else:
        data, edges = (processor
                    .create_graph(unit_ball=params['distance'], k_neigh=params['knn'], sub=params['subset'], star=params['star'], **params['kwargs'])
                    .build())
        folds = processor.cross_validate(k_folds=params['folds'], stratify=data.y)


    edges = {k: remove_duplicate_edges(edges[k]) for k in edges.keys()}

    A = df.drop(columns=['Target']).columns.get_loc(sensitive_var)
    sensitive_col = data.x[:, A]  # [num_nodes]

    numeric_idx = [df.drop(columns=['Target']).columns.get_loc(col) for col in processor.numeric]

    data.x = data.x.to(device)
    data.y = data.y.to(device)
    edge_index = data.edge_index.to(device)

    # Transform to undirected graph
    if graph_dataset_flag:
        edge_index = remove_duplicate_edges(edge_index)
        edge_index = to_undirected(edge_index)
        data.edge_index = edge_index
    else:
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = to_undirected(edge_index)
        data.edge_index = edge_index

    sensitive_col = sensitive_col.to(device)
    s = 1.0 * (sensitive_col == 1) - 1.0 * (sensitive_col == 0)

    #============================================================================
    #                           MODEL SELECTION
    #============================================================================

    preprocessor = None
    postprocessor = None
    fairdrop = None

    #--------------------------------------------------
    #                       AIF360
    #--------------------------------------------------

    if 'rw' in model_str:
        from aif360.algorithms.preprocessing import Reweighing
        preprocessor = Reweighing(
            unprivileged_groups=[{'sensitive': 0}],
            privileged_groups=[{'sensitive': 1}]
            )
    elif adversarial_flag:
        from aif360.algorithms.inprocessing import AdversarialDebiasing
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()

        def modelClass(x):
            sess = tf.Session()
            return AdversarialDebiasing(
                unprivileged_groups=[{'sensitive': 0}],
                privileged_groups=[{'sensitive': 1}],
                sess=sess,
                **x
            )

    elif 'ro' in model_str:
        from aif360.algorithms.postprocessing import RejectOptionClassification
        postprocessor = RejectOptionClassification(
            unprivileged_groups=[{'sensitive': 0}], 
            privileged_groups=[{'sensitive': 1}],
            metric_name=model_args["metric_name"],
            metric_ub=model_args["metric_ub"], metric_lb=model_args["metric_lb"]
            )
        metric_name = model_args["metric_name"]
        metric_ub = model_args["metric_ub"]
        metric_lb = model_args["metric_lb"]
        del model_args["metric_name"]
        del model_args["metric_ub"]
        del model_args["metric_lb"]

    #--------------------------------------------------
    #             GRAPH PRE PROCESSING
    #--------------------------------------------------

    if undersampling_flag:
        from models.models import get_undersampled_mask

    elif fairdrop_flag:
        from models.models import FairDrop
        data.sensitive_attr = sensitive_col

    #--------------------------------------------------
    #                       SKLEARN
    #--------------------------------------------------

    if 'logreg' in model_str:
        from sklearn.linear_model import LogisticRegression
        model_args = {
            'C': model_args['C'], 'solver': model_args['solver'], 'max_iter': model_args['max_iter']
            }
        modelClass = LogisticRegression

    elif 'xgb' in model_str:
        from xgboost import XGBClassifier
        modelClass = XGBClassifier

    elif 'lgbm' in model_str:
        from lightgbm import LGBMClassifier
        modelClass = LGBMClassifier

    elif 'mlp' in model_str:
        from sklearn.neural_network import MLPClassifier
        modelClass = MLPClassifier


    #--------------------------------------------------
    #                   GRAPH MODELS
    #--------------------------------------------------
    
    elif 'gcn' in model_str and not ablation_flag:
        from models.models import GCN
        modelClass = GCN

    elif 'gat' in model_str and not ablation_flag:
        from models.models import GAT
        modelClass = GAT

    elif 'sage' in model_str:
        from models.models import SAGE
        modelClass = SAGE

    elif 'h2gcn' in model_str:
        from models.models import H2GCN
        modelClass = H2GCN

    elif model_str == 'fairgnn':
        from models.models import FairGNN
        data.sensitive_attr = sensitive_col
        modelClass = FairGNN

    elif model_str == 'nifty':
        from models.models import NIFTY
        data.A = A
        modelClass = NIFTY

    elif model_str == 'fairsin':
        from models.models import FairSIN
        data.sensitive_attr = sensitive_col
        modelClass = FairSIN

#    elif model_str == 'bind':
#        from models.models import BINDStructureLearner
#        modelClass = BINDStructureLearner

    #--------------------------------------------------
    #                       NSD
    #--------------------------------------------------
    elif model_str == 'polydia':
        from models.fair_filter import PolyFilterDiscreteDiagSheafDiffusion
        modelClass = PolyFilterDiscreteDiagSheafDiffusion

    elif model_str == 'polybun':
        from models.fair_filter import PolyFilterDiscreteBundleSheafDiffusion
        modelClass = PolyFilterDiscreteBundleSheafDiffusion

    elif model_str == 'polygen':
        from models.fair_filter import PolyFilterDiscreteGeneralSheafDiffusion
        modelClass = PolyFilterDiscreteGeneralSheafDiffusion

    elif model_str == 'specdia':
        from models.fair_filter import SpectrumFilterDiscreteDiagSheafDiffusion
        modelClass = SpectrumFilterDiscreteDiagSheafDiffusion
#        model_args["vectors"] = sensitive_col

    elif model_str == 'specbun':
        from models.fair_filter import SpectrumFilterDiscreteBundleSheafDiffusion
        modelClass = SpectrumFilterDiscreteBundleSheafDiffusion
#        model_args["vectors"] = sensitive_col

    elif model_str == 'specgen':
        from models.fair_filter import SpectrumFilterDiscreteGeneralSheafDiffusion
        modelClass = SpectrumFilterDiscreteGeneralSheafDiffusion
#        model_args["vectors"] = sensitive_col

    elif model_str == 'vectdia':
        from models.fair_filter import VectorFilterDiscreteDiagSheafDiffusion
        modelClass = VectorFilterDiscreteDiagSheafDiffusion
        model_args["vectors"] = sensitive_col

    elif model_str == 'vectbun':
        from models.fair_filter import VectorFilterDiscreteBundleSheafDiffusion
        modelClass = VectorFilterDiscreteBundleSheafDiffusion
        model_args["vectors"] = sensitive_col

    elif model_str == 'vectgen':
        from models.fair_filter import VectorFilterDiscreteGeneralSheafDiffusion
        modelClass = VectorFilterDiscreteGeneralSheafDiffusion
        model_args["vectors"] = sensitive_col

    elif model_str == 'dia':
        from NSD.models.disc_models import DiscreteDiagSheafDiffusion
        modelClass = DiscreteDiagSheafDiffusion

    elif model_str == 'bun':
        from NSD.models.disc_models import DiscreteBundleSheafDiffusion
        modelClass = DiscreteBundleSheafDiffusion

    elif model_str == 'gen':
        from NSD.models.disc_models import DiscreteGeneralSheafDiffusion
        modelClass = DiscreteGeneralSheafDiffusion

    #--------------------------------------------------
    #                       Custom GNNs
    #--------------------------------------------------

    elif model_str == 'polygat':
        from models.fair_filter import PolyFilterGAT
        modelClass = PolyFilterGAT

    elif model_str == 'polygcn':
        from models.fair_filter import PolyFilterGCN
        modelClass = PolyFilterGCN

    elif model_str == 'specgat':
        from models.fair_filter import SpectrumFilterGAT
        modelClass = SpectrumFilterGAT

    elif model_str == 'specgcn':
        from models.fair_filter import SpectrumFilterGCN
        modelClass = SpectrumFilterGCN

    elif model_str == 'vectgat':
        from models.fair_filter import VectorFilterGAT
        modelClass = VectorFilterGAT
        model_args["vectors"] = sensitive_col

    elif model_str == 'vectgcn':
        from models.fair_filter import VectorFilterGCN
        modelClass = VectorFilterGCN
        model_args["vectors"] = sensitive_col

    elif model_str == 'custgat':
        from models.fair_filter import CustomGAT
        modelClass = CustomGAT

    elif model_str == 'custgcn':
        from models.fair_filter import CustomGCN
        modelClass = CustomGCN

    #============================================================================
    #                           TRAINING
    #============================================================================

    results = {}

    # test-val-train split for graph models
    if not params['kfold'] and not sklearn_flag:

        model_args.update({
            'num_features': nfeat,
            'input_dim': nfeat,
            'num_classes': 2,
            'output_dim': 1,
            'num_nodes': nobs,
            'graph_size': nobs,
            'device': device
        })

        if fairdrop_flag:
            fd = FairDrop(model_args).to(device)
            fd.train()

        if undersampling_flag:
            data.train_mask = get_undersampled_mask(data.y, sensitive_col, data.train_mask)

        print(f'Training\n' + '-'*20)
        if graph_model_flag:
            model = modelClass(model_args).to(device)
        else:
            model = modelClass(edge_index, model_args).to(device)

        criterion = choose_criterion(REWEIGHTING, data, data.train_mask, device)

        trainer = SheafTrainer(
            data,
            model,
            optimizer,
            criterion,
            sensitive_col,
            device = device,
            metric = params['metric'],
            k = params['k'],
            precomputed = False,
            distances = None,
            uses_edge_index = graph_model_flag,
            additional_loss = additional_loss_flag,
            nifty_flag = nifty_flag,
            modular_flag = modular_flag,
        )
        print_str = 'Results for validation set - UNFILTERED' if model_str in filters else 'Results for validation set' 
        print(print_str)
        val_results = trainer.train_model(
                train_mask=data.train_mask,
                val_mask=data.val_mask,
                eval = 'val',
                full_trace = params['trace'],
                epochs = params['epochs'],
                fairdrop = fairdrop,
                early_stop = params['early_stop'],
                stop_epochs = params['patience'],
                )
        print_str = 'Results for test set - UNFILTERED' if model_str in filters else 'Results for test set' 
        print(print_str)
        test_results = trainer.evaluate()   
        print('='*20)

        if compare:
            results['unfiltered'] = {'val': val_results, 'test': test_results}

        if model_str in filters and 'poly' in model_str:
            z = s.repeat_interleave(model.final_d)
            model.set_filter(
                z,
                k_eig=model_args['k_eig'], select=model_args['k_select'],
                K_cheb=model_args['K_cheb'],
                gamma=model_args['gamma'], sigma=model_args['sigma'], 
                n_points=10000000, mode="topk", nonzero=True, weighted=True, epsilon=0.01
                )
        
        elif model_str in filters and 'vect' in model_str:
            model.set_filter(val=True)

        elif model_str in filters and 'spec' in model_str:
            z = s.repeat_interleave(model.final_d)
            model.set_filter(
                z,
                f=model_args['f'],
                k_eig=model_args['k_eig'],
                select=model_args['k_select'],
                mode="topk",
                weighted=True
                )

        elif model_str not in filters:
            results = {'val': val_results, 'test': test_results}

        if model_str in filters:
            print('Results for validation set - FILTERED')
            val_results = trainer.evaluate(test=False)   
            print('Results for test set - FILTERED')
            test_results = trainer.evaluate()   
            print('='*20)
            if compare:
                results['filtered'] = {'val': val_results, 'test': test_results}
            else:
                results = {'val': val_results, 'test': test_results}


    # kfold cv training loop for graph models.
    elif params['kfold'] and not sklearn_flag:

        model_args.update({
            'num_features': nfeat,
            'input_dim': nfeat,
            'num_classes': 2,
            'output_dim': 1,
            'num_nodes': nobs,
            'graph_size': nobs,
            'device': device
        })

        all_val_results = []
        all_test_results = []

        for i, (train_mask, val_mask, test_mask) in enumerate(folds):
            print(f"Fold {i+1}")
            data.test_mask = test_mask

            if fairdrop_flag:
                fd = FairDrop(model_args).to(device)
                fd.train()

            if undersampling_flag:
                train_mask = get_undersampled_mask(data.y, sensitive_col, train_mask)
            
            if graph_model_flag:
                model = modelClass(model_args).to(device)
            else:
                model = modelClass(edge_index, model_args).to(device)         

            criterion = choose_criterion(REWEIGHTING, data, train_mask, device)

            trainer = SheafTrainer(
                data,
                model,
                optimizer,
                criterion,
                sensitive_col,
                device=device,
                metric=params['metric'],
                k=params['k'],
                numeric=numeric_idx,
                precomputed = False,
                distances = None,
                uses_edge_index = graph_model_flag,
                additional_loss = additional_loss_flag,
                nifty_flag = nifty_flag,
                modular_flag = modular_flag,
            )

            val_results = trainer.train_model(
                train_mask=train_mask,
                val_mask=val_mask,
                eval='val',
                full_trace=params['trace'],
                epochs=params['epochs'],
                early_stop = params['early_stop'],
                stop_epochs = params['patience'],
                fairdrop = fairdrop,
            )
            test_results = trainer.evaluate()

            if model_str in filters and 'poly' in model_str:
                z = s.repeat_interleave(model.final_d)
                model.set_filter(
                    z,
                    k_eig=model_args['k_eig'], select=model_args['k_select'],
                    K_cheb=model_args['K_cheb'],
                    gamma=model_args['gamma'], sigma=model_args['sigma'], 
                    n_points=100000, mode="topk", nonzero=True, weighted=True, epsilon=1e-5
                    )
            
            elif model_str in filters and 'vect' in model_str:
                model.set_filter(val=True)

            elif model_str in filters and 'spec' in model_str:
                z = s.repeat_interleave(model.final_d)
                model.set_filter(
                    z,
                    f=model_args['f'],
                    k_eig=model_args['k_eig'],
                    select=model_args['k_select'],
                    mode="topk",
                    weighted=True
                    )

            elif model_str not in filters:
                results = {'val': val_results, 'test': test_results}

            if model_str in filters:
                print('Results for validation set - FILTERED')
                val_results = trainer.evaluate(test=False)   
                print('Results for test set - FILTERED')
                test_results = trainer.evaluate()   
                print('='*20)

            all_val_results.append(val_results)
            all_test_results.append(test_results)

        avg_val_results = {k: np.mean([r[k] for r in all_val_results]) for k in all_val_results[0]}
        std_val_results = {k: np.std([r[k] for r in all_val_results]) for k in all_val_results[0]}
        print("\n" + "="*50 + "\n")
        print("Cross-Validation Results (Validation):\n")
        for key in avg_val_results:
            print(f"{key}: \t {avg_val_results[key]:.3f} ± {std_val_results[key]:.3f}")
        print("\n" + "="*50 + "\n")

        avg_test_results = {k: np.mean([r[k] for r in all_test_results]) for k in all_val_results[0]}
        std_test_results = {k: np.std([r[k] for r in all_test_results]) for k in all_val_results[0]}
        print("Cross-Validation Results (Test):\n")
        for key in avg_test_results:
            print(f"{key}: \t {avg_test_results[key]:.3f} ± {std_test_results[key]:.3f}")

        results = {
            "val_fold_results": all_val_results,
            "test_fold_results": all_test_results,
            "avg_val_results": avg_val_results,
            "avg_test_results": avg_test_results,
            "std_val_results": std_val_results,
            "std_test_results": std_test_results
        }

    # test-val-train split for sklearn models
    elif not params['kfold'] and sklearn_flag:
        indices = None

        if adversarial_flag:
            tf.reset_default_graph()
            model = modelClass(model_args)
        else:
            model = modelClass(**model_args)
            
        val_results, test_results, indices = train_sklearn(
            model, data, sensitive_col,
            data.train_mask, data.val_mask, data.test_mask,
            indices=indices, k=params["k"], metric=params["metric"],
            trace=True, numeric_cols=numeric_idx,
            adversarial=adversarial_flag,
            Reweighing=preprocessor,
            RejectOption=postprocessor
            )
        results = {'val': val_results, 'test': test_results}

    # kfold cv training loop with a benchmark (mlp, logreg, xgb,...).
    elif params['kfold'] and sklearn_flag:
        all_val_results = []
        all_test_results = []
        indices = None

        for i, (train_mask, val_mask, test_mask) in enumerate(folds):
            print(f"Fold {i+1}")
            if adversarial_flag:
                tf.reset_default_graph()
                model = modelClass(model_args)
            else:
                model = modelClass(**model_args)

            val_results, test_results, indices = train_sklearn(
                model, data, sensitive_col,
                train_mask, val_mask, test_mask,
                indices=indices, k=params["k"], metric=params["metric"],
                trace=True, numeric_cols=numeric_idx,
                adversarial=adversarial_flag,
                Reweighing=preprocessor,
                RejectOption=postprocessor
                )
            all_val_results.append(val_results)
            all_test_results.append(test_results)

        avg_val_results = {k: np.mean([r[k] for r in all_val_results]) for k in all_val_results[0]}
        std_val_results = {k: np.std([r[k] for r in all_val_results]) for k in all_val_results[0]}

        print("\n" + "="*50 + "\n")
        print("Cross-Validation Results (Validation):\n")
        for key in avg_val_results:
            print(f"{key}: \t {avg_val_results[key]:.3f} ± {std_val_results[key]:.3f}")
        print("\n" + "="*50 + "\n")

        avg_test_results = {k: np.mean([r[k] for r in all_test_results]) for k in all_val_results[0]}
        std_test_results = {k: np.std([r[k] for r in all_test_results]) for k in all_val_results[0]}
        print("Cross-Validation Results (Test):\n")
        for key in avg_test_results:
            print(f"{key}: \t {avg_test_results[key]:.3f} ± {std_test_results[key]:.3f}")

        results = {
            "val_fold_results": all_val_results,
            "test_fold_results": all_test_results,
            "avg_val_results": avg_val_results,
            "avg_test_results": avg_test_results,
            "std_val_results": std_val_results,
            "std_test_results": std_test_results
        }

    #============================================================================
    #                           DATA STORAGE
    #============================================================================
    if 'ro' in model_str:
        model_args.update({
            'metric_name': metric_name,
            'metric_ub': metric_ub,
            'metric_lb': metric_lb,
        })
    results['args'] = {
        **model_args,
        **params,
    }

    if "vect" in model_str:
        del results['args']['vectors']

    return results