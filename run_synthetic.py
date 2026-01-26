#============================================================================
#                           IMPORTS AND LIBRARIES
#============================================================================

import random
from typing import Dict
import argparse
import json
import os

# Machine learning and data science
import numpy as np
import pandas as pd

# Pytorch
import torch
import torch.optim as optim
from torch_geometric.utils import remove_self_loops, to_undirected

# Custom imports
from utils.data_processing import remove_duplicate_edges, SBMProcessing
from utils.constants import (
    additional_loss, sklearn_models, filters
)
from utils.training import SheafTrainer

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

def run_experiment_synthetic(
        model_str: str,
        model_args: Dict,
        params: Dict,
        simul_params: Dict,
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

    device = torch.device(params['device'])

    processor = SBMProcessing(
        n_nodes_per_block = simul_params['n_nodes_per_block'],
        p_intra = simul_params['p_intra'],
        p_inter = simul_params['p_inter'],
        bias_strength = simul_params['bias_strength'],
        corr_xy =  simul_params['corr_xy'],
        corr_xs = simul_params['corr_xs'],
        n_feat = simul_params['n_feat'],
        device=device
        )

    #============================================================================
    #                           DATA PROCESSING
    #============================================================================

    df = processor.df
    nobs, nfeat = df.values.shape
    nfeat -= 1
    sensitive_var = processor.sensitive
    sensitive = df[[sensitive_var]]

    data, edges = (processor
                .create_graph(unit_ball=params['distance'], k_neigh=params['knn'], sub=params['subset'], star=params['star'], cache=False, **params['kwargs'])
                .split(train_ratio=params['train_ratio'], val_ratio=params['val_ratio'], test_ratio=params['test_ratio'])
                .build())


    edges = {k: remove_duplicate_edges(edges[k]) for k in edges.keys()}

    A = df.drop(columns=['Target']).columns.get_loc(sensitive_var)
    sensitive_col = data.x[:, A]  # [num_nodes]

    numeric_idx = [df.columns.get_loc(col) for col in processor.numeric]

    data.x = data.x.to(device)
    data.y = data.y.to(device)
    edge_index = data.edge_index.to(device)

    # Transform to undirected graph
    edge_index = remove_duplicate_edges(edge_index)
    edge_index = to_undirected(edge_index)
    data.edge_index = edge_index

    sensitive_col = sensitive_col.to(device)
    s = 1.0 * (sensitive_col == 1) - 1.0 * (sensitive_col == 0)

    #============================================================================
    #                           MODEL SELECTION
    #============================================================================

    #--------------------------------------------------
    #                       NSD
    #--------------------------------------------------
    if model_str == 'polydia':
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

    elif model_str == 'specbun':
        from models.fair_filter import SpectrumFilterDiscreteBundleSheafDiffusion
        modelClass = SpectrumFilterDiscreteBundleSheafDiffusion

    elif model_str == 'specgen':
        from models.fair_filter import SpectrumFilterDiscreteGeneralSheafDiffusion
        modelClass = SpectrumFilterDiscreteGeneralSheafDiffusion

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

    #============================================================================
    #                           TRAINING
    #============================================================================

    results = {}

    # test-val-train split for graph models

    model_args.update({
        'num_features': nfeat,
        'input_dim': nfeat,
        'num_classes': 2,
        'output_dim': 1,
        'num_nodes': nobs,
        'graph_size': nobs,
        'device': device
    })

    print(f'Training\n' + '-'*20)
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
        uses_edge_index = False,
        additional_loss = False,
        nifty_flag = False,
        modular_flag = False,
    )
    print_str = 'Results for validation set - UNFILTERED'
    print(print_str)
    val_results = trainer.train_model(
            train_mask=data.train_mask,
            val_mask=data.val_mask,
            eval = 'val',
            full_trace = params['trace'],
            epochs = params['epochs'],
            fairdrop = None,
            early_stop = params['early_stop'],
            stop_epochs = params['patience'],
            )
    print_str = 'Results for test set - UNFILTERED'
    print(print_str)
    test_results = trainer.evaluate()   
    print('='*20)

    if compare:
        results['unfiltered'] = {'val': val_results, 'test': test_results}

    if model_str in filters and 'poly' in model_str:
        s = s.repeat_interleave(model.final_d)
        model.set_filter(
            s,
            k_eig=model_args['k_eig'], select=model_args['k_select'],
            K_cheb=model_args['K_cheb'],
            gamma=model_args['gamma'], sigma=model_args['sigma'], 
            n_points=100000, mode="topk", nonzero=True, weighted=True, epsilon=1e-5
            )
    
    elif model_str in filters and 'vect' in model_str:
        model.set_filter(val=True)

    elif model_str in filters and 'spec' in model_str:
        s = s.repeat_interleave(model.final_d)
        model.set_filter(
            s,
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

    #============================================================================
    #                           DATA STORAGE
    #============================================================================
    results['args'] = {
        **model_args,
        **params,
    }

    if "vect" in model_str:
        del results['args']['vectors']

    return results

def get_fixed_configs(args):
    maxim = 2.0
    bias = (args.bias - 0.5)*2.0
    simul_params = {
        'n_nodes_per_block': 500,
        'p_intra': 0.06 + (1-bias/6) * 0.1,
        'p_inter': 0.04,
        'bias_strength': args.bias,
        'corr_xy': maxim(1-bias/4),
        'corr_xs': maxim*bias,
        'n_feat': 32
    }

    params = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'lr': 0.01,
        'weight_decay': 0.0005,
        'beta1': 0.9,
        'beta2': 0.999,
        'epochs': 20,
        'patience': 30,
        'early_stop': True,
        'trace': False,
        'train_ratio': 0.8,
        'val_ratio': 0.,
        'test_ratio': 0.2,
        'distance': False,
        'knn': False,
        'subset': False,
        'star': False,
        'k': 5,
        'metric': 'euclidean',
        'kwargs': {'k': 5, 'metric': 'euclidean'}
    }

    model_args = {
        'hidden_channels': 32,
        'layers': 4,
        'input_dropout': 0.05,
        'dropout': 0.5,
        'd': 3,
        'use_act': False,
        'sheaf_act': 'id',
        'left_weights': True,
        'right_weights': True,
        'sparse_learner': False,
        'second_linear': True,
        'linear': False,
        'orth': 'householder',
        'edge_weights': False,
        'normalised': True,
        'deg_normalised': False,
        'add_lp': False,
        'add_hp': False,
        'max_t': 1.0,
        'k_eig': 100,
        'k_select': 10,
        'K_cheb': 71,
        'gamma': 0.6*bias,
        'sigma': 0.001,
        'f': 10.0*bias,
        'reg_strength': 30*bias,
#        'gamma': 2.0*bias+0.5,
#        'sigma': 0.001,
#        'f': 1.5*bias+0.5,
#        'reg_strength': 1000*bias,
    }

    if 'poly' in args.model:
        model_args.update({'k_select': 2})
    elif 'spec' in args.model:
        model_args.update({'k_select': 15})

    return params, simul_params, model_args

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bias', type=float, required=True, help='Bias level (0.5 - 1.0)')
    parser.add_argument('--model', type=str, default='polydia', help='Model name')
    parser.add_argument('--seeds', type=int, default=10, help='Number of seeds to run')
    parser.add_argument('--output_dir', type=str, default='results/synthetic', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== Starting experiment: Model={args.model} | Bias={args.bias} ===")
    
    aggregated_results = {
        'val_acc': [], 'test_acc': [],
        'val_ind': [], 'test_ind': [],
        'val_dist': [], 'test_dist': []
    }

    for seed in range(args.seeds):
        print(f" >> Seed {seed+1}/{args.seeds}")
        
        params, simul_params, model_args = get_fixed_configs(args)
        params['seed'] = seed

        try:
            res = run_experiment_synthetic(
                model_str=args.model,
                model_args=model_args,
                params=params,
                simul_params=simul_params,
                compare=True
            )

            data_res = res['filtered'] if 'filtered' in res else res
            
            aggregated_results['val_acc'].append(data_res['val'].get('acc', 0))
            aggregated_results['test_acc'].append(data_res['test'].get('acc', 0))
            aggregated_results['val_ind'].append(data_res['val'].get('ind', 0)) 
            aggregated_results['test_ind'].append(data_res['test'].get('ind', 0))

            val_dist = np.sqrt((1-data_res['val'].get('acc', 0))**2 + data_res['val'].get('ind', 0)**2)
            test_dist = np.sqrt((1-data_res['test'].get('acc', 0))**2 + data_res['test'].get('ind', 0)**2)
            aggregated_results['val_dist'].append(val_dist) 
            aggregated_results['test_dist'].append(test_dist)

        except Exception as e:
            print(f"Error in seed {seed}: {e}")
            continue

    final_output = {
        'metrics': aggregated_results,
        'summary': {
            'mean_test_acc': float(np.mean(aggregated_results['test_acc'])),
            'std_test_acc': float(np.std(aggregated_results['test_acc'])),
            'mean_test_ind': float(np.mean(aggregated_results['test_ind'])),
            'std_test_ind': float(np.std(aggregated_results['test_ind'])),
            'mean_test_dist': float(np.mean(aggregated_results['test_dist'])),
            'std_test_dist': float(np.std(aggregated_results['test_dist']))
        }
    }

    filename = f"{args.output_dir}/res_{args.model}_bias_{args.bias}.json"
    with open(filename, 'w') as f:
        json.dump(final_output, f, indent=4)

    print(f"Finished. Saved to {filename}")
    print(f"Mean Test Acc: {final_output['summary']['mean_test_acc']:.4f}")

if __name__ == "__main__":
    main()