import argparse
import json
import os
import copy

import optuna
import torch
import tensorflow.compat.v1 as tf

from utils.constants import tabular_datasets
from run_filtering import run_experiment

tf.disable_eager_execution()

BETA1 = 0.9
BETA2 = 0.999

TRACE = False
KFOLD = False

FOLDS = 5
EPOCHS = 10
EPOCHS_FINAL = 40
SEED = 42

K = 10
METRIC = "euclidean"

TRAIN_RATIO = 0.5
TEST_RATIO = 0.25
VAL_RATIO = 0.25

FAIRNESS_THRESHOLD = 0.2
FAIRNESS_METRIC_KEY = "val_ind"

EARLY_STOP = True
PATIENCE = 40

ABLATION_MODELS = [
    'custgat', 'custgcn',
    'polygat', 'polygcn',
    'specgat', 'specgcn',
    'vectgat', 'vectgcn'
]

def get_search_space(trial, model_name, dataset):
    # --- Common parameters ---
    params = {
        'lr': trial.suggest_float('lr', 1e-4, 5e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        'kfold': False,
        'folds': FOLDS,
        'beta1': BETA1,
        'beta2': BETA2,
        'epochs': EPOCHS,
        'seed': SEED,
        'early_stop': EARLY_STOP,
        'patience': PATIENCE,
    }

    model_params = {}

    ablation_flag = model_name in ABLATION_MODELS
    graph_flag = any([name in model_name for name in ['gcn', 'gat', 'sage', 'h2gcn']]) and not ablation_flag
    sheaf_flag = any([name in model_name for name in ['dia', 'bun', 'gen']])

    if 'poly' in model_name:
        model_params.update({
            'k_eig': trial.suggest_categorical("k_eig", [100, 200, 300]),
            'k_select': trial.suggest_categorical("k_select", [1, 5, 10]),
            'K_cheb': trial.suggest_categorical("K_cheb", [20, 31, 50, 61, 70, 81, 93]),
            'gamma': trial.suggest_float("gamma", 1e-1, 1, log=True),
            'sigma': trial.suggest_float("sigma", 1e-4, 1e-1, log=True)
        })
    elif 'vect' in model_name:
        model_params.update({
            "reg_strength": trial.suggest_float("reg_strength", 1e-2, 1e5, log=True)
        })
    elif 'spec' in model_name:
        model_params.update({
            'k_eig': trial.suggest_categorical("k_eig", [100, 200, 300]),
            'f': trial.suggest_float("f", 1e-1, 10, log=True),
            'k_select': trial.suggest_categorical("k_select", [1, 5, 10, 20, 30])
        })

    # NSD
    if sheaf_flag or ablation_flag:
        model_params.update({
            'layers': trial.suggest_int('layers', 1, 6),
            'hidden_channels': trial.suggest_categorical('hidden_channels', [2, 4, 8, 16, 32, 64]),
            'input_dropout': trial.suggest_float('input_dropout', 0.0, 0.7),
            'dropout': trial.suggest_float('dropout', 0.0, 0.7),
            'right_weights': trial.suggest_categorical('right_weights', [True, False]),
            'sparse_learner': trial.suggest_categorical('sparse_learner', [True, False]),
            'use_act': trial.suggest_categorical('use_act', [True, False]),
            'sheaf_act': trial.suggest_categorical('sheaf_act', ['id', 'tanh', 'elu']),
            'add_lp': False,
            'add_hp': False,
#            'add_lp': trial.suggest_categorical('add_lp', [True, False]),
#            'add_hp': trial.suggest_categorical('add_hp', [True, False]),
            'second_linear': trial.suggest_categorical('second_linear', [True, False]),
            'linear': trial.suggest_categorical('linear', [True, False]),
            'max_t': 1.0,
            'orth': 'matrix_exp',
            'edge_weights': False
        })

        if ablation_flag:
            model_params['d'] = 1
            model_params['normalised'] = True
            model_params['deg_normalised'] = False
            model_params['left_weights'] = False
        else:
            model_params['d'] = trial.suggest_int('d', 2, 3)
            model_params['left_weights'] = trial.suggest_categorical('left_weights', [True, False]),

        if 'dia' in model_name:
            normalised = trial.suggest_categorical('normalised', [True, False])
            model_params.update({
                'normalised': normalised,
                'deg_normalised': not normalised,
            })

        elif 'bun' in model_name:
            model_params.update({
                'deg_normalised': False,
                'normalised': True,
                'orth': trial.suggest_categorical('orth', ['matrix_exp', 'cayley', 'euler']),
                'edge_weights': trial.suggest_categorical('edge_weights', [True, False])
            })

        elif 'gen' in model_name:
            normalised = trial.suggest_categorical('normalised', [True, False])
            model_params.update({
                'normalised': normalised,
                'deg_normalised': not normalised,
            })

    # Graph models
    elif graph_flag:
        model_params.update({
            'hidden_dim': trial.suggest_categorical('hidden_dim', [8, 16, 32, 64, 128]),
            'num_layers': trial.suggest_int('layers', 2, 5),
            'activation': trial.suggest_categorical('activation', ['relu', 'elu', 'leaky_relu']),
            'dropout': trial.suggest_float('dropout', 0.2, 0.6)
        })
        if 'gat' in model_name:
            model_params['heads'] = trial.suggest_categorical('heads', [1, 4, 8])

    # Graph fairness
    elif model_name in ['fairgnn', 'fairsin', 'bind', 'nifty']:
        model_params.update({
            'hidden_dim': trial.suggest_categorical('hidden_dim', [16, 32, 64, 128]),
            'num_layers': trial.suggest_int('layers', 2, 5),
            'activation': trial.suggest_categorical('activation', ['relu', 'elu', 'leaky_relu']),
            'dropout': trial.suggest_float('dropout', 0.2, 0.6),
            'type': trial.suggest_categorical('type', ['gcn', 'gat', 'sage'])
        })
        if model_name == 'fairgnn':
            model_params.update({
                'adv_coeff': trial.suggest_float('adv_coeff', 1e-2, 1e2, log=True),
                'cov_coeff': trial.suggest_float('ortho_coeff', 1e-2, 1e2, log=True)
            })
        elif model_name == 'fairsin':
            model_params.update({
                'adv_coeff': trial.suggest_float('adv_coef', 1e-2, 1e2, log=True),
            })
        else:
            model_params.update({
                'coeff': trial.suggest_float('coef', 1e-3, 1e1, log=True),
            })
    
    # XGBoost / LightGBM
    elif 'xgb' in model_name or 'lgbm' in model_name:
        model_params.update({
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        })
        if 'lgbm' in model_name:
            model_params['num_leaves'] = trial.suggest_int('num_leaves', 20, 100)

    # MLP
    elif 'mlp' in model_name:
        hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128, 256])
        layers = trial.suggest_int('layers', 2, 4)
        hidden_layer_sizes = [hidden_dim for _ in range(layers)]
        model_params.update({
            'hidden_layer_sizes': hidden_layer_sizes,
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'solver': 'adam',
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
            'learning_rate_init': params['lr'],
            'beta_1': params['beta1'],
            'beta_2': params['beta2'],
        })

    elif model_name == 'adversarial':
        model_params.update({
            'adversary_loss_weight': trial.suggest_float('adversary_loss_weight', 1e-3, 1e2, log=True),
            'classifier_num_hidden_units': trial.suggest_categorical('classifier_num_hidden_units', [8, 16, 32, 64]),
            'num_epochs': params['epochs']*10,
            'scope_name': 'debiased_classifier',
            'debias': True,
        })

    # LogReg
    elif 'logreg' in model_name:
        model_params.update({
            'C': trial.suggest_float('C', 1e-3, 1e2, log=True),
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear']),
            'max_iter': 1000
        })

    if 'ro' in model_name:
        fairness_margin = trial.suggest_float('fairness_margin', 0.01, 0.2)
        model_params.update({
            'metric_name': 'Statistical parity difference',
            'metric_ub': fairness_margin,
            'metric_lb': -fairness_margin
        })

    if 'fd' in model_name:
        model_params.update({
            'drop_edge_rate': trial.suggest_float('drop_edge_rate', 0.1, 0.4),
            'target_homophily': False
        })

    return params, model_params


def setup_run_config(trial, dataset, model_str):
    params, model_args = get_search_space(trial, model_str, dataset)

    if dataset in tabular_datasets:
        KNN = True
    else:
        KNN = False

    params.update({
        'name': dataset,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'distance': False,
        'subset': False,
        'knn': KNN,
        'star': False,
        'folds': FOLDS,
        'delta': 0.,
        'k': K,
        'kwargs': {'k': K, 'metric': METRIC,},
        'metric': METRIC,
        'train_ratio': TRAIN_RATIO,
        'test_ratio': TEST_RATIO,
        'val_ratio': VAL_RATIO,
        'trace': False,
        'epochs': EPOCHS,
    })
    
    return params, model_args



def objective(trial, dataset, model_str):

    if 'adversarial' in model_str:
        tf.reset_default_graph()

    params, model_args = setup_run_config(trial, dataset, model_str)
    
    params['kfold'] = False 

    try:
        results = run_experiment(
            model_str=model_str, 
            params=params, 
            model_args=model_args
        )

        if results and 'val' in results:
            for key in results:
                if isinstance(results[key], dict):
                    for key2 in results[key]:
                        trial.set_user_attr(f"{key}_{key2}", results[key][key2])
                else:
                    trial.set_user_attr(f"{key}", results[key])
            
            return results['val']['balacc']
            
    except RuntimeError as e:
        error_msg = str(e)
        if "out of memory" in error_msg:
            torch.cuda.empty_cache()
            print(f"Trial {trial.number} failed due to OOM.")
            raise optuna.TrialPruned()
        elif "linalg" in error_msg and ("converge" in error_msg or "101" in error_msg):
            print(f"Trial {trial.number} failed due to Linalg convergence error (ill-conditioned matrix).")
            return 0.0
        else:
            raise e
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 0.0
    
    return 0.0


def run_final_workflow(study, dataset, model_str):
    print("\n" + "="*40)
    print(" SELECTING BEST MODEL WITH FAIRNESS CONSTRAINTS")
    print("="*40)

    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    candidates = []

    for t in trials:
        fair_val = abs(t.user_attrs.get(FAIRNESS_METRIC_KEY, 1.0))
        balacc_val = t.value
        
        if fair_val <= FAIRNESS_THRESHOLD:
            candidates.append((balacc_val, t))
            
    if candidates:
        best_trial = max(candidates, key=lambda x: x[0])[1]
        print(f"Selected Trial {best_trial.number} (Balacc: {best_trial.value:.4f}, Fair: {best_trial.user_attrs.get(FAIRNESS_METRIC_KEY):.4f})")
    else:
        print("No trial met the fairness threshold. Selecting minimal unfairness.")
        best_trial = min(trials, key=lambda t: abs(t.user_attrs.get(FAIRNESS_METRIC_KEY, 1.0)))
        print(f"Fallback Trial {best_trial.number} (Balacc: {best_trial.value:.4f}, Fair: {best_trial.user_attrs.get(FAIRNESS_METRIC_KEY):.4f})")

    best_params = best_trial.params
    fixed_trial = optuna.trial.FixedTrial(best_params)
    
    params_base, model_args_base = setup_run_config(fixed_trial, dataset, model_str)

    print(f"\n[1/2] Executing {FOLDS}-Fold Cross Validation...")
    
    params_kfold = copy.deepcopy(params_base)
    params_kfold['kfold'] = True
    params_kfold['folds'] = FOLDS
    params_kfold['epochs'] = EPOCHS_FINAL

    if 'adversarial' in model_str:
        tf.reset_default_graph()
    elif 'ro' in model_str:
        params_kfold.update({'metric_name': 'Statistical parity difference'})
    
    try:
        results_kfold = run_experiment(
            model_str=model_str, 
            params=params_kfold, 
            model_args=copy.deepcopy(model_args_base)
        )
        print("Results K-Fold:", results_kfold)

    except Exception as e:
        print(f"Error in final K-fold: {e}")

    print(f"\n[2/2] Executing Final Training (Full Dataset)...")
    
    params_full = copy.deepcopy(params_base)
    params_full['kfold'] = False
    params_full['epochs'] = EPOCHS_FINAL
    params_full['train_ratio'] = TRAIN_RATIO + VAL_RATIO
    params_full['val_ratio'] = 0.
    params_full['test_ratio'] = TEST_RATIO
    
    try:
        if 'adversarial' in model_str:
            tf.reset_default_graph()
        elif 'ro' in model_str:
            params_full.update({'metric_name': 'Statistical parity difference'})
        results_full = run_experiment(
            model_str=model_str, 
            params=params_full, 
            model_args=copy.deepcopy(model_args_base)
        )
        print("Results Full Train:", results_full)
        

        if args.model in ABLATION_MODELS:
            output_path = f'results/filter/ablation/{study.study_name}_final_run.json'
        else:
            output_path = f'results/filter/experiment/{study.study_name}_final_run.json'

        with open(output_path, 'w') as f:
            final_data = {
                'best_params': best_params,
                'selection_metrics': {
                    'trial_number': best_trial.number,
                    'validation_balacc': best_trial.value,
                    'validation_fairness': best_trial.user_attrs.get(FAIRNESS_METRIC_KEY)
                },
                'kfold_metrics': results_kfold if 'results_kfold' in locals() else None,
                'full_run_metrics': results_full
            }
            json.dump(final_data, f, indent=4)
        print(f"Final results stored at: {output_path}")
        
    except Exception as e:
        print(f"Error in full training: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optuna training')
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Name of the dataset (e.g.: german, credit, pokec)')
    parser.add_argument('--model', type=str, required=True, 
                        help='Name of the model (e.g.: FairGNN, NIFTY, BIND)')
    parser.add_argument('--n_trials', type=int, required=False,
                    help='Name of the model (e.g.: FairGNN, NIFTY, BIND)')
    parser.set_defaults(n_trials=50)
    args = parser.parse_args()

    study_name = f"{args.model}_{args.dataset}"
    storage_name = f"sqlite:///storage/{study_name}.db"

    study = optuna.create_study(
        study_name=study_name, 
        direction="maximize", 
        storage=storage_name,
        load_if_exists=True
    )

    concrete_objective = lambda x: objective(x, dataset=args.dataset, model_str=args.model)
    print("Initialize optimization")
    study.optimize(concrete_objective, n_trials=args.n_trials)

    print("-" * 20)
    print("Best params:")
    print(study.best_params)
    print(f"Best values: {study.best_value}")

    if args.model in ABLATION_MODELS:
        folder = 'results/filter/ablation'
    else:
        folder = 'results/filter/experiment'
    os.makedirs(folder, exist_ok=True)

    with open(f'{folder}/{study_name}_best.json', 'w') as f:
        json.dump(study.best_params, f)

    run_final_workflow(study, args.dataset, args.model)