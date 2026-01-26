models = [
    'logreg', 'xgb', 'lgbm', 'mlp',                 # Tabular models
    'rw_logreg', 'rw_xgb', 'rw_lgbm',               # Reweighting
    'adversarial',                                  # Adversarial learning
    'ro_logreg', 'ro_xgb', 'ro_lgbm',               # Reject option
    'gcn', 'gat', 'sage', 'h2gcn',                  # Graph models
    'us_gcn', 'us_gat', 'us_sage', 'us_h2gcn',      # Undersampling
    'fd_gcn', 'fd_gat', 'fd_sage', 'fd_h2gcn',      # Fair Drop
    'fairgnn', 'nifty', 'fairsin', 'bind',          # Fair graph models
    'dia', 'bun', 'gen',                            # NSD
    'polydia', 'polybun', 'polygen',                # Polynomial filter
    'vectdia', 'vectbun', 'vectgen',                # Vector filter
    'specdia', 'specbun', 'specgen',                # Spectral Projection filter
]

graph_models = [
    'gcn', 'gat', 'sage', 'h2gcn',                  # Graph models
    'us_gcn', 'us_gat', 'us_sage', 'us_h2gcn',      # Undersampling
    'fd_gcn', 'fd_gat', 'fd_sage', 'fd_h2gcn',      # Fair Drop
    'fairgnn', 'nifty', 'fairsin', 'bind',          # Fair graph models
]

additional_loss = ['nifty']                         # Fair graph models

nsd = [
    'dia', 'bun', 'gen',                            # NSD
    'polydia', 'polybun', 'polygen',                # Polynomial filter
    'vectdia', 'vectbun', 'vectgen',                # Vector filter
    'specdia', 'specbun', 'specgen',                # Spectral Projection filter
]

filters = [
    'polydia', 'polybun', 'polygen',                # Polynomial filter
    'vectdia', 'vectbun', 'vectgen',                # Vector filter
    'specdia', 'specbun', 'specgen',                # Spectral Projection filter
]

sklearn_models = [
    'logreg', 'xgb', 'lgbm', 'mlp',                 # Tabular models
    'rw_logreg', 'rw_xgb', 'rw_lgbm',               # Reweighting
    'adversarial',                                  # Adversarial learning
    'ro_logreg', 'ro_xgb', 'ro_lgbm',               # Reject option
]

aif360 = [
    'rw_logreg', 'rw_xgb', 'rw_lgbm',               # Reweighting
    'adversarial',                                  # Adversarial learning
    'ro_logreg', 'ro_xgb', 'ro_lgbm',               # Reject option
]

modular = ['fairgnn', 'fairsin']

graph_datasets = ['nba', 'pokec_n', 'pokec_z', 'pokec_n_large', 'pokec_z_large']

tabular_datasets = ["german", "compass", "adult", "pakdd", "gmsc", "taiwan"]

datasets = graph_datasets + tabular_datasets