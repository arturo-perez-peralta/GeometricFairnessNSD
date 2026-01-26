import argparse

def get_runexp_args():
    parser = argparse.ArgumentParser(
        description="Run a simple classification experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--subset', action='store_true')
    parser.add_argument('--no-subset', dest='subset', action='store_false')
    parser.set_defaults(subset=False)

    parser.add_argument('--distance', action='store_true')
    parser.add_argument('--no-distance', dest='distance', action='store_false')
    parser.set_defaults(distance=False)

    parser.add_argument('--knn', action='store_true')
    parser.add_argument('--no-knn', dest='knn', action='store_false')
    parser.set_defaults(knn=False)
    
    parser.add_argument('--star', action='store_true')
    parser.add_argument('--no-star', dest='star', action='store_false')
    parser.set_defaults(star=False)
    
    parser.add_argument(
        '--delta',
        type=float,
        required=False,
        help='Value of the distance in the local unit distance topology'
    )

    parser.add_argument(
        '--k',
        type=int,
        required=False,
        help='Number of neighbors for the local knn topology'
    )

    parser.add_argument(
        '--layers',
        type=int,
        default=1,
        help='Number of layers of the discrete sheaf diffusion'
    )

    parser.add_argument(
        '--time',
        type=float,
        default=1.0,
        help='Integration time of the continuous sheaf diffusion'
    )

    parser.add_argument(
        '--coef',
        type=float,
        default=1e-3,
        help='Coefficient of the laplacian'
    )

    parser.add_argument(
        '--knn_weight',
        type=float,
        default=1.,
        help='Weight of knn topology'
    )

    parser.add_argument(
        '--unit_weight',
        type=float,
        default=1.,
        help='Weight of unit ball topology'
    )

    parser.add_argument(
        '--subset_weight',
        type=float,
        default=1.,
        help='Weight of global topology'
    )

    parser.add_argument(
        '--metric',
        type=str,
        default='euclidean',
        help='Metric used for consistency, knn,...'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Name of the dataset to experiment on'
    )

    parser.add_argument(
        '--diffusion',
        type=str,
        required=True,
        help='Type of sheaf diffusion (either pre, in or post)'
    )

    parser.add_argument('--disc', action='store_true')
    parser.add_argument('--no-disc', dest='disc', action='store_false')
    parser.set_defaults(disc=False)

    parser.add_argument('--kfold', action='store_true')
    parser.add_argument('--no-kfold', dest='kfold', action='store_false')
    parser.set_defaults(disc=False)

    parser.add_argument('--save', action='store_true')
    parser.add_argument('--no-save', dest='save', action='store_false')
    parser.set_defaults(save=False)

    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--no-trace', dest='trace', action='store_false')
    parser.set_defaults(trace=False)

    parser.add_argument(
        '--folds',
        type=int,
        default=5,
        help='Value of folds for cross validation'
    )

    parser.add_argument(
        '--k_con',
        type=int,
        default=True,
        help='Value of k to compute consistency'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs /10'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        dest='lr',
        help='Learning rate'
    )

    parser.add_argument(
        '--beta1',
        type=float,
        default=0.9,
        dest='beta1',
        help='Beta 1'
    )

    parser.add_argument(
        '--beta2',
        type=float,
        default=0.999,
        dest='beta2',
        help='Beta 2'
    )
    
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-2,
        dest='decay',
        help='Weight decay'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        help='Device (cuda or cpu)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed for reproductibility'
    )

    parser.add_argument(
        '--name',
        type=str,
        required=False,
        help='Name of the file to save the results.'
    )
    
    return parser.parse_args()


def get_neural_args():
    parser = argparse.ArgumentParser(
        description="Run a simple classification experiment with neural sheaf diffusion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--subset', action='store_true')
    parser.add_argument('--no-subset', dest='subset', action='store_false')
    parser.set_defaults(subset=False)

    parser.add_argument('--knn', action='store_true')
    parser.add_argument('--no-knn', dest='knn', action='store_false')
    parser.set_defaults(knn=False)

    parser.add_argument('--distance', action='store_true')
    parser.add_argument('--no-distance', dest='distance', action='store_false')
    parser.set_defaults(distance=False)
    
    parser.add_argument('--star', action='store_true')
    parser.add_argument('--no-star', dest='star', action='store_false')
    parser.set_defaults(star=False)
    
    parser.add_argument(
        '--delta',
        type=float,
        required=False,
        help='Value of the distance in the local unit distance topology'
    )

    parser.add_argument(
        '--k',
        type=int,
        required=False,
        help='Number of neighbors for the local knn topology'
    )

    parser.add_argument(
        '--layers',
        type=int,
        default=1,
        help='Number of layers of the discrete sheaf diffusion'
    )

    parser.add_argument(
        '--time',
        type=float,
        default=1.0,
        help='Integration time of the continuous sheaf diffusion'
    )

    parser.add_argument(
        '--coef',
        type=float,
        default=1e-3,
        help='Coefficient of the laplacian'
    )

    parser.add_argument(
        '--metric',
        type=str,
        default='euclidean',
        help='Metric used for consistency, knn,...'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        required=False,
        help='Name of the file to save the results.'
    )

    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Name of the dataset to experiment on'
    )

    parser.add_argument(
        '--diffusion',
        type=str,
        required=True,
        help='Type of sheaf diffusion (either pre, in or post)'
    )

    parser.add_argument('--disc', action='store_true')
    parser.add_argument('--no-disc', dest='disc', action='store_false')
    parser.set_defaults(disc=False)

    parser.add_argument('--kfold', action='store_true')
    parser.add_argument('--no-kfold', dest='kfold', action='store_false')
    parser.set_defaults(disc=False)

    parser.add_argument('--save', action='store_true')
    parser.add_argument('--no-save', dest='save', action='store_false')
    parser.set_defaults(save=False)

    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--no-trace', dest='trace', action='store_false')
    parser.set_defaults(trace=False)

    parser.add_argument('--nsd', action='store_true')
    parser.add_argument('--no-nsd', dest='nsd', action='store_false')
    parser.set_defaults(nsd=False)

    parser.add_argument('--mlp', action='store_true')
    parser.add_argument('--no-mlp', dest='mlp', action='store_false')
    parser.set_defaults(mlp=False)

    parser.add_argument('--xgb', action='store_true')
    parser.add_argument('--no-xgb', dest='xgb', action='store_false')
    parser.set_defaults(xgb=False)

    parser.add_argument('--logreg', action='store_true')
    parser.add_argument('--no-logreg', dest='logreg', action='store_false')
    parser.set_defaults(logreg=False)

    parser.add_argument(
        '--folds',
        type=int,
        default=5,
        help='Value of folds for cross validation'
    )

    parser.add_argument(
        '--k_con',
        type=int,
        default=True,
        help='Value of k to compute consistency'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs /10'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        dest='lr',
        help='Learning rate'
    )

    parser.add_argument(
        '--beta1',
        type=float,
        default=0.9,
        dest='beta1',
        help='Beta 1'
    )

    parser.add_argument(
        '--beta2',
        type=float,
        default=0.999,
        dest='beta2',
        help='Beta 2'
    )
    
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-2,
        dest='decay',
        help='Weight decay'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        help='Device (cuda or cpu)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed for reproductibility'
    )
    
    return parser.parse_args()



def get_shap_args():
    parser = argparse.ArgumentParser(
        description="Obtain SHAP values for explainability purposes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--subset', action='store_true')
    parser.add_argument('--no-subset', dest='subset', action='store_false')
    parser.set_defaults(subset=False)

    parser.add_argument('--distance', action='store_true')
    parser.add_argument('--no-distance', dest='distance', action='store_false')
    parser.set_defaults(distance=False)

    parser.add_argument('--knn', action='store_true')
    parser.add_argument('--no-knn', dest='knn', action='store_false')
    parser.set_defaults(knn=False)
    
    parser.add_argument('--star', action='store_true')
    parser.add_argument('--no-star', dest='star', action='store_false')
    parser.set_defaults(star=False)
    
    parser.add_argument(
        '--delta',
        type=float,
        required=False,
        help='Value of the distance in the local unit distance topology'
    )

    parser.add_argument(
        '--k',
        type=int,
        required=False,
        help='Number of neighbors for the local knn topology'
    )

    parser.add_argument(
        '--layers',
        type=int,
        default=1,
        help='Number of layers of the discrete sheaf diffusion'
    )

    parser.add_argument(
        '--time',
        type=float,
        default=1.0,
        help='Integration time of the continuous sheaf diffusion'
    )

    parser.add_argument(
        '--coef',
        type=float,
        default=1e-3,
        help='Coefficient of the laplacian'
    )

    parser.add_argument(
        '--metric',
        type=str,
        default='euclidean',
        help='Metric used for consistency, knn,...'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Name of the dataset to experiment on'
    )

    parser.add_argument(
        '--diffusion',
        type=str,
        required=True,
        help='Type of sheaf diffusion (either pre, in or post)'
    )

    parser.add_argument('--disc', action='store_true')
    parser.add_argument('--no-disc', dest='disc', action='store_false')
    parser.set_defaults(disc=False)

    parser.add_argument('--save', action='store_true')
    parser.add_argument('--no-save', dest='save', action='store_false')
    parser.set_defaults(save=False)

    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--no-trace', dest='trace', action='store_false')
    parser.set_defaults(trace=False)

    parser.add_argument(
        '--k_con',
        type=int,
        default=True,
        help='Value of k to compute consistency'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs /10'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        dest='lr',
        help='Learning rate'
    )

    parser.add_argument(
        '--beta1',
        type=float,
        default=0.9,
        dest='beta1',
        help='Beta 1'
    )

    parser.add_argument(
        '--beta2',
        type=float,
        default=0.999,
        dest='beta2',
        help='Beta 2'
    )
    
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-2,
        dest='decay',
        help='Weight decay'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        help='Device (cuda or cpu)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed for reproductibility'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        required=False,
        help='Name of the file to save the results.'
    )

    return parser.parse_args()



