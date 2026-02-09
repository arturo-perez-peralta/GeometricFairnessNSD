#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Data processing utils.
This module implements different classes which abstract away the data processing pipeline, which in our case is comprised of:
    1. Loading the data.
    2. Preprocesing (one-hot encoding, NA imputation,...).
    3. Graph creation.
    4. kfold / train-test split.

This is done through a series of classes which implement different procedures, abstracting from the general (graph creation, splits,...) to the particular (preprocessing of each dataset).

Methods
-------
    1. remove_duplicate_edges (credit https://github.com/twitter-research/neural-sheaf-diffusion/tree/master).

Classes
-------
    1. GraphBuilder: Given a pandas dataset it automatically transforms it to pytorch and provides methods for graph building (knn, unit ball, subset and star topologies).
    2. DataProcessing: Extends GraphBuilder, it first loads the data which will later be fed to the GraphBuilder. Implements kfold and train-test split.

The following classes implement the process of loading and preprocessing each dataset. This is done through the download and preprocess methods 
hinted at in the DataProcessing class.
    3. GermanProcessing: loading and preprocessing of german dataset.
    4. AdultProcessing: loading and adult of homecredit dataset.
    5. CompassProcessing: loading and preprocessing of compass dataset.
    6. HomecreditProcessing: loading and preprocessing of homecredit dataset.
    7. SimulationProcessing: simulation of an artificial dataset.
"""

#===========================================
#                   IMPORTS
#===========================================

# Standard library
from typing import Union, Callable, Optional
import os
import hashlib

# Data processing
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import NearestNeighbors

# Torch
import torch
from torch_geometric.data import Data

#===========================================
#             HELPER FUNCTIONS
#===========================================

def remove_duplicate_edges(edge_index):
    """Given a edge_index tensor, removes duplicate edges (i.e. if (u,v) and (v,u) both apppear in edge_index, it removes one of them).
    Credit: https://github.com/twitter-research/neural-sheaf-diffusion/tree/master
    """
    processed_edges = set()
    new_edge_index = []

    for e in range(edge_index.size(1)):
        source, target = sorted((edge_index[0, e].item(), edge_index[1, e].item()))
        if (source, target) in processed_edges:
            continue
        processed_edges.add((source, target))
        new_edge_index.append([source, target])
    return torch.tensor(new_edge_index, dtype=torch.long, device=edge_index.device).t()


# Helper for generating a synthetic dataset for groups of individuals  (Gaussian mixture)
def make_gaussian_mixture(ns, mus, sigmas, random_state=None):
    """
    ns: number of observations of each group ( N = sum(ns) )
    mus : means for each group (each mean should be a 1D array-like of length L)
    sigmas : covs of each group (each sigma should either a number (homogenoues cov), a 1D-array of length L (diagonal), or an LxL cov matrix

    Notes: 
    L: number of features
    ns, mus, sigmas should be of size Ng (number of groups)
    """    
    rng = np.random.default_rng(random_state)
    
    L = len(mus[0])
    features = []
    group_indexes = []

    for i, (n, μ, Σ) in enumerate(zip(ns, mus, sigmas)):
        μ = np.asarray(μ)
        Σ = np.asarray(Σ)
        if Σ.ndim == 0: # covariances are scalars
            Σ = Σ * np.eye(L)
        elif Σ.ndim == 1: # covariances are vectors
            Σ = np.diag(Σ)

        # generate features for current group
        X = rng.multivariate_normal(mean=μ, cov=Σ, size=n)
        features.append(X)
        group_indexes.append(i * np.ones(n, dtype=int))

    # flatten everything
    features = np.vstack(features)
    group_indexes = np.concatenate(group_indexes)
    return features, group_indexes
    

def make_gaussian_mixture_sens(ns, mus, sigmas, psens, gaps, random_state=None):
    """
    ns: number of observations of each group ( N = sum(ns) )
    mus : means for each group (each mean should be a 1D array-like of length L)
    sigmas : covs of each group (each sigma should either a number (homogenoues cov), a 1D-array of length L (diagonal), or an LxL cov matrix

    Notes: 
    L: number of features
    ns, mus, sigmas should be of size Ng (number of groups)
    """    
    rng = np.random.default_rng(random_state)
    
    L = len(mus[0])
    features = []
    group_indexes = []
    sensitive = np.random.binomial(n=1, p=psens, size=sum(ns))

    for i, (n, μ, Σ, gap) in enumerate(zip(ns, mus, sigmas, gaps)):
        μ = np.asarray(μ) + gap * (sensitive[i] - 0.5)
        Σ = np.asarray(Σ)
        if Σ.ndim == 0: # covariances are scalars
            Σ = Σ * np.eye(L)
        elif Σ.ndim == 1: # covariances are vectors
            Σ = np.diag(Σ)

        # generate features for current group
        X = rng.multivariate_normal(mean=μ, cov=Σ, size=n)
        features.append(X)
        group_indexes.append(i * np.ones(n, dtype=int))

    # flatten everything
    features = np.vstack(features)
    features = np.concatenate((features, sensitive), axis=1)
    group_indexes = np.concatenate(group_indexes)
    return features, group_indexes


#===========================================
#             MAIN UTILS
#===========================================

def data_loader(data, device, n=5000, p=0.5):

    #       Synthetic graphs
    #---------------------------------

    if data == 'german':
        processor = GermanProcessing(device=device)
    elif data == 'adult':
        processor = AdultProcessing(device=device)
    elif data == 'compass':
        processor = CompassProcessing(device=device)
    elif data == 'pakdd':
        processor = PakddProcessing(device=device)
    elif data == 'gmsc':
        processor = GMSCProcessing(device=device)
    elif data == 'taiwan':
        processor = TaiwanProcessing(device=device)

    #       Social
    #---------------------------------

    elif data == 'nba':
        processor = NBAProcessing(device=device)
    elif data == 'pokec_n':
        processor = PokecNProcessing(device=device)
    elif data == 'pokec_z':
        processor = PokecZProcessing(device=device)
    elif data == 'pokec_n_large':
        processor = PokecNLargeProcessing(device=device)
    elif data == 'pokec_z_large':
        processor = PokecZLargeProcessing(device=device)

    #       Simulation
    #---------------------------------
    elif data == 'simulation':
        processor = SimulationProcessing(n=n, p=p, device=device)

    # elif args['data'] == 'simulation2':
    #     from utils.data_processing import SimulationBlasProcessing
    #     n, p = 5000, 0.5
    #     processor = SimulationBlasProcessing(n=n, p=p, device=device)

    return processor


#===========================================
#             CLASSES
#===========================================


class GraphBuilder:
    """Class that transform a given dataset to pytorch and provides the user with methods for fair topology creation.
    """
    def __init__(
            self,
            dataset: Union[Data, pd.DataFrame],
            feature_columns: Optional[list[str]] = None,
            device = torch.device("cuda")
            ):
        """Constructor of the class. Stores the input arguments and runs the _preliminary method to transform the dataset into a pytorch geometric Data structure.
        """
        self.original_dataset = dataset                                                         # original dataset.

        if isinstance(dataset, pd.DataFrame):
            self.feature_columns = feature_columns or [column for column in dataset.columns]    # columns.
        else:
            self.feature_columns = feature_columns

        self.device = device                                                                    # device.

        self.data, self.pos, self.existing_edges = self._preliminary(dataset)                   # runs _preliminary.
        self.new_edges_log = {}                                                                 # will store which edges are added when creating the fair graphs.


    def _preliminary(self, dataset: Union[Data, pd.DataFrame]):
        """Private method that handles the different data inputs a user might be interested in using, although in practise we only use DataFrames.
        The logic is the following:
            1. If the dataset is a Pytorch Geometric Data object do nothing.
            2. If the dataset is a pandas DataFrame, store the features and labels as a Data object, there are no existing edges.
        """

        # If the dataset is Pytorch Geometric Data object simply store existing edges.
        if isinstance(dataset, Data):
            data = dataset
            features = data.x.cpu().numpy()
            existing_edges = set(map(tuple, data.edge_index.T.tolist()))

        # If the dataset is a pandas DataFrame store features and labels in a Data object, there are no existing edges.
        else:
            if hasattr(dataset, "values"):
                if isinstance(dataset, pd.DataFrame) and 'Target' in dataset.columns:
                    dataset = dataset.reset_index(drop = True)
                    target = dataset['Target'].values.astype(np.int64)
                    features = dataset.drop(columns=['Target']).values.astype(np.float32)
                else:
                    raise ValueError("Dataset must be a DataFrame with a 'Target' column.")
            else:
                raise ValueError("Unsupported dataset format for graph construction.")

            data = Data(
                x=torch.tensor(features, dtype=torch.float),
                y=torch.tensor(target, dtype=torch.long)
            )
            if self.edges is None:
                data.edge_index = torch.empty((2, 0), dtype=torch.long)
                existing_edges = set()
            else:
                data.edge_index = torch.tensor(self.edges)
#                data.edge_index = remove_duplicate_edges(data.edge_index)
                existing_edges = set(map(tuple, data.edge_index.T.tolist()))
        
        data.to(self.device)
        return data, features, existing_edges

    def _ending(self, new_edges, name: str):
        """Private method ran after adding each type of edge. Sends all tensors to the desired device and stores them in the log.
        """

        # If there are new edges, add them to the edge_index tensor.
        if new_edges:
            new_edge_tensor = torch.tensor(new_edges, dtype=torch.long).T.to(self.device)
            if hasattr(self.data, 'edge_index'):
                self.data.edge_index = torch.cat([self.data.edge_index, new_edge_tensor], dim=1).to(self.device)
            else:
                self.data.edge_index = new_edge_tensor.to(self.device).to(self.device)

        # in other case add and empty tensor with the correct dimensions.
        else:
            if not hasattr(self.data, 'edge_index'):
                self.data.edge_index = torch.empty((2, 0), dtype=torch.long).to(self.device)
            new_edge_tensor = torch.empty((2, 0), dtype=torch.long).to(self.device)

        # Send all new tensors to the proper device.
        self.data.edge_index = self.data.edge_index.to(self.device)
        new_edge_tensor = new_edge_tensor.to(self.device)
        self.data = self.data.to(self.device)

        # Add the new edges to the log with the corresponding label.
        self.new_edges_log[name] = new_edge_tensor
        return new_edge_tensor



    def add_knn_edges(self, k: int, metric: Union[str, Callable] = 'euclidean'):
        """Method that creates the knn graph with k neighbors and using a given metric (either available in sklearn as a string or passing a custom method).
        """
        # Initialize knn model.
        # When using the mahalanobis distance we need to compute the inverse of the covariance matrix.
        if metric == 'mahalanobis':
            VI = np.linalg.pinv(np.cov(self.pos.T))
            nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric, metric_params = {'VI': VI})  # +1 node itself.
        else:
            nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric)  # +1 node itself.

        # Fit and extract the knn.
        nbrs.fit(self.pos)
        distances, indices = nbrs.kneighbors(self.pos)

        # Progressively add the edges corresponding to the nearest neighbors.
        new_edges = []
        n = len(self.pos)

        for i in range(n):
            for j in indices[i][1:]:  # skip own node (i).
                if (i, j) not in self.existing_edges and (j, i) not in self.existing_edges:
                    new_edges.extend([[i, j], [j, i]])

        self._ending(new_edges, 'knn')
        return self


    def add_unit_ball_edges(self, radius: float, metric: Union[str, Callable] = 'euclidean'):
        """Method that creates the unit ball graph with a given radius and using a given metric
        (either available in sklearn as a string or passing a custom method).
        """

        # Fit spatial knn.
        nbrs = NearestNeighbors(metric=metric, algorithm='auto', n_jobs=-1)
        nbrs.fit(self.pos)

        # Determine a radius using the given relative radius.
        distances, _ = nbrs.kneighbors(self.pos, n_neighbors=min(50, len(self.pos)))
        # max_distance = np.max(distances)
        # true_radius = max_distance * radius
        all_distances = distances[:, 1:].ravel()
        true_radius = np.quantile(all_distances, radius/self.df.shape[0])

        # Find neighbors inside the radius.
        neighbors_within_radius = nbrs.radius_neighbors(self.pos, radius=true_radius, return_distance=False)

        # Build the edges.
        new_edges = []
        for i, neighs in enumerate(neighbors_within_radius):
            for j in neighs:
                if i != j and (i, j) not in self.existing_edges and (j, i) not in self.existing_edges:
                    new_edges.extend([[i, j], [j, i]])

        self._ending(new_edges, 'unit_ball')
        return self


    def add_subset(self, subset: dict, aggregator_edges: list[tuple]):
        """Method that creates the subset graph given a dictionary with each subset
        and a graph on the aggregators.
        """

        # Initialize all aggregator edges.
        start_idx = self.data.num_nodes
        agg_nodes = list(subset.keys())
        agg_ids = {agg_nodes[i]: i + start_idx for i in range(len(agg_nodes))}

        # Add subset edges connecting all observations with their aggregators.
        new_edges = []
        for group_id, node_ids in subset.items():
            agg = agg_ids[group_id]
            for nid in node_ids:
                if (nid, agg) not in self.existing_edges and (agg, nid) not in self.existing_edges:
                    new_edges.extend([[nid, agg], [agg, nid]])

        # Add edges connecting aggregators.
        for i, j in aggregator_edges:
            a, b = agg_ids[i], agg_ids[j]
            if (a, b) not in self.existing_edges and (b, a) not in self.existing_edges:
                new_edges.extend([[a, b], [b, a]])

        # Initialize aggregators as the average of the nodes of each subset
        # feat_dim = self.data.x.size(1)
        # x_agg = torch.zeros((len(agg_nodes), feat_dim), dtype=self.data.x.dtype).to(self.device)
        x_agg = torch.stack([
            self.data.x[nodes].mean(dim=0)
            for nodes in subset.values()
        ]).to(self.device)
        y_agg = torch.zeros((len(agg_nodes)), dtype=self.data.y.dtype).to(self.device)

        # Add this new data
        self.data.x = torch.cat([self.data.x, x_agg], dim=0).to(self.device)
        self.data.y = torch.cat([self.data.y, y_agg], dim=0).to(self.device)

        self._ending(new_edges, 'subset')
        return self

    def add_star_node(self, aggregator_id: Optional[int] = None):
        """Method that creates the star graph.
        """

        # Add the aggregator id and edges to all nodes.
        num_nodes = self.data.num_nodes
        aggregator_id = aggregator_id or num_nodes
        new_edges = []

        for nid in range(num_nodes):
            if (nid, aggregator_id) not in self.existing_edges and (aggregator_id, nid) not in self.existing_edges:
                new_edges.extend([[nid, aggregator_id], [aggregator_id, nid]])

        # Initialize the aggregator a the average of all nodes.
        # feat_dim = self.data.x.size(1)
        # x_agg = torch.zeros((1, feat_dim), dtype=self.data.x.dtype).to(self.device)
        x_agg = self.data.x.mean(dim=0).to(self.device)
        y_agg = torch.zeros((1), dtype=self.data.y.dtype).to(self.device)

        # Concatenate aggregator to the original data.
        self.data.x = torch.cat([self.data.x, x_agg], dim=0).to(self.device)
        self.data.y = torch.cat([self.data.y, y_agg], dim=0).to(self.device)

        # Add edges to the log.
        self._ending(new_edges, 'star')
        return self

    def build(self):
        """Return the data and edges log."""
        return self.data, self.new_edges_log

    def get_data(self):
        """Return the data."""
        return self.data

    def get_new_edges(self) -> list[torch.Tensor]:
        """"Return the edges log."""
        return self.new_edges_log
    
    def erase_graph(self):
        """Erase the graph so another graph may be built."""
        self.data.edge_index = torch.empty((2, 0), dtype=torch.long).to(self.device)
        self.new_edges_log = {}
        self.existing_edges = set()
        return self




class DataProcessing(GraphBuilder):
    """Class that extends the GraphBuilder to integrate data utils, namely kfolds, train-val-test split and an easy interace for
    graph building.
    """
    def __init__(self, device):
        """Constructor. download and preprocess are methods that obtain and process a dataframe and need to be implemented by a 
        class extension."""
        self.edges = None
        self.download().preprocess()
        super().__init__(dataset = self.df, device = device)
    
    def create_graph(
            self,
            unit_ball: bool = False,
            k_neigh: bool = False,
            sub: bool = False,
            star: bool = False,
            cache: bool = True,
            **kwargs
            ):
        """Function used to create the graph according to user specifications. Relies on the add_blank_edges methods
        from the GraphBuilder class."""
        
        params = {
            'unit_ball': unit_ball,
            'k_neigh': k_neigh,
            'sub': sub,
            'star': star,
            'data': self.data,
            **kwargs
        }
        
        param_str = str(sorted(params.items()))
        file_id = hashlib.md5(param_str.encode('utf-8')).hexdigest()
        cache_dir = "cache_graphs"
        os.makedirs(cache_dir, exist_ok=True)
        file_path = os.path.join(cache_dir, f"{file_id}.pt")

        # Use cache if available
        if os.path.exists(file_path) and self.data != 'simul':
            self.data, self.new_edges_log = torch.load(file_path, map_location=self.device)
            self.new_edges = self.new_edges_log
            return self

        # unit ball topology
        if unit_ball:
            radius = kwargs['radius']
            metric = kwargs['metric']
            self.add_unit_ball_edges(radius = radius)

        # knn topology
        if k_neigh:
            k = kwargs['k']
            metric = kwargs['metric']
            self.add_knn_edges(k = k, metric = metric)

        # subset topology
        if sub:
            subset = kwargs['subset']
            aggregator_edges = kwargs['aggregator_edges']
            self.add_subset(subset = subset, aggregator_edges = aggregator_edges)

        # Star topology
        if star:
            self.add_star_node()

        self.data, self.new_edges = self.build()
        
        # Cache
        if cache and self.data != 'simul':
            torch.save((self.data, self.new_edges_log), file_path)
        
        return self

    def split(self, train_ratio=0.5, val_ratio=0.1, test_ratio=0.4, stratify=None, seed=42):
        """Obtain train, val, test masks."""

        data = self.data 
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"

        # Extract indices.
        num_nodes = data.num_nodes
        indices = np.arange(num_nodes)

        # If given labels, use a stratification strategy.
        stratify_labels = stratify if stratify is not None else None

        # Perform split-
        train_idx, tmp_idx = train_test_split(indices, train_size=train_ratio, stratify=stratify_labels, random_state=seed)
        if val_ratio == 0.:
            test_idx = tmp_idx
            val_idx = tmp_idx
        else:
            # Repeat with the testval set.
            stratify_tmp = stratify[tmp_idx] if stratify is not None else None
            val_size = val_ratio / (val_ratio + test_ratio)
            val_idx, test_idx = train_test_split(tmp_idx, train_size=val_size, stratify=stratify_tmp, random_state=seed)

        # Store masks.
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool).to(self.device)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool).to(self.device)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool).to(self.device)
        data.train_mask[train_idx] = True
        data.val_mask[val_idx] = True
        data.test_mask[test_idx] = True
        self.data = data

        return self
    
    def cross_validate(self, k_folds=5, seed=42, stratify=None, test_ratio=0.2):
        """Preform kfold split with a hold-out set.        
        """
        # Extract indices.
        num_nodes = self.data.num_nodes
        indices = np.arange(num_nodes)

        # If given labels, adopt a stratify strategy.
        if stratify is not None:
            y_full = stratify.cpu().numpy() if torch.is_tensor(stratify) else stratify
        else:
            y_full = None

        # Obtain hold-out set.
        n_test = int(num_nodes * test_ratio)
        rng = np.random.RandomState(seed)
        test_idx_fixed = rng.choice(indices, size=n_test, replace=False)
        trainval_idx = np.setdiff1d(indices, test_idx_fixed)

        #  If using a stratify strategy, use the generated vector.
        if y_full is not None:
            y_trainval = y_full[trainval_idx]
        else:
            y_trainval = None

        # Test masks.
        test_mask_const = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask_const[test_idx_fixed] = True

        # Perform kfold split.
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        folds = []
        for train_idx_rel, val_idx_rel in kf.split(trainval_idx, y_trainval):
            # Create train and val masks, store them in an array.
            train_idx = trainval_idx[train_idx_rel]
            val_idx = trainval_idx[val_idx_rel]
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[train_idx] = True
            val_mask[val_idx] = True

            folds.append((train_mask, val_mask, test_mask_const.clone()))

        return folds

    def get_df(self):
        """Return the original dataset."""
        return self.df


#==========================================================
#                   CONCRETE CLASSES
#==========================================================

#------------------------------
#          German
#------------------------------


class GermanProcessing(DataProcessing):
    """Class implementing processing of the German Dataset."""

    def download(self) -> pd.DataFrame:
        """Download the German dataset."""

        self.data = 'german'

        self.categorical_maps = {
            'Status': {'A11': 1, 'A12': 2, 'A13': 3, 'A14': 4},
            'CreditHistory': {'A30': 1, 'A31': 2, 'A32': 3, 'A33': 4, 'A34': 5},
            'Purpose': {'A40': 1, 'A41': 2, 'A42': 3, 'A43': 4, 'A44': 5, 'A45': 6, 'A46': 7, 'A47': 8, 'A48': 9, 'A49': 10, 'A410': 11},
            'Savings': {'A61': 1, 'A62': 2, 'A63': 3, 'A64': 4, 'A65': 5},
            'EmploymentSince': {'A71': 1, 'A72': 2, 'A73': 3, 'A74': 4, 'A75': 5},
            'PersonalStatusSex': {'A91': 1, 'A92': 2, 'A93': 3, 'A94': 4, 'A95': 5},
            'OtherDebtors': {'A101': 1, 'A102': 2, 'A103': 3},
            'Property': {'A121': 1, 'A122': 2, 'A123': 3, 'A124': 4},
            'OtherInstallmentPlans': {'A141': 1, 'A142': 2, 'A143': 3},
            'Housing': {'A151': 1, 'A152': 2, 'A153': 3},
            'Job': {'A171': 1, 'A172': 2, 'A173': 3, 'A174': 4},
            'Telephone': {'A191': 1, 'A192': 2},
            'ForeignWorker': {'A201': 1, 'A202': 2}
        }

        # Categorical and numeric variables.
        self.numeric = ['Duration', 'CreditAmount', 'InstallmentRate', 'ResidenceSince', 'NumberCredits', 'PeopleLiable']
        self.categoricals = list(self.categorical_maps.keys())
        self.column_names = [
            "Status", "Duration", "CreditHistory", "Purpose", "CreditAmount", "Savings",
            "EmploymentSince", "InstallmentRate", "PersonalStatusSex", "OtherDebtors",
            "ResidenceSince", "Property", "Age", "OtherInstallmentPlans", "Housing",
            "NumberCredits", "Job", "PeopleLiable", "Telephone", "ForeignWorker", "Target"
        ]

        # Load the data, store the dataframe.
        df = pd.read_csv('datasets/german/german.data', sep=' ', header=None, names=self.column_names)
        df['Target'] = df['Target'].map({1: 1, 2: 0})
        self.df = df.copy()
        return self

    def preprocess(self) -> pd.DataFrame:
        """Basic pre-processing: one-hot encoding of numeric variables, treatment of sensitive variable."""

        df = self.df

        for col, mapping in self.categorical_maps.items():
           df[col] = df[col].map(mapping)

        df['Age'] = df['Age'].apply(lambda x: 1 if x >= 25 else 0)
        # df[self.numeric_features] = scaler.fit_transform(df[self.numeric_features])
        df['Sex'] = df['PersonalStatusSex'].apply(lambda x: 1 if x in [2, 5] else 0)

        target = df['Target']
        df = df.drop(columns=['Target'])
        df = pd.get_dummies(df, columns=self.categorical_maps.keys(), prefix_sep='_')
        df["Target"] = target.astype(np.int64)

        self.sensitive = "Age"
        self.df = df
        return self


#------------------------------
#          Adult
#------------------------------


class AdultProcessing(DataProcessing):
    """Class implementing the processing of the Adult dataset."""
    def download(self):
        """Download the Adult dataset."""

        self.data = 'adult'

        df = pd.read_csv(
            "datasets/adult/adult.data",
            header=None,
            names=[
                "Age", "Workclass", "fnlwgt", "Education", "EducationNum", "MaritalStatus",
                "Occupation", "Relationship", "Race", "Sex", "CapitalGain",
                "CapitalLoss", "HoursPerWeek", "NativeCountry", "Target"
            ],
            na_values=" ?"
        )
        df = df.dropna().reset_index(drop=True)
        df["Target"] = df["Target"].str.strip().map({">50K": 1, "<=50K": 0})

        self.sensitive = "Sex"
        self.df = df
        return self

    def preprocess(self):
        """Basic pre-processing: one-hot encoding of numeric variables, treatment of sensitive variable."""
        df = self.df.copy()
        self.categoricals = ["Workclass", "Education", "MaritalStatus", "Occupation", "Relationship", "Race", "Sex", "NativeCountry"]
        self.numeric = ["Age", "EducationNum", "CapitalGain", "CapitalLoss", "HoursPerWeek"]
        df = df.drop(columns=["fnlwgt"])

        df = pd.get_dummies(df, columns=self.categoricals, drop_first=True)
        df = df.rename(columns={"Sex_ Male": "Sex"})
        self.df = df
        return self
    

#------------------------------
#          Compas
#------------------------------


class CompassProcessing(DataProcessing):
    def download(self):
        self.data = 'compas'
        
        df = pd.read_csv("datasets/compas/compas-scores-two-years.csv")

        df = df[
            (df["days_b_screening_arrest"] <= 30) &
            (df["days_b_screening_arrest"] >= -30) &
            (df["is_recid"] != -1) &
            (df["c_charge_degree"] != "O") &
            (df["score_text"] != "N/A")
        ]

        df['c_jail_in'] = pd.to_datetime(df['c_jail_in'], errors='coerce')
        df['c_jail_out'] = pd.to_datetime(df['c_jail_out'], errors='coerce')
        df['days_in_jail'] = (df['c_jail_out'] - df['c_jail_in']).dt.days.clip(lower=0)
        
        self.df = df
        return self

    def preprocess(self):
        self.sensitive = 'race'

        df = self.df.copy()
        df = df.rename(columns={'two_year_recid': 'Target'})
        
        keep_cols = [
            'age', 'c_charge_degree', 'race', 'age_cat', 'sex', 
            'priors_count', 'days_b_screening_arrest', 'days_in_jail', 
            'Target'
        ]
        
        df = df[keep_cols]

        df['race'] = (df['race'] == 'African-American')

        self.numeric = [
            'age', 
            'priors_count', 
            'days_b_screening_arrest', 
            'days_in_jail'
        ]
        
        self.categoricals = ['c_charge_degree', 'age_cat', 'sex']

        scaler = MinMaxScaler()
        df_numeric = pd.DataFrame(scaler.fit_transform(df[self.numeric]), columns=self.numeric, index=df.index)
        df_categorical = pd.get_dummies(df[self.categoricals], drop_first=True)
        
        df_processed = pd.concat([df_numeric, df_categorical, df[['race', 'Target']]], axis=1)
        
        self.df = df_processed
        return self
    
#------------------------------
#          Pakdd
#------------------------------

class PakddProcessing(DataProcessing):
    """Class implementing processing of the Pakdd dataset."""
    def download(self):
        """Downlaod Pakdd dataset."""
        self.data = 'pakdd'
        df = pd.read_csv("datasets/pakdd/pkdd.csv")
        df["Target"] = df["BAD"] == "GOOD"
        df = df.drop(columns = ["BAD"])
        self.df = df
        return self

    def preprocess(self):
        """Basic pre-processing: one-hot encoding of numeric variables, treatment of sensitive variable."""
        df = self.df.copy()
        self.categoricals = [
            "PAYMENT_DAY", "APPLICATION_SUBMISSION_TYPE", "SEX", "MARITAL_STATUS", 
            "STATE_OF_BIRTH", "NATIONALITY", "RESIDENTIAL_STATE", "RESIDENCE_TYPE", 
            "COMPANY", "PROFESSIONAL_STATE", "PROFESSION_CODE", "OCCUPATION_TYPE", 
            "MATE_PROFESSION_CODE", "EDUCATION_LEVEL2", "PRODUCT",
            "FLAG_RESIDENCIAL_PHONE", "FLAG_EMAIL", "FLAG_VISA", 
            "FLAG_MASTERCARD", "FLAG_PROFESSIONAL_PHONE",
        ]
        self.numeric = [
            "QUANT_DEPENDANTS", "MONTHS_IN_RESIDENCE", "PERSONAL_MONTHLY_INCOME", 
            "OTHER_INCOMES", "QUANT_SPECIAL_BANKING_ACCOUNTS", "QUANT_CARS", 
            "CREDIT_AMNT"
        ]

        df['AGE'] = df['AGE'] >= 25

        for col in self.numeric:
            df[col].fillna(df[col].median(), inplace=True)

        df = pd.get_dummies(df, columns=self.categoricals, drop_first=True)

        self.sensitive = "AGE"
        self.df = df
        return self
    
#------------------------------
#          GMSC
#------------------------------

class GMSCProcessing(DataProcessing):
    """Class implementing processing of the GMSC dataset."""
    def download(self):
        """Downlaod GMSC dataset."""
        self.data = 'gmsc'
        df = pd.read_csv("datasets/gmsc/gmsc.csv")
        df["Target"] = df["BAD"] == "GOOD"
        df = df.drop(columns = ["BAD"])
        self.df = df
        return self

    def preprocess(self):
        """Basic pre-processing: one-hot encoding of numeric variables, treatment of sensitive variable."""
        df = self.df.copy()
        self.categoricals = [
            "UnknownNumberOfDependents", "UnknownMonthlyIncome", "NoDependents", "NoIncome", 
            "ZeroDebtRatio", "UnknownIncomeDebtRatio", "WeirdRevolvingUtilization", 
            "ZeroRevolvingUtilization", "HasRevolvingLines", "HasRealEstateLoans", 
            "HasMultipleRealEstateLoans", "EligibleSS", "DTIOver33", "DTIOver43", 
            "Never30.59DaysPastDueNotWorse", "Never60.89DaysPastDueNotWorse", 
            "Never90DaysLate", "Weird0999Utilization", "FullUtilization", "ExcessUtilization", 
            "Never30.89DaysPastDueNotWorse", "NeverPastDue", "AnyOpenCreditLinesOrLoans", 
            "Has.Dependents"
        ]

        exclude_cols = ["BAD", "Target"] + self.categoricals
        self.numeric = [c for c in df.columns if c not in exclude_cols]

        df['age'] = df['age'] >= 25

        for col in self.numeric:
            df[col].fillna(df[col].median(), inplace=True)

        df = pd.get_dummies(df, columns=self.categoricals, drop_first=True)

        self.sensitive = "age"
        self.df = df
        return self
    
#------------------------------
#          Taiwan
#------------------------------

class TaiwanProcessing(DataProcessing):
    """Class implementing processing of the Taiwan dataset."""
    def download(self):
        """Downlaod Taiwan dataset."""
        self.data = 'taiwan'
        df = pd.read_excel('datasets/taiwan/taiwan.xls', header=1)
        df["Target"] = df["default payment next month"]
        df = df.drop(columns=["default payment next month", "ID"])
        self.df = df
        return self

    def preprocess(self):
        """Basic pre-processing: one-hot encoding of numeric variables, treatment of sensitive variable."""
        df = self.df.copy()
        self.categoricals = [
            'EDUCATION', 'MARRIAGE', 'SEX',
            'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
        self.numeric = [
            'LIMIT_BAL',
            'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
            'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
        ]

        df['AGE'] = df['AGE'] >= 25

        for col in self.numeric:
            df[col].fillna(df[col].median(), inplace=True)

        df = pd.get_dummies(df, columns=self.categoricals, drop_first=True)

        self.sensitive = "AGE"
        self.df = df
        return self

#------------------------------
#          Homecredit
#------------------------------


class HomecreditProcessing(DataProcessing):
    """Class implementing processing of the Homecredit dataset."""
    def download(self):
        """Downlaod Homecredit dataset."""
        self.data = 'homecredit'
        df = pd.read_csv("datasets/homecredit/application_train.csv")
        df = df[[
            "TARGET", "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
            "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
            "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
            "REGION_POPULATION_RELATIVE", "DAYS_BIRTH", "DAYS_EMPLOYED", "CNT_FAM_MEMBERS"
        ]]
        df["Target"] = df["TARGET"]
        self.df = df
        return self

    def preprocess(self):
        """Basic pre-processing: one-hot encoding of numeric variables, treatment of sensitive variable."""
        df = self.df.copy()
        self.categoricals = [
            "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
            "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE"
        ]
        self.numeric = [
            "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
            "REGION_POPULATION_RELATIVE", "DAYS_EMPLOYED", "CNT_FAM_MEMBERS"
        ]

        df['AGE'] = -df['DAYS_BIRTH'].astype('float') / 365
        df = df.drop(columns=['DAYS_BIRTH'])
        df['AGE'] = df['AGE'] >= 25

        for col in self.numeric:
            df[col].fillna(df[col].median(), inplace=True)

        df = pd.get_dummies(df, columns=self.categoricals, drop_first=True)

        self.sensitive = "AGE"
        self.df = df
        return self
    

#------------------------------
#          Simulation (first)
#------------------------------


class SimulationProcessing(DataProcessing):
    """Class implementing the simulation scheme outlined in the paper."""
    def __init__(self, n, p, device):
        """Constructor (initialize p and n)"""
        self.n = n
        self.p = p
        super().__init__(device = device)

    def download(self):
        """Simulate the process outlined in the paper."""
        self.data = 'simul'
        a = np.random.binomial(1, self.p, self.n)
        v = np.random.normal(loc=a, scale=1, size=self.n)
        u = np.random.normal(loc=v, scale=1, size=self.n)
        w = np.random.normal(loc=v, scale=1, size=self.n)
        t = np.random.normal(loc=-0.5, scale=1, size=self.n)
        per1 = np.random.normal(loc=0, scale=2, size=self.n)
        per2 = np.random.normal(loc=0, scale=2, size=self.n)
        per3 = np.random.normal(loc=0, scale=2, size=self.n)

        logits = 0.5*w + 0.5*t
        y = np.array(1/(1 + np.exp(-logits)) > 0.5, dtype=int)

        self.df = pd.DataFrame({
            'a': a,
            'u': u,
            't': t,
            'per1': per1,
            'per2': per2,
            'per3': per3,
            'Target': y,
        })
        self.categoricals = ['Target']
        self.numeric = ['a', 'u', 't', 'per1', 'per2', 'per3']
        self.sensitive = 'a'
        return self

    def preprocess(self):
        return self
    

#------------------------------
#          Simulation (SBM)
#------------------------------

class SBMProcessing(DataProcessing):
    """Class implementing the SBM simulation scheme."""
    def __init__(self, n_nodes_per_block, p_intra, p_inter, bias_strength, corr_xy, corr_xs, n_feat, device):
        """Constructor (initialize p and n)"""
        self.data = 'simul'
        self.n_nodes_per_block = n_nodes_per_block
        self.p_intra = p_intra
        self.p_inter = p_inter
        self.bias_strength = bias_strength
        self.corr_xs = corr_xs
        self.corr_xy = corr_xy
        self.n_feat = n_feat
        super().__init__(device = device)

    def download(self):
        """Simulate the process outlined in the paper."""
        sizes = [self.n_nodes_per_block, self.n_nodes_per_block]

        probs = [[self.p_intra, self.p_inter], 
                [self.p_inter, self.p_intra]]
        
        G = nx.stochastic_block_model(sizes, probs, seed=42)
        
        for node in G.nodes():
            block = G.nodes[node]['block'] # 0 o 1
            
            if np.random.rand() < self.bias_strength:
                s = block
            else:
                s = 1 - block
                
            G.nodes[node]['sensitive_attr'] = s

        n = len(G.nodes())
        
        y = np.array([G.nodes[node]['block'] for node in G.nodes()])
        x = np.random.randn(n, self.n_feat)

        if self.corr_xy > 0:
            x[y == 1] += self.corr_xy
        
        if self.corr_xs > 0:
            x[s == 1] += self.corr_xs
        
        s = np.array([G.nodes[node]['sensitive_attr'] for node in G.nodes()])

        self.df = pd.DataFrame(x, columns=[f"feat_{i}" for i in range(x.shape[1])])
        self.df['Target'] = y
        self.df['Sens'] = s
        self.edges = np.array(G.edges).T

        self.categoricals = ['Target', 'Sens']
        self.numeric = [f"feat_{i}" for i in range(x.shape[1])]
        self.sensitive = 'Sens'

        return self

    def preprocess(self):
        return self


#------------------------------
#          Simulation (Blas)
#------------------------------


class SimulationBlasProcessing(DataProcessing):
    """Class implementing a more complete simulation scheme."""
    def __init__(self, ns, mus, sigmas, psens, gap, weights, seed, device, threshold=0.):
        """Constructor (initialize p and n)"""
        self.ns = ns
        self.sigmas = sigmas
        self.mus = mus
        self.psens = psens
        self.gap = gap
        self.seed = seed
        self.weights = weights
        self.threshold = threshold
        super().__init__(device = device)

    def download(self):
        """Simulate the process outlined in the paper."""
        self.data = 'simul'
        features, group_indexes = make_gaussian_mixture_sens(
            ns=self.ns, mus=self.mus, sigmas=self.sigmas,
            psens=self.psens, gap=self.gap, random_state=self.seed
            )

        logits = features @ self.weights
        y = np.array(1/(1 + np.exp(-logits)) > self.threshold, dtype=int)

        num_generic = features.shape[1] - 1
        col_names = [f"feat_{i}" for i in range(num_generic)]
        col_names.append("Sensitive")

        df = pd.DataFrame(features, columns=col_names)
        df['Target'] = y

        self.df = df
        self.categoricals = ['Target']
        self.numeric = col_names
        self.sensitive = 'Sensitive'
        return self

    def preprocess(self):
        return self

#------------------------------
#          Pokec - n
#------------------------------


class PokecNProcessing(DataProcessing):
    """Class implementing processing of the Pokec-n Dataset."""

    def download(self) -> pd.DataFrame:
        """Download the Pokec-n dataset."""
        self.data = 'pokecn'
        # Load all data.
        edges = np.load("datasets/pokecn/pokec_n/region_job_2_2_edges.npy")
        features = np.load("datasets/pokecn/pokec_n/region_job_2_2_features.npy")
        labels = np.load("datasets/pokecn/pokec_n/region_job_2_2_labels.npy")
        sens = np.load("datasets/pokecn/pokec_n/region_job_2_2_sens.npy")

        # Load the data, store the dataframe.
        df = pd.DataFrame(features, columns=[f"feat_{i}" for i in range(features.shape[1])])
        self.numeric = list(df.columns)
        df['Target'] = labels
        df['Sensitive'] = sens

        self.edges = edges
        self.sensitive = "Sensitive"
        self.df = df.copy()
        return self

    def preprocess(self) -> pd.DataFrame:
        """Basic pre-processing: one-hot encoding of numeric variables, treatment of sensitive variable."""
        return self
    

#------------------------------
#          Pokec - n - large
#------------------------------


class PokecNLargeProcessing(DataProcessing):
    """Class implementing processing of the Pokec-n Large Dataset."""

    def download(self) -> pd.DataFrame:
        """Download the Pokec-n large dataset."""

        # embeddings = pd.read_csv("embedding", sep=" ", skiprows=1, header=None)
        # ids = embeddings[0].values
        # nfeatures = embeddings.iloc[:, 1:].values
        self.data = 'pokecnl'
        # Load dataframe
        df = pd.read_csv("datasets/pokecnl/pokec_n_large/region_job_2.csv") 
        df = df.rename(columns={'I_am_working_in_field': 'Target'})
        df['Target'] = df['Target'] > 1
        df = df.dropna()
        df = df.reset_index(drop=True)
        
        # Load edges
        edges = pd.read_csv("datasets/pokecnl/pokec_n_large/region_job_2_relationship.txt", sep="\t", header=None)
        mapping = {old_id: i for i, old_id in enumerate(df['user_id'])}
        edges[0] = edges[0].map(mapping)
        edges[1] = edges[1].map(mapping)
        edges = edges.dropna()
        edges = np.array(edges.values).T
        df = df.drop(columns=['user_id'])

        #Store data        
        self.edges = edges
        self.numeric = [c for c in df.columns if c != 'Target']
        self.sensitive = 'region'
        self.df = df
        return self

    def preprocess(self) -> pd.DataFrame:
        """Basic pre-processing: one-hot encoding of numeric variables, treatment of sensitive variable."""
        return self

#------------------------------
#          Pokec - z
#------------------------------


class PokecZProcessing(DataProcessing):
    """Class implementing processing of the Pokec-z Dataset."""

    def download(self) -> pd.DataFrame:
        """Download the Pokec-z dataset."""
        self.data = 'pokecz'
        # Load all data.
        edges = np.load("datasets/pokecz/pokec_z/region_job_1_edges.npy")
        features = np.load("datasets/pokecz/pokec_z/region_job_1_features.npy")
        labels = np.load("datasets/pokecz/pokec_z/region_job_1_labels.npy")
        sens = np.load("datasets/pokecz/pokec_z/region_job_1_sens.npy")

        # Load the data, store the dataframe.
        df = pd.DataFrame(features, columns=[f"feat_{i}" for i in range(features.shape[1])])
        self.numeric = list(df.columns)
        df['Target'] = labels
        df['Sensitive'] = sens

        self.edges = edges
        self.sensitive = "Sensitive"
        self.df = df.copy()
        return self

    def preprocess(self) -> pd.DataFrame:
        """Basic pre-processing: one-hot encoding of numeric variables, treatment of sensitive variable."""
        return self

#------------------------------
#          Pokec - z - large
#------------------------------


class PokecZLargeProcessing(DataProcessing):
    """Class implementing processing of the Pokec-z large Dataset."""

    def download(self) -> pd.DataFrame:
        """Download the Pokec-z large dataset."""
        self.data = 'pokeczl'
        # embeddings = pd.read_csv("embedding", sep=" ", skiprows=1, header=None)
        # ids = embeddings[0].values
        # nfeatures = embeddings.iloc[:, 1:].values
        
        # Load dataframe
        df = pd.read_csv("datasets/pokeczl/pokec_z_large/region_job.csv") 
        df = df.rename(columns={'I_am_working_in_field': 'Target'})
        df['Target'] = df['Target'] > 1
        df = df.dropna()
        df = df.reset_index(drop=True)
        
        # Load edges
        edges = pd.read_csv("datasets/pokeczl/pokec_z_large/region_job_relationship.txt", sep="\t", header=None)
        mapping = {old_id: i for i, old_id in enumerate(df['user_id'])}
        edges[0] = edges[0].map(mapping)
        edges[1] = edges[1].map(mapping)
        edges = edges.dropna()
        edges = np.array(edges.values).T
        df = df.drop(columns=['user_id'])

        #Store data        
        self.edges = edges
        self.numeric = [c for c in df.columns if c != 'Target']
        self.sensitive = 'region'
        self.df = df
        return self

    def preprocess(self) -> pd.DataFrame:
        """Basic pre-processing: one-hot encoding of numeric variables, treatment of sensitive variable."""
        return self
    
#------------------------------
#          NBA
#------------------------------


class NBAProcessing(DataProcessing):
    """Class implementing processing of the NBA Dataset."""

    def download(self) -> pd.DataFrame:
        """Download the NBA dataset."""
        self.data = 'nba'
        # embeddings = pd.read_csv("nba.embedding", sep=" ", skiprows=1, header=None)
        # ids = embeddings[0].values
        # nfeatures = embeddings.iloc[:, 1:].values
        
        # Load dataframe
        df = pd.read_csv("datasets/nba/nba.csv") 
        df['Target'] = df['SALARY'] == 1

        mapping = {old_id: i for i, old_id in enumerate(df['user_id'])}
        
        # Load edges
        edges = pd.read_csv("datasets/nba/nba_relationship.txt", sep="\t", header=None)
        edges[0] = edges[0].map(mapping)
        edges[1] = edges[1].map(mapping)
        edges = edges.dropna()
        edges = np.array(edges.values).T

        df = df.drop(columns=['user_id', 'SALARY'])

        #Store data        
        self.edges = edges
        self.numeric = [c for c in df.columns if c != 'Target']
        self.sensitive = 'country'
        self.df = df
        return self

    def preprocess(self) -> pd.DataFrame:
        """Basic pre-processing: one-hot encoding of numeric variables, treatment of sensitive variable."""
        return self