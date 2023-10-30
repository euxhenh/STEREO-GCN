import json
import logging
import os
import pickle
from functools import cached_property
from pathlib import Path
from typing import List

import anndata
import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse
from scipy.spatial.distance import cdist
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize as _normalize
from torch.utils.data import Dataset
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import dense_to_sparse

from ..utils.search import search_nonlin_fn

logger = logging.getLogger(__name__)


class DatasetMixin(Dataset):
    """Helper properties"""

    def __init__(self):
        self.x_seq = []
        self.y_seq = []

    def __getitem__(self, index):
        return self.x_seq, self.y_seq  # Returns lists of tensors

    def __len__(self):
        return 1  # There is only one time series

    @property
    def x_seq_full(self):
        """Will return x_seq concatenated by y_seq[-1], i.e., the entire
        time series from point 0 to T+1.
        """
        return [*self.x_seq, self.y_seq[-1]]

    @property
    def n_nodes_seq(self) -> List[int]:
        return [x.shape[0] for x in self.x_seq]

    @property
    def n_seq(self) -> int:
        return len(self.x_seq)

    @property
    def in_features(self) -> int:
        return self.x_seq[0].shape[1]

    @property
    def out_features(self) -> int:
        return self.in_features

    def save(self, ckpt: str) -> None:
        """Dump dataset."""
        d = {}
        d['x_seq'] = self.x_seq
        d['y_seq'] = self.y_seq
        d = self.on_save(d)

        with open(ckpt, 'wb') as f:
            pickle.dump(d, f)

    def on_save(self, d: dict) -> dict:
        """Update d for derived classes."""
        pass

    @classmethod
    def load_from_checkpoint(cls, ckpt: str):
        """Load from ckpt"""
        obj = cls()

        with open(ckpt, 'rb') as f:
            d = pickle.load(f)
        for k, val in d.items():
            setattr(obj, k, val)

        return obj

    @staticmethod
    def normalize_samples(x):
        """Normalize so that each sample has unit length."""
        return _normalize(x, 'l1')

    def correlation_matrices(self):
        A_corr = []
        for x, y in zip(self.x_seq, self.y_seq):
            # x is the source and we want sources as columns so (y, x)
            A = cdist(y, x, metric='correlation')
            A = torch.from_numpy(A).float() / 2
            A_corr.append(A)
        return A_corr


class HubSimulatedData(DatasetMixin):
    """
    Following the assumption that gene regulatory networks are
    approximately scale-free, we generate random graphs where a small
    number of nodes (transcription factors) regulate all other genes. We
    call these regulatory nodes hubs. We sample without replacement a set
    of hubs H uniformly over [n] of size n_hubs. A random network is then
    constructed by adding an edge with probability `edge_prob`. For all
    added edges, we sample their weight uniformly at random from [0, 1]. To
    construct a sequence of T smoothly-changing graphs, we generate two
    such random matrices A_1 and A_T and interpolate between them to
    generate a sequence As.

    In order to distinguish between the different regulatory sources, hub
    node features are drawn at random from different normal distributions
    (std=1) centered at the vertices of a k-dimensional hypercube. All
    other node features are sampled independently from a continuous uniform
    distribution (0, 1). Finally, response node features Y are computed as
    Y = `nonlin`(A @ X) @ W depending on the value of `nonlin` and
    `apply_linear_transformatin`.

    Adjacency matrices are in 'target_to_source' format.

    Parameters
    ----------
    n_nodes: int
        Total number of nodes to use (including hub nodes).
    n_features: int
        Number of features to generate for the feature matrices X.
    n_hubs: int
        Number of hubs to select from n_nodes.
    min_n_hubs_per_trg: int
        Every non-hub node will be regulated by at least this many nodes.
    max_n_hubs_per_trg: int or None
        Every non-hub node will be regulated by at most this many nodes. If
        None, will not set an upper limit.
    n_redundant_features: int
        See sklearn.datasets.make_classification.
    n_seq: int
        Number of time points to generate.
    x_init: str
        Method of initializing non-hub nodes. Will get attr from np.random.
    preserve_hubs: bool
        If True, then both the first and last adjacency matries will
        contain the same set of hubs. Otherwise, these sets will be
        different.
    normalize, add_self_loops: bool
        Whether to fill diagonal with 1's and normalize the adjacency
        matrices as done in the graph convolutional operator.
    nonlin: str
        The nonlinearity to apply. `Identity` is equivalent to no
        nonlinearity.
    logical_nonlin: str
        A numpy function that can be applied to an axis. If this is not
        None, then `nonlin` above will not be used. E.g., if this is 'max',
        then instead of applying a linear combination of src nodes of the
        form a_1 * s_1 + ... + a_T * s_T, will instead take the max as in
        elementwise_max(a_1 * s_1, ..., a_T * s_T).
    edge_prob: float
        Will add an edge between a hub node and a non-hub node with this
        probability.
    add_noise: bool
        If True, will add noise as in Y = nonlin(A @ X) + noise.
    apply_linear_transformation: bool
        If True, will compute Y = nonlin(A @ X) @ W for a random W.
    static_W: bool
        Only used if `apply_linear_transformation` is True. If True, will
        use the same W for all time points.
    indep_x: bool
        If True, for each time point will draw a new random X, otherwise
        will use the one from the last time point (y).
    """

    def __init__(
        self,
        n_nodes: int = 100,
        n_features: int = 9,
        *,
        n_hubs: int | None = None,
        min_n_hubs_per_trg: int = 1,
        max_n_hubs_per_trg: int | None = None,
        n_redundant_features: int = 0,
        n_seq: int = 5,
        x_init: str = 'random',
        preserve_hubs: bool = True,
        normalize: bool = False,
        add_self_loops: bool = False,
        nonlin: str = 'Identity',
        logical_nonlin: str | None = None,  # max, min, or other np func
        edge_prob: float = 0.1,
        add_noise: bool = False,
        apply_linear_transformation: bool = False,
        static_W: bool = True,
        indep_x: bool = False,
    ):
        self.n_nodes = n_nodes
        self.n_hubs = n_hubs or max(n_nodes // 10, 1)
        self.min_n_hubs_per_trg = min_n_hubs_per_trg
        self.max_n_hubs_per_trg = max_n_hubs_per_trg
        self.n_redundant_features = n_redundant_features
        self.x_init = x_init
        self.preserve_hubs = preserve_hubs
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.nonlin = search_nonlin_fn(nonlin)
        self.logical_nonlin = logical_nonlin
        self.edge_prob = edge_prob
        self.add_noise = add_noise
        self.apply_linear_transformation = apply_linear_transformation
        self.indep_x = indep_x
        assert n_seq >= 1
        assert self.n_hubs < self.n_nodes
        assert self.min_n_hubs_per_trg <= self.n_hubs
        assert self.n_redundant_features <= n_features
        assert (
            self.max_n_hubs_per_trg is None
            or self.max_n_hubs_per_trg >= self.min_n_hubs_per_trg
        )

        # select random nodes as hubs
        self.hub_ids_0 = self.random_hubs()
        self.hub_ids_seq = [self.hub_ids_0]

        self.A_seq = [A_0 := self.get_random_A(self.hub_ids_0)]
        if n_seq >= 2:
            # generate new hubs if not preserving
            self.hub_ids_T = self.hub_ids_0 if preserve_hubs else self.random_hubs()
            A_T = self.get_random_A(self.hub_ids_T)
            # interpolate between first and last time point
            self.A_seq = list(torch.from_numpy(np.linspace(A_0, A_T, n_seq)))
            all_hub_ids = np.union1d(self.hub_ids_0, self.hub_ids_T)
            # create [hubs_0, ... hubs_0_and_T x (n_seq - 2) ..., hubs_T]
            self.hub_ids_seq.extend([*([all_hub_ids] * (n_seq - 2)), self.hub_ids_T])

        assert len(self.A_seq) == len(self.hub_ids_seq) == n_seq

        if apply_linear_transformation:
            self.W_seq = [W_0 := self.random_W(n_features)]
            if static_W:
                self.W_seq = self.W_seq * n_seq
            elif n_seq >= 2:
                W_T = self.random_W(n_features)
                self.W_seq = list(torch.from_numpy(np.linspace(W_0, W_T, n_seq)))
            assert len(self.W_seq) == n_seq
        else:
            self.W_seq = None

        if self.indep_x:
            self.x_seq = [self.get_random_x(hub_ids, n_features)
                          for hub_ids in self.hub_ids_seq]
        else:  # only one time point here, will be populated later
            self.x_seq = [self.get_random_x(self.hub_ids_seq[0], n_features)]
        self.y_seq = []

        for i, A in enumerate(self.A_seq):
            x = self.x_seq[i]
            # apply nonlin or logical nonlin
            if logical_nonlin is None:
                y = A @ x
                y = self.nonlin(y)
            else:
                y = torch.einsum('ij, jk -> ijk', A, x)
                y = getattr(y, logical_nonlin)(dim=1).values  # dim = 1

            if apply_linear_transformation:  # multiply by W
                y = y @ self.W_seq[i]
            if add_noise:  # add gaussian noise
                y = y + torch.randn_like(y) / 64

            self.y_seq.append(y)
            if not self.indep_x and i != len(self.A_seq) - 1:
                x = y.clone()
                # fill next x with new hub features
                x = self.fill_hubs(x, self.hub_ids_seq[i], n_features)
                self.x_seq.append(x)

        assert len(self.y_seq) == len(self.x_seq) == n_seq

    def random_hubs(self):
        hubs = np.random.choice(self.n_nodes, size=self.n_hubs, replace=False)
        hubs.sort()
        return hubs

    @staticmethod
    def random_W(n_features):
        """Random W from uniform(-0.5, 0.5)"""
        return torch.randn((n_features, n_features)) / 9

    def get_random_x(self, hub_ids, n_features):
        """Hub node features are initialized as random gaussians from the
        vertices of a hypercude. All other nodes are initialized at random.
        """
        x = getattr(np.random, self.x_init)(size=(self.n_nodes, n_features))
        x = torch.from_numpy(x).float()
        x = self.fill_hubs(x, hub_ids, n_features)
        return x

    def fill_hubs(self, x, hub_ids, n_features):
        x[hub_ids] = torch.from_numpy(make_classification(
            n_samples=len(hub_ids), n_features=n_features,
            n_informative=n_features - self.n_redundant_features,
            n_redundant=self.n_redundant_features, n_classes=len(hub_ids),
            n_clusters_per_class=1, hypercube=False, class_sep=5)[0]).float()
        return x

    def get_random_A(self, hub_ids):
        """Random adjacency matrix. Edges are assigned between a hub and a
        non-hub at random with probability `self.edge_prob`.
        Returns an adjacency matrix in 'target_to_source' format.
        """
        A = torch.zeros((self.n_nodes, self.n_nodes))
        A[hub_ids, hub_ids] = 1

        for node in range(self.n_nodes):
            if node in hub_ids:
                continue
            sample = torch.rand(len(hub_ids))
            # Repeat until we get the required min number of hubs
            while self.is_outside_range(
                (sample < self.edge_prob).sum(),
                self.min_n_hubs_per_trg,
                self.max_n_hubs_per_trg,
            ):
                sample = torch.rand(len(hub_ids))
            sources = hub_ids[sample < self.edge_prob]
            A[node, sources] = 1

        if self.normalize:
            edge_index, edge_weight = dense_to_sparse(A.T)
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight,
                num_nodes=A.shape[0],
                add_self_loops=self.add_self_loops,
            )
            # 'target_to_source' mode
            A[edge_index[1], edge_index[0]] = edge_weight
        return A

    @staticmethod
    def is_outside_range(a: float, m: float | None = None, M: float | None = None):
        if m is not None and a < m:
            return True
        if M is not None and a > M:
            return True
        return False

    def on_save(self, d: dict) -> dict:
        """Store matrices and hubs"""
        super().on_save(d)
        d['A_seq'] = self.A_seq
        d['hub_ids_seq'] = self.hub_ids_seq
        d['W_seq'] = self.W_seq
        return d


class GeneRegMixin(DatasetMixin):
    tf_path: str = 'data/TF_TRRUST.txt'
    # tf_path: str = 'data/TF_names.txt'
    snc_path: str = 'data/senescence_list.json'

    @property
    def highly_variable(self) -> np.ndarray:
        """Return a boolean array of highly var genes in adata."""
        if 'highly_variable' in self.adata.var:
            return self.adata.var['highly_variable'].to_numpy()
        return None

    @property
    def adata(self) -> anndata.AnnData:
        """Return the human lung cell atlas."""
        if not hasattr(self, '_adata'):
            self._adata = anndata.read(os.path.expanduser(self.adata_path), 'r')
        return self._adata

    @property
    def adata_gene_names(self) -> np.ndarray:
        """Return array of gene names in adata."""
        if 'feature_name' in self.adata.var:
            genes = self.adata.var['feature_name'].to_numpy().astype(str)
        else:
            genes = self.adata.var_names.to_numpy().astype(str)
        return genes

    @cached_property
    def tfs(self):
        """Get a list of all known TFs"""
        # TF source: http://humantfs.ccbr.utoronto.ca/download.php
        tf_names = open(os.path.expanduser(self.tf_path), 'r').readlines()
        tf_names = [tf.strip() for tf in tf_names]
        # remove Zinc finger transcription factors
        # tf_names = [tf for tf in tf_names if not tf.startswith("ZNF")]
        return np.asarray(tf_names)

    @cached_property
    def snc_sets(self) -> dict:
        """Return a dict of senescence marker sets."""
        if not hasattr(self, '_snc_sets'):
            self._snc_sets = json.load(open(self.snc_path, "r"))
        return self._snc_sets

    @cached_property
    def snc_markers(self) -> np.ndarray:
        """Return a list of all senescence markers."""
        return np.unique(np.concatenate(list(self.snc_sets.values())))

    @cached_property
    def snc_tfs(self) -> np.ndarray:
        """Return an array of senescence markers that are also TFs."""
        return np.intersect1d(self.tfs, self.snc_markers)

    @property
    def gene_is_tf(self):
        """Return a boolean mask of same shape as self.genes."""
        return np.in1d(self.genes, self.tfs)

    @property
    def olap_tfs(self) -> np.ndarray:
        """Return a list of TFs also found in our data."""
        return np.intersect1d(self.genes, self.tfs)

    @property
    def olap_snc_markers(self) -> np.ndarray:
        """Return a list of snc markers that are in genes."""
        return np.intersect1d(self.genes, self.snc_markers)

    @property
    def gene_is_snc_marker(self) -> np.ndarray:
        """Return a boolean mask of genes that are markers."""
        return np.in1d(self.genes, self.snc_markers)

    @property
    def olap_snc_tfs(self) -> np.ndarray:
        """Return a list of TFs that are also senescence markers and
        overlap with self.genes.
        """
        return np.intersect1d(self.genes, self.snc_tfs)

    @property
    def gene_is_snc_tf(self) -> np.ndarray:
        """Return a boolean mask of genes that are markers."""
        return np.in1d(self.genes, self.snc_tfs)

    def to_keep(
        self,
        highly_variable: bool = False,
        keep_snc_markers: bool = False,
        tfs_only: bool = False,
        targets_only: bool = False,
    ):
        _to_keep = np.full(self.adata.shape[1], True)
        genes = self.adata_gene_names

        if highly_variable:
            _to_keep[~self.highly_variable] = False
        if tfs_only:
            _to_keep[~np.in1d(genes, self.tfs)] = False
        if keep_snc_markers:
            _to_keep[np.in1d(genes, self.snc_markers)] = True
        if targets_only:
            df, A, _ = load_trrust()
            df = df[(np.in1d(df[0], genes) & (np.in1d(df[1], genes)))]
            nodes = np.unique(np.concatenate([df[0].unique(), df[1].unique()]))
            _to_keep[~np.in1d(genes, nodes)] = False

        return _to_keep


class GeneRegPseudotimeDataset(GeneRegMixin):
    """Gene expression data with ordered cells for each time point.
    """

    def __init__(
        self,
        name,
        n_seq: int,
        adata_path: str,
        path: str = "data/polys",
        highly_variable: bool = False,
        keep_snc_markers: bool = False,
        targets_only: bool = False,
        n_features: int = 10,
        normalize: bool = False,
    ):
        super().__init__()
        path = Path(path)
        self.adata_path = adata_path

        genes_to_keep = self.to_keep(
            highly_variable=highly_variable,
            keep_snc_markers=keep_snc_markers,
            targets_only=targets_only,
        )
        self.genes = self.adata_gene_names
        if False in genes_to_keep:
            self.genes = self.genes[genes_to_keep]

        print(f"Keeping {genes_to_keep.sum()} genes.")

        for i in range(n_seq):
            x = np.load(open(path / f'{name}{i}.npz', "rb"))
            # skip point first and last point
            ticks = np.linspace(1, x.shape[1] - 1, n_features, dtype=int)
            x = x[:, ticks]
            assert x.shape[1] == n_features
            if False in genes_to_keep:
                x = x[genes_to_keep]
            self.x_seq.append(torch.from_numpy(x).float())

        if normalize:
            self.x_seq = [self.normalize_samples(x) for x in self.x_seq]
            self.x_seq = [torch.from_numpy(x).float() for x in self.x_seq]

        self.genes_to_keep = genes_to_keep

        self.y_seq = self.x_seq[1:]
        self.x_seq = self.x_seq[:-1]


class PBMCGeneRegPseudotimeDataset(GeneRegPseudotimeDataset):

    def tvgl_prep(self):
        """Prepare data for TVGL by return original samples."""
        x_seq = []
        for i in range(self.n_seq):
            x = self.adata[self.adata.obs['timepoint'] == i + 1, self.genes_to_keep[:, None]].X
            if issparse(x):
                x = x.toarray()
            if x.shape[0] > 1000:
                # subsample if too many samples
                x = x[np.random.choice(x.shape[0], size=1000, replace=False)]
            x_seq.append(x)
        y_seq = [np.full(x.shape[0], t) for t, x in enumerate(x_seq)]
        return np.vstack(x_seq), np.concatenate(y_seq)


def load_trrust(
    path='data/networks/trrust_rawdata.human.tsv',
    res_genes=None,
):
    """Return known TF gene interactions from TRRUST database."""
    df = pd.read_csv(os.path.expanduser(path), sep='\t', header=None)

    if res_genes is not None:
        # Both TF and gene should be in res_genes
        df = df[(np.in1d(df[0], res_genes)) & (np.in1d(df[1], res_genes))]

    le = LabelEncoder()
    all_genes = np.concatenate([df[0], df[1]])
    le.fit(all_genes)
    sources = le.transform(df[0])
    targets = le.transform(df[1])
    weights = np.zeros(len(sources))
    weights[df[2] == 'Repression'] = -1
    weights[df[2] == 'Activation'] = 1
    # to separate unknown from zero
    weights[~np.in1d(df[2], ['Repression', 'Activation'])] = 0.00001

    n_genes = max(sources.max(), targets.max()) + 1
    A = np.zeros((n_genes, n_genes))
    A[targets, sources] = weights  # target-to-source mode
    A = torch.from_numpy(A).float()
    genes = le.classes_

    return df, A, genes


def load_fibroblasts_gt(path='data/networks/fibroblast-normal-panda.txt'):
    """Return the true Adjacency matrix of interactions."""

    df = pd.read_csv(os.path.expanduser(path), sep='\t', header=None)

    le = LabelEncoder()
    all_genes = np.concatenate([df[0], df[1]])
    le.fit(all_genes)
    sources = le.transform(df[0])
    targets = le.transform(df[1])
    weights = df[2].to_numpy()

    n_genes = max(sources.max(), targets.max()) + 1
    A = np.zeros((n_genes, n_genes))
    A[targets, sources] = weights  # target-to-source mode
    A = torch.from_numpy(A).float()
    genes = le.classes_

    return df, A, genes
