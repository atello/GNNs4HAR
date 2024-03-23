import numpy as np
import pickle
import torch
import networkx as nx
from sklearn.covariance import GraphicalLasso, MinCovDet, EmpiricalCovariance, LedoitWolf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr
from torch_geometric.data import Data


def edge_calculation(x_data, method="pearson", corr_threshold=0.2, n_neighbors=3, corr_mtrx=None, **kwargs):
    """
    Calculates the correlation between signals in the input data
    :param x_data: raw data signals
    :param method: the method used to calculate the correlation between signals., e.g., pearson, knn_clustering
    :param corr_threshold: correlation threshold if pearson is used (Default 0.2). If corr >= threshold an edge is set
    :param n_neighbors: number of neighbors if knn method is used. Default knn=3
    :param weighted: whether to add correlation values as edge_weights or just {0,1}
    :param corr_mtrx: pre-calculated correlation matrix
    :return: dict edges where the key is a tuple (i,j) and the value is the weight between that pair of nodes
    """

    edges = {}

    if method == "pearson":
        if corr_mtrx is None:
            corr = np.corrcoef(x_data)
        else:
            corr = corr_mtrx

        for s_i in range(len(corr)):
            for s_j in range(len(corr)):
                if s_i != s_j and np.abs(corr[s_i, s_j]) >= corr_threshold:
                    edges[(s_i, s_j)] = np.abs(corr[s_i, s_j]) if kwargs["weighted"] else 1

    elif method == "knn":
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(x_data)
        distances, indices = nbrs.kneighbors(X=x_data)

        for s_i in range(len(indices)):
            for s_j in range(1, len(indices[0])):
                edges[(s_i, indices[s_i, s_j])] = np.abs(1 - distances[s_i, s_j]) if kwargs["weighted"] else 1

    elif method == "cosine":
        corr = cosine_similarity(x_data)

        for s_i in range(len(corr)):
            for s_j in range(len(corr)):
                if s_i != s_j and corr[s_i, s_j] >= corr_threshold:
                    edges[(s_i, s_j)] = np.abs(corr[s_i, s_j])

    elif method == "graph_lasso":

        cov = MinCovDet(random_state=42, assume_centered=True).fit(x_data.T)
        cov = EmpiricalCovariance(assume_centered=True).fit(x_data.T)
        corr = cov.covariance_
        # spearman = spearmanr(x_data.T).correlation
        # corr = np.corrcoef(x_data)
        #
        for s_i in range(len(corr)):
            for s_j in range(len(corr)):
                if s_i != s_j and corr[s_i, s_j] >= corr_threshold:
                    edges[(s_i, s_j)] = np.abs(corr[s_i, s_j])

    elif method == "fc":
        if corr_mtrx is None:
            corr = np.corrcoef(x_data)
        else:
            corr = corr_mtrx

        minv = np.min(np.abs(corr))
        maxv = np.max(np.abs(corr))
        for s_i in range(len(corr)):
            for s_j in range(len(corr)):
                if s_i != s_j and np.abs(corr[s_i, s_j]) >= corr_threshold:
                    edges[(s_i, s_j)] = np.abs(corr[s_i, s_j])

    return edges


def signals_to_graph(data, labels, save_as=None, **kwargs):
    """
    Converts the signals' matrix (MxN), where M is the number of samples and N in the number of signals per sample, into
    an undirected graph. Each signal represents a node in the graph. An edge between to signals is considered if the
    pearson correlation is greater or equal than certain threshold "c" (default c=0.2).
    :param data: The signals matrix (MxN)
    :param labels: The labels for each graph (1xN)
    :param method: the method used to calculate the correlation between signals., e.g., pearson, knn_clustering
    :param corr_threshold: Correlation threshold for establishing edges between signals
    :param save_as: full path to save the generated graphs as pickle file
    :return: A graph representation of the raw data signals.
    """

    graphs = []
    corr_by_activity = kwargs.get("corr_by_activity", {})
    for sample_i in range(len(data)):
        sample = data[sample_i].T  # use Transpose
        label = labels[sample_i]

        if kwargs.get("corr_all_data", False):

            assert kwargs.get("edges", None) is not None, 'Sparse Adj. Matrix calculated from correlation from the' \
                                                          'entire ds_name is missing '
            edges = kwargs.get("edges")
        else:
            edges = edge_calculation(x_data=sample, **kwargs)

        if len(edges) == 0:
            continue

        nodes = set([e[0] for e in edges] + [e[1] for e in edges])
        nodes = {n: sample[n, :] for n in nodes}

        g = {"nodes": nodes, "edges": edges, "label": label}
        graphs.append(g)

    if save_as:
        pickle.dump(obj=graphs, file=open(save_as, 'wb'))

    return graphs


def graph_dict_to_torch(graphs, save_as=None):
    torch_graphs = []

    for g in graphs:
        nodes = g["nodes"]

        sensors_dict = {key: i for i, key in enumerate(nodes.keys())}

        edge_idxs = torch.tensor([[sensors_dict[k[0]], sensors_dict[k[1]]] for k in g["edges"].keys()],
                                 dtype=torch.long)

        edge_attr = torch.tensor([[weight] for weight in g["edges"].values()], dtype=torch.float)

        x = torch.tensor(np.asarray([nodes[key] for key in nodes]), dtype=torch.float)

        y = torch.tensor(np.asarray([g["label"]]), dtype=torch.long)

        graph = Data(x=x, edge_index=edge_idxs.t().contiguous(), edge_attr=edge_attr, y=y)
        torch_graphs.append(graph)

    if save_as:
        pickle.dump(obj=torch_graphs, file=open(save_as, 'wb'))

    return torch_graphs
