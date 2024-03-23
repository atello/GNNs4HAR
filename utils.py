import numpy as np
import torch
import random

from torch import Tensor
from torch_geometric.utils import dropout_edge


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gaussian_kernel(X):
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float)
    pairwise_dists = torch.sum((X.unsqueeze(1) - X.unsqueeze(0)) ** 2, dim=-1)
    exponent = -pairwise_dists / (X.std() ** 2)  # (X.std() ** 2)
    return torch.exp(exponent)


def compute_edges(similarity_matrix, threshold=0.2):
    edge_index = []
    edge_weight = []
    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix)):
            if i != j and np.abs(similarity_matrix[i, j]) > threshold:
                edge_index.append([i, j])
                edge_weight.append(np.abs(similarity_matrix[i, j]))

    # unique nodes in edge_index to remove isolated nodes
    unique_nodes = np.unique(edge_index)
    nodes_dict = {key: i for i, key in enumerate(unique_nodes)}
    edge_index = torch.tensor([[nodes_dict[e[0]], nodes_dict[e[1]]] for e in edge_index], dtype=torch.long)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    return edge_index, edge_weight, unique_nodes


def InfoNCEloss(pos_view1, pos_view2, neg_v1, temperature=0.07) -> Tensor:

    z = torch.cat((pos_view1, pos_view2), dim=0)
    batch_size = pos_view1.size(0)
    sim_matrix = cosine_similarity_matrix(z)
    positives_sim = torch.diag(sim_matrix[batch_size:, :batch_size])

    # idx = torch.ones(sim_matrix.size(0) // 2, sim_matrix.size(0) // 2, dtype=torch.bool)
    # idx[torch.eye(sim_matrix.size(0) // 2, dtype=torch.bool)] = 0
    # negatives_sim = sim_matrix[batch_size:, :batch_size][idx].view(batch_size, batch_size - 1)

    z_neg_v1 = torch.cat((pos_view1, neg_v1), dim=0)
    sim_matrix_v1 = cosine_similarity_matrix(z_neg_v1)
    negatives_sim = sim_matrix_v1[batch_size:, :batch_size]
    # negatives_sim = torch.hstack((sim_matrix_v1[batch_size:, :batch_size], negatives_sim))

    positive_exp = torch.exp(positives_sim / temperature)
    positive_exp = torch.where(torch.isnan(positive_exp) | torch.isinf(positive_exp),
                               torch.zeros_like(positive_exp),
                               positive_exp)

    negative_exp = torch.exp(negatives_sim / temperature)
    negative_exp = torch.where(torch.isnan(negative_exp) | torch.isinf(negative_exp),
                               torch.zeros_like(negative_exp),
                               negative_exp)

    negative_sum = torch.sum(negative_exp, dim=1)
    negative_sum = torch.where(torch.isnan(negative_sum) | torch.isinf(negative_sum),
                               torch.zeros_like(negative_sum),
                               negative_sum)

    loss = -torch.log(positive_exp / (positive_exp + negative_sum))
    loss = torch.where(torch.isnan(loss) | torch.isinf(loss),
                       torch.zeros_like(loss),
                       loss)
    loss = loss.mean()

    return loss


def corrupt_graph(x, edge_index, corruption):
    # edge_mask = torch.ones(edge_index.size(1), dtype=torch.bool)
    if corruption == "nodes":  # random permutation of nodes
        perm_idx = torch.randperm(x.size(0))
        corrupted = (x[perm_idx], edge_index)
    elif corruption == "edge_index":  # shuffles the dest nodes from edge_index
        edge_perm_idx = torch.randperm(edge_index[1].size(0))
        edge_index[1] = edge_index[1, edge_perm_idx]
        corrupted = (x, edge_index)
    else:
        perm_idx = torch.randperm(x.size(0))
        edge_perm_idx = torch.randperm(edge_index[1].size(0))
        edge_index[1] = edge_index[1, edge_perm_idx]
        corrupted = (x[perm_idx], edge_index)
    return corrupted


def cosine_similarity_matrix(matrix):
    dot_product = torch.matmul(matrix, matrix.T)
    norms = torch.norm(matrix, dim=1, keepdim=True)
    norms_product = torch.matmul(norms, norms.T)
    cosine_similarity = dot_product / norms_product

    return cosine_similarity


def get_sensor_names(ds_name: str):
    sensor_labels = []
    if ds_name.lower() == "ucihar":
        sensor_labels = [
            "body_acc_x",
            "body_acc_y",
            "body_acc_z",
            "body_gyro_x",
            "body_gyro_y",
            "body_gyro_z",
            "total_acc_x",
            "total_acc_y",
            "total_acc_z"
        ]
        remove_list = [5]
    elif ds_name.lower() == "mhealth":
        sensor_labels = [
            'chest_Acc_X', 'chest_Acc_Y', 'chest_Acc_Z',
            'ankle_Acc_X', 'ankle_Acc_Y', 'ankle_Acc_Z',
            'ankle_Gyr_X', 'ankle_Gyr_Y', 'ankle_Gyr_Z',
            'ankle_Mag_X', 'ankle_Mag_Y', 'ankle_Mag_Z',
            'wrist_Acc_X', 'wrist_Acc_Y', 'wrist_Acc_Z',
            'wrist_Gyr_X', 'wrist_Gyr_Y', 'wrist_Gyr_Z',
            'wrist_Mag_X', 'wrist_Mag_Y', 'wrist_Mag_Z'
        ]
        remove_list = [11]
    elif ds_name.lower() == "pamap2":
        sensor_labels = [
            'hand_Acc_X', 'hand_Acc_Y', 'hand_Acc_Z',
            'hand_Gyr_X', 'hand_Gyr_Y', 'hand_Gyr_Z',
            'hand_Mag_X', 'hand_Mag_Y', 'hand_Mag_Z',
            'chest_Acc_X', 'chest_Acc_Y', 'chest_Acc_Z',
            'chest_Gyr_X', 'chest_Gyr_Y', 'chest_Gyr_Z',
            'chest_Mag_X', 'chest_Mag_Y', 'chest_Mag_Z',
            'ank_Acc_X', 'ank_Acc_Y', 'ank_Acc_Z',
            'ank_Gyr_X', 'ank_Gyr_Y', 'ank_Gyr_Z',
            'ank_Mag_X', 'ank_Mag_Y', 'ank_Mag_Z'
        ]
        remove_list = [9, 19, 20, 22]

    labels_dict = {k: v for k, v in enumerate(sensor_labels)}
    [labels_dict.pop(k) for k in remove_list]
    keep_dict = {k: item[1] for k, item in enumerate(labels_dict.items())}
    return keep_dict