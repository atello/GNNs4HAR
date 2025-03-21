import logging
import os
import pickle
from copy import deepcopy

from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import utils

logging.basicConfig(level=logging.INFO)


def load_data(path):
    all_subject_data = []

    # column_names = [
    #     'chest_Acc_X', 'chest_Acc_Y', 'chest_Acc_Z',
    #     'electro_1', 'electro_2',
    #     'lf_ank_Acc_X', 'lf_ank_Acc_Y', 'lf_ank_Acc_Z',
    #     'lf_Gyr_X', 'lf_Gyr_Y', 'lf_Gyr_Z',
    #     'lf_Mag_X', 'lf_Mag_Y', 'lf_Mag_Z',
    #     'wr_Acc_X', 'wr_Acc_Y', 'wr_Acc_Z',
    #     'wr_Gyr_X', 'wr_Gyr_Y', 'wr_Gyr_Z',
    #     'wr_Mag_X', 'wr_Mag_Y', 'wr_Mag_Z', 'label'
    # ]
    #
    # columns_to_keep = [
    #     'chest_Acc_X', 'chest_Acc_Y', 'chest_Acc_Z',
    #     'lf_ank_Acc_X', 'lf_ank_Acc_Y', 'lf_ank_Acc_Z',
    #     'lf_Gyr_X', 'lf_Gyr_Y', 'lf_Gyr_Z',
    #     'lf_Mag_X', 'lf_Mag_Y', 'lf_Mag_Z',
    #     'wr_Acc_X', 'wr_Acc_Y', 'wr_Acc_Z',
    #     'wr_Gyr_X', 'wr_Gyr_Y', 'wr_Gyr_Z',
    #     'wr_Mag_X', 'wr_Mag_Y', 'wr_Mag_Z', 'label'
    # ]

    for file_i in os.listdir(path):
        if "_ideal" not in file_i:  # and "_self" not in file_i:
            continue
        subject_path = path + "/" + file_i
        logging.info(f"loading '{subject_path}' file")
        subject_data = pd.read_csv(subject_path, sep='\t', header=None)

        k = 2
        columns_to_keep = list(range(2, 11))
        for i in range(8):
            for j in range(k, k + 9):
                columns_to_keep.append(j + 13)
            k = k + 13
        columns_to_keep.append(subject_data.shape[1] - 1)

        subject_data = subject_data.iloc[:, columns_to_keep]

        assert np.sum(subject_data.isna().sum()) == 0, "NAN Values"

        subject_data["user"] = int(file_i.split("_")[0][7:])

        all_subject_data.append(subject_data)

    all_subject_data = pd.concat(all_subject_data, axis=0)
    all_subject_data.reset_index(drop=True, inplace=True)

    # remove the transition data, label=0
    activities_to_keep = [1, 2, 3, 5, 8, 11, 20, 26, 28, 33]
    activities_to_delete = [i for i in range(34) if i not in activities_to_keep]
    for i in activities_to_delete:
        all_subject_data.drop(index=list(all_subject_data[all_subject_data.iloc[:, -2] == i].index), inplace=True)
        all_subject_data.reset_index(drop=True, inplace=True)

    X = np.asarray(all_subject_data)
    y = np.asarray(all_subject_data.iloc[:, -2])
    groups = np.asarray(all_subject_data["user"]).astype(int)

    # sgkf = StratifiedGroupKFold(n_splits=3, random_state=None, shuffle=False)
    # train_idx, test_idx = next(sgkf.split(X, y, groups))
    # groups2 = groups[train_idx]
    # train_idx, val_idx = next(sgkf.split(X[train_idx], y[train_idx], groups2))
    #
    # print(f"train: {np.unique(groups2[train_idx])}, \n"
    #       f"val: {np.unique(groups2[val_idx])}, \n"
    #       f"test: {np.unique(groups[test_idx])}")

    val_subjects = [4, 6, 10, 11]
    test_subjects = [1, 7, 8, 9, 12, 14]
    train_subjects = list(set(range(18)).difference(set(val_subjects).union(test_subjects)))

    # train: [2  3  5 13 15 16 17],
    # val: [4  6 10 11],
    # test: [1  7  8  9 12 14]

    # train_mask used to calculate stats for data normalization from training set.
    train_mask = np.in1d(groups, train_subjects)
    val_mask = np.in1d(groups, val_subjects)
    test_mask = np.in1d(groups, test_subjects)

    splits = {
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask
    }

    x_copy = X[train_mask].copy().astype(float)
    mean, std = x_copy[:, :-2].mean(axis=0), x_copy[:, :-2].std(axis=0)
    minv, maxv = x_copy[:, :-2].min(axis=0), x_copy[:, :-2].max(axis=0)

    stats = {
        "mean": mean.reshape(-1, 1),
        "std": std.reshape(-1, 1),
        "min": minv.reshape(-1, 1),
        "max": maxv.reshape(-1, 1)
    }

    X[:, :-2] = (X[:, :-2] - mean) / std

    return X, y, splits, stats


# Windows per user and activity
def create_samples_windows(x_data, y, window_size, step):
    x_data = x_data.astype(float)
    users = np.sort(np.unique(x_data[:, -1])).astype(int)
    act_dict = {int(act): i for i, act in enumerate(np.sort(np.unique(y)))}
    X_win = []
    X_group = []
    y_win = []
    for usr_i in users:
        idx = np.where(x_data[:, -1] == usr_i)[0]
        usr_samples = x_data[idx]
        act_labels = np.sort(np.unique(usr_samples[:, -2]))
        for label in act_labels:
            idx_act = np.where(usr_samples[:, -2] == label)[0]
            samples = usr_samples[idx_act, :-2]

            for i in range(0, len(samples), step):
                if i + window_size <= len(samples):
                    window = samples[i: i + window_size]
                    assert len(window) > 0, f"Empty window. User: {usr_i}, label: {label}"
                    X_win.append(window)
                    y_win.append(act_dict[label])
                    X_group.append(usr_i)
                else:
                    break

    return np.array(X_win).astype(float), np.array(y_win).astype(int), np.array(X_group).astype(int)


def to_pyg_graphs(data, labels, groups):
    graphs = []

    for idx in range(data.shape[0]):
        win = data[idx].T

        g = Data(x=torch.tensor(win, dtype=torch.float),
                 y=torch.tensor(labels[idx], dtype=torch.long),
                 subject=torch.tensor(groups[idx], dtype=torch.long))
        graphs.append(g)
    return graphs


def create_graphs_for_ensembles(x_data, y, graphs, threshold=0.2):
    """
    Create graphs for training N ensemble binary models. N is calculated from the unique labels in 'y'.
    A different similarity matrix based on Pearson Correlation coefficient is calculated for each activity. Then,
    'y' is set to 1 for all the windows that corresponds to the activity for which the graphs are being created,
    and 0 otherwise.

    @param x_data: raw input data from the training set
    @param y: labels of raw input data from the training set
    @param graphs:  pre-computed graphs with no-edges, in PyG format
    @param threshold: an edge is created if the correlation coefficient is above 'threshold'
    @return: A dictionary with activity-label as key, which values are the graphs and its corresponding
    correlation coefficient for each activity.
    """
    activities = np.sort(np.unique(y))
    act_dict = {int(act): i for i, act in enumerate(activities)}
    graphs_per_activity = {}

    for act in tqdm(activities):
        gtemp = deepcopy(graphs)

        idx = np.where(y == act)[0]
        signals = x_data[idx]
        corr = np.corrcoef(signals.T)

        for g in gtemp:
            edge_index, edge_weight, unique_nodes = utils.compute_edges(corr, threshold)
            g.x = g.x[unique_nodes]
            g.edge_index = edge_index.t().contiguous()
            g.edge_weight = edge_weight
            g.y = 1 if g.y == act_dict[act] else 0

        graphs_per_activity[act_dict[act]] = {"graphs": deepcopy(gtemp), "corr": corr}
    return graphs_per_activity


def create_window_graph(no_edges_graph, sim_mtrx, threshold):
    edge_index, edge_weight, unique_nodes = utils.compute_edges(sim_mtrx, threshold)

    graph = Data(x=no_edges_graph.x[unique_nodes],
                 edge_index=edge_index.t().contiguous(),
                 edge_weight=edge_weight,
                 y=no_edges_graph.y,
                 subject=no_edges_graph.subject)
    return graph


def load_preprocessed_data(raw_data_path, **kwargs):
    X, y, splits, stats = load_data(raw_data_path)

    logging.info("Creating windows")
    X_win, y_win, groups = create_samples_windows(X, y, window_size=128, step=64)

    logging.info("Creating PyG graphs")
    no_edges_graphs = to_pyg_graphs(data=X_win, labels=y_win, groups=groups)

    # used only for computing the similarity matrices
    x_train = X[splits["train_mask"], :-2].astype(float)
    y_train = y[splits["train_mask"]].astype(int)

    ds_variant = kwargs["ds_variant"]
    threshold = kwargs["threshold"]

    if "corrcoef" in ds_variant:
        sim_mtrx = np.corrcoef(x_train.T)
    elif "gaussian" in ds_variant:
        sim_mtrx = utils.gaussian_kernel(x_train.T).numpy()
    elif ds_variant == "ensemble":
        ret = create_graphs_for_ensembles(x_data=x_train, y=y_train, graphs=no_edges_graphs)
    elif ds_variant == "no_edges":
        ret = no_edges_graphs

    if "corrcoef" in ds_variant or "gaussian" in ds_variant:
        graphs = []
        try:
            for g in tqdm(no_edges_graphs):
                if "_win" in ds_variant:
                    sim_mtrx = np.corrcoef(g.x) if "corrcoef" in ds_variant else utils.gaussian_kernel(g.x).numpy()
                full_graph = create_window_graph(g.clone(), sim_mtrx, threshold)
                graphs.append(full_graph)
        except RuntimeWarning as rtw:
            print(rtw)
        ret = graphs

    return ret, stats


def get_ts_data(raw_data_path, preproc_data_file, win_size, step, train_subjects, val_subjects, test_subjects):

    if os.path.isfile(preproc_data_file):
        print("loading pre-processed data file")
        ts_data = pickle.load(open(preproc_data_file, 'rb'))
        x_train = ts_data["x_train"]
        x_val = ts_data["x_val"]
        x_test = ts_data["x_test"]
        y_train = ts_data["y_train"]
        y_val = ts_data["y_val"]
        y_test = ts_data["y_test"]
    else:
        X, y, splits, stats = load_data(raw_data_path)
        X_win, y_win, groups = create_samples_windows(X, y, window_size=win_size, step=step)

        train_mask = np.in1d(groups, train_subjects)
        val_mask = np.in1d(groups, val_subjects)
        test_mask = np.in1d(groups, test_subjects)

        x_train = X_win[train_mask]
        x_val = X_win[val_mask]
        x_test = X_win[test_mask]

        y_train = y_win[train_mask]
        y_val = y_win[val_mask]
        y_test = y_win[test_mask]

        ts_data = {
            "x_train": x_train,
            "x_val": x_val,
            "x_test": x_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test
        }

        torch.save(ts_data, preproc_data_file)

        pickle.dump(
            obj=ts_data,
            file=open(preproc_data_file, 'wb')
        )

    return x_train, x_val, x_test, y_train, y_val, y_test