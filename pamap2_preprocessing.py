import logging
import os
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import utils

logging.basicConfig(level=logging.INFO)


def load_data(path, fillnan="interpolate"):
    all_subject_data = []

    columns_to_keep = (
            [1] +  # label
            list(range(4, 7)) + list(range(10, 16)) +  # IMU hand
            list(range(21, 24)) + list(range(27, 33)) +  # IMU chest
            list(range(38, 41)) + list(range(44, 50))  # IMU ankle
    )

    for file_i in tqdm(os.listdir(path)):
        subject_path = f"{path}/{file_i}"
        logging.info(f"loading '{subject_path}' file")
        subject_data = pd.read_csv(subject_path, sep=' ', header=None)
        subject_data = subject_data.iloc[:, columns_to_keep]
        subject_data.drop(index=list(subject_data[subject_data.iloc[:, 0] == 0].index), inplace=True)
        subject_data.reset_index(drop=True, inplace=True)

        if fillnan == "interpolate":
            logging.info("Filling missing values, interpolation.")
            subject_data.interpolate(method="linear", axis=0, inplace=True)
        elif fillnan == "zero":
            logging.info("Filling missing values, zeros.")
            subject_data.fillna(0, inplace=True)
        elif fillnan == "dropna":
            logging.info("Deleting rows with NaN values")
            subject_data.dropna(axis=0, inplace=True)

        subject_data["user"] = int(file_i.split(".")[0][-3:])
        all_subject_data.append(subject_data)

    all_subject_data = pd.concat(all_subject_data, axis=0)
    all_subject_data.reset_index(drop=True, inplace=True)

    # remove the transition data, label=0
    all_subject_data.drop(index=list(all_subject_data[all_subject_data.iloc[:, 0] == 0].index), inplace=True)
    all_subject_data.reset_index(drop=True, inplace=True)

    X = np.asarray(all_subject_data)
    y = np.asarray(all_subject_data.iloc[:, 0])
    groups = np.asarray(all_subject_data["user"]).astype(int)

    train_subjects = [102, 104, 106, 108, 109]
    val_subjects = [101, 107]
    test_subjects = [103, 105]

    trmask, valmask, testmask = get_split_masks(train_subjects=train_subjects,
                                                val_subjects=val_subjects,
                                                test_subjects=test_subjects,
                                                subjects=groups)

    x_copy = X[trmask].astype(float)
    mean, std = x_copy[:, 1:-1].mean(axis=0), x_copy[:, 1:-1].std(axis=0)
    minv, maxv = x_copy[:, 1:-1].min(axis=0), x_copy[:, 1:-1].max(axis=0)

    stats = {
        "mean": mean.reshape(-1, 1),
        "std": std.reshape(-1, 1),
        "min": minv.reshape(-1, 1),
        "max": maxv.reshape(-1, 1)
    }

    splits = {
        "train_mask": trmask,
        "val_mask": valmask,
        "test_mask": testmask
    }
    X[:, 1:-1] = (X[:, 1:-1] - mean) / std

    return X, y, splits, stats


# Windows per user and activity
def create_samples_windows(x_data, y, window_size, step):
    x_data = x_data.astype(float)
    users = np.sort(np.unique(x_data[:, -1])).astype(int)
    act_dict = {int(act): i for i, act in enumerate(np.sort(np.unique(y)))}
    X_win = []
    X_group = []
    y_win = []
    for usr_i in tqdm(users):
        idx = np.where(x_data[:, -1] == usr_i)[0]
        usr_samples = x_data[idx]
        act_labels = np.sort(np.unique(usr_samples[:, 0]))
        for label in act_labels:
            idx_act = np.where(usr_samples[:, 0] == label)[0]
            samples = usr_samples[idx_act, 1:-1]

            for i in range(0, len(samples), step):
                if i + window_size <= len(samples):
                    window = samples[i: i + window_size]
                    assert len(window) > 0, f"Empty window. User: {usr_i}, label: {label}"
                    X_win.append(window)
                    y_win.append(act_dict[label])
                    X_group.append(usr_i)
                else:
                    continue

    return np.array(X_win).astype(float), np.array(y_win).astype(int), np.array(X_group).astype(int)


def to_pyg_graphs(data, labels, groups):
    graphs = []

    for idx in tqdm(range(data.shape[0])):
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


def get_split_masks(train_subjects, val_subjects, test_subjects, subjects):
    train_mask = np.in1d(subjects, train_subjects)
    val_mask = np.in1d(subjects, val_subjects)
    test_mask = np.in1d(subjects, test_subjects)
    return train_mask, val_mask, test_mask


def load_preprocessed_data(raw_data_path, **kwargs):

    ds_variant = kwargs["ds_variant"]
    threshold = kwargs["threshold"]
    fillnan = kwargs["fillnan"]
    win_size = kwargs.get("win_size", 512)
    win_step = kwargs.get("win_step", 100)

    X, y, splits, stats = load_data(raw_data_path, fillnan=fillnan)

    logging.info("Creating windows")
    X_win, y_win, groups = create_samples_windows(X, y, window_size=win_size, step=win_step)

    logging.info("Creating PyG graphs")
    no_edges_graphs = to_pyg_graphs(data=X_win, labels=y_win, groups=groups)

    x_train = X[splits["train_mask"], 1:-1].astype(float)
    y_train = y[splits["train_mask"]].astype(int)

    if "corrcoef" in ds_variant:
        sim_mtrx = np.corrcoef(x_train.T)
    elif "gaussian" in ds_variant:
        sim_mtrx = utils.gaussian_kernel(x_train.T).numpy()
    elif "ensemble" in ds_variant:
        logging.info("Transforming graphs to binary per base model for ensemble")
        ret = create_graphs_for_ensembles(x_data=x_train, y=y_train, graphs=no_edges_graphs, threshold=threshold)
    elif "no_edges" in ds_variant:
        ret = no_edges_graphs

    if "corrcoef" in ds_variant or "gaussian" in ds_variant:
        graphs = []
        for g in tqdm(no_edges_graphs):
            if "_win" in ds_variant:
                sim_mtrx = np.corrcoef(g.x) if "corrcoef" in ds_variant else utils.gaussian_kernel(g.x).numpy()
            full_graph = create_window_graph(g.clone(), sim_mtrx, threshold)
            graphs.append(full_graph)
        ret = graphs

    return ret
