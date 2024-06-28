import logging

from matplotlib import pyplot as plt

from mhealth_dataset import MHealthDataset
from torch_geometric.loader import DataLoader
import numpy as np

from pamap2_dataset import Pamap2Dataset
import utils

logging.basicConfig(level=logging.NOTSET)


if __name__ == "__main__":
    DSNAME = "PAMAP2"
    ACTIVITY = 1

    dataset = Pamap2Dataset(root='data/PAMAP2/corrcoef_all', variant="corrcoef_all", fillnan="interpolate", activity=ACTIVITY)

    train = DataLoader(dataset[dataset.train_mask], batch_size=64, shuffle=True)
    val = DataLoader(dataset[dataset.val_mask], batch_size=64, shuffle=False)
    test = DataLoader(dataset[dataset.test_mask], batch_size=64, shuffle=False)

    if DSNAME == "PAMAP2":
        ticks = [
            'hand_Acc_X', 'hand_Acc_Y', 'hand_Acc_Z',
            'hand_Acc2_X', 'hand_Acc2_Y', 'hand_Acc2_Z',
            'hand_Gyr_X', 'hand_Gyr_Y', 'hand_Gyr_Z',
            'hand_Mag_X', 'hand_Mag_Y', 'hand_Mag_Z',
            'chest_Acc_X', 'chest_Acc_Y', 'chest_Acc_Z',
            'chest_Acc2_X', 'chest_Acc2_Y', 'chest_Acc2_Z',
            'chest_Gyr_X', 'chest_Gyr_Y', 'chest_Gyr_Z',
            'chest_Mag_X', 'chest_Mag_Y', 'chest_Mag_Z',
            'ank_Acc_X', 'ank_Acc_Y', 'ank_Acc_Z',
            'ank_Acc2_X', 'ank_Acc2_Y', 'ank_Acc2_Z',
            'ank_Gyr_X', 'ank_Gyr_Y', 'ank_Gyr_Z',
            'ank_Mag_X', 'ank_Mag_Y', 'ank_Mag_Z'
        ]

    elif DSNAME == "MHEALTH":
        ticks = [
            'chest_Acc_X', 'chest_Acc_Y', 'chest_Acc_Z',
            'lft_ank_Acc_X', 'lft_ank_Acc_Y', 'lft_ank_Acc_Z',
            'lft_ank_Gyr_X', 'lft_ank_Gyr_Y', 'lft_ank_Gyr_Z',
            'lft_ank_Mag_X', 'lft_ank_Mag_Y', 'lft_ank_Mag_Z',
            'rght_wrist_Acc_X', 'rght_wrist_Acc_Y', 'rght_wrist_Acc_Z',
            'rght_wrist_Gyr_X', 'rght_wrist_Gyr_Y', 'rght_wrist_Gyr_Z',
            'rght_wrist_Mag_X', 'rght_wrist_Mag_Y', 'rght_wrist_Mag_Z'
        ]
    elif DSNAME == "UCIHAR":
        ticks = [
            'body_Acc_X', 'body_Acc_Y', 'body_Acc_Z',
            'body_Gyr_X', 'body_Gyr_Y', 'body_Gyr_Z',
            'total_Acc_X', 'total_Acc_Y', 'total_Acc_Z'
        ]

    mhealth_correlations = MHealthDataset(root=f'data/{DSNAME}/ensemble', variant="ensemble", all_corr_matrices=True)
    corr_matrx = mhealth_correlations.all_corr_matrices[ACTIVITY]

    edge_index, _, _ = utils.compute_edges(corr_matrx, 0.2)
    adj = np.zeros((21, 21))
    for idx in edge_index.numpy():
        adj[idx[0], idx[1]] = np.abs(corr_matrx[idx[0], idx[1]])

    minv, maxv = np.min(adj), np.max(adj)
    adj = (adj-minv) / (maxv - minv)

    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(adj, cmap=plt.cm.viridis)
    ax.set_xticks(range(0, len(ticks)), labels=[t for t in ticks], rotation=45, ha="right", rotation_mode="anchor",
                  fontsize=16)
    ax.set_yticks(range(len(ticks)), labels=[t for t in ticks], fontsize=20)
    plt.rcParams.update({'font.size': 22})
    plt.rc('axes', labelsize=18)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)  # fontsize of the tick labels
    # fig.colorbar(im, ax=ax)
    plt.title(mhealth.activity_name)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.show()
