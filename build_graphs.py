import argparse
import logging

import networkx as nx
import numpy as np

from pamap2_dataset import Pamap2Dataset
from mhealth_dataset import MHealthDataset
from realdisp_dataset import RealDispDataset
from ucihar_dataset import UCIHARDataset
from torch_geometric.utils import to_networkx, to_undirected

logging.basicConfig(level=logging.NOTSET)

parser = argparse.ArgumentParser()
parser.add_argument('--ds_name', default="mhealth", type=str)
parser.add_argument('--ds_variant', type=str)
parser.add_argument('--fillnan', type=str)
parser.add_argument('--corr_threshold', default=0.2, type=float)
parser.add_argument('--win_size', default=512, type=int)
parser.add_argument('--win_step', default=100, type=int)
args = parser.parse_args()

if __name__ == "__main__":
    assert args.ds_name.lower() in ["mhealth", "pamap2", "ucihar", "realdisp"], f"Dataset {args.ds_name} not supported"

    dataset_class = {
        "ucihar": UCIHARDataset,
        "mhealth": MHealthDataset,
        "pamap2": Pamap2Dataset,
        "realdisp": RealDispDataset
    }

    dataset = dataset_class[args.ds_name.lower()](root=f'data/{args.ds_name}/{args.ds_variant}/',
                                                  variant=args.ds_variant,
                                                  fillnan=args.fillnan,
                                                  threshold=args.corr_threshold,
                                                  win_size=None,
                                                  win_step=None)

    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index)
    nx_graph = to_networkx(data).to_undirected()
    longest_path = np.max([len(c) for c in sorted(nx.connected_components(nx_graph), key=len, reverse=True)])
    print(f"longest path: {longest_path}")

    exit(0)

