import argparse
import os

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
from matplotlib import pyplot as plt
from torch_geometric.explain import Explainer, PGExplainer
from torch_geometric.nn import to_captum_model, to_captum_input
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected, to_dense_adj

import models
import utils
from mhealth_dataset import MHealthDataset
from pair_data import create_data_pairs
from pamap2_dataset import Pamap2Dataset
from ucihar_dataset import UCIHARDataset
from captum.attr import Saliency, IntegratedGradients
from collections import defaultdict

matplotlib.use('svg')


def load_data(ds_name, variant, fillnan=None, batch_size=64, threshold=0.2):
    assert ds_name.lower() in ["mhealth", "pamap2", "ucihar"], f"Dataset {ds_name} not supported"

    root_dir = f'./data/{ds_name}/{variant}'
    if ds_name.lower() == "mhealth":
        dataset_class = MHealthDataset
    elif ds_name.lower() == "ucihar":
        dataset_class = UCIHARDataset
    elif ds_name.lower() == "pamap2":
        dataset_class = Pamap2Dataset

    if "contrastive" in variant:
        ds1 = dataset_class(root=f"{root_dir}/corrcoef_all{variant.split('_')[1]}",
                            variant=f"corrcoef_all{variant}",
                            fillnan=fillnan,
                            threshold=threshold)
        test1 = ds1[ds1.test_mask]

        ds2 = dataset_class(root=f"{root_dir}/corrcoef_win{variant.split('_')[1]}",
                            variant=f"corrcoef_win{variant}",
                            fillnan=fillnan,
                            threshold=threshold)
        test2 = ds2[ds2.test_mask]

        test = DataLoader([create_data_pairs(pair[0], pair[1]) for pair in zip(test1, test2)],
                          batch_size=batch_size,
                          follow_batch=['x_s', 'x_t'],
                          shuffle=False)
    else:
        ds = dataset_class(root=root_dir, variant=variant, fillnan=fillnan, threshold=threshold)
        test = DataLoader(ds[ds.test_mask], batch_size=batch_size, shuffle=False)

    return test


def plot_adj_matrix(ds_name, edges, edge_weights, node_weights, label, variant):
    sensor_names = utils.get_sensor_names(ds_name=ds_name)
    size = len(sensor_names.items())

    adj_mtrx = np.zeros((size, size))
    ticks = np.arange(size)

    lbl_id, label = label

    # lines 70 - 74 fix a bug of matplotlib. It allows matplotlib to apply the font size settings.
    params = {'font.size': 13,
              'axes.labelsize': 16, 'xtick.labelsize': 15, 'ytick.labelsize': 15}
    plt.rcParams.update(params)
    plt.plot()

    for idx in range(edges.shape[1]):
        adj_mtrx[(edges[0, idx], edges[1, idx])] = np.round(edge_weights[idx], 2)

    adj_mtrx = adj_mtrx / adj_mtrx.max()

    fig = plt.figure(figsize=(10, 8))
    im = plt.imshow(adj_mtrx, cmap=plt.cm.viridis)
    fig.colorbar(im, orientation='vertical')

    plt.xticks(ticks, labels=[v for k, v in sensor_names.items()], rotation=90, ha='center')
    plt.yticks(ticks, labels=[v for k, v in sensor_names.items()])

    plt.title(label)
    fig.tight_layout()

    params = {'font.size': 13,
              'axes.labelsize': 16, 'xtick.labelsize': 15, 'ytick.labelsize': 15}
    plt.rcParams.update(params)
    plt.savefig(f'./figs/adj_matrices/{ds_name}/{variant}_{lbl_id:02d}_{label}.svg', format="svg"),

    fig = plt.figure(figsize=(6, 6))
    graph = nx.from_numpy_array(adj_mtrx)
    graph = nx.relabel_nodes(graph, sensor_names)
    pos = nx.circular_layout(graph)
    nx.draw_networkx(graph, pos, node_color=node_weights, node_size=800, cmap=plt.cm.Blues,
                     with_labels=True)

    for edge in graph.edges(data='weight'):
        nx.draw_networkx_edges(graph, pos, edgelist=[edge], width=edge[2]*20)

    plt.title(label)
    plt.tight_layout()
    plt.savefig(f'./figs/adj_matrices/{ds_name}/graph_{variant}_{lbl_id:02d}_{label}.svg', format="svg"),
    plt.show()


# utils.seed_all(123456)
parser = argparse.ArgumentParser()

"""
    PARAMS
"""
parser.add_argument('--ds_name', help="value for ds_name variant", type=str)
parser.add_argument('--ds_variant', help="value for variant", type=str)
parser.add_argument('--fillnan', help="method used to fill NaN values", type=str)
parser.add_argument('--model_name', help="value for model", type=str)
parser.add_argument('--postfix', help="value for model", type=str)
parser.add_argument('--heads', help="number of heads", type=int)
parser.add_argument('--epochs', help="value for epochs", type=int)
parser.add_argument('--batch_size', help="value for batch_size", type=int)
parser.add_argument('--lr', help="value for init_lr", type=float)
parser.add_argument('--w_decay', help="value for weight_decay", type=float)
parser.add_argument('--lr_reduce_factor', help="value for lr_reduce_factor", type=float)
parser.add_argument('--lr_reduce_factor_ft', help="value for lr_reduce_factor_ft", type=float)
parser.add_argument('--lr_schedule_patience', help="value for lr_schedule_patience", type=int)
parser.add_argument('--min_lr', help="value for min_lr", type=float)

"""
    GNN PARAMS
"""
parser.add_argument('--num_layers', help="value for num_layers", type=int)
parser.add_argument('--input_dim', help="value for input_dim", type=int)
parser.add_argument('--hidden_dim', help="value for hidden_dim", type=int)
parser.add_argument('--out_dim', help="value for out_dim", type=int)
parser.add_argument('--aggr', help="value for convolution layer pooling ", type=str)
parser.add_argument('--global_pooling', help="value for global pooling function", type=str)
parser.add_argument('--conv_dropout', help="value for convolution layer dropout", type=float)
parser.add_argument('--classifier_dropout', help="value for dropout", type=float)
parser.add_argument('--batch_norm', help="value for batch_norm", action="store_true")
parser.add_argument('--corruption', choices=['all', 'nodes', 'edge_index'],
                    help="corruption method ['all', 'nodes', 'edge_index']", default=None, type=str)
parser.add_argument('--log_wandb', help="save logs in wandb", action="store_true")
args = parser.parse_args()

device = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = vars(args)
params["is_explainer"] = True

assert args.model_name in ["transf", "gat", "gat2", "graphconv", "gine"], \
    f"Model {args.model_name} not implemented."

if args.model_name == "transf":
    model = models.TransfNet(input_dim=args.input_dim,
                             hc=args.hidden_dim,
                             heads=2,
                             out_channels=args.out_dim).to(device)
elif args.model_name == "gat":
    model = models.GATNet(input_dim=args.input_dim,
                          hc=args.hidden_dim,
                          heads=2,
                          out_channels=args.out_dim).to(device)
elif args.model_name == "gat2":
    model = models.GAT2Net(params=params).to(device)
elif args.model_name == "graphconv":
    if "contrastive" in args.ds_variant:
        global_encoder = models.GraphConvEncoder(net_params=params).to(device)
        win_encoder = models.GraphConvEncoder(net_params=params).to(device)
        model = models.HARGMIdual(global_encoder=global_encoder,
                                  win_encoder=win_encoder,
                                  hidden_channels=args.hidden_dim,
                                  out_dim=args.out_dim,
                                  dropout=args.classifier_dropout,
                                  corruption=args.corruption).to(device)
    else:
        model = models.GraphConvNet(net_params=params).to(device)
elif args.model_name == "gine":
    model = models.GINENet(net_params=params).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)

test_loader = load_data(ds_name=args.ds_name,
                        variant=args.ds_variant,
                        fillnan=args.fillnan,
                        batch_size=args.batch_size)

postfix = args.postfix

if "contrastive" in args.ds_variant:
    variant_name = f"contrastive{args.ds_variant.split('_')[1]}_{args.fillnan}_{args.corruption}" \
        if args.ds_name == "PAMAP2" else f"contrastive{args.ds_variant.split('_')[1]}_{args.corruption}"
else:
    variant_name = f"{args.ds_variant}_{args.fillnan}" if args.ds_name == "PAMAP2" else args.ds_variant
dirs = {
    "best_training_model": f"out/{args.ds_name}/checkpoints/best_training_{variant_name}_{args.model_name}_{postfix}.pth",
    "final_model": f"out/{args.ds_name}/checkpoints/final_{variant_name}_{args.model_name}_{postfix}.pth",
    "training_acc_loss": f"out/{args.ds_name}/training/acc_losses_{variant_name}_{args.model_name}_{postfix}.pkl",
    "results": f"out/{args.ds_name}/training/results_{variant_name}_{args.model_name}_{postfix}.pkl"
}

for path in dirs:
    if not os.path.isfile(dirs[path]):
        os.makedirs(os.path.dirname(dirs[path]), exist_ok=True)

config = params
config["dirs"] = dirs

# load best model during training
checkpoint = torch.load(dirs["final_model"], map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.eval()

path = f'explanations/{args.ds_name}_{args.ds_variant}_{args.model_name}_{postfix}'
explainer_fname = os.path.join(path, "explainer_run.pt")
os.makedirs(path, exist_ok=True)

dataset = test_loader.dataset

labels = dataset.y
labels_dict = dataset.activity_labels
explanations = {}
np.random.seed(123456)

for lbl in np.unique(labels):
    evaluated_graph = np.random.choice(np.where(labels == lbl)[0])
    data = dataset[evaluated_graph]
    # batch = torch.zeros(data.x.shape[0], dtype=int)

    # Edge explainability
    # ===================

    # Captum assumes that for all given input tensors, dimension 0 is
    # equal to the number of samples. Therefore, we use unsqueeze(0).
    captum_model = to_captum_model(model, mask_type='edge')
    edge_mask = torch.ones(data.edge_index.shape[1], requires_grad=True)

    ig = IntegratedGradients(captum_model)
    ig_attr_edge1, delta = ig.attribute(edge_mask.unsqueeze(0), target=data.y,
                                        baselines=torch.zeros(edge_mask.unsqueeze(0).shape),
                                        additional_forward_args=(data.x, data.edge_index, data.edge_weight, data.batch),
                                        internal_batch_size=1,
                                        n_steps=200,
                                        return_convergence_delta=True)

    # Scale attributions to [0, 1]:
    ig_attr_edge = ig_attr_edge1.squeeze(0).abs()
    ig_attr_edge = ig_attr_edge / ig_attr_edge.max()

    topk = ig_attr_edge.topk(k=10)

    edges = data.edge_index[:, topk[1]]
    edge_attr = topk[0]

    # edges = data.edge_index[:, ig_attr_edge.sort(descending=True)[1][:10]]
    # edge_attr = ig_attr_edge.sort(descending=True)[0][:10]

    edges, edge_attr = to_undirected(edges, edge_attr, reduce="mean")

    edges = edges.cpu().detach().numpy()
    weights = edge_attr.cpu().detach().numpy()
    # edges = data.edge_index.cpu().detach().numpy()
    # edge_weights = ig_attr_edge.cpu().detach().numpy()

    # # Node explainability
    # # ===================

    captum_model = to_captum_model(model, mask_type='node')

    ig = IntegratedGradients(captum_model)
    ig_attr_node1 = ig.attribute(data.x.unsqueeze(0), target=data.y,
                                 additional_forward_args=(data.edge_index, data.edge_weight, data.batch),
                                 internal_batch_size=1)

    # Scale attributions to [0, 1]:
    ig_attr_node = ig_attr_node1.squeeze(0).abs().sum(dim=1)
    ig_attr_node /= ig_attr_node.max()

    # Node and edge explainability
    # ============================

    captum_model = to_captum_model(model, mask_type='node_and_edge')

    ig = IntegratedGradients(captum_model)
    ig_attr_node2, ig_attr_edge2 = ig.attribute(
        inputs=(data.x.unsqueeze(0), edge_mask.unsqueeze(0)), target=data.y,
        baselines=(torch.zeros(data.x.unsqueeze(0).shape), torch.zeros(edge_mask.unsqueeze(0).shape)),
        additional_forward_args=(data.edge_index, data.edge_weight, data.batch),
        internal_batch_size=1)

    # Scale attributions to [0, 1]:
    ig_attr_node = ig_attr_node2.squeeze(0).abs().sum(dim=1)
    ig_attr_node /= ig_attr_node.max()
    ig_attr_edge = ig_attr_edge2.squeeze(0).abs()
    ig_attr_edge /= ig_attr_edge.max()

    weights = ig_attr_edge.cpu().detach().numpy()
    node_weights = ig_attr_node.cpu().detach().numpy()

    plot_adj_matrix(ds_name=args.ds_name, edges=edges, edge_weights=weights, node_weights=node_weights,
                    label=(lbl, labels_dict[lbl]), variant=f"{variant_name}_{args.model_name}")

    print("DONE")
