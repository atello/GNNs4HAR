import argparse
import logging
import os
import pickle
from datetime import datetime

from torch import nn

import wandb

import mhealth_preprocessing
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import models
from mhealth_dataset import MHealthDataset
from pamap2_dataset import Pamap2Dataset
from ucihar_dataset import UCIHARDataset
from realdisp_dataset import RealDispDataset
import utils

logging.basicConfig(level=logging.INFO, format="'%(levelname)s:%(message)s'")


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else scheduler,
        'loss': loss,
    }, path)


def load_checkpoint(path):
    return torch.load(path)


# def train_one_epoch(loader, model, optimizer, criterion, device=torch.device("cpu")):
#     model.train()
#
#     total_loss = 0
#     total_correct = 0
#     for data in loader:  # Iterate in batches over the training ds_name.
#         sample_weight = torch.tensor(compute_sample_weight(class_weight="balanced", y=data.y), dtype=float)
#         data = data.to(device)
#         optimizer.zero_grad()  # Clear gradients.
#         out = model(data.x, data.edge_index, data.edge_weight, data.batch)  # Perform a single forward pass.
#
#         criterion.weight = sample_weight.to(device)
#         loss = criterion(out.view(-1), data.y.float())  # Compute the loss.
#         pred = out.sigmoid().round().float()  # out.sigmoid(dim=1)  # Use the class with the highest probability.
#
#         loss.backward()  # Derive gradients.
#         optimizer.step()  # Update parameters based on gradients.
#         total_loss += float(loss) * data.num_graphs
#         total_correct += int((pred.view(-1) == data.y).sum())  # Check against ground-truth labels.
#
#     return total_loss / len(loader.dataset), total_correct / len(loader.dataset)


@torch.no_grad()
def test_one_epoch(loader, model, device=torch.device("cpu")):
    model.eval()

    target = []
    predicted = []
    total_correct = 0
    total_loss = 0
    for data in loader:  # Iterate in batches over the training/test ds_name.
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_weight, data.batch)
        pred = out.sigmoid().view(-1)

        predicted.extend(pred.detach().cpu().numpy())
        target.extend(data.y.detach().cpu().numpy())

    return target, predicted


def load_data(ds_name, ds_variant, fillnan=None, batch_size=64, threshold=0.2):
    assert ds_name.lower() in ["mhealth", "pamap2", "ucihar", "realdisp"], f"Dataset {ds_name} not supported"

    if ds_name.lower() == "mhealth":
        dataset_class = MHealthDataset
    elif ds_name.lower() == "ucihar":
        dataset_class = UCIHARDataset
    elif ds_name.lower() == "pamap2":
        dataset_class = Pamap2Dataset
    elif ds_name.lower() == "realdisp":
        dataset_class = RealDispDataset

    # load the dataset that contains the test data

    no_edges_data = dataset_class(root=f'./data/{ds_name}/no_edges', fillnan=fillnan, variant="no_edges")
    test_data = no_edges_data[no_edges_data.test_mask]

    logging.info("Loading testing data")
    # load the correlation matrices used to create the adjacency matrices for each base model
    dataset = dataset_class(root=f'data/{ds_name}/{ds_variant}', variant=ds_variant, fillnan=fillnan,
                            threshold=threshold, all_corr_matrices=True)
    corr_matrices = dataset.all_corr_matrices

    logging.info("Creating graphs from test data")
    graphs_per_activity = {}
    for key in tqdm(corr_matrices):
        sim_mtrx = corr_matrices[key]
        graphs = []
        for g in test_data:
            graph = mhealth_preprocessing.create_window_graph(g.clone(), sim_mtrx, 0.2)
            graphs.append(graph)
        graphs_per_activity[key] = DataLoader(graphs, batch_size=batch_size, shuffle=False)

    return graphs_per_activity


if __name__ == "__main__":
    utils.seed_all(123456)
    os.environ["WANDB_API_KEY"] = "43fd1443427b4cd3b65de44e599483daa7a7dc2b"

    parser = argparse.ArgumentParser()

    """
        PARAMS
    """
    parser.add_argument('--ds_name', help="value for ds_name variant", type=str)
    parser.add_argument('--ds_variant', help="value for variant", type=str)
    parser.add_argument('--activity', help="value for variant", type=str)
    parser.add_argument('--fillnan', choices=['interpolate', 'dropna', 'zero'],
                        help="method for filling nan values", default=None, type=str)
    parser.add_argument('--model_name', help="value for model", type=str)
    parser.add_argument('--batch_size', help="value for batch_size", type=int)
    parser.add_argument('--heads', help="value for batch_size", default=-1, type=int)
    parser.add_argument('--log_wandb', help="value for batch_norm", action="store_true")

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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = vars(args)
    params["is_explainer"] = False

    if args.ds_name == "PAMAP2":
        assert args.fillnan is not None, "One option for parameter 'fillnan': ['interpolate', 'dropna', 'zero'] is " \
                                       "required for PAMAP2 dataset."

    if args.model_name in ["gat", "gat2"]:
        assert args.heads > 0, "Number of heads is not provided"
    else:
        params.pop("heads")

    try:

        assert args.model_name in ["transf", "gat", "gat2", "graphconv", "gine"], \
            f"Model {args.model_name} not implemented."

        if args.model_name == "transf":
            model = models.TransfNet(params=params).to(device)
        elif args.model_name == "gat":
            model = models.GATNet(params=params).to(device)
        elif args.model_name == "gat2":
            model = models.GAT2Net(params=params).to(device)
        elif args.model_name == "graphconv":
            model = models.GraphConvNet(net_params=params).to(device)
        elif args.model_name == "gine":
            model = models.GINENet(net_params=params).to(device)

        model.lin1 = nn.Linear(model.lin1.in_features, model.lin1.out_features * 4)
        model.lin2 = nn.Linear(model.lin2.in_features * 4, 1)
        model = model.to(device)

        test_loaders = load_data(ds_name=args.ds_name,
                                 ds_variant=args.ds_variant,
                                 fillnan=args.fillnan,
                                 batch_size=args.batch_size)

        criterion = torch.nn.BCEWithLogitsLoss().to(device)

        postfix = datetime.today().strftime('%Y%m%d_%H%M%S') if args.log_wandb else "TEST"

        dirs = {
            "base_models": f"out/{args.ds_name}/ensembles/checkpoints",
            "ensemble_results": f"out/{args.ds_name}/ensembles/testing/results_{args.ds_variant}_{args.model_name}_{postfix}.pkl"
        }

        for path in dirs:
            if not os.path.isfile(dirs[path]):
                os.makedirs(os.path.dirname(dirs[path]), exist_ok=True)

        config = params
        config["dirs"] = dirs

        wandb_run = None
        if args.log_wandb:
            wandb_run = wandb.init(
                # Set the project where this run will be logged
                project="GNNsHAR",
                save_code=True,
                group="final_models_evaluation",
                tags=[args.ds_name, args.ds_variant, args.model_name],
                name=f"{args.ds_name}_{args.ds_variant}_{args.model_name}_{postfix}",
                # Track hyperparameters and run metadata
                config=config,
            )

        logging.info("Model evaluation results")
        logging.info('-' * 89)

        predicted_probs = []
        true_labels = []
        bm_accuracies = []
        base_models_file_names = os.listdir(f'{dirs["base_models"]}/')
        base_models_file_names = [f for f in base_models_file_names if f'final_{args.ds_variant}_bm_' in f]
        for loader_key in test_loaders:
            filename = [f for f in base_models_file_names if
                        f'final_{args.ds_variant}_bm_{loader_key}_{args.model_name}' in f]
            checkpoint = torch.load(f'{dirs["base_models"]}/{filename[0]}')
            model.load_state_dict(checkpoint['model_state_dict'])

            model.eval()
            # evaluate final model on hold out test set
            y_test, y_pred = test_one_epoch(
                loader=test_loaders[loader_key], model=model, device=device
            )
            true_labels.append(y_test)
            predicted_probs.append(y_pred)

        y_test = np.array(true_labels[0])
        y_pred = np.argmax(predicted_probs, axis=0)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred, normalize=True)
        precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        results = {
            "y_test": y_test,
            "y_pred": y_pred,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        filepath = dirs["ensemble_results"]
        pickle.dump(obj=results, file=open(filepath, 'wb'))

        if wandb_run is not None:
            wandb_run.log({
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }, commit=True)

        print("accuracy: {}".format(accuracy))
        print("f_score: {}".format(f1))

        logging.info('-' * 89)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Model initialization stopped by user (Ctrl + C)')
