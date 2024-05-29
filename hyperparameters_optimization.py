import argparse

import logging
import os
import pickle
import numpy as np
import torch
import models
import utils

from functools import partial
from datetime import datetime
from ray import tune
from ray.air import session
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from sklearn.utils import compute_class_weight
from torch_geometric.loader import DataLoader

from mhealth_dataset import MHealthDataset
from pamap2_dataset import Pamap2Dataset
from ucihar_dataset import UCIHARDataset
from realdisp_dataset import RealDispDataset

# logging.disable(logging.INFO)
logging.basicConfig(level=logging.INFO)


def save_checkpoint(model, optimizer, epoch, loss, acc, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'acc': acc
    }, path)


def load_data(ds_name, variant, batch_size):
    assert ds_name.lower() in ["mhealth", "pamap2", "ucihar", "realdisp"], f"Dataset {ds_name} not supported"

    root = os.path.abspath("/home/andres/Dropbox/PhD Smart Environments - RUG/Experiments/GNNs/gnn_ensemble/data")

    fillnan = None
    if ds_name.lower() == "mhealth":
        root_dir = f'{root}/{ds_name}/{variant}'
        dataset_class = MHealthDataset
    elif ds_name.lower() == "ucihar":
        root_dir = f'{root}/{ds_name}/{variant}'
        dataset_class = UCIHARDataset
    elif ds_name.lower() == "realdisp":
        root_dir = f'{root}/{ds_name}/{variant}'
        dataset_class = RealDispDataset
    elif ds_name.lower() == "pamap2":
        root_dir = f"{root}/{ds_name}/{'_'.join(variant.split('_')[:-1])}"
        fillnan = variant.split('_')[-1]
        variant = '_'.join(variant.split('_')[:-1])
        dataset_class = Pamap2Dataset

    dataset = dataset_class(root=root_dir, variant=variant, fillnan=fillnan, threshold=0.2)

    # ds_train = dataset[dataset.train_mask].copy()
    # ds_train = np.asarray([w.T.numpy() for w in ds_train.data.x.reshape(-1, 21, 128)]).reshape(-1, 21)
    # mean, std = ds_train.mean(axis=0, keepdims=True), ds_train.std(axis=0, keepdims=True)
    # dataset.data.x = (dataset.data.x.reshape(-1, 23, 512) - mean.reshape(-1,1)) / std.reshape(-1,1)
    # dataset.data.x = dataset.data.x.reshape(-1,512)

    # ds_train = dataset[dataset.train_mask].copy()
    # ds_train = ds_train.data.x.reshape(-1, 512).clone()
    # mean, std = ds_train.mean(axis=0, keepdims=True), ds_train.std(axis=0, keepdims=True)
    # dataset.data.x = (dataset.data.x - mean) / std

    train = DataLoader(dataset[dataset.train_mask], batch_size=batch_size, shuffle=True)
    val = DataLoader(dataset[dataset.val_mask], batch_size=batch_size, shuffle=False)
    test = DataLoader(dataset[dataset.test_mask], batch_size=batch_size, shuffle=False)

    return train, val, test


def train_one_epoch(loader, model, optimizer, criterion, device=torch.device("cpu")):
    model.train()

    total_loss = 0
    total_correct = 0
    for data in loader:  # Iterate in batches over the training ds_name.
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_weight, data.batch)  # Perform a single forward pass.

        loss = criterion(out, data.y)  # Compute the loss.
        pred = out.argmax(dim=1)  # Use the class with the highest probability.

        loss.backward()  # Derive gradients.
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

        total_loss += float(loss) * data.num_graphs
        total_correct += int((pred == data.y).sum())  # Check against ground-truth labels.

    return total_loss / len(loader.dataset), total_correct / len(loader.dataset)


@torch.no_grad()
def test_one_epoch(loader, model, criterion, device=torch.device("cpu")):
    model.eval()

    target = []
    predicted = []
    total_correct = 0
    total_loss = 0
    for data in loader:  # Iterate in batches over the training/test ds_name.
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_weight, data.batch)
        pred = out.argmax(dim=1)  # Use the class with the highest probability.

        loss = criterion(out, data.y)  # Compute the loss.
        total_loss += float(loss) * data.num_graphs
        total_correct += int((pred == data.y).sum())  # Check against ground-truth labels.

        predicted.extend(pred.detach().cpu().numpy())
        target.extend(data.y.detach().cpu().numpy())

    return total_loss / len(loader.dataset), total_correct / len(loader.dataset), target, predicted


def trainer(config):
    # trial_id, params = train_params
    params = config
    params["is_explainer"] = False

    # # Initialize a new aim Run
    # postfix = datetime.today().strftime('%Y%m%d_%H%M')
    # trial_name = "_".join(key + "_" + str(value) for key, value in params.items())
    # trial_name = f"trial_{str(trial_id).zfill(3)}_{trial_name}_{postfix}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # assert params["model_name"] in ["transf", "gat", "gat2", "graphconv", "gine"], \
    #     f"Model {params["model_name"]} not implemented."

    if params["model_name"] == "transf":
        model = models.TransfNet(input_dim=params["input_dim"],
                                 hc=params["hidden_dim"],
                                 heads=2,
                                 out_channels=params["out_dim"]).to(device)
    elif params["model_name"] == "gat":
        model = models.GATNet(input_dim=params["input_dim"],
                              hc=params["hidden_dim"],
                              heads=2,
                              out_channels=params["out_dim"]).to(device)
    elif params["model_name"] == "gat2":
        model = models.GAT2Net(params=params).to(device)
    elif params["model_name"] == "graphconv":
        model = models.GraphConvNet(net_params=params).to(device)
    elif args.model_name == "mpool":
        model = models.MPoolGNN(net_params=params).to(device)
    elif params["model_name"] == "gine":
        model = models.GINENet(net_params=params).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["w_decay"])

    train_loader, val_loader, test_loader = load_data(ds_name=params["ds_name"],
                                                      variant=params["ds_variant"],
                                                      batch_size=params["batch_size"])

    all_labels = train_loader.dataset.data.y[train_loader.dataset.train_mask]
    class_weight = torch.tensor(compute_class_weight(class_weight="balanced",
                                                     classes=np.unique(all_labels),
                                                     y=all_labels.numpy()), dtype=torch.float)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight).to(device)

    dirs = {
        "best_trial_models": f"checkpoints/best_trial.pth",
        "training_acc_loss": f"training/acc_losses.pkl"
    }

    for path in dirs:
        if not os.path.isfile(dirs[path]):
            os.makedirs(os.path.dirname(dirs[path]), exist_ok=True)

    #########################################################################

    # checkpoint = session.get_checkpoint()
    #
    # if checkpoint:
    #     checkpoint_state = checkpoint.to_dict()
    #     start_epoch = checkpoint_state["epoch"]
    #     model.load_state_dict(checkpoint_state["net_state_dict"])
    #     optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    # else:
    #     start_epoch = 0

    train_loss_arr = []
    train_acc_arr = []
    val_loss_arr = []
    val_acc_arr = []

    best_epoch = 0
    best_loss = np.inf
    best_acc = 0
    no_improve = 0
    for epoch in range(0, params["epochs"]):
        tr_loss, tr_acc = train_one_epoch(
            loader=train_loader, model=model, optimizer=optimizer, criterion=criterion, device=device
        )
        train_loss_arr.append(tr_loss)
        train_acc_arr.append(tr_acc)

        val_loss, val_acc, _, _ = test_one_epoch(
            loader=val_loader, model=model, criterion=criterion, device=device
        )
        val_loss_arr.append(val_loss)
        val_acc_arr.append(val_acc)

        if val_loss < best_loss:
            no_improve = 0
            best_epoch = epoch
            best_loss = val_loss
            best_acc = val_acc

            save_checkpoint(
                model=model, optimizer=optimizer, epoch=best_epoch, loss=best_loss,
                acc=best_acc, path=f"{dirs['best_trial_models']}"
            )

        else:
            no_improve += 1

        training_acc_loss = {
            "train_loss": train_loss_arr,
            "train_acc": train_acc_arr,
            "val_loss": val_loss_arr,
            "val_acc": val_acc_arr
        }

        pickle.dump(
            obj=training_acc_loss,
            file=open(dirs["training_acc_loss"], 'wb')
        )

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be accessed through in ``get_checkpoint()``
        # in future iterations.
        # Not
        # e to save a file like checkpoint, you still need to put it under a directory
        # to construct a checkpoint.
        # with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        #     path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
        #     torch.save(
        #         (model.state_dict(), optimizer.state_dict()), path
        #     )
        #     checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
        session.report(
            {"tr_loss": tr_loss, "tr_acc": tr_acc, "val_loss": val_loss, "val_acc": val_acc, "best_loss": best_loss}
        )


if __name__ == "__main__":
    utils.seed_all(123456)

    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name', help="value for ds_name variant", type=str)
    parser.add_argument('--ds_variant', help="value for variant", type=str)
    parser.add_argument('--model_name', help="value for model", type=str)
    parser.add_argument('--epochs', help="value for epochs", type=int)
    parser.add_argument('--batch_size', help="value for batch_size", type=int)
    parser.add_argument('--num_layers', help="value for num_layers", type=int)
    parser.add_argument('--input_dim', help="value for input_dim", type=int)
    parser.add_argument('--out_dim', help="value for out_dim", type=int)
    args = parser.parse_args()

    config = {
        "lr": tune.qloguniform(1e-4, 1e-2, 1e-5),
        "w_decay": tune.qloguniform(1e-5, 1e-3, 1e-6),
        "conv_dropout": tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        "classifier_dropout": tune.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
        "heads": tune.choice([1, 2, 4]),
        "batch_norm": tune.choice([True, False]),
        "hidden_dim": tune.choice([64, 128]),
        "aggr": tune.choice(["add", "mean", "max"]),
        "global_pooling": tune.choice(["add", "mean", "max"]),
        "ds_name": args.ds_name,
        "ds_variant": args.ds_variant,
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "input_dim": args.input_dim,
        "out_dim": args.out_dim,
        "num_layers": args.num_layers
    }

    if args.model_name not in ["transf", "gat", "gat2"]:
        config.pop("heads")

    postfix = datetime.today().strftime('%Y%m%d_%H%M%S')

    hyperopt_search = HyperOptSearch(metric="val_loss", mode="min")

    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='val_loss',
        mode='min',
        max_t=200,
        grace_period=10,
        reduction_factor=2,
    )

    reporter = CLIReporter(
        metric_columns=["tr_loss", "tr_acc", "val_loss", "val_acc", "best_loss"])

    result = tune.run(
        partial(trainer),
        search_alg=hyperopt_search,
        resources_per_trial={"gpu": 0.25},
        config=config,
        num_samples=500,
        scheduler=scheduler,
        progress_reporter=reporter,
        log_to_file=True,
        local_dir=os.path.abspath(
            f"/home/andres/ray_results/hyperparams_optim/{config['ds_name']}_{config['ds_variant']}_{config['model_name']}/"),
        callbacks=[WandbLoggerCallback(
            project="GNNsHAR",
            group=f"{config['ds_name']}_{config['ds_variant']}_{config['model_name']}_{postfix}",
            excludes=["time_total_s", "training_iteration", "time_since_restore", "iterations_since_restore",
                      "time_this_iter_s", "timestamp"]
        )]
    )

    best_trial = result.get_best_trial("val_loss", "min")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["val_loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["val_acc"]))
