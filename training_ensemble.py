import argparse
import logging
import os
import pickle
from datetime import datetime

import wandb

import models

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.utils import compute_sample_weight, compute_class_weight
from torch_geometric.loader import DataLoader
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


def train_one_epoch(loader, model, optimizer, criterion, device=torch.device("cpu")):
    model.train()

    total_loss = 0
    total_correct = 0
    for data in loader:  # Iterate in batches over the training ds_name.
        # sample_weight = torch.tensor(compute_sample_weight(class_weight="balanced", y=data.y), dtype=float)
        data = data.to(device)
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x, data.edge_index, data.edge_weight, data.batch)  # Perform a single forward pass.
        pred = out.sigmoid().round().float()  # out.sigmoid(dim=1)  # Use the class with the highest probability.

        # criterion.weight = sample_weight.to(device)
        loss = criterion(out.view(-1), data.y.float())  # Compute the loss.

        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        total_loss += float(loss) * data.num_graphs
        total_correct += int((pred.view(-1) == data.y).sum())  # Check against ground-truth labels.

    return total_loss / len(loader.dataset), total_correct / len(loader.dataset)


@torch.no_grad()
def test_one_epoch(loader, model, criterion, device=torch.device("cpu")):
    model.eval()

    target = []
    predicted = []
    total_correct = 0
    total_loss = 0
    for data in loader:  # Iterate in batches over the training/test ds_name.
        # sample_weight = torch.tensor(compute_sample_weight(class_weight="balanced", y=data.y), dtype=float)
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_weight, data.batch)
        pred = out.sigmoid().round().float()  # out.argmax(dim=1)  # Use the class with the highest probability.

        # criterion.weight = sample_weight.to(device)
        loss = criterion(out.view(-1), data.y.float())  # Compute the loss.

        total_loss += float(loss) * data.num_graphs
        total_correct += int((pred.view(-1) == data.y).sum())  # Check against ground-truth labels.

        predicted.extend(pred.detach().cpu().numpy())
        target.extend(data.y.detach().cpu().numpy())

    return total_loss / len(loader.dataset), total_correct / len(loader.dataset), target, predicted


def train_model(train_loader, val_loader, model, optimizer, scheduler, criterion, device, params, dirs, wandb_run):
    try:
        # logging.info("Training start")

        train_loss_arr = []
        train_acc_arr = []
        val_loss_arr = []
        val_acc_arr = []

        best_epoch = 0
        best_loss = np.inf
        no_improve = 0
        for epoch in range(1, params["epochs"] + 1):
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

            if scheduler is not None:
                scheduler.step(val_loss)

            if val_loss < best_loss:
                no_improve = 0
                best_loss = val_loss
                best_epoch = epoch

                save_checkpoint(
                    model=model, optimizer=optimizer, scheduler=scheduler, epoch=best_epoch, loss=val_loss,
                    path=dirs["best_training_model"]
                )
            else:
                no_improve += 1

            if params["log_wandb"]:
                wandb_run.log({"tr_loss": tr_loss, "tr_acc": tr_acc, "val_loss": val_loss, "val_acc": val_acc,
                               "best_loss": best_loss, "best_epoch": epoch}, commit=True)

            if args.scheduler:
                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print(f"\n!! Min Learning rate {params['min_lr']} reached. Training stopped!")
                    break

            if val_acc == 1:
                print(f"\n!! Max validation accuracy reached. Training stopped!")
                break

            if no_improve >= params["patience_tr"] and scheduler is None:
                print(f"\n!! No loss improvement. Training stopped!")
                break

            if epoch == 1 or epoch % 5 == 0:
                print(
                    f'Epoch: {epoch:03d}, '
                    f'Train Acc: {tr_acc:.4f}, Train Loss: {tr_loss:.4f}, '
                    f'Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}, '
                    f'Best Loss: {best_loss:.4f}, Best epoch: {best_epoch:03d}'
                )
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

        logging.info("Training done, accuracies and losses saved.")

    except KeyboardInterrupt:
        print('-' * 89)
        print('Training early stopped by user (Ctrl + C)')
        exit(1)


def update_best_model(new_data, val_loader, model, optimizer, scheduler, criterion, device, params, dirs):
    # Load best model for continuing training on the validation data for the final model
    checkpoint = torch.load(dirs["best_training_model"])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    for pgroup in optimizer.param_groups:
        pgroup['lr'] = pgroup['lr'] * 0.1
        # pgroup['weight_decay'] *= params["wd_reduce_factor_ft"]

    TRAIN_EPOCHS = epoch if epoch < 100 else 100
    PATIENCE = 10

    train_loader = DataLoader(new_data, batch_size=params["batch_size"], shuffle=True)

    try:
        model.train()

        train_loss_arr = []
        train_acc_arr = []
        val_loss_arr = []
        val_acc_arr = []

        best_loss = loss
        best_epoch = 0
        no_improve = 0
        for epoch in range(1, TRAIN_EPOCHS + 1):
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
                best_loss = val_loss
                best_epoch = epoch

                save_checkpoint(
                    model=model, optimizer=optimizer, scheduler=scheduler, epoch=best_epoch, loss=val_loss,
                    path=dirs["final_model"]
                )

                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= PATIENCE or val_acc == 1 or epoch == TRAIN_EPOCHS:
                if best_loss == loss:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    epoch = checkpoint['epoch']
                    loss = checkpoint['loss']

                    save_checkpoint(
                        model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, loss=loss,
                        path=dirs["final_model"]
                    )
                print(f"\n!! No loss improvement. Training stopped!")
                break

            if epoch == 1 or epoch % 5 == 0:
                print(
                    f'Epoch: {epoch:03d}, '
                    f'Train Acc: {tr_acc:.4f}, Train Loss: {tr_loss:.4f}, '
                    f'Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}, '
                    f'Best Loss: {best_loss:.4f}'
                )

    except KeyboardInterrupt:
        print('-' * 89)
        print('Model update early stopped by user (Ctrl + C)')


def model_evaluation(loader, model, criterion, device, dirs, wandb_run, eval_set="test_set"):
    # load best model during training
    if eval_set == "validation_set":
        checkpoint = torch.load(dirs["best_training_model"])
        filepath = dirs["results_val"]
    else:
        checkpoint = torch.load(dirs["final_model"])
        filepath = dirs["results_test"]

    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    # evaluate final model on hold out test set
    val_loss, val_acc, y_test, y_pred = test_one_epoch(
        loader=loader, model=model, criterion=criterion, device=device
    )

    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    precision = precision_score(y_test, y_pred, average="binary", zero_division=0)
    recall = recall_score(y_test, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)

    results = {
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    if wandb_run is not None:
        wandb_run.log({
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }, commit=True)

    pickle.dump(obj=results, file=open(filepath, 'wb'))

    print(f"balanced accuracy: {balanced_accuracy:.4f}, "
          f"accuracy: {accuracy:.4f}, "
          f"f_score: {f1:.4f}, "
          f"precision: {precision:.4f}, "
          f"recall: {recall:.4f}")


def load_data(ds_name, variant, activity, fillnan=None, batch_size=64, threshold=0.2):
    assert ds_name.lower() in ["mhealth", "pamap2", "ucihar", "realdisp"], f"Dataset {ds_name} not supported"

    root_dir = f'./data/{ds_name}/{variant}'
    if ds_name.lower() == "mhealth":
        dataset_class = MHealthDataset
    elif ds_name.lower() == "ucihar":
        dataset_class = UCIHARDataset
    elif ds_name.lower() == "pamap2":
        dataset_class = Pamap2Dataset
    elif ds_name.lower() == "realdisp":
        dataset_class = RealDispDataset

    dataset = dataset_class(root=root_dir, variant=variant, fillnan=fillnan, activity=activity, threshold=threshold)

    train = DataLoader(dataset[dataset.train_mask], batch_size=batch_size, shuffle=True)
    val = DataLoader(dataset[dataset.val_mask], batch_size=batch_size, shuffle=False)
    test = DataLoader(dataset[dataset.test_mask], batch_size=batch_size, shuffle=False)

    return train, val, test


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
    parser.add_argument('--epochs', help="value for epochs", type=int)
    parser.add_argument('--batch_size', help="value for batch_size", type=int)
    parser.add_argument('--heads', help="value for batch_size", default=-1, type=int)
    parser.add_argument('--lr', help="value for init_lr", type=float)
    parser.add_argument('--w_decay', help="value for weight_decay", type=float)
    parser.add_argument('--lr_reduce_factor', help="value for lr_reduce_factor", required=False, type=float)
    parser.add_argument('--lr_schedule_patience', help="value for lr_schedule_patience", required=False, type=int)
    parser.add_argument('--min_lr', help="value for min_lr", required=False, type=float)
    parser.add_argument('--patience_tr', help="value for patience_tr", default=-1, type=int)
    parser.add_argument('--scheduler', help="value for scheduler", action="store_true")

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
    parser.add_argument('--postfix', help="best model id", type=str)
    parser.add_argument('--log_wandb', help="value for batch_norm", action="store_true")
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

    if args.scheduler:
        assert (args.lr_reduce_factor is not None and
                args.lr_schedule_patience is not None and
                args.min_lr is not None), ("parameters 'lr_reduce_factor', "
                                           "'lr_schedule_patience', and 'min_lr' are mandatory to use scheduler")
    else:
        assert args.patience_tr > 0, "Parameter 'patience_tr' used for early stopping is missing"

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

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)

        if args.scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                   factor=args.lr_reduce_factor,
                                                                   patience=args.lr_schedule_patience,
                                                                   verbose=True)
        else:
            scheduler = None

        train_loader, val_loader, test_loader = load_data(ds_name=args.ds_name,
                                                          variant=args.ds_variant,
                                                          fillnan=args.fillnan,
                                                          activity=args.activity,
                                                          batch_size=args.batch_size)

        # class weighting
        all_labels = train_loader.dataset.data.y[train_loader.dataset.train_mask]
        neg, pos = np.unique(all_labels, return_counts=True)[1]
        pos_weight = torch.tensor(neg / pos,
                                  dtype=torch.float)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)

        postfix = datetime.today().strftime('%Y%m%d_%H%M%S') if args.log_wandb else "TEST"

        dirs = {
            "best_training_model": f"out/{args.ds_name}/ensembles/checkpoints/best_training_{args.ds_variant}_bm_{args.activity}_{args.model_name}_{postfix}.pt",
            "final_model": f"out/{args.ds_name}/ensembles/checkpoints/final_{args.ds_variant}_bm_{args.activity}_{args.model_name}_{postfix}.pt",
            "training_acc_loss": f"out/{args.ds_name}/ensembles/training/acc_losses_{args.ds_variant}_bm_{args.activity}_{args.model_name}_{postfix}.pkl",
            "results_val": f"out/{args.ds_name}/ensembles/training/results_val_{args.ds_variant}_bm_{args.activity}_{args.model_name}_{postfix}.pkl",
            "results_test": f"out/{args.ds_name}/ensembles/training/results_test_{args.ds_variant}_bm_{args.activity}_{args.model_name}_{postfix}.pkl"
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
                group="base_models_ensemble",
                tags=[args.ds_name, args.ds_variant, args.model_name],
                name=f"{args.ds_name}_{args.ds_variant}_{args.model_name}_bm{int(args.activity):02d}_{postfix}",
                # Track hyperparameters and run metadata
                config=config,
            )

        variant_name = f"corrcoef_all_{args.fillnan}" if args.ds_name == "PAMAP2" else "corrcoef_all"
        checkpoint = torch.load(f"out/{args.ds_name}/checkpoints/best_training_{variant_name}_{args.model_name}_"
                                f"{args.postfix}.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.lin1 = nn.Linear(model.lin1.in_features, model.lin1.out_features * 4)
        model.lin2 = nn.Linear(model.lin2.in_features * 4, 1)
        model = model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(model)
        print("Model parameters: ", sum(p.numel() for p in model.parameters()))

        exit(0)

        logging.info(f"Training {args.ds_name}_{args.ds_variant}_{args.model_name}")
        logging.info('-' * 89)

        train_model(
            train_loader=train_loader, val_loader=val_loader, model=model, optimizer=optimizer, criterion=criterion,
            scheduler=scheduler, device=device, params=params, dirs=dirs, wandb_run=wandb_run
        )

        logging.info("Results: validation set")
        logging.info('-' * 89)
        model_evaluation(
            loader=val_loader, model=model, criterion=criterion, device=device, dirs=dirs, wandb_run=wandb_run,
            eval_set="validation_set"
        )
        logging.info('-' * 89)

        logging.info("Updating the best model with training + validation set")

        new_data = train_loader.dataset + val_loader.dataset
        update_best_model(
            new_data=new_data, val_loader=val_loader, model=model, optimizer=optimizer, criterion=criterion,
            scheduler=scheduler, device=device, params=params, dirs=dirs
        )

        logging.info("Results: test set")
        logging.info('-' * 89)
        model_evaluation(
            loader=test_loader, model=model, criterion=criterion, device=device, dirs=dirs, wandb_run=wandb_run,
            eval_set="test_set"
        )
        logging.info('-' * 89)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Model initialization stopped by user (Ctrl + C)')
        exit(1)
