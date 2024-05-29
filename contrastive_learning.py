import argparse
import configparser
import logging
import os
import pickle
from datetime import datetime
from argparse import Namespace

import numpy as np
import torch

import wandb
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.utils import compute_class_weight
from torch_geometric.loader import DataLoader

import models
from pair_data import create_data_pairs, PairData
from mhealth_dataset import MHealthDataset
from pamap2_dataset import Pamap2Dataset
from ucihar_dataset import UCIHARDataset
from realdisp_dataset import RealDispDataset
import utils

# logging.disable(logging.INFO)
logging.basicConfig(level=logging.INFO)


def save_checkpoint(model, optimizer, scheduler, epoch, loss, acc, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else scheduler,
        'epoch': epoch,
        'loss': loss,
        'acc': acc
    }, path)


def load_data(ds_name, ds_variant, is_dual, fillnan=None, batch_size=64, threshold=0.2):
    assert ds_name.lower() in ["mhealth", "pamap2", "ucihar", "realdisp"], f"Dataset {ds_name} not supported"

    root_dir = f'./data/{ds_name}'
    if ds_name.lower() == "mhealth":
        dataset_class = MHealthDataset
    elif ds_name.lower() == "ucihar":
        dataset_class = UCIHARDataset
    elif ds_name.lower() == "pamap2":
        dataset_class = Pamap2Dataset
    elif ds_name.lower() == "realdisp":
        dataset_class = RealDispDataset

    variant = f"corrcoef_all_{ds_variant}" if ds_variant is not None else "corrcoef_all"
    ds1 = dataset_class(root=f"{root_dir}/{variant}",
                        variant=f"{variant}",
                        fillnan=fillnan,
                        threshold=threshold)

    train1 = ds1[ds1.train_mask]
    val1 = ds1[ds1.val_mask]
    test1 = ds1[ds1.test_mask]

    if is_dual:
        variant = f"corrcoef_win_{ds_variant}" if ds_variant is not None else "corrcoef_win"
        ds2 = dataset_class(root=f"{root_dir}/{variant}",
                            variant=f"{variant}",
                            fillnan=fillnan,
                            threshold=threshold)

        train2 = ds2[ds2.train_mask]
        val2 = ds2[ds2.val_mask]
        test2 = ds2[ds2.test_mask]

        train = DataLoader([create_data_pairs(pair[0], pair[1]) for pair in zip(train1, train2)],
                           batch_size=batch_size,
                           follow_batch=['x_s', 'x_t'],
                           shuffle=True)
        val = DataLoader([create_data_pairs(pair[0], pair[1]) for pair in zip(val1, val2)],
                         batch_size=batch_size,
                         follow_batch=['x_s', 'x_t'],
                         shuffle=False)
        test = DataLoader([create_data_pairs(pair[0], pair[1]) for pair in zip(test1, test2)],
                          batch_size=batch_size,
                          follow_batch=['x_s', 'x_t'],
                          shuffle=False)
    else:
        train = DataLoader(ds1[ds1.train_mask], batch_size=batch_size, shuffle=True)
        val = DataLoader(ds1[ds1.val_mask], batch_size=batch_size, shuffle=False)
        test = DataLoader(ds1[ds1.test_mask], batch_size=batch_size, shuffle=False)

    return train, val, test


def train_one_epoch(loader, model, optimizer, criterion, device=torch.device("cpu")):
    model.train()

    total_loss = 0
    total_correct = 0
    for data in loader:  # Iterate in batches over the training ds_name.
        optimizer.zero_grad()  # Clear gradients.

        if isinstance(data, PairData):
            data.x_s = data.x_s.to(device)
            data.edge_index_s = data.edge_index_s.to(device)
            data.edge_weight_s = data.edge_weight_s.to(device)
            data.x_s_batch = data.x_s_batch.to(device)
            data.y_s = data.y_s.to(device)

            data.x_t = data.x_t.to(device)
            data.edge_index_t = data.edge_index_t.to(device)
            data.edge_weight_t = data.edge_weight_t.to(device)
            data.x_t_batch = data.x_t_batch.to(device)
            data.y_t = data.y_t.to(device)

            out, contrastive_views = model(data.x_s, data.edge_index_s, data.edge_weight_s, data.x_s_batch,
                                           data.x_t, data.edge_index_t, data.edge_weight_t, data.x_t_batch)
            y_true = data.y_s

        else:
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            data.edge_weight = data.edge_weight.to(device)
            data.batch = data.batch.to(device)
            data.y = data.y.to(device)
            y_true = data.y

            out, contrastive_views = model(data.x, data.edge_index, data.edge_weight, data.batch)

        pred = out.argmax(dim=1)  # Use the class with the highest probability.

        contrastive_loss = utils.InfoNCEloss(*contrastive_views)
        classification_loss = criterion(out, y_true)  # Compute the loss.
        # accuracy_loss = 1 - (int((pred == y_true).sum()) / data.num_graphs)

        # loss = contrastive_loss + (0.4 * classification_loss) + (0.6 * accuracy_loss)
        loss = contrastive_loss + classification_loss

        loss.backward()  # Derive gradients.
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()  # Update parameters based on gradients.

        total_loss += float(loss) * data.num_graphs
        total_correct += int((pred == y_true).sum())  # Check against ground-truth labels.

    return total_loss / len(loader.dataset), total_correct / len(loader.dataset)


@torch.no_grad()
def test_one_epoch(loader, model, criterion, device=torch.device("cpu")):
    model.eval()

    target = []
    predicted = []
    total_correct = 0
    total_loss = 0

    for data in loader:  # Iterate in batches over the training/test ds_name.
        if isinstance(data, PairData):
            data.x_s = data.x_s.to(device)
            data.edge_index_s = data.edge_index_s.to(device)
            data.edge_weight_s = data.edge_weight_s.to(device)
            data.x_s_batch = data.x_s_batch.to(device)
            data.y_s = data.y_s.to(device)

            data.x_t = data.x_t.to(device)
            data.edge_index_t = data.edge_index_t.to(device)
            data.edge_weight_t = data.edge_weight_t.to(device)
            data.x_t_batch = data.x_t_batch.to(device)
            data.y_t = data.y_t.to(device)

            out = model(data.x_s, data.edge_index_s, data.edge_weight_s, data.x_s_batch,
                        data.x_t, data.edge_index_t, data.edge_weight_t, data.x_t_batch)
            y_true = data.y_s

        else:
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            data.edge_weight = data.edge_weight.to(device)
            data.batch = data.batch.to(device)
            data.y = data.y.to(device)
            y_true = data.y

            out = model(data.x, data.edge_index, data.edge_weight, data.batch)

        loss = criterion(out, y_true)  # Compute the loss.
        pred = out.argmax(dim=1)  # Use the class with the highest probability.

        total_loss += float(loss) * data.num_graphs
        total_correct += int((pred == y_true).sum())  # Check against ground-truth labels.

        predicted.extend(pred.detach().cpu().numpy())
        target.extend(y_true.detach().cpu().numpy())

    return total_loss / len(loader.dataset), total_correct / len(loader.dataset), target, predicted


if __name__ == "__main__":
    utils.seed_all(123456)

    parser = argparse.ArgumentParser()
    """
        PARAMS
    """
    parser.add_argument('--ds_name', help="value for ds_name", type=str)
    parser.add_argument('--ds_variant', help="value for ds_variant", default=None, type=str)
    parser.add_argument('--fillnan', choices=['interpolate', 'dropna', 'zero'],
                        help="method for filling nan values", default=None, type=str)
    parser.add_argument('--model_name', help="value for model", type=str)
    parser.add_argument('--heads', help="number of heads", type=int)
    parser.add_argument('--epochs', help="value for epochs", type=int)
    parser.add_argument('--batch_size', help="value for batch_size", type=int)
    parser.add_argument('--lr', help="value for init_lr", type=float)
    parser.add_argument('--w_decay', help="value for weight_decay", type=float)
    parser.add_argument('--lr_reduce_factor', help="value for lr_reduce_factor", required=False, type=float)
    parser.add_argument('--lr_schedule_patience', help="value for lr_schedule_patience", required=False, type=int)
    parser.add_argument('--min_lr', help="value for min_lr", required=False, type=float)
    parser.add_argument('--patience_tr', help="value for patience_tr", default=-1, type=int)
    parser.add_argument('--scheduler', help="value for scheduler", action="store_true")
    parser.add_argument('--scheduler_algo', choices=['rop', 'cosann'],
                        help="method for lr scheduler algorithm", default=None, type=str)
    parser.add_argument('--use_sgd', help="use sgd instead of Adam", action="store_true")

    """
        GNN PARAMS
    """
    parser.add_argument('--notes', help="notes about the model", default="no comments", type=str)
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
    parser.add_argument('--contrast_method', choices=['single', 'dual'],
                        help="contrastive method ['single', 'dual']", default=None, type=str)
    parser.add_argument('--log_wandb', help="save logs in wandb", action="store_true")
    args = parser.parse_args()

    params = vars(args)

    if args.ds_name == "PAMAP2":
        assert args.fillnan is not None, "One option for parameter 'fillnan': ['interpolate', 'dropna', 'zero'] is " \
                                         "required for PAMAP2 dataset."

    if args.model_name in ["gat", "gat2", "gatres", "transf"]:
        assert args.heads > 0, "Number of heads is not provided"
    else:
        params.pop("heads")

    assert args.corruption is not None, "Parameter '--corruption' for generating the negative samples is missing." \
                                        "['all', 'node_features', 'edge_index']"

    assert args.contrast_method is not None, "Parameter '--contrast_method' is missing ['single', 'dual']"

    if args.scheduler:
        assert (args.lr_reduce_factor is not None and
                args.lr_schedule_patience is not None and
                args.min_lr is not None), "parameters 'lr_reduce_factor', 'lr_schedule_patience', and 'min_lr' are " \
                                          "mandatory to use scheduler"
        assert args.scheduler_algo is not None, "Parameter 'scheduler_algo' is missing. Choose ['rop', 'cosann'], " \
                                                "{rop: ReduceOnPlateau, cosann: CosineAnnealingLR}"
    else:
        assert args.patience_tr > 0, "Parameter 'patience_tr' used for early stopping is missing"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_name == "graphconv":
        global_encoder = models.GraphConvEncoder(net_params=params).to(device)
        win_encoder = models.GraphConvEncoder(net_params=params).to(device)
    else:
        global_encoder = models.GAT2Enconder(params=params).to(device)
        win_encoder = models.GAT2Enconder(params=params).to(device)

    if args.contrast_method == "dual":
        model = models.HARGMIdual(global_encoder=global_encoder,
                                  win_encoder=win_encoder,
                                  hidden_channels=args.hidden_dim,
                                  out_dim=args.out_dim,
                                  dropout=args.classifier_dropout,
                                  corruption=args.corruption).to(device)
    else:
        model = models.HARGMIsingle(encoder=global_encoder,
                                    hidden_channels=args.hidden_dim,
                                    out_dim=args.out_dim,
                                    dropout=args.classifier_dropout,
                                    corruption=args.corruption).to(device)

    print(model)
    print("Model parameters: ", sum(p.numel() for p in model.parameters()))

    if args.use_sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)

    if args.scheduler:
        if args.scheduler_algo == "rop":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                   factor=args.lr_reduce_factor,
                                                                   patience=args.lr_schedule_patience,
                                                                   verbose=True)
        elif args.scheduler_algo == "cosann":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=args.lr_schedule_patience,
                                                                   eta_min=args.min_lr)
    else:
        scheduler = None

    train_loader, val_loader, test_loader = load_data(ds_name=args.ds_name,
                                                      ds_variant=args.ds_variant,
                                                      is_dual=True if args.contrast_method == "dual" else False,
                                                      fillnan=args.fillnan,
                                                      batch_size=args.batch_size)

    if args.contrast_method == "dual":
        all_labels = np.array([pair.y_s.item() for pair in train_loader.dataset])
    else:
        all_labels = train_loader.dataset.y.numpy()

    # class_weight = torch.tensor(compute_class_weight(class_weight="balanced",
    #                                                  classes=np.unique(all_labels),
    #                                                  y=all_labels), dtype=torch.float)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    postfix = datetime.today().strftime('%Y%m%d_%H%M%S') if args.log_wandb else "TEST"

    variant_name = f"contrastive_{args.ds_variant}" if args.ds_variant is not None else "contrastive"
    variant_name = f"{variant_name}_{args.fillnan}_{args.corruption}" \
        if args.ds_name == "PAMAP2" else f"{variant_name}_{args.corruption}"
    dirs = {
        "best_training_model": f"out/{args.ds_name}/checkpoints/best_{variant_name}_{args.model_name}_{postfix}.pth",
        "final_model": f"out/{args.ds_name}/checkpoints/final_{variant_name}_{args.model_name}_{postfix}.pth",
        "training_acc_loss": f"out/{args.ds_name}/training/acc_losses_{variant_name}_{args.model_name}_{postfix}.pkl",
        "results": f"out/{args.ds_name}/training/results_{variant_name}_{args.model_name}_{postfix}.pkl"
    }

    for path in dirs:
        if not os.path.isfile(dirs[path]):
            os.makedirs(os.path.dirname(dirs[path]), exist_ok=True)

    config = params
    config["dirs"] = dirs

    if args.log_wandb:
        wandb_run = wandb.init(
            # Set the project where this run will be logged
            project="GNNsHAR",
            save_code=True,
            group="final_models_evaluation",
            tags=[args.ds_name, f"{variant_name}", args.model_name],
            name=f"{args.ds_name}_{variant_name}_{args.model_name}_{postfix}",
            # Track hyperparameters and run metadata
            config=config,
        )

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
            best_loss = val_loss
            best_acc = val_acc
            best_epoch = epoch

            save_checkpoint(
                model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, loss=val_loss, acc=best_acc,
                path=dirs["best_training_model"]
            )
        else:
            no_improve += 1

        if args.log_wandb:
            wandb_run.log({"tr_loss": tr_loss, "tr_acc": tr_acc, "val_loss": val_loss, "val_acc": val_acc,
                           "best_loss": best_loss, "best_epoch": best_epoch})

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                scheduler.step()

        if val_acc == 1:
            print(f"\n!! Max validation accuracy reached. Training stopped!")
            break

        if no_improve >= params["patience_tr"] and scheduler is None:
            print(f"\n!! No loss improvement. Training stopped!")
            break

        if epoch % 5 == 0:
            print(
                f'Epoch: {epoch:03d}, '
                f'Train Acc: {tr_acc:.4f}, Train Loss: {tr_loss:.4f}, '
                f'Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}, '
                f'Best Loss: {best_loss:.4f}'
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

    # **************************************** updating model ****************************************

    checkpoint = torch.load(dirs["best_training_model"])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    for pgroup in optimizer.param_groups:
        pgroup['lr'] = pgroup['lr'] * 0.1

    TRAIN_EPOCHS = epoch if epoch < 100 else 100
    PATIENCE = 10

    new_data = train_loader.dataset + val_loader.dataset
    train_loader = DataLoader(new_data, batch_size=args.batch_size, follow_batch=['x_s', 'x_t'], shuffle=True)

    train_loss_arr = []
    train_acc_arr = []
    val_loss_arr = []
    val_acc_arr = []

    best_epoch = 0
    best_loss = loss
    best_acc = 0
    no_improve = 0
    for epoch in range(0, TRAIN_EPOCHS):
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
            best_loss = val_loss
            best_acc = val_acc

            save_checkpoint(
                model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, loss=val_loss, acc=best_acc,
                path=dirs["final_model"]
            )
        else:
            no_improve += 1

        if no_improve >= PATIENCE or val_acc == 1 or epoch == TRAIN_EPOCHS:
            if best_loss == loss:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if scheduler is not None:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                epoch = checkpoint['epoch']
                loss = checkpoint['loss']

                save_checkpoint(
                    model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, loss=loss,
                    path=dirs["final_model"]
                )
            print(f"\n!! Max validation accuracy reached. Training stopped!")
            break

        if epoch % 5 == 0:
            print(
                f'Epoch: {epoch:03d}, '
                f'Train Acc: {tr_acc:.4f}, Train Loss: {tr_loss:.4f}, '
                f'Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}, '
                f'Best Loss: {best_loss:.4f}'
            )

    # ***************************** final model evaluation *****************************************

    # load best model during training
    checkpoint = torch.load(dirs["final_model"])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.eval()
    # evaluate final model on hold out test set
    val_loss, val_acc, y_test, y_pred = test_one_epoch(
        loader=test_loader, model=model, criterion=criterion, device=device
    )

    accuracy = accuracy_score(y_test, y_pred, normalize=True)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    mtrcs = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    precision = mtrcs[0]
    recall = mtrcs[1]
    f1_score = mtrcs[2]

    results = {
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

    if args.log_wandb:
        wandb_run.log({
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }, commit=True)

        filepath = dirs["results"]
        pickle.dump(obj=results, file=open(filepath, 'wb'))

    print(f"balanced accuracy: {balanced_accuracy:.4f}, "
          f"accuracy: {accuracy:.4f}, "
          f"f_score: {f1_score:.4f}, "
          f"precision: {precision:.4f}, "
          f"recall: {recall:.4f}")
