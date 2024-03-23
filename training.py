import torch
import torch.nn.functional as F
import numpy as np
import logging
import os
import pickle
from sklearn.metrics import (accuracy_score,
                             balanced_accuracy_score,
                             precision_recall_fscore_support)
from sklearn.utils import compute_sample_weight
from torch_geometric.loader import DataLoader


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
    for i, data in enumerate(loader):  # Iterate in batches over the training ds_name.
        optimizer.zero_grad()  # Clear gradients.

        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_weight, data.batch)  # Perform a single forward pass.
        pred = out.argmax(dim=1)  # Use the class with the highest probability.

        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()  # Update parameters based on gradients.

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


def train_model(train_loader, val_loader, model, optimizer, scheduler, criterion, device, params, dirs, wandb_run):
    try:
        # logging.info("Training start")

        train_loss_arr = []
        train_acc_arr = []
        val_loss_arr = []
        val_acc_arr = []

        best_loss = np.inf
        best_epoch = 0
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
                               "best_loss": best_loss, "best_epoch": best_epoch}, commit=True)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                    scheduler.step()
                # if new_lr != last_lr and epoch > 1:
                #     print('-' * 89)
                #     print(f"Loss does not improve, new learning rate: {new_lr}")
                #     print('-' * 89)
                #     last_lr = new_lr
                # if optimizer.param_groups[0]['lr'] < params['min_lr']:
                #     print(f"\n!! Min Learning rate {params['min_lr']} reached. Training stopped!")
                #     break

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

    except KeyboardInterrupt:
        print('-' * 89)
        print('Training early stopped by user (Ctrl + C)')
        exit(1)


def update_best_model(new_data, val_loader, model, optimizer, scheduler, criterion, device, params, dirs):
    # Load best model for continuing training on the validation data for the final model
    checkpoint = torch.load(dirs["best_training_model"])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    for pgroup in optimizer.param_groups:
        pgroup['lr'] = pgroup['lr'] * 0.1
        # pgroup['weight_decay'] *= params["wd_reduce_factor_ft"]

    TRAIN_EPOCHS = epoch if epoch < 100 else 100
    PATIENCE = 10

    # rnd_training = random.sample(graphs_training, int(len(graphs_training)*0.1))  # 10% of training graphs at random
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

                save_checkpoint(
                    model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, loss=val_loss,
                    path=dirs["final_model"]
                )

                no_improve = 0
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


def evaluate_final_model(loader, model, optimizer, criterion, device, dirs, wandb_run):
    # load best model during training
    checkpoint = torch.load(dirs["final_model"])
    # checkpoint = torch.load(dirs["best_training_model"])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.eval()
    # evaluate final model on hold out test set
    val_loss, val_acc, y_test, y_pred = test_one_epoch(
        loader=loader, model=model, criterion=criterion, device=device
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

    if wandb_run is not None:
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
