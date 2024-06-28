import argparse
import copy
import os
import pickle

import pamap2_preprocessing as pamap
import mhealth_preprocessing as mhealth
import ucihar_preprocessing as ucihar
import realdisp_preprocessing as realdisp
import numpy as np
import torch

import utils
import wandb

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.utils import compute_class_weight
from time_series_dataset import TSPytorchDataset
from datetime import datetime
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

class LSTM(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(n_channels, 64, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(64, 181)
        self.classifier = nn.Linear(181, n_classes)


    def forward(self, x):

        x, _ = self.lstm(x)
        x = F.dropout(x, p=0.46892, training=self.training)
        x = x[:,-1,:]
        x = self.fc1(x)
        out = self.classifier(x)

        return out


class CNN(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=n_channels, out_channels=64, kernel_size=7, stride=3)
        self.pool1 = nn.MaxPool1d(kernel_size=7, stride=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=3)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1, stride=3)
        self.pool3 = nn.MaxPool1d(kernel_size=1, stride=3)

        self.fc1 = nn.Linear(64, 512)
        self.classifier = nn.Linear(512, n_classes)


    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.conv1(x).relu()
        x = self.pool1(x)
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x).relu()
        x = self.pool2(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv3(x).relu()
        x = self.pool3(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        out = self.classifier(x)

        return out


class CNNLSTM4l(nn.Module):
    def __init__(self, n_channels, n_classes, win_size):
        super(CNNLSTM4l, self).__init__()

        self.conv1 = nn.Conv1d(n_channels, 507, kernel_size=3)
        self.conv2 = nn.Conv1d(507, 111, kernel_size=3)
        self.conv3 = nn.Conv1d(111, 468, kernel_size=3)
        self.conv4 = nn.Conv1d(468, 509, kernel_size=3)
        # self.conv1 = nn.Conv1d(n_channels, 128, kernel_size=3)
        # self.conv2 = nn.Conv1d(128, 128, kernel_size=3)
        # self.conv3 = nn.Conv1d(128, 128, kernel_size=3)
        # self.conv4 = nn.Conv1d(128, 128, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)

        dim = (win_size - 2 - 2 - 2 - 2) // 2

        self.lstm = nn.LSTM(dim, 256, num_layers=1, batch_first=True)  # hd=772
        self.classifier = nn.Linear(256, n_classes)

        # self.lstm = nn.LSTM(dim, 772, num_layers=1, batch_first=True)  # hd=772
        # self.classifier = nn.Linear(772, n_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = self.conv3(x).relu()
        x = self.conv4(x).relu()
        x = F.dropout(x, p=0.00952, training=self.training)
        x = self.pool(x)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = x[:,-1,:]
        x = F.dropout(x, p=0.27907, training=self.training)
        out = self.classifier(x)

        return out


class DeepConvLSTM(nn.Module):
    def __init__(self, n_channels, n_classes, conv_kernels=64, kernel_size=5, LSTM_units=128, dropout=0.5):
        super(DeepConvLSTM, self).__init__()

        self.conv1 = nn.Conv2d(1, conv_kernels, (kernel_size, 1))
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (kernel_size, 1))

        self.dropout = dropout
        self.lstm = nn.LSTM(n_channels * conv_kernels, LSTM_units, num_layers=2, batch_first=True)

        self.classifier = nn.Linear(LSTM_units, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = self.conv3(x).relu()
        x = self.conv4(x).relu()

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = F.dropout(x, p=self.dropout, training=self.training)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = x[:,-1,:]
        out = self.classifier(x)

        return out


def train_one_epoch(data_loader, model, loss_function, optimizer, device):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * num_batches

    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss


def test_one_epoch(loader, model, criterion, device):
    num_batches = len(loader)
    total_loss = 0
    total_correct = 0
    target = []
    predicted = []

    model.eval()
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            output = model(X)
            pred = output.argmax(dim=1)  # Use the class with the highest probability.

            total_loss += criterion(output, y).item() * num_batches
            total_correct += int((pred == y).sum())  # Check against ground-truth labels.

            predicted.extend(pred.detach().cpu().numpy())
            target.extend(y.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    avg_acc = total_correct / len(loader.dataset)
    return avg_loss, avg_acc, target, predicted


def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)


def load_dataset(ds_name, win_size, step):
    PATH = f"./data/{args.ds_name.upper()}/raw_data"
    PREPROC_DATA_FILE = f"./data/{args.ds_name}/ts_data/{args.ds_name.upper()}_ts.pt"

    if ds_name.lower() == "mhealth":
        val_subjects = [6, 10]
        test_subjects = [2, 9]
        train_subjects = [1, 3, 4, 5, 7, 8]
        dataset_module = mhealth

    elif ds_name.lower() == "ucihar":
        val_subjects = [2, 6, 12, 19, 26]
        test_subjects = [5, 8, 10, 14, 20, 21]
        train_subjects = [1, 3, 4, 7, 9, 11, 13, 15, 16, 17, 18, 22, 23, 24, 25, 27, 28, 29, 30]
        dataset_module = ucihar

    elif ds_name.lower() == "pamap2":
        val_subjects = [101, 107]
        test_subjects = [103, 105]
        train_subjects = [102, 104, 106, 108, 109]
        dataset_module = pamap

    elif ds_name.lower() == "realdisp":
        val_subjects = [4, 6, 10, 11]
        test_subjects = [1, 7, 8, 9, 12, 14]
        train_subjects = list(set(range(18)).difference(set(val_subjects).union(test_subjects)))
        dataset_module = realdisp

    x_train, x_val, x_test, y_train, y_val, y_test = dataset_module.get_ts_data(raw_data_path=PATH,
                                                                                preproc_data_file=PREPROC_DATA_FILE,
                                                                                win_size=win_size,
                                                                                step=step,
                                                                                train_subjects=train_subjects,
                                                                                val_subjects=val_subjects,
                                                                                test_subjects=test_subjects)
    train = TSPytorchDataset(x_train, y_train)
    val = TSPytorchDataset(x_val, y_val)
    test = TSPytorchDataset(x_test, y_test)

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, y_train


if __name__ == "__main__":
    utils.seed_all(123456)

    parser = argparse.ArgumentParser()

    """
        PARAMS
    """
    parser.add_argument('--ds_name', help="value for ds_name variant", type=str)
    parser.add_argument('--ds_variant', help="value for variant", type=str)
    parser.add_argument('--model_name', help="value for model", type=str)
    parser.add_argument('--epochs', help="value for epochs", type=int)
    parser.add_argument('--batch_size', help="value for batch_size", type=int)
    parser.add_argument('--lr', help="value for init_lr", type=float)
    parser.add_argument('--patience_tr', help="value for patience_tr", default=-1, type=int)
    parser.add_argument('--win_size', help="value for window size", type=int)
    parser.add_argument('--step', help="value for time step for the sliding windows", type=int)

    """
        MODEL PARAMS
    """
    parser.add_argument('--input_dim', help="value for input_dim", type=int)
    parser.add_argument('--hidden_dim', help="value for hidden_dim", type=int)
    parser.add_argument('--out_dim', help="value for out_dim", type=int)
    parser.add_argument('--dropout', help="value for convolution layer dropout", type=float)
    parser.add_argument('--log_wandb', help="save logs in wandb", action="store_true")
    args = parser.parse_args()

    train_loader, val_loader, test_loader, all_labels = load_dataset(
        ds_name=args.ds_name,
        win_size=args.win_size,
        step=args.step,
    )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_name.lower() == "deepconvlstm":
        model = DeepConvLSTM(n_channels=args.input_dim, n_classes=args.out_dim, LSTM_units=args.hidden_dim).to(device)
    elif args.model_name.lower() == "cnn":
        model = CNN(n_channels=args.input_dim, n_classes=args.out_dim).to(device)
    elif args.model_name.lower() == "cnnlstm4l":
        model = CNNLSTM4l(n_channels=args.input_dim, n_classes=args.out_dim, win_size=args.win_size).to(device)
    else:
        model = LSTM(n_channels=args.input_dim, n_classes=args.out_dim).to(device)

    print("Model parameters: ", sum(p.numel() for p in model.parameters()))

    # class weighting
    all_labels = all_labels
    class_weight = torch.tensor(compute_class_weight(class_weight="balanced",
                                                     classes=np.unique(all_labels),
                                                     y=all_labels), dtype=torch.float)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    postfix = datetime.today().strftime('%Y%m%d_%H%M%S') if args.log_wandb else "TEST"

    variant_name = f"{args.ds_variant}_interpolate" if args.ds_name == "PAMAP2" else args.ds_variant
    dirs = {
        "final_model": f"out/{args.ds_name}/checkpoints/final_{variant_name}_{args.model_name}_{postfix}.pth",
        "training_acc_loss": f"out/{args.ds_name}/training/acc_losses_{variant_name}_{args.model_name}_{postfix}.pkl",
        "results": f"out/{args.ds_name}/training/results_{variant_name}_{args.model_name}_{postfix}.pkl"
    }

    for path in dirs:
        if not os.path.isfile(dirs[path]):
            os.makedirs(os.path.dirname(dirs[path]), exist_ok=True)

    config = vars(args)
    config["dirs"] = dirs

    wandb_run = None
    if args.log_wandb:
        wandb_run = wandb.init(
            # Set the project where this run will be logged
            project="GNNsHAR",
            save_code=True,
            group="final_models_evaluation",
            tags=[args.ds_name, args.ds_variant, args.model_name],
            name=f"{args.ds_name}_{variant_name}_{args.model_name}_{postfix}",
            # Track hyperparameters and run metadata
            config=config,
        )

    ## model training

    no_improvement = 0
    best_loss = np.inf
    best_epoch = 0
    best_model = copy.deepcopy(model)
    for ix_epoch in range(500):
        trn_loss = train_one_epoch(train_loader, model, criterion, optimizer=optimizer, device=device)
        val_loss, val_acc, _, _ = test_one_epoch(val_loader, model, criterion, device)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = ix_epoch
            no_improvement = 0

            save_checkpoint(
                model=model, optimizer=optimizer, epoch=best_epoch, loss=val_loss,
                path=dirs["final_model"]
            )
        else:
            no_improvement += 1

        if args.log_wandb:
            wandb_run.log({"tr_loss": trn_loss, "val_loss": val_loss, "val_acc": val_acc,
                           "best_loss": best_loss, "best_epoch": best_epoch}, commit=True)

        if ix_epoch == 0 or (ix_epoch + 1) % 5 == 0:
            print(
                f'Epoch: {ix_epoch:03d}, '
                f"train_loss: {trn_loss:.4f}, "
                f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, "
                f"best_epoch: {best_epoch:03d}, best_lost: {best_loss:.4f}"
            )

        if no_improvement >= args.patience_tr:
            print(f"\n!! No loss improvement. Training stopped!")
            break

    print("\n\n-------- Final model test --------")

    # load best model during training
    checkpoint = torch.load(dirs["final_model"])
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    val_loss, val_acc, y_test, y_pred = test_one_epoch(loader=test_loader, model=model,
                                                       criterion=criterion, device=device)

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
