import argparse
import logging
import os

import numpy as np
import torch

import wandb
from sklearn.utils import compute_class_weight
from datetime import datetime

from ucihar_dataset import UCIHARDataset
import utils
import models
import training as training
from mhealth_dataset import MHealthDataset
from pamap2_dataset import Pamap2Dataset
from realdisp_dataset import  RealDispDataset

from torch_geometric.loader import DataLoader

logging.basicConfig(level=logging.INFO, format="'%(levelname)s:%(message)s'")


def load_data(ds_name, variant, fillnan=None, batch_size=64, threshold=0.2):
    assert ds_name.lower() in ["mhealth", "pamap2", "ucihar", "realdisp"], f"Dataset {ds_name} not supported"

    # if ds_name.lower() == "mhealth":
    #     root_dir = f'./data/{ds_name}/{variant}'
    #     dataset_class = MHealthDataset
    # elif ds_name.lower() == "ucihar":
    #     root_dir = f'./data/{ds_name}/{variant}'
    #     dataset_class = UCIHARDataset
    # elif ds_name.lower() == "pamap2":
    #     root_dir = f"./data/{ds_name}/{'_'.join(variant.split('_')[:-1])}"
    #     fillnan = variant.split('_')[-1]
    #     variant = '_'.join(variant.split('_')[:-1])
    #     dataset_class = Pamap2Dataset

    root_dir = f'./data/{ds_name}/{variant}'
    if ds_name.lower() == "mhealth":
        dataset_class = MHealthDataset
    elif ds_name.lower() == "ucihar":
        dataset_class = UCIHARDataset
    elif ds_name.lower() == "pamap2":
        dataset_class = Pamap2Dataset
    elif ds_name.lower() == "realdisp":
        dataset_class = RealDispDataset

    dataset = dataset_class(root=root_dir, variant=variant, fillnan=fillnan, threshold=threshold)

    # dataset = dataset_class(root=root_dir, variant=variant, fillnan=fillnan, threshold=0.2)

    train = DataLoader(dataset[dataset.train_mask], batch_size=batch_size, shuffle=True)
    val = DataLoader(dataset[dataset.val_mask], batch_size=batch_size, shuffle=False)
    test = DataLoader(dataset[dataset.test_mask], batch_size=batch_size, shuffle=False)

    return train, val, test


if __name__ == "__main__":
    utils.seed_all(123456)

    parser = argparse.ArgumentParser()

    """
        PARAMS
    """
    parser.add_argument('--ds_name', help="value for ds_name variant", type=str)
    parser.add_argument('--ds_variant', help="value for variant", type=str)
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
    parser.add_argument('--num_layers', help="value for num_layers", type=int)
    parser.add_argument('--input_dim', help="value for input_dim", type=int)
    parser.add_argument('--hidden_dim', help="value for hidden_dim", type=int)
    parser.add_argument('--out_dim', help="value for out_dim", type=int)
    parser.add_argument('--aggr', help="value for convolution layer pooling ", type=str)
    parser.add_argument('--global_pooling', help="value for global pooling function", type=str)
    parser.add_argument('--conv_dropout', help="value for convolution layer dropout", type=float)
    parser.add_argument('--classifier_dropout', help="value for dropout", type=float)
    parser.add_argument('--batch_norm', help="value for batch_norm", action="store_true")
    parser.add_argument('--log_wandb', help="save logs in wandb", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = vars(args)
    params["is_explainer"] = False  # only used when running model explainer python script

    if args.ds_name == "PAMAP2":
        assert args.fillnan is not None, "One option for parameter 'fillnan': ['interpolate', 'dropna', 'zero'] is " \
                                         "required for PAMAP2 dataset."

    if args.model_name in ["gat", "gat2", "gatres", "transf"]:
        assert args.heads > 0, "Number of heads is not provided"
    else:
        params.pop("heads")

    if args.scheduler:
        assert (args.lr_reduce_factor is not None and
                args.lr_schedule_patience is not None and
                args.min_lr is not None), "parameters 'lr_reduce_factor', 'lr_schedule_patience', and 'min_lr' are " \
                                          "mandatory to use scheduler"
        assert args.scheduler_algo is not None, "Parameter 'scheduler_algo' is missing. Choose ['rop', 'cosann'], " \
                                                "{rop: ReduceOnPlateau, cosann: CosineAnnealingLR}"
    else:
        assert args.patience_tr > 0, "Parameter 'patience_tr' used for early stopping is missing"

    try:

        assert args.model_name in ["transf", "gat", "gatres", "gat2", "graphconv", "gine",
                                   "mpool", "resgcnn_pm", "resgcnn_mh"], f"Model {args.model_name} not implemented."

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
        elif args.model_name == "gatres":
            model = models.GATResClassifier(input_dim=args.input_dim,
                                            num_channels=args.hidden_dim,
                                            heads=args.heads,
                                            out_dim=args.out_dim,
                                            global_pooling=args.global_pooling,
                                            dropout=args.classifier_dropout).to(device)
        elif args.model_name == "gat2":
            model = models.GAT2Net(params=params).to(device)
        elif args.model_name == "mpool":
            model = models.MPoolGNN(net_params=params).to(device)
        elif args.model_name == "graphconv":
            model = models.GraphConvNet(net_params=params).to(device)
        elif args.model_name == "gine":
            model = models.GINENet(net_params=params).to(device)
        elif args.model_name == "resgcnn_pm":
            model = models.ResGCNN_pamap2(device).to(device)
        elif args.model_name == "resgcnn_mh":
            model = models.ResGCNN_mhealth(device).to(device)

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

        print(model)
        print("Model parameters: ", sum(p.numel() for p in model.parameters()))

        exit(0)

        train_loader, val_loader, test_loader = load_data(ds_name=args.ds_name,
                                                          variant=args.ds_variant,
                                                          fillnan=args.fillnan,
                                                          batch_size=args.batch_size)

        # class weighting
        all_labels = train_loader.dataset.data.y[train_loader.dataset.train_mask]
        class_weight = torch.tensor(compute_class_weight(class_weight="balanced",
                                                         classes=np.unique(all_labels),
                                                         y=all_labels.numpy()), dtype=torch.float)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weight).to(device)

        # criterion = torch.nn.CrossEntropyLoss().to(device)

        postfix = datetime.today().strftime('%Y%m%d_%H%M%S') if args.log_wandb else "TEST"

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

        logging.info(f"Training {args.ds_name}_{args.ds_variant}")
        logging.info('-' * 89)

        training.train_model(
            train_loader=train_loader, val_loader=val_loader, model=model, optimizer=optimizer, criterion=criterion,
            scheduler=scheduler, device=device, params=params, dirs=dirs, wandb_run=wandb_run
        )

        logging.info("Updating the best model with training + validation set")

        new_data = train_loader.dataset + val_loader.dataset
        training.update_best_model(
            new_data=new_data, val_loader=val_loader, model=model, optimizer=optimizer, criterion=criterion,
            scheduler=scheduler, device=device, params=params, dirs=dirs
        )

        logging.info("Model evaluation results")
        logging.info('-' * 89)
        training.evaluate_final_model(
            loader=test_loader, model=model, optimizer=optimizer, criterion=criterion, device=device, dirs=dirs,
            wandb_run=wandb_run
        )
        logging.info('-' * 89)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Model initialization stopped by user (Ctrl + C)')
