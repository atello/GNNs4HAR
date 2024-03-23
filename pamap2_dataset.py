import numpy as np
import torch
import os
import os.path as osp
import shutil
import zipfile

import pamap2_preprocessing
from torch_geometric.data import InMemoryDataset, download_url


class Pamap2Dataset(InMemoryDataset):
    def __init__(self, root, variant, transform=None, pre_transform=None, pre_filter=None, **kwargs):

        self.NUM_ACTIVITIES = 12
        self.activity_labels = {k: v for k, v in enumerate(
            ['Lying down', 'Sitting', 'Standing', 'Walking', 'Running', 'Cycling', 'Nordic Walk', 'Walking Upstairs',
             'Walking Downstairs', 'Vacuum Cleaning', 'Ironing', 'Rope Jumping']
        )}
        self.params = {k: v for k, v in kwargs.items()}
        self.variant = variant

        super().__init__(root, transform, pre_transform, pre_filter)

        # For PyG>=2.4:
        # self.load(self.processed_paths[0])
        # For PyG<2.4:
        # self.data, self.slices = torch.load(self.processed_paths[0])

        self.train_mask = None
        self.val_mask = None
        self.test_mask = None
        self.all_corr_matrices = {}

        if "ensemble" in variant:
            if self.params.get("all_corr_matrices", False):
                self.all_corr_matrices = torch.load(f"{self.processed_dir}/all_corr_matrices.pt")
            else:
                self.activity = self.params.get("activity",
                                                0)  # if no activity id is passed, it returns activity {0: Standing}
                data_dict = torch.load(f"{self.processed_dir}/pamap2_bm_{self.activity}_{self.params['fillnan']}.pt")
                self.data, self.slices = self.collate(data_dict["graphs"])
                self.activity_name = data_dict["activity_name"]
                self.get_splits()
        else:
            self.load(self.processed_paths[0])
            self.get_splits()

    @property
    def raw_dir(self) -> str:
        return osp.join("/".join(self.root.split("/")[:-1]), "raw_data")

    @property
    def raw_file_names(self):
        return [f'subject{i}.dat' for i in range(101, 110)]

    @property
    def processed_file_names(self):
        if "ensemble" in self.variant:
            return [f"pamap2_bm_{i}_{self.params['fillnan']}.pt" for i in range(self.NUM_ACTIVITIES)] + [
                "all_corr_matrices.pt"]
        else:
            return [f'pamap2_{self.variant}_{self.params["fillnan"]}.pt']

    def process(self):
        # Read data into huge `Data` list.

        data_list = pamap2_preprocessing.load_preprocessed_data(raw_data_path=self.raw_dir,
                                                                ds_variant=self.variant,
                                                                fillnan=self.params["fillnan"],
                                                                threshold=self.params.get("threshold", None),
                                                                win_size=self.params["win_size"],
                                                                win_step=self.params["win_step"])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # For PyG>=2.4:
        # self.save(data_list, self.processed_paths[0])
        # For PyG<2.4:
        # torch.save(self.collate(data_list), self.processed_paths[0])

        if "ensemble" in self.variant:
            if isinstance(data_list, dict):
                all_corr_matrices = {}
                for key in data_list.keys():
                    filename = f"{self.processed_dir}/pamap2_bm_{key}_{self.params['fillnan']}.pt"
                    file = {
                        "activity_name": self.activity_labels[key],
                        "graphs": data_list[key]["graphs"]
                    }
                    torch.save(file, filename)
                    all_corr_matrices[key] = data_list[key]["corr"]
                torch.save(all_corr_matrices, f"{self.processed_dir}/all_corr_matrices.pt")
        else:
            self.save(data_list, self.processed_paths[0])

    # def get_dataset(self, name):
    #     idx = (self.processed_paths == name)
    #     return self.processed_paths[idx]

    def get_splits(self):
        val_subjects = [101, 107]
        test_subjects = [103, 105]
        train_subjects = [102, 104, 106, 108, 109]
        self.train_mask = np.in1d(self.data.subject.numpy(), train_subjects)
        self.val_mask = np.in1d(self.data.subject.numpy(), val_subjects)
        self.test_mask = np.in1d(self.data.subject.numpy(), test_subjects)
