import numpy as np
import torch
import os
import os.path as osp
import shutil
import zipfile
import realdisp_preprocessing
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset, download_url


class RealDispDataset(InMemoryDataset):
    def __init__(self, root, variant, transform=None, pre_transform=None, pre_filter=None, **kwargs):

        self.NUM_ACTIVITIES = 10
        self.activity_labels = {k: v for k, v in enumerate(
            ['Walking', 'Jogging', 'Running', 'Jumping', 'Jump rope', 'Waist bends forward', 'Arms up',
             'Knees to breast', 'Knees Bending', 'Cycling']
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

        if variant == "ensemble":
            if self.params.get("all_corr_matrices", False):
                self.all_corr_matrices = torch.load(f"{self.processed_dir}/all_corr_matrices.pt")
            else:
                self.activity = self.params.get("activity",
                                                0)  # if no activity id is passed, it returns activity {0: Standing}
                data_dict = torch.load(f"{self.processed_dir}/realdisp_ensemble_bm_{self.activity}.pt")
                self.data, self.slices = self.collate(data_dict["graphs"])
                self.activity_name = data_dict["activity_name"]
                # self.correlation_matrix = data_dict["correlation_matrix"]
                self.get_splits()
        else:
            # self.data, self.slices = torch.load(self.processed_paths[0])
            self.load(self.processed_paths[0])
            self.get_splits()

    @property
    def raw_dir(self) -> str:
        return osp.join("/".join(self.root.split("/")[:-1]), "raw_data")

    @property
    def raw_file_names(self):
        return [f'subject{i}_ideal.log' for i in range(1, 18)] + [f'subject{i}_self.log' for i in range(1, 18)]

    @property
    def processed_file_names(self):
        if self.variant == "ensemble":
            return [f"realdisp_ensemble_bm_{i}.pt" for i in range(self.NUM_ACTIVITIES)] + ["all_corr_matrices.pt"]
        else:
            return [f'realdisp_{self.variant}.pt']

    # def download(self):
    #     # Download to `self.raw_dir`.
    #     file_name = "mhealth+dataset.zip"
    #     url = f"https://archive.ics.uci.edu/static/public/319/{file_name}"
    #     download_url(url, self.raw_dir)
    #     with zipfile.ZipFile(f"{self.raw_dir}/{file_name}", "r") as zip_ref:
    #         zip_ref.extractall(f"{self.raw_dir}")
    #
    #     extracted_dir = f"{self.raw_dir}/MHEALTHDATASET/"
    #     for fname in os.listdir(extracted_dir):
    #         if fname.split(".")[1] == "log":
    #             file = f"{extracted_dir}/{fname}"
    #             shutil.move(file, self.raw_dir)
    #     shutil.rmtree(extracted_dir)
    #     os.remove(f"{self.raw_dir}/{file_name}")

    def process(self):
        # Read data into huge `Data` list.

        data_list, _ = realdisp_preprocessing.load_preprocessed_data(raw_data_path=self.raw_dir,
                                                                     ds_variant=self.variant,
                                                                     threshold=self.params.get("threshold", None))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # For PyG>=2.4:
        # self.save(data_list, self.processed_paths[0])
        # For PyG<2.4:
        # torch.save(self.collate(data_list), self.processed_paths[0])

        if self.variant == "ensemble":
            if isinstance(data_list, dict):
                all_corr_matrices = {}
                for key in data_list.keys():
                    filename = f"{self.processed_dir}/realdisp_ensemble_bm_{key}.pt"
                    file = {
                        "activity_name": self.activity_labels[key],
                        "graphs": data_list[key]["graphs"]
                    }
                    torch.save(file, filename)
                    all_corr_matrices[key] = data_list[key]["corr"]
                torch.save(all_corr_matrices, f"{self.processed_dir}/all_corr_matrices.pt")
        else:
            torch.save(self.collate(data_list), self.processed_paths[0])

    def get_dataset(self, name):
        idx = (self.processed_paths == name)
        return self.processed_paths[idx]

    def get_splits(self):
        val_subjects = [4, 6, 10, 11]
        test_subjects = [1, 7, 8, 9, 12, 14]
        train_subjects = list(set(range(18)).difference(set(val_subjects).union(test_subjects)))
        self.train_mask = np.in1d(self.data.subject.numpy(), train_subjects)
        self.val_mask = np.in1d(self.data.subject.numpy(), val_subjects)
        self.test_mask = np.in1d(self.data.subject.numpy(), test_subjects)
