from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from openstl.api import BaseExperiment
from openstl.utils import create_parser, default_parser

FEATURES = [
    "todays_day_fires",
    "todays_day_frp",
    "todays_night_fires",
    "todays_night_frp",
    "ndvi",
    "landcover",
    "elevation",
    "population",
    "temperature_2m",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "total_precipitation_sum",
]


class WildfireDataset(Dataset):
    def __init__(self, data_dir, idx_in, idx_out):
        """
        Custom PyTorch Dataset for wildfire data
        Args:
            data_dir (str): Directory containing chip data
        """
        super().__init__()
        self.data_dir = data_dir
        self.data_list = []
        self.label_list = []
        for chip_dir in Path(data_dir).iterdir():
            input = defaultdict(list)
            output = []
            for daily in sorted(chip_dir.iterdir()):
                if not daily.is_dir():
                    continue
                for feature in FEATURES:
                    input[feature].append(np.load(daily / f"{feature}.npy"))
            output.append(np.load(chip_dir / "tomorrows_fires.npy"))
            data = np.stack(list(input.values()), axis=1)
            labels = np.array(output)[:, np.newaxis, :, :]
            labels = np.repeat(labels, len(FEATURES), axis=1)
            self.data_list.append(data)
            self.label_list.append(labels)
        self.idx_in = idx_in
        self.idx_out = idx_out
        self.mean = self.data_list.mean(axis=(0, 1, 3, 4), keepdims=True)
        self.std = self.data_list.std(axis=(0, 1, 3, 4), keepdims=True)
        self.transform = None

    def __len__(self):
        """
        Returns the total number of chips in the dataset
        """
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Loads and returns a single sample's features and label

        Args:
            idx (int): Index of the chip to load

        Returns:
            dict: A dictionary containing loaded features and label
        """
        data, labels = self.data_list[idx], self.label_list[idx]
        # Get input sequence: shape will be [timesteps, num_of_features, height, width]
        input = torch.tensor(data[self.idx_in], dtype=torch.float32)
        output = torch.tensor(labels[self.idx_out], dtype=torch.float32)

        return input, output


batch_size = 64
pre_seq_length = 5
aft_seq_length = 1

train_set = WildfireDataset("./dataset/train", [-4, -3, -2, -1, 0], [0])
val_set = WildfireDataset("./dataset/val", [-4, -3, -2, -1, 0], [0])
test_set = WildfireDataset("./dataset/test", [-4, -3, -2, -1, 0], [0])
dataloader_train = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, pin_memory=True
)
dataloader_val = torch.utils.data.DataLoader(
    val_set, batch_size=batch_size, shuffle=True, pin_memory=True
)
dataloader_test = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=True, pin_memory=True
)

custom_training_config = {
    "pre_seq_length": pre_seq_length,
    "aft_seq_length": aft_seq_length,
    "total_length": pre_seq_length + aft_seq_length,
    "batch_size": batch_size,
    "val_batch_size": batch_size,
    "epoch": 100,
    "lr": 0.001,
    "metrics": ["mse", "mae"],  # FIXME: use precision/recall/F1 instead
    "ex_name": "custom_exp",
    "dataname": "custom",
    "in_shape": [10, 12, 64, 64],
}

custom_model_config = {
    # For MetaVP models, the most important hyperparameters are:
    # N_S, N_T, hid_S, hid_T, model_type
    "method": "ConvLSTM",
    # Users can either using a config file or directly set these hyperparameters
    # 'config_file': 'configs/custom/example_model.py',
    # Here, we directly set these parameters
    # reverse scheduled sampling
    "reverse_scheduled_sampling": 0,
    "r_sampling_step_1": 25000,
    "r_sampling_step_2": 50000,
    "r_exp_alpha": 5000,
    # scheduled sampling
    "scheduled_sampling": 1,
    "sampling_stop_iter": 50000,
    "sampling_start_value": 1.0,
    "sampling_changing_rate": 0.00002,
    # model
    "num_hidden": "128,128,128,128",
    "filter_size": 5,
    "stride": 1,
    "patch_size": 2,
    "layer_norm": 0,
}


args = create_parser().parse_args([])
config = args.__dict__

# update the training config
config.update(custom_training_config)
# update the model config
config.update(custom_model_config)
# fulfill with default values
default_values = default_parser()
for attribute in default_values.keys():
    if config[attribute] is None:
        config[attribute] = default_values[attribute]

exp = BaseExperiment(
    args,
    dataloaders=(dataloader_train, dataloader_val, dataloader_test),
    strategy="auto",
)

exp.train()
exp.test()
