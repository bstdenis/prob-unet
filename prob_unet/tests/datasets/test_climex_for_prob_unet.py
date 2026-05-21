import itertools
import pickle
from pathlib import Path

import cftime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import xarray
from torchmetrics.image import StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure
from torch.utils import data as td


from resoterre.config_utils import config_from_yaml
from resoterre.ml.data_loader_utils import normalize, inverse_normalize
from resoterre.plots.nd_plots import nd_ax_plot

from prob_unet.datasets import climex_for_prob_unet
from prob_unet.ml.probabilistic_unet import ProbabilisticUnet


class SkeletonDataset(td.Dataset):
    def __init__(self, path_daily_data, path_daily_coarse_data, path_neighbors, train_years, validation_years,
                 test_years, num_neighbors=0, active_split_name="train", debug_max_sample=None):
        self.path_daily_data = path_daily_data
        self.path_daily_coarse_data = path_daily_coarse_data
        self.path_neighbors = path_neighbors
        self.num_neighbors = num_neighbors
        self.active_split_name = active_split_name
        nc_files = sorted(list(Path(path_daily_data).glob("*.nc")))
        self.train_files = [f for f in nc_files if int(f.stem.split("_")[-1]) in train_years]
        self.val_files = [f for f in nc_files if int(f.stem.split("_")[-1]) in validation_years]
        self.test_files = [f for f in nc_files if int(f.stem.split("_")[-1]) in test_years]
        self.num_train_samples = len(self.train_files) * 365 * (num_neighbors + 1)
        self.num_val_samples = len(self.val_files) * 365 * (num_neighbors + 1)
        self.num_test_samples = len(self.test_files) * 365
        # if self.active_split_name == "train":
        #     self.starting_idx = 0
        # elif self.active_split_name == "val":
        #     self.starting_idx = 12
        # elif self.active_split_name == "test":
        #     self.starting_idx = 20
        # else:
        #     raise ValueError(f"Unsupported split name: {self.active_split_name}")
        ds = xarray.open_dataset(Path(self.path_daily_data, "kda_daily_pr_1961.nc"), engine="h5netcdf",
                                 decode_times=False)
        self.save_data = {"rlat": ds["rlat"].values, "rlon": ds["rlon"].values, "lat": ds["lat"].values,
                          "lon": ds["lon"].values, "height": ds["height"].values,
                          "rotated_pole_attrs": ds["rotated_pole"].attrs,
                          "gattrs": ds.attrs,
                          "rlat_attrs": ds["rlat"].attrs, "rlon_attrs": ds["rlon"].attrs, "lat_attrs": ds["lat"].attrs,
                          "lon_attrs": ds["lon"].attrs, "height_attrs": ds["height"].attrs,
                          "time_attrs": ds["time"].attrs, "pr_attrs": ds["pr"].attrs}
        ds.close()
        with open(Path(self.path_neighbors, 
                       "climex_pattern_search_5sims_djf", "results", "climex_neighbors.pkl"), "rb") as f:
            self.neighbors_djf = pickle.load(f)
        with open(Path(self.path_neighbors, 
                       "climex_pattern_search_5sims_jja", "results", "climex_neighbors.pkl"), "rb") as f:
            self.neighbors_jja = pickle.load(f)
        with open(Path(self.path_neighbors, 
                       "climex_pattern_search_5sims_mam", "results", "climex_neighbors.pkl"), "rb") as f:
            self.neighbors_mam = pickle.load(f)
        with open(Path(self.path_neighbors, 
                       "climex_pattern_search_5sims_son", "results", "climex_neighbors.pkl"), "rb") as f:
            self.neighbors_son = pickle.load(f)
        self.debug_max_sample = debug_max_sample

    def __len__(self):
        if self.active_split_name == "train":
            if self.debug_max_sample is not None:
                return min(self.num_train_samples, self.debug_max_sample)
            return self.num_train_samples
        elif self.active_split_name == "val":
            if self.debug_max_sample is not None:
                return min(self.num_val_samples, self.debug_max_sample)
            return self.num_val_samples
        elif self.active_split_name == "test":
            if self.debug_max_sample is not None:
                return min(self.num_test_samples, self.debug_max_sample)
            return self.num_test_samples
        else:
            raise ValueError(f"Unsupported split name: {self.active_split_name}")
    
    def get_neighbors_list_for_date(self, member, year, month, day):
        if month in [12, 1, 2]:
            return self.neighbors_djf.get((member, year, month, day), [])
        elif month in [3, 4, 5]:
            return self.neighbors_mam.get((member, year, month, day), [])
        elif month in [6, 7, 8]:
            return self.neighbors_jja.get((member, year, month, day), [])
        elif month in [9, 10, 11]:
            return self.neighbors_son.get((member, year, month, day), [])
        else:
            raise ValueError(f"Invalid month: {month}")
    
    def get_active_file(self, file_idx):
        if self.active_split_name == "train":
            return self.train_files[file_idx]
        elif self.active_split_name == "val":
            return self.val_files[file_idx]
        elif self.active_split_name == "test":
            return self.test_files[file_idx]
        else:
            raise ValueError(f"Unsupported split name: {self.active_split_name}")

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset with length {len(self)}")
        if self.active_split_name == "test":
            neighbor_idx = 0
            daily_count = idx
        else:
            neighbor_idx = idx % (self.num_neighbors + 1)
            daily_count = idx // (self.num_neighbors + 1)
        netcdf_idx = daily_count % 365
        file_count = daily_count // 365

        active_file = self.get_active_file(file_count)
        member = active_file.stem.split("_")[0]
        coarse_file_name = f"{member}_daily_coarse_pr_{active_file.stem.split('_')[-1]}.nc"
        ds_input = xarray.open_dataset(Path(self.path_daily_coarse_data, coarse_file_name), engine="h5netcdf",
                                       decode_times=False)
        input_data = normalize(ds_input["pr"].values[netcdf_idx, :, :], valid_min=0.0, valid_max=0.001,
                               log_normalize=True, log_offset=1e-12)
        cf_datetime = cftime.num2date(ds_input["time"].values[netcdf_idx], ds_input["time"].attrs["units"],
                                      calendar=ds_input["time"].attrs.get("calendar", "standard"))

        if neighbor_idx == 0:
            ds_output = xarray.open_dataset(active_file, engine="h5netcdf", decode_times=False)
            target_data = ds_output["pr"].values[netcdf_idx, :, :]
        else:
            neighbors_list = self.get_neighbors_list_for_date(
                member, cf_datetime.year, cf_datetime.month, cf_datetime.day)
            neighbor = neighbors_list[neighbor_idx - 1]
            ds_output = xarray.open_dataset(
                Path(self.path_daily_data, f"{neighbor[0]}_daily_pr_{neighbor[1]}.nc"),
                engine="h5netcdf", decode_times=False)
            cf_datetime_output = cftime.num2date(
                ds_output["time"].values[:], ds_output["time"].attrs["units"],
                calendar=ds_output["time"].attrs.get("calendar", "standard"))
            target_date = cftime.datetime(neighbor[1], neighbor[2], neighbor[3], 0, 30,
                                          calendar=ds_output["time"].attrs.get("calendar", "standard"))
            date_idx = next(i for i, d in enumerate(cf_datetime_output) if d == target_date)
            target_data = ds_output["pr"].values[date_idx, :, :]

        target_data = normalize(target_data, valid_min=0.0, valid_max=0.001,
                                log_normalize=True, log_offset=1e-12)
        idx_data = {"input_first_block": input_data,
                    "target": target_data,
                    "year": cf_datetime.year,
                    "month": cf_datetime.month,
                    "day": cf_datetime.day}
        ds_output.close()
        ds_input.close()
        return idx_data


def output_training_figures(config, count, input_data, target_data, output_data):
    if count in [0, 1, 2, 5, 10, 50, 100, 1000]:
        n_rows = input_data.shape[0]
        n_cols = 3
        fig = plt.figure(figsize=(12, 4 * n_rows))
        for i in range(n_rows):
            ax1 = fig.add_subplot(n_rows, n_cols, i * n_cols + 1)
            ax1.set_title("Input")
            ax1.pcolormesh(input_data[i, 0, :, :].cpu().detach().numpy(), vmin=-1, vmax=1)
            ax2 = fig.add_subplot(n_rows, n_cols, i * n_cols + 2)
            ax2.set_title("Target")
            ax2.pcolormesh(target_data[i, 0, :, :].cpu().detach().numpy(), vmin=-1, vmax=1)
            ax3 = fig.add_subplot(n_rows, n_cols, i * n_cols + 3)
            ax3.set_title("Output")
            ax3.pcolormesh(output_data[i, 0, :, :].cpu().detach().numpy(), vmin=-1, vmax=1)
        plt.tight_layout()
        plt.savefig(Path(config.path_model_output, f"training_visualization_{count}.png"))
        plt.close(fig)


def test_climex_walking_skeleton():
    # ToDo: this is a workaround to not expose local paths in the code base.
    config = config_from_yaml(climex_for_prob_unet.ProbUnetClimexConfig,
                              Path(Path.home(), "tmp", "climex_skeleton_config.yaml"))

    # Step 1: Merge hourly files into daily
    for member in config.members:
        for year in range(config.starting_training_year, config.ending_training_year + 1):
            climex_for_prob_unet.climex_hourly_to_daily_single_year_to_disk(config, member, year)
    
    # Step 1.1: Create smoothed version of daily precip
    for member in config.members:
        for year in range(config.starting_training_year, config.ending_training_year + 1):
            climex_for_prob_unet.climex_upscale_single_year_to_disk(config, member, year)
    
    # Step 2: Need a dataloader that can read this file and feed it into a UNet for training.
    # ToDo: there may need to be some smarter shuffling
    #       (e.g. load a number of different netcdf files and shuffle in this subset, there's a name for that...)
    skeleton_dataset = SkeletonDataset(path_daily_data=config.path_daily_output,
                                       path_daily_coarse_data=config.path_daily_coarse_output,
                                       path_neighbors=config.path_neighbors,
                                       train_years=[1961],
                                       validation_years=[],
                                       test_years=[],
                                       num_neighbors=2,
                                       active_split_name="train",
                                       debug_max_sample=None)
    if config.device == "cuda":
        data_loader = td.DataLoader(skeleton_dataset, batch_size=16, shuffle=False, pin_memory=True)
    else:
        data_loader = td.DataLoader(skeleton_dataset, batch_size=8, shuffle=False)

    # Step 3: Train a probabilistic UNet on this data and validate that it can learn something meaningful.
    # ToDo: consider moving the latent space before the resolution increase layers.
    prob_unet = ProbabilisticUnet(
        in_channels=1, out_channels=1, num_latent_dimensions=2, depth=2,
        initial_nb_of_hidden_channels=8, kernel_size=3, resolution_increase_layers=2)
    prob_unet.to(config.device)
    # ToDo: this needs to have its own path
    path_prob_unet_model = Path(config.path_model_output, "prob_unet_model.pth")
    criterion = nn.MSELoss()
    ssim_loss = MultiScaleStructuralSimilarityIndexMeasure(data_range=(-1.0, 1.0), kernel_size=7).to(config.device)
    if not path_prob_unet_model.is_file():
        optimizer = optim.Adam(prob_unet.parameters(), lr=0.01)
        prob_unet.train()
        losses = []
        kl_losses = []
        reconstruction_losses = []
        for epoch in range(64):  # ToDo: more epochs
            for item in data_loader:
                optimizer.zero_grad()
                input_data = item["input_first_block"].unsqueeze(1).to(config.device)  # Add channel dimension
                target_data = item["target"].unsqueeze(1).to(config.device)  # Add channel dimension
                output = prob_unet(input_data, target_data=target_data, mode="train")  # Add channel dimension
                output_training_figures(config=config, count=len(losses),
                                        input_data=input_data,
                                        target_data=target_data, output_data=output)
                # reconstruction_loss = criterion(output.squeeze(1), target_data)
                reconstruction_loss = 1 - ssim_loss(output, target_data)
                kl_loss = prob_unet.kl_divergence()
                loss = reconstruction_loss + kl_loss
                loss.backward()
                optimizer.step()
                reconstruction_losses.append(reconstruction_loss.item())
                kl_losses.append(kl_loss.item())
                losses.append(loss.item())
        torch.save(prob_unet.state_dict(), path_prob_unet_model)
    else:        
        prob_unet.load_state_dict(torch.load(path_prob_unet_model))
    prob_unet.eval()


    # Step 4: Output some inference with latent space control
    # Using train set for dummy inference, but should be test set after validation
    data_loader_test = td.DataLoader(
        SkeletonDataset(path_daily_data=config.path_daily_output,
                        path_daily_coarse_data=config.path_daily_coarse_output,
                        path_neighbors=config.path_neighbors,
                        train_years=[],
                        validation_years=[],
                        test_years=[1962],
                        active_split_name="test"),
        batch_size=4, shuffle=False)
    for item in data_loader_test:
        for n, latent_space_coords in enumerate(itertools.product(*config.latent_space_discretization)):
            # ToDo: call probabilistic UNet with a specified latent vector to control the output
            # result = unet(item["input_first_block"].unsqueeze(1))  # Add channel dimension
            # Dummy latent space variation
            # result += n / 20.0
            result = prob_unet(item["input_first_block"].unsqueeze(1), mode="inference",
                               latent_coord=torch.Tensor(latent_space_coords))  # Add channel dimension
            z = criterion(result.squeeze(1), item["target"]).item()
            for i in range(result.shape[0]):
                result_data = inverse_normalize(result[i].cpu().detach().numpy(), known_min=0.0, known_max=0.001,
                                                log_normalize=True, log_offset=1e-12) 
                # ToDo: this has to have its own path
                climex_for_prob_unet.save_result_from_dataset_item(
                    config.path_inference_output, data_loader_test.dataset, {x: item[x][i] for x in item}, result_data,
                    latent_space_idx=n, latent_space_coords=latent_space_coords)
        break  # ToDo: remove

    # Step 5: Compute some metrics

    # Step 6: visualize some results
    ds = xarray.open_dataset(Path(config.path_inference_output, "inference_1962_01_01_1.nc"), engine="h5netcdf")
    fig1 = plt.figure(figsize=(12, 12))
    ax1 = fig1.add_subplot(1, 1, 1)
    nd_ax_plot(ax1, fig1, ds["pr"].values[0, :, :], "", vmin=0, vmax=0.00012, reverse_i=True)
    fig1.savefig(Path(config.path_inference_output, "test_visualization.png"))
    plt.close(fig1)

    # ds = xarray.open_dataset(Path(config.path_inference_output, "target_1962_01_21_None.nc"), engine="h5netcdf")
    # fig1 = plt.figure(figsize=(12, 12))
    # ax1 = fig1.add_subplot(1, 1, 1)
    # nd_ax_plot(ax1, fig1, ds["pr"].values[0, :, :], "", vmin=0, vmax=0.00012, reverse_i=True)
    # fig1.savefig(Path(config.path_inference_output, "test_visualization_target.png"))
    # plt.close(fig1)

    fig_final = plt.figure(figsize=(20, 10))
    ax1 = fig_final.add_subplot(3, 4, 1)
    ds1 = xarray.open_dataset(Path(config.path_daily_coarse_output, "kda_daily_coarse_pr_1962.nc"), engine="h5netcdf",
                                decode_times=False)
    ax1.pcolormesh(ds1["pr"].values[0, :, :], vmin=0, vmax=0.00012)
    ax1.set_title("Input")
    ax2 = fig_final.add_subplot(3, 4, 5)
    ds2 = xarray.open_dataset(Path(config.path_daily_output, "kda_daily_pr_1962.nc"), engine="h5netcdf",
                              decode_times=False)
    ax2.pcolormesh(ds2["pr"].values[0, :, :], vmin=0, vmax=0.00012)
    ax2.set_title("Target")
    for n, latent_space_coords in enumerate(itertools.product(*config.latent_space_discretization)):
        idx_2d = np.unravel_index(n, (len(config.latent_space_discretization[0]), len(config.latent_space_discretization[1])))
        i = idx_2d[0]
        j = idx_2d[1] + 1
        ax = fig_final.add_subplot(3, 4, i * 4 + j + 1)
        ds_n = xarray.open_dataset(Path(config.path_inference_output, f"inference_1962_01_01_{n}.nc"), engine="h5netcdf",
                                   decode_times=False)
        data_n = ds_n["pr"].values[0, :, :]
        # data = np.random.rand(128, 128) * 0.00003 * (latent_space_coords[0] + 1) * (latent_space_coords[1] + 1)  # Dummy data with some variation
        ax.pcolormesh(data_n, vmin=0, vmax=0.00012)
        ax.set_title(f"Sample {i * 4 + j}")
    fig_final.savefig(Path(config.path_inference_output, "test_visualization_final.png"))
    plt.close(fig_final)
    assert Path(config.path_inference_output, "test_visualization_final.png").is_file()
