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
from torch.utils import data as td


from resoterre.config_utils import config_from_yaml
from resoterre.ml.data_loader_utils import normalize, inverse_normalize
from resoterre.plots.nd_plots import nd_ax_plot

from prob_unet.datasets import climex_for_prob_unet
from prob_unet.ml.probabilistic_unet import ProbabilisticUnet


class SkeletonDataset(td.Dataset):
    def __init__(self, path_data, path_neighbors, active_split_name="train"):
        self.path_data = path_data
        self.path_neighbors = path_neighbors
        self.num_neighbors = 2
        self.active_split_name = active_split_name
        if self.active_split_name == "train":
            self.starting_idx = 0
        elif self.active_split_name == "val":
            self.starting_idx = 12
        elif self.active_split_name == "test":
            self.starting_idx = 20
        else:
            raise ValueError(f"Unsupported split name: {self.active_split_name}")
        ds = xarray.open_dataset(Path(self.path_data, "kda_daily_pr_1961.nc"), engine="h5netcdf",
                                 decode_times=False)
        self.save_data = {"rlat": ds["rlat"].values, "rlon": ds["rlon"].values, "lat": ds["lat"].values,
                          "lon": ds["lon"].values, "height": ds["height"].values,
                          "rotated_pole_attrs": ds["rotated_pole"].attrs,
                          "gattrs": ds.attrs,
                          "rlat_attrs": ds["rlat"].attrs, "rlon_attrs": ds["rlon"].attrs, "lat_attrs": ds["lat"].attrs,
                          "lon_attrs": ds["lon"].attrs, "height_attrs": ds["height"].attrs,
                          "time_attrs": ds["time"].attrs, "pr_attrs": ds["pr"].attrs}
        ds.close()
        with open(str(self.path_neighbors), "rb") as f:
            self.neighbors = pickle.load(f)

    def __len__(self):
        if self.active_split_name == "train":
            return 12 * (self.num_neighbors + 1)
        elif self.active_split_name == "val":
            return 8 * (self.num_neighbors + 1)
        elif self.active_split_name == "test":
            return 8 * (self.num_neighbors + 1)
        else:
            raise ValueError(f"Unsupported split name: {self.active_split_name}")


    def __getitem__(self, idx):
        neighbor_idx = idx % (self.num_neighbors + 1)
        file_count = idx // (self.num_neighbors + 1)
        netcdf_idx = self.starting_idx + file_count

        ds_input = xarray.open_dataset(
            Path(self.path_data, "kda_daily_pr_1961_coarse.nc"),
            engine="h5netcdf", decode_times=False)
        input_data = normalize(ds_input["pr"].values[netcdf_idx, :, :], valid_min=0.0, valid_max=0.001,
                               log_normalize=True, log_offset=1e-12)
        cf_datetime = cftime.num2date(ds_input["time"].values[netcdf_idx], ds_input["time"].attrs["units"],
                                      calendar=ds_input["time"].attrs.get("calendar", "standard"))

        if neighbor_idx == 0:
            ds_output = xarray.open_dataset(
                Path(self.path_data, "kda_daily_pr_1961.nc"),
                engine="h5netcdf", decode_times=False)
            target_data = ds_output["pr"].values[netcdf_idx, :, :]
        else:
            neighbors_list = self.neighbors[("kda", cf_datetime.year, cf_datetime.month, cf_datetime.day)]
            neighbor = neighbors_list[neighbor_idx - 1]
            ds_output = xarray.open_dataset(
                Path(self.path_data, f"{neighbor[0]}_daily_pr_{neighbor[1]}.nc"),
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


def test_climex_walking_skeleton():
    # ToDo: this is a workaround to not expose local paths in the code base.
    config = config_from_yaml(climex_for_prob_unet.ProbUnetClimexConfig,
                              Path(Path.home(), "tmp", "climex_skeleton_config.yaml"))

    # Step 1: Merge hourly files into daily
    for member in config.members:
        for year in range(config.starting_training_year, config.ending_training_year + 1):
            climex_for_prob_unet.climex_hourly_to_daily_single_year_to_disk(config, member, year)
    
    # Step 1.1: Create smoothed version of daily precip
    # ToDo: write into a single function call
    for member in config.members:
        for year in range(config.starting_training_year, config.ending_training_year + 1):
            path_coarse_input = Path(config.path_output, f"{member}_daily_{config.variable_name}_{year}_coarse.nc")
            if not path_coarse_input.is_file():
                ds_coarse = climex_for_prob_unet.climex_upscale(config.path_output, member, year)
                ds_coarse.to_netcdf(path_coarse_input, engine="h5netcdf",
                                    encoding={"pr": {"chunksizes": (365, 32, 32)}})
    
    # Step 2: Need a dataloader that can read this file and feed it into a UNet for training.
    # ToDo: there may need to be some smarter shuffling
    #       (e.g. load a number of different netcdf files and shuffle in this subset, there's a name for that...)
    skeleton_dataset = SkeletonDataset(path_data=config.path_output, path_neighbors=config.path_neighbors,
                                       active_split_name="train")
    data_loader = td.DataLoader(skeleton_dataset, batch_size=4, shuffle=True)

    # Step 3: Train a probabilistic UNet on this data and validate that it can learn something meaningful.
    # ToDo: consider moving the latent space before the resolution increase layers.
    prob_unet = ProbabilisticUnet(
        in_channels=1, out_channels=1, num_latent_dimensions=2, depth=2,
        initial_nb_of_hidden_channels=8, kernel_size=3, resolution_increase_layers=2)
    path_prob_unet_model = Path(config.path_output, "prob_unet_model.pth")
    criterion = nn.MSELoss()
    if not path_prob_unet_model.is_file():
        optimizer = optim.Adam(prob_unet.parameters(), lr=0.01)
        prob_unet.train()
        losses = []
        kl_losses = []
        reconstruction_losses = []
        for epoch in range(80):  # ToDo: more epochs
            for item in data_loader:
                optimizer.zero_grad()
                output = prob_unet(item["input_first_block"].unsqueeze(1),
                                   target_data=item["target"].unsqueeze(1), mode="train")  # Add channel dimension
                reconstruction_loss = criterion(output.squeeze(1), item["target"])
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
        SkeletonDataset(path_data=config.path_output, path_neighbors=config.path_neighbors, active_split_name="train"),
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
                climex_for_prob_unet.save_result_from_dataset_item(
                    config.path_output, data_loader_test.dataset, {x: item[x][i] for x in item}, result_data,
                    latent_space_idx=n, latent_space_coords=latent_space_coords)
        break  # ToDo: remove

    # Step 5: Compute some metrics

    # Step 6: visualize some results
    ds = xarray.open_dataset(Path(config.path_output, "inference_1961_01_01_1.nc"), engine="h5netcdf")
    fig1 = plt.figure(figsize=(12, 12))
    ax1 = fig1.add_subplot(1, 1, 1)
    nd_ax_plot(ax1, fig1, ds["pr"].values[0, :, :], "", vmin=0, vmax=0.00012, reverse_i=True)
    fig1.savefig(Path(config.path_output, "test_visualization.png"))
    plt.close(fig1)

    ds = xarray.open_dataset(Path(config.path_output, "target_1961_01_21_None.nc"), engine="h5netcdf")
    fig1 = plt.figure(figsize=(12, 12))
    ax1 = fig1.add_subplot(1, 1, 1)
    nd_ax_plot(ax1, fig1, ds["pr"].values[0, :, :], "", vmin=0, vmax=0.00012, reverse_i=True)
    fig1.savefig(Path(config.path_output, "test_visualization_target.png"))
    plt.close(fig1)

    fig_final = plt.figure(figsize=(20, 10))
    ax1 = fig_final.add_subplot(3, 4, 1)
    ds1 = xarray.open_dataset(Path(config.path_output, "kda_daily_pr_1961_coarse.nc"), engine="h5netcdf",
                                decode_times=False)
    ax1.pcolormesh(ds1["pr"].values[0, :, :], vmin=0, vmax=0.00012)
    ax1.set_title("Input")
    ax2 = fig_final.add_subplot(3, 4, 5)
    ds2 = xarray.open_dataset(Path(config.path_output, "kda_daily_pr_1961.nc"), engine="h5netcdf",
                              decode_times=False)
    ax2.pcolormesh(ds2["pr"].values[0, :, :], vmin=0, vmax=0.00012)
    ax2.set_title("Target")
    for n, latent_space_coords in enumerate(itertools.product(*config.latent_space_discretization)):
        idx_2d = np.unravel_index(n, (len(config.latent_space_discretization[0]), len(config.latent_space_discretization[1])))
        i = idx_2d[0]
        j = idx_2d[1] + 1
        ax = fig_final.add_subplot(3, 4, i * 4 + j + 1)
        ds_n = xarray.open_dataset(Path(config.path_output, f"inference_1961_01_01_{n}.nc"), engine="h5netcdf",
                                   decode_times=False)
        data_n = ds_n["pr"].values[0, :, :]
        # data = np.random.rand(128, 128) * 0.00003 * (latent_space_coords[0] + 1) * (latent_space_coords[1] + 1)  # Dummy data with some variation
        ax.pcolormesh(data_n, vmin=0, vmax=0.00012)
        ax.set_title(f"Sample {i * 4 + j}")
    fig_final.savefig(Path(config.path_output, "test_visualization_final.png"))
    plt.close(fig_final)
    assert Path(config.path_output, "test_visualization_final.png").is_file()
