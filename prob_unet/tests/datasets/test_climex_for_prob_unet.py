import itertools
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import xarray
from torch.utils import data as td


from resoterre.config_utils import config_from_yaml
from resoterre.ml.data_loader_utils import inverse_normalize
from resoterre.plots.nd_plots import nd_ax_plot

from prob_unet.datasets import climex_for_prob_unet
from prob_unet.ml.climex_training import climex_training


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

    prob_unet = climex_training(config)
    prob_unet.eval()

    # Step 4: Output some inference with latent space control
    # Using train set for dummy inference, but should be test set after validation
    data_loader_test = td.DataLoader(
        climex_for_prob_unet.ClimexDataset(
            path_daily_data=config.path_daily_output,
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
            input_data = item["input_first_block"].unsqueeze(1).to(config.device)
            result = prob_unet(input_data, mode="inference", latent_coord=torch.Tensor(latent_space_coords))
            # z = criterion(result.squeeze(1), item["target"].to(config.device)).item()
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
