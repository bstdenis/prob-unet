import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cftime
import numpy as np
import xarray
from torch.utils import data as td

from resoterre.config_utils import config_from_yaml
from resoterre.data_management.netcdf_utils import CFVariables, netcdf_defaults
from resoterre.ml.data_loader_utils import normalize, inverse_normalize


@dataclass(frozen=True, slots=True)
class ProbUnetClimexConfig:
    experiment_name: str
    path_daily_output: Path
    path_daily_coarse_output: Path
    path_model_output: Path
    path_inference_output: Path
    path_climex: Path
    path_neighbors: Path
    variable_name: str
    starting_training_year: int
    ending_training_year: int
    members: list = field(default_factory=list)
    overwrite_existing_daily_files: bool = False
    overwrite_existing_coarse_files: bool = False
    output_original_grid_coarse_files: bool = False
    num_latent_dimensions: int = 2
    unet_depth: int = 2
    depth_of_latent_injection: int = 0  # 0 means in the last layer, 1 means one layer before last, etc.
    initial_nb_of_hidden_channels: int = 8
    unet_kernel_size: int = 3
    learning_rate: float = 0.01
    mse_weight: float = 1.0
    ssim_weight: float = 0.0
    kl_weight: float = 1.0
    latent_space_discretization: list = field(default_factory=list)
    training_batch_size: int = 32
    device: str = "cpu"
    num_epochs: int = 10
    restart_training: bool = False
    inference_batch_size: int = 32
    inference_device: str = "cpu"


def prob_unet_climex_parse_config(config: ProbUnetClimexConfig | Path | str) -> ProbUnetClimexConfig:
    if isinstance(config, ProbUnetClimexConfig):
        return config
    else:
        return config_from_yaml(ProbUnetClimexConfig, config)


def climex_hourly_to_daily_single_year(path_climex, member, year):
    nc_files = []
    for m in range(1, 13):
        nc_files.append(
            f"{path_climex}/{member}/series/{year:04d}{m:02d}/pr_{member}_{year:04d}{m:02d}_se.nc")
    ds = xarray.open_mfdataset(nc_files, decode_times=False)
    # ToDo: spatial index should be arguments
    hourly_pr = ds["pr"].values[:, 80:208, 100:228]
    if hourly_pr.shape[0] % 24 != 0:
        raise ValueError("Time dimension is not a multiple of 24, cannot reshape to daily.")
    hourly_pr_reshape = hourly_pr.reshape((hourly_pr.shape[0] // 24, 24, 128, 128))
    daily_pr = hourly_pr_reshape.mean(axis=1)

    cf_attrs = {
        "Conventions": "CF-1.13",
        "title": "ClimEx",
        "history": f"Created on {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')} from original data",
        # "institution": "",
        "source": "CRCM5",
        # "comment": "",
        "references": "http://www.ouranos.ca",}
    for key, value in ds.attrs.items():
        if (key not in cf_attrs) and (key not in ["contact", "reference"]):
            cf_attrs[key] = value
    
    cf_coordinates = CFVariables()
    cf_coordinates.add(
            "rlat",
            data=ds["rlat"].values[80:208],
            dtype=np.float32,
            attributes={key: value for key, value in ds["rlat"].attrs.items() if key not in ["actual_range"]},
        )
    cf_coordinates.add(
            "rlon",
            data=ds["rlon"].values[100:228],
            dtype=np.float32,
            attributes={key: value for key, value in ds["rlon"].attrs.items() if key not in ["actual_range"]},
        )
    for key in ["lat", "lon"]:
        cf_coordinates.add(
            key,
            dims=("rlat", "rlon"),
            data=ds[key].values[80:208, 100:228],
            dtype=np.float32,
            attributes=netcdf_defaults[f"{key}_attributes"],
        )
    cf_coordinates.add(
            "height",
            data=ds["height"].values,
            dtype=np.float32,
            attributes=ds["height"].attrs,
        )
    cf_coordinates.add(
            "time",
            data=ds["time"].values[::24],
            dtype=np.float64,
            attributes=ds["time"].attrs,
        )
    time_bnds = np.zeros((ds["time_bnds"].shape[0] // 24, 2), dtype=np.float64)
    time_bnds[:, 0] = ds["time_bnds"].values[::24, 0]
    time_bnds[:, 1] = ds["time_bnds"].values[23::24, 1]
    cf_coordinates.add(
            "time_bnds",
            dims=("time", "bnds"),
            data=time_bnds,
            dtype=np.float64,
            attributes=ds["time_bnds"].attrs,
        )
    cf_coordinates.add(
        "rotated_pole",
        dims = (),
        data = np.array(0, dtype=np.int8),
        attributes = ds["rotated_pole"].attrs,
    )

    cf_variables = CFVariables()
    cf_variables.add(
        "pr",
        dims=("time", "rlat", "rlon"),
        data=daily_pr.astype(np.float32),
        attributes=ds["pr"].attrs,)

    ds = xarray.Dataset(data_vars=cf_variables, coords=cf_coordinates, attrs=cf_attrs)

    return ds


def climex_upscale(path_data, member, year, original_grid=False):
    ds = xarray.open_dataset(Path(path_data, f"{member}_daily_pr_{year:04d}.nc"), decode_times=False,
                             engine="h5netcdf")
    # ToDo: spatial index should be arguments
    pr_reshaped = ds["pr"].values.reshape(ds["pr"].shape[0], 32, 4, 32, 4)
    pr_coarse = pr_reshaped.mean(axis=(2, 4))
    if original_grid:
        pr_coarse = np.repeat(np.repeat(pr_coarse, 4, axis=1), 4, axis=2)
        rlat_coarse = ds["rlat"].values
        rlon_coarse = ds["rlon"].values
        lat_coarse = ds["lat"].values
        lon_coarse = ds["lon"].values
    else:
        rlat_reshaped = ds["rlat"].values.reshape(128 // 4, 4)
        rlat_coarse = rlat_reshaped.mean(axis=1)
        rlon_reshaped = ds["rlon"].values.reshape(128 // 4, 4)
        rlon_coarse = rlon_reshaped.mean(axis=1)
        lat_reshaped = ds["lat"].values.reshape(128 // 4, 4, 128 // 4, 4)
        lat_coarse = lat_reshaped.mean(axis=(1, 3))
        lon_reshaped = ds["lon"].values.reshape(128 // 4, 4, 128 // 4, 4)
        lon_coarse = lon_reshaped.mean(axis=(1, 3))

    cf_attrs = {
        "Conventions": "CF-1.13",
        "title": "ClimEx",
        "history": f"Created on {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
        # "institution": "",
        "source": "CRCM5",
        # "comment": "",
        "references": "http://www.ouranos.ca",}
    for key, value in ds.attrs.items():
        cf_attrs[key] = value
    
    cf_coordinates = CFVariables()
    cf_coordinates.add(
            "rlat",
            data=rlat_coarse,
            dtype=np.float32,
            attributes={key: value for key, value in ds["rlat"].attrs.items() if key not in ["actual_range"]},
        )
    cf_coordinates.add(
            "rlon",
            data=rlon_coarse,
            dtype=np.float32,
            attributes={key: value for key, value in ds["rlon"].attrs.items() if key not in ["actual_range"]},
        )
    for key in ["lat", "lon"]:
        cf_coordinates.add(
            key,
            dims=("rlat", "rlon"),
            data=lat_coarse if key == "lat" else lon_coarse,
            dtype=np.float32,
            attributes=netcdf_defaults[f"{key}_attributes"],
        )
    cf_coordinates.add(
            "height",
            data=ds["height"].values,
            dtype=np.float32,
            attributes=ds["height"].attrs,
        )
    cf_coordinates.add(
            "time",
            data=ds["time"].values,
            dtype=np.float64,
            attributes=ds["time"].attrs,
        )
    cf_coordinates.add(
            "time_bnds",
            dims=("time", "bnds"),
            data=ds["time_bnds"].values,
            dtype=np.float64,
            attributes=ds["time_bnds"].attrs,
        )
    cf_coordinates.add(
        "rotated_pole",
        dims = (),
        data = np.array(0, dtype=np.int8),
        attributes = ds["rotated_pole"].attrs,
    )

    cf_variables = CFVariables()
    cf_variables.add(
        "pr",
        dims=("time", "rlat", "rlon"),
        data=pr_coarse.astype(np.float32),
        attributes=ds["pr"].attrs,)

    ds = xarray.Dataset(data_vars=cf_variables, coords=cf_coordinates, attrs=cf_attrs)

    return ds


def save_result_from_dataset_item(path_output, dataset, item, result, prefix="inference", latent_space_idx=None,
                                  latent_space_coords=None):
    cf_attrs = {
        "Conventions": "CF-1.13",
        "title": "ClimEx",
        "history": f"Created on {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
        # "institution": "",
        # "source": "CRCM5",
        # "comment": "",
        "references": "http://www.ouranos.ca",}
    for key, value in dataset.save_data["gattrs"].items():
        if (key not in cf_attrs) and (key not in ["contact", "reference"]):
            cf_attrs[key] = value
    
    cf_coordinates = CFVariables()
    cf_coordinates.add(
            "rlat",
            data=dataset.save_data["rlat"],
            dtype=np.float32,
            attributes={key: value for key, value in dataset.save_data["rlat_attrs"].items() if key not in ["actual_range"]},
        )
    cf_coordinates.add(
            "rlon",
            data=dataset.save_data["rlon"],
            dtype=np.float32,
            attributes={key: value for key, value in dataset.save_data["rlon_attrs"].items() if key not in ["actual_range"]},
        )
    for key in ["lat", "lon"]:
        cf_coordinates.add(
            key,
            dims=("rlat", "rlon"),
            data=dataset.save_data[key],
            dtype=np.float32,
            attributes=dataset.save_data[f"{key}_attrs"],
        )
    cf_coordinates.add(
            "height",
            data=dataset.save_data["height"],
            dtype=np.float32,
            attributes=dataset.save_data["height_attrs"],
        )
    cf_coordinates.add(
            "time",
            data=[cftime.DatetimeNoLeap(item["year"], item["month"], item["day"])],
            dtype=np.float64,
            attributes={k: v for k, v in dataset.save_data["time_attrs"].items() if k not in ["calendar", "units"]},
        )
    cf_coordinates.add(
        "rotated_pole",
        dims = (),
        data = np.array(0, dtype=np.int8),
        attributes = dataset.save_data["rotated_pole_attrs"],
    )
    if latent_space_coords is not None:
        cf_coordinates.add(
            "latent_space_coords",
            data = np.array(latent_space_coords, dtype=np.float32),
            attributes = {},
        )

    cf_variables = CFVariables()
    cf_variables.add(
        "pr",
        dims=("time", "rlat", "rlon"),
        data=result.astype(np.float32),
        attributes=dataset.save_data["pr_attrs"],)

    ds = xarray.Dataset(data_vars=cf_variables, coords=cf_coordinates, attrs=cf_attrs)
    file_name = f"{prefix}_{item['year']}_{item['month']:02d}_{item['day']:02d}_{latent_space_idx}.nc"
    ds.to_netcdf(Path(path_output, file_name), engine="h5netcdf",
                 encoding={"pr": {"chunksizes": (1, 128, 128)}})


def climex_hourly_to_daily_single_year_to_disk(config, member, year):
    config = prob_unet_climex_parse_config(config)
    if config.variable_name != "pr":
        raise NotImplementedError("This function is currently only verified for 'pr' variable.")
    path_sample_input = Path(config.path_daily_output, f"{member}_daily_{config.variable_name}_{year}.nc")
    if not path_sample_input.is_file() or config.overwrite_existing_daily_files:
        ds_daily = climex_hourly_to_daily_single_year(config.path_climex, member, year)
        path_sample_input.parent.mkdir(parents=True, exist_ok=True)
        ds_daily.to_netcdf(path_sample_input, engine="h5netcdf", encoding={"pr": {"chunksizes": (128, 128, 128)}})
    # ToDo: sample figure logic based on configurable options
    # ds = xarray.open_dataset(path_sample_input, engine="h5netcdf", decode_times=False)
    # fig1 = plt.figure(figsize=(12, 12))
    # ax1 = fig1.add_subplot(1, 1, 1)
    # ax1.pcolormesh(ds["lon"].values[0, :], ds["lat"].values[:, 0], ds["pr"].values[0, :, :], vmin=0, vmax=0.00012,
    #                shading="nearest")
    # ax1.set_xlabel("Longitude at bottom")
    # ax1.set_ylabel("Latitude at left")
    # fig1.savefig(Path(config.path_output, f"test_daily_pr_{member}_{year}.png"))
    # plt.close(fig1)


def climex_upscale_single_year_to_disk(config, member, year):
    config = prob_unet_climex_parse_config(config)
    if config.variable_name != "pr":
        raise NotImplementedError("This function is currently only verified for 'pr' variable.")
    path_sample_input = Path(config.path_daily_coarse_output,
                             f"{member}_daily_coarse_{config.variable_name}_{year}.nc")
    if not path_sample_input.is_file() or config.overwrite_existing_coarse_files:
        ds_coarse = climex_upscale(config.path_daily_output, member, year)
        path_sample_input.parent.mkdir(parents=True, exist_ok=True)
        ds_coarse.to_netcdf(path_sample_input, engine="h5netcdf", encoding={"pr": {"chunksizes": (365, 32, 32)}})
    if config.output_original_grid_coarse_files:
        path_sample_input = Path(config.path_daily_coarse_output,
                                 f"{member}_daily_coarse_original_grid_{config.variable_name}_{year}.nc")
        if not path_sample_input.is_file() or config.overwrite_existing_coarse_files:
            ds_coarse = climex_upscale(config.path_daily_output, member, year, original_grid=True)
            path_sample_input.parent.mkdir(parents=True, exist_ok=True)
            ds_coarse.to_netcdf(path_sample_input, engine="h5netcdf", encoding={"pr": {"chunksizes": (128, 128, 128)}})


class ClimexDataset(td.Dataset):
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
