from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cftime
import numpy as np
import xarray

from resoterre.data_management.netcdf_utils import CFVariables, netcdf_defaults


@dataclass(frozen=True, slots=True)
class ProbUnetClimexConfig:
    path_output: Path
    path_climex: Path
    path_neighbors: Path
    variable_name: str
    starting_training_year: int
    ending_training_year: int
    members: list = field(default_factory=list)
    overwrite_existing_daily_files: bool = False
    latent_space_discretization: list = field(default_factory=list)


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


def climex_upscale(path_data, member, year):
    ds = xarray.open_dataset(Path(path_data, f"{member}_daily_pr_{year:04d}.nc"), decode_times=False)
    # ToDo: spatial index should be arguments
    pr_reshaped = ds["pr"].values.reshape(ds["pr"].shape[0], 32, 4, 32, 4)
    pr_coarse = pr_reshaped.mean(axis=(2, 4))
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
    if config.variable_name != "pr":
        raise NotImplementedError("This function is currently only verified for 'pr' variable.")
    path_sample_input = Path(config.path_output, f"{member}_daily_{config.variable_name}_{year}.nc")
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
