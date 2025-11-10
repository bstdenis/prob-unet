import calendar
from pathlib import Path

import numpy as np
import xarray
from prob_unet.patterns.feature_extraction import custom_features
from resoterre.ml.data_loader_utils import normalize


climex_simulations = ["kda", "kdb", "kdc", "kdd", "kde", "kdf", "kdg", "kdh", "kdi", "kdj", "kdk", "kdl", "kdm", "kdn",
                      "kdo", "kdp", "kdq", "kdr", "kds", "kdt", "kdu", "kdv", "kdw", "kdx", "kdy", "kdz", "kea", "keb",
                      "kec", "ked", "kee", "kef", "keg", "keh", "kei", "kej", "kek", "kel", "kem", "ken", "keo", "kep",
                      "keq", "ker", "kes", "ket", "keu", "kev", "kew", "kex"]


def list_monthly_files(path_climex, variables, simulations=None, years=None, months=None,
                       all_variables_available=False):
    simulations = simulations or climex_simulations
    years = years or list(range(1950, 2100))
    months = months or list(range(1, 13))
    if all_variables_available and (len(variables) > 1):
        raise NotImplementedError()
    
    d = {}
    for variable in variables:
        d[variable] = {}
        for simulation in simulations:
            path_simulation = Path(path_climex, simulation, "series")
            d[variable][simulation] = []
            for year in years:
                for month in months:
                    file_name = f"{variable}_{simulation}_{year}{month:02d}_se.nc"
                    path_nc_file = Path(path_simulation, f"{year}{month:02d}", file_name)
                    if path_nc_file.is_file():
                        d[variable][simulation].append(str(path_nc_file))
    return d


def list_daily_timesteps(path_climex, variables, simulations=None, years=None, months=None):
    d = list_monthly_files(path_climex, variables, simulations, years, months)
    for variable in d:
        for simulation in d[variable]:
            daily_files = []
            for monthly_file in d[variable][simulation]:
                year = int(Path(monthly_file).name.split('_')[2][:4])
                month = int(Path(monthly_file).name.split('_')[2][4:6])
                days_in_month = calendar.monthrange(year, month)[1]
                for day in range(1, days_in_month + 1):
                    if (month == 2) and (day == 29):
                        # No Feb 29 in climex
                        continue
                    daily_files.append((monthly_file, day))
            d[variable][simulation] = daily_files
    return d


def split_daily_timesteps_in_batch(daily_timesteps_dict, batch_size):
    if len(daily_timesteps_dict) > 1:
        raise NotImplementedError("Batching only implemented for one variable at a time.")
    batches = []
    count = 0
    current_batch = []
    for variable in daily_timesteps_dict:
        for simulation in daily_timesteps_dict[variable]:
            for path_nc_file, day in daily_timesteps_dict[variable][simulation]:
                year = int(Path(path_nc_file).name.split('_')[2][:4])
                month = int(Path(path_nc_file).name.split('_')[2][4:6])
                current_batch.append((variable, simulation, year, month, day))
                count += 1
                if count >= batch_size:
                    batches.append(current_batch)
                    current_batch = []
                    count = 0
    if current_batch:
        batches.append(current_batch)
    return batches


def pr_daily_feature_extraction(path_climex, sim, year, month, day, i_min, i_max, j_min, j_max, upscale_factor=4,
                                feature_mode='16d'):
    nc_file = Path(path_climex, f"{sim}/series/{year}{month:02d}/pr_{sim}_{year}{month:02d}_se.nc")
    ds = xarray.open_dataset(nc_file, decode_times=False)
    pr = ds['pr'].values  # (time, lat, lon)
    data = pr[(day - 1) * 24: day * 24, i_min:i_max, j_min:j_max].mean(0)
    feature_data = normalize(data, mode=(0, 1), valid_min=0, valid_max=1e-1, log_normalize=True, log_offset=1e-8)
    upscale_x, upscale_y = (feature_data.shape[0] // upscale_factor, feature_data.shape[1] // upscale_factor)
    upscaled_data = feature_data.reshape(upscale_x, 4, upscale_y, 4).mean(axis=(1, 3))
    features = custom_features(upscaled_data, mode=feature_mode)
    return features


def pr_multiple_samples_batch(path_climex, list_of_candidates, i_min, i_max, j_min, j_max, upscale_factor=4,
                              feature_mode='16d'):
    # (sim, year, month, day)
    list_of_features = []
    for candidate in list_of_candidates:
        sim, year, month, day = candidate
        list_of_features.append(pr_daily_feature_extraction(path_climex, sim, year, month, day,
                                                            i_min, i_max, j_min, j_max, upscale_factor, feature_mode))
    return np.array(list_of_features)
