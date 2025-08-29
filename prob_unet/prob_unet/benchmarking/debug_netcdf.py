import datetime
import functools
from pathlib import Path

import cartopy.crs as ccrs
import numpy as np
import xarray

from prob_unet.climatology.climatology_io import DailyClimatologyFile, HourlyClimatologyFile


@functools.lru_cache(maxsize=2)
def compute_mrcc_grid(rlon_idx_min=None, rlon_idx_max=None, rlat_idx_min=None, rlat_idx_max=None):
    rlon = np.arange(2.695007, 33.385010 + 0.055, 0.11)
    rlat = np.arange(-12.615000, 18.075000 + 0.055, 0.11)
    if rlon_idx_min or rlon_idx_max is not None:
        if (rlon_idx_min is not None) and (not 0 <= rlon_idx_min < len(rlon)):
            raise ValueError("rlon_idx_min is out of bounds.")
        if (rlon_idx_max is not None) and (not 0 <= rlon_idx_max < len(rlon)):
            raise ValueError("rlon_idx_max is out of bounds.")
        rlon = rlon[slice(rlon_idx_min, rlon_idx_max)]
    if rlat_idx_min or rlat_idx_max is not None:
        if (rlat_idx_min is not None) and (not 0 <= rlat_idx_min < len(rlat)):
            raise ValueError("rlat_idx_min is out of bounds.")
        if (rlat_idx_max is not None) and (not 0 <= rlat_idx_max < len(rlat)):
            raise ValueError("rlat_idx_max is out of bounds.")
        rlat = rlat[slice(rlat_idx_min, rlat_idx_max)]

    rlon_mesh, rlat_mesh = np.meshgrid(rlon, rlat)
    rlon_flatten = rlon_mesh.flatten()
    rlat_flatten = rlat_mesh.flatten()

    pole_longitude = 83.0
    pole_latitude = 42.5

    rotated_pole = ccrs.RotatedPole(pole_longitude=pole_longitude, pole_latitude=pole_latitude)
    pc = ccrs.PlateCarree()
    rotated_coords = np.array([rlon_flatten, rlat_flatten]).T
    lonlat = pc.transform_points(rotated_pole, rotated_coords[:, 0], rotated_coords[:, 1])
    longitude = lonlat[:, 0].reshape(rlat.shape[0], rlon.shape[0])
    latitude = lonlat[:, 1].reshape(rlat.shape[0], rlon.shape[0])
    return rlon, rlat, longitude, latitude


def dummy_climatology_netcdf(
        variable_name='dummy', units='1', frequency='daily', calendar='gregorian', season=None, month=1, day=1, hour=0,
        rlon_idx_min=None, rlon_idx_max=None, rlat_idx_min=None, rlat_idx_max=None):
    rlon, rlat, longitude, latitude = compute_mrcc_grid(rlon_idx_min=rlon_idx_min, rlon_idx_max=rlon_idx_max,
                                                        rlat_idx_min=rlat_idx_min, rlat_idx_max=rlat_idx_max)
    if frequency == 'daily':
        time_units = f'days since 0000-{str(month).zfill(2)}-{str(day).zfill(2)} 00:00:00'
    elif frequency == 'hourly':
        time_units = f'hours since 0000-{str(month).zfill(2)}-{str(day).zfill(2)} {str(hour).zfill(2)}:00:00'
    else:
        raise NotImplementedError()
    time_var = xarray.Variable(dims=('time',),
                               data=np.array([0], dtype=np.float32),
                               attrs={'standard_name': 'time', 'long_name': 'time', 'calendar': calendar,
                                      'axis': 'T', 'time_units': time_units},
                                 encoding={'dtype': np.float32})
    rlon_var = xarray.Variable(dims=('rlon',),
                               data=rlon,
                               attrs={'units': 'degrees', 'standard_name': 'grid_longitude', 'axis': 'X'},
                               encoding={'dtype': np.float32})
    rlat_var = xarray.Variable(dims=('rlat',),
                               data=rlat,
                               attrs={'units': 'degrees', 'standard_name': 'grid_latitude', 'axis': 'Y'},
                               encoding={'dtype': np.float32})
    lon_var = xarray.Variable(dims=('rlat', 'rlon'),
                              data=longitude,
                              attrs={'units': 'degrees_east', 'standard_name': 'longitude'},
                              encoding={'dtype': np.float32})
    lat_var = xarray.Variable(dims=('rlat', 'rlon'),
                              data=latitude,
                              attrs={'units': 'degrees_north', 'standard_name': 'latitude'},
                              encoding={'dtype': np.float32})
    rotated_pole_var = xarray.Variable(dims=(),
                                       data=0,
                                       attrs={'grid_mapping_name': 'rotated_latitude_longitude',
                                              'earth_radius': 6370997.0,
                                              'grid_north_pole_latitude': 42.5,
                                              'grid_north_pole_longitude': 83.0,
                                              'north_pole_grid_longitude': 0})

    dims = ('rlat', 'rlon')
    data_shape = (len(rlat), len(rlon))
    data = np.random.random(data_shape)
    climatology_variable = xarray.Variable(dims=dims,
                                           data=data,
                                           attrs={'units': units,
                                                  'grid_mapping': 'rotated_pole'},
                                           encoding={'dtype': np.float32})
    data_vars = {variable_name: climatology_variable}
    cf_coordinates = {'time': time_var,
                      'rlon': rlon_var,
                      'rlat': rlat_var,
                      'lon': lon_var,
                      'lat': lat_var,
                      'rotated_pole': rotated_pole_var}
    cf_attrs = {'Conventions': 'CF-1.6'}
    ds = xarray.Dataset(data_vars=data_vars, coords=cf_coordinates, attrs=cf_attrs)
    return ds


def dummy_climatology_files_on_disk(
        tmp_dir, variable_names='dummy', units='1', frequency='daily', calendar='gregorian', season=None, month=1,
        day=1, hour=0, rlon_idx_min=None, rlon_idx_max=None, rlat_idx_min=None, rlat_idx_max=None, num_files=1):
    if isinstance(variable_names, str):
        variable_names = [variable_names]
    if isinstance(units, str):
        units = [units] * len(variable_names)
    output_files = []
    for variable_name, local_units in zip(variable_names, units):
        current_datetime = datetime.datetime(2000, month, day, hour)
        for _ in range(num_files):
            ds = dummy_climatology_netcdf(variable_name=variable_name, units=local_units, frequency=frequency,
                                          calendar=calendar, season=season, month=current_datetime.month,
                                          day=current_datetime.day, hour=current_datetime.hour,
                                          rlon_idx_min=rlon_idx_min, rlon_idx_max=rlon_idx_max,
                                          rlat_idx_min=rlat_idx_min, rlat_idx_max=rlat_idx_max)

            if frequency == 'daily':
                output_files.append(DailyClimatologyFile(
                    climatology_directory=tmp_dir, variable_name=variable_name, month=current_datetime.month,
                    day=current_datetime.day, prefix='test_'))
                time_delta_kwargs = {'days': 1}
            elif frequency == 'hourly':
                output_files.append(HourlyClimatologyFile(
                    climatology_directory=tmp_dir, variable_name=variable_name, month=current_datetime.month,
                    day=current_datetime.day, hour=current_datetime.hour,
                    prefix='test_'))
                time_delta_kwargs = {'hours': 1}
            else:
                raise NotImplementedError()
            Path(output_files[-1].path_climatology_file).parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(output_files[-1].path_climatology_file)
            current_datetime += datetime.timedelta(**time_delta_kwargs)
            if current_datetime.year > 2000:
                raise ValueError("num_files is too large for the given month/day/hour combination.")
    return output_files
