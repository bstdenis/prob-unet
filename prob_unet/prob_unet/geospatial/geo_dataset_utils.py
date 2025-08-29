import cartopy.crs as ccrs
import random
import xarray

import numpy as np

from prob_unet.geospatial import geo_utils


def get_dataset_projection(nc_file=None, ds=None):
    if ds is None:
        ds = xarray.open_dataset(nc_file)
    # ToDo: this is only one possible case. Also, the grid may be variable specific with each variable having a
    #     grid_mapping attribute pointing to a variable with the grid information.
    if ('rotated_pole' in ds.data_vars) or ('rotated_pole' in ds.coords):
        rotated_pole = ds['rotated_pole']
        if 'longitude_of_prime_meridian' in rotated_pole.attrs:
            central_rotated_longitude = rotated_pole.longitude_of_prime_meridian.item()
        else:
            central_rotated_longitude = 0.0
        return ccrs.RotatedPole(pole_longitude=rotated_pole.grid_north_pole_longitude.item(),
                                pole_latitude=rotated_pole.grid_north_pole_latitude.item(),
                                central_rotated_longitude=central_rotated_longitude)
    return ccrs.PlateCarree()


def get_random_variable_name(ds):
    data_vars = list(ds.data_vars)
    valid_data_vars = []
    for data_var in data_vars:
        # ToDo: there are many more exceptions to be considered, also inspecting variables dimensions could help
        if data_var == 'rotated_pole':
            continue
        valid_data_vars.append(data_var)
    return random.choice(valid_data_vars)


def get_random_time_index(ds):
    return random.randint(0, len(ds['time']) - 1)


def spatial_subset_using_bounding_box(ds, lon_min, lon_max, lat_min, lat_max):
    isel_kwargs = geo_utils.xarray_spatial_subset_using_bounding_box_isel_kwargs(
        ds, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max)
    return ds.isel(**isel_kwargs)


def spatial_subset(ds, lon_min=None, lon_max=None, lat_min=None, lat_max=None):
    if (lon_min is not None) and (lon_max is not None) and (lat_min is not None) and (lat_max is not None):
        return spatial_subset_using_bounding_box(ds, lon_min, lon_max, lat_min, lat_max)
    else:
        raise NotImplementedError()


def open_dataset_for_regrid(filename_or_obj, cat_id=None, cat_domain=None, allow_cat_overwrite=False, **kwargs):
    # Fix necessary attributes for xscen.regrid to work properly
    # Now supporting multiple input files?
    if isinstance(filename_or_obj, (list, tuple)):
        ds = xarray.open_mfdataset(filename_or_obj, decode_timedelta=False, **kwargs)
    else:
        ds = xarray.open_dataset(filename_or_obj, decode_timedelta=False, **kwargs)
    if cat_id is not None:
        if (not allow_cat_overwrite) and ('cat:id' in ds.attrs) and (ds.attrs['cat:id'] != cat_id):
            raise ValueError(f"cat:id already exists in the dataset attributes: {ds.attrs['cat:id']}")
        ds.attrs['cat:id'] = cat_id
    if cat_domain is not None:
        if (not allow_cat_overwrite) and ('cat:domain' in ds.attrs) and (ds.attrs['cat:domain'] != cat_domain):
            raise ValueError(f"cat:domain already exists in the dataset attributes: {ds.attrs['cat:domain']}")
        ds.attrs['cat:domain'] = cat_domain
    if 'rlat' in ds:
        if 'axis' not in ds['rlat'].attrs:
            raise NotImplementedError("rlat variable does not have 'axis' attribute")
    elif 'lat' in ds:
        if 'axis' not in ds['lat'].attrs:
            ds['lat'].attrs['axis'] = 'Y'
    if 'rlon' in ds:
        if 'axis' not in ds['rlon'].attrs:
            raise NotImplementedError("rlon variable does not have 'axis' attribute")
    elif 'lon' in ds:
        if 'axis' not in ds['lon'].attrs:
            ds['lon'].attrs['axis'] = 'X'
    return ds


class GeoDatasetAxData:
    def __init__(self, lon=None, lat=None, gridmap_data=None, nc_file=None, zoom=None, var_name=None, time_index=None):
        self.lon = lon
        self.lat = lat
        self.gridmap_data = gridmap_data
        self.data_projection = None
        self.zoom = zoom
        self.var_name = var_name
        self.time_index = time_index
        if nc_file is not None:
            self.ax_data_from_netcdf(nc_file)

    def ax_data_from_netcdf(self, nc_file):
        ds = xarray.open_dataset(nc_file)
        if self.zoom is not None:
            ds = spatial_subset(ds, **self.zoom)
            self.lon = np.array([self.zoom['lon_min'], self.zoom['lon_max']])
            self.lat = np.array([self.zoom['lat_min'], self.zoom['lat_max']])
        else:
            self.lon = ds['lon'].data
            self.lat = ds['lat'].data
        if self.var_name is None:
            self.var_name = get_random_variable_name(ds)
        if self.time_index is None:
            self.time_index = get_random_time_index(ds)
        self.gridmap_data = ds[self.var_name].isel(time=self.time_index)
        self.data_projection = get_dataset_projection(ds=ds)


class GeoDatasetFigData:
    def __init__(self, ax_data=None, nc_files=None, zooms=None, var_names=None, default_time_indices=None,
                 time_indices=None):
        if (var_names is not None) and (time_indices is not None):
            raise NotImplementedError()
        if ax_data is not None:
            # nb of rows and columns and keys could be inferred from ax_data
            raise NotImplementedError()
        self.ax_data = {} if ax_data is None else ax_data
        self.nb_of_rows = 1 if zooms is None else len(zooms)
        self.row_keys = [None] if zooms is None else list(zooms.keys())
        self.zooms = {} if zooms is None else zooms
        self.var_names = var_names
        self.default_time_indices = default_time_indices
        self.time_indices = time_indices
        if var_names is not None:
            self.nb_of_columns = len(var_names)
            self.column_keys = var_names
        elif time_indices is not None:
            self.nb_of_columns = len(time_indices)
            self.column_keys = time_indices
        else:
            self.nb_of_columns = 1
            self.column_keys = [None]
        if nc_files is not None:
            self.fig_data_from_netcdf(nc_files)

    def fig_data_from_netcdf(self, nc_files):
        for row_key in self.row_keys:
            for i, column_key in enumerate(self.column_keys):
                grid_key = (row_key, column_key)
                if isinstance(nc_files, dict):
                    nc_file = nc_files[grid_key]
                else:
                    nc_file = nc_files
                if self.var_names is None:
                    var_name = None
                else:
                    var_name = column_key
                if self.time_indices is None:
                    time_index = self.default_time_indices[i]
                else:
                    time_index = column_key
                self.ax_data[(row_key, column_key)] = GeoDatasetAxData(
                    nc_file=nc_file, zoom=self.zooms[row_key], var_name=var_name, time_index=time_index)


# ToDo: if this is useful, move to geodataset_plot_data.py
class MultiGeoDatasetPlotData:
    def __init__(self, lon=None, lat=None, data=None, gridmap_data=None, nc_files=None, var_names=None,
                 time_indices=None):
        self.lon_min = None
        self.lon_max = None
        self.lat_min = None
        self.lat_max = None
        self.vmin = None
        self.vmax = None
        self.gridmap_data = {} if gridmap_data is None else gridmap_data
        self.data_projections = {}
        self.var_names = var_names
        self.time_indices = time_indices
        if nc_files is not None:
            self.plot_data_from_netcdf(nc_files)

    def plot_data_from_netcdf(self, nc_files):
        for key, nc_file in nc_files.items():
            ds = xarray.open_dataset(nc_file)
            lon_min = ds['lon'].data.min()
            if (self.lon_min is None) or (lon_min < self.lon_min):
                self.lon_min = lon_min
            lon_max = ds['lon'].data.max()
            if (self.lon_max is None) or (lon_max > self.lon_max):
                self.lon_max = lon_max
            lat_min = ds['lat'].data.min()
            if (self.lat_min is None) or (lat_min < self.lat_min):
                self.lat_min = lat_min
            lat_max = ds['lat'].data.max()
            if (self.lat_max is None) or (lat_max > self.lat_max):
                self.lat_max = lat_max
            # ToDo: this is a placeholder, the actual data should be selected based on the variable name and time index
            if self.var_names.get(key, None) is None:
                self.var_names[key] = get_random_variable_name(ds)
            # if self.time_index is None:
            #     self.time_index = get_random_time_index(ds)
            self.gridmap_data[key] = ds[self.var_names[key]].isel(time=self.time_indices[key])
            v_min = self.gridmap_data[key].min().item()
            if (self.vmin is None) or (v_min < self.vmin):
                self.vmin = v_min
            v_max = self.gridmap_data[key].max().item()
            if (self.vmax is None) or (v_max > self.vmax):
                self.vmax = v_max
            self.data_projections[key] = get_dataset_projection(ds=ds)


def xarray_regridding_mask(ds_input, ds_regrid):
    grid_input = geo_utils.xarray_to_grid(ds_input)
    grid_regrid = geo_utils.xarray_to_grid(ds_regrid)
    return geo_utils.regridding_mask(grid_input, grid_regrid)


def xarray_extract_slices_lon_lat(ds, lon_lat_extent=None, preserve_dimension=False):
    slices_dict = {}
    if ('lon' in ds) and ('lat' in ds):
        if (ds['lon'].dims == ('rlat', 'rlon')) and (ds['lat'].dims == ('rlat', 'rlon')):
            grid = geo_utils.StructuredGridLonLat(lon=ds['lon'].values, lat=ds['lat'].values,
                                                  lon_dims=('rlat', 'rlon'), lat_dims=('rlat', 'rlon'))
            d0_min_idx, d0_max_idx, d1_min_idx, d1_max_idx = grid.subset_idx(lon_lat_extent)
            if (not preserve_dimension) and (d0_min_idx == d0_max_idx):
                slices_dict['rlat'] = d0_min_idx
            else:
                slices_dict['rlat'] = slice(d0_min_idx, d0_max_idx + 1)
            if (not preserve_dimension) and (d1_min_idx == d1_max_idx):
                slices_dict['rlon'] = d1_min_idx
            else:
                slices_dict['rlon'] = slice(d1_min_idx, d1_max_idx + 1)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    return slices_dict


def xarray_extract_slices(ds, dims=None, lon_lat_extent=None, preserve_dimension=False, explicit_slices=None):
    # dims can be provided if they are specific to a variable to skip unnecessary dimensions
    dims = dims or ds.dims
    if preserve_dimension is True:
        preserve_dimension = list(ds.dims)
    elif preserve_dimension is False:
        preserve_dimension = []
    explicit_slices = explicit_slices or {}
    slices_dict = {}
    for key, value in dims.items():
        if value == 1:
            if key in preserve_dimension:
                slices_dict[key] = slice(0, 1)
            else:
                slices_dict[key] = 0
        elif key == 'time':
            raise NotImplementedError()
        elif key in ['lon', 'longitude', 'rlon', 'lat', 'latitude', 'rlat']:
            slices_dict.update(xarray_extract_slices_lon_lat(
                ds, lon_lat_extent=lon_lat_extent, preserve_dimension=preserve_dimension))
        elif key in explicit_slices:
            slices_dict[key] = explicit_slices[key]
        else:
            raise NotImplementedError()
    return slices_dict


def xarray_extract(ds, variable_name, lon_lat_extent=None, explicit_slices=None):
    dims = {k: v for k, v in ds.dims.items() if k in ds[variable_name].dims}
    slices_dict = xarray_extract_slices(ds, dims=dims, lon_lat_extent=lon_lat_extent, explicit_slices=explicit_slices)
    slices = tuple([slices_dict[dim] for dim in ds[variable_name].dims])
    return ds[variable_name][slices].values
