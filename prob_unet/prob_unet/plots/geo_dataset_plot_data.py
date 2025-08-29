import xarray

from prob_unet.geospatial import geo_dataset_utils


class GeoDatasetPlotData:
    def __init__(self, var_name=None, vertical_dimension_name=None, vertical_index=None, nc_file=None):
        self.lon = None
        self.lat = None
        self.var_name = var_name
        self.time_index = None
        self.vertical_dimension_name = vertical_dimension_name
        self.vertical_index = vertical_index
        self.gridmap_data = None
        self.data_projection = None
        if nc_file is not None:
            self.plot_data_from_netcdf(nc_file)

    def plot_data_from_netcdf(self, nc_file):
        ds = xarray.open_dataset(nc_file)
        self.lon = ds['lon'].data
        self.lat = ds['lat'].data
        if self.var_name is None:
            self.var_name = geo_dataset_utils.get_random_variable_name(ds)
        isel_kwargs = {}
        if ('time' in ds) and (self.time_index is None):
            self.time_index = geo_dataset_utils.get_random_time_index(ds)
            if 'time' in ds[self.var_name].dims:
                isel_kwargs['time'] = self.time_index
        if self.vertical_index is not None:
            if self.vertical_dimension_name in ds[self.var_name].dims:
                isel_kwargs[self.vertical_dimension_name] = self.vertical_index
        elif 'pres' in ds[self.var_name].dims:
            isel_kwargs['pres'] = 0
        if isel_kwargs:
            self.gridmap_data = ds[self.var_name].isel(**isel_kwargs)
        else:
            self.gridmap_data = ds[self.var_name]
        self.data_projection = geo_dataset_utils.get_dataset_projection(ds=ds)
