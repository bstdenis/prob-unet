import itertools
import random
from math import prod

import numpy as np
import shapely
import xarray as xr

from shapely.geometry import Point, Polygon


class NDimensionalSample:
    def __init__(self, shape, threshold_for_sampling=1000000, threshold_for_replacement=10000000):
        self.shape = shape
        self.size = prod(shape)
        self.threshold_for_sampling = threshold_for_sampling
        self.threshold_for_replacement = threshold_for_replacement
        if self.size <= self.threshold_for_sampling:
            self.mode = 'complete'
        elif self.size <= self.threshold_for_replacement:
            self.mode = 'sample without replacement'
        else:
            self.mode = 'sample with replacement'

    def iterator(self):
        if self.size <= self.threshold_for_sampling:
            itertools_product = itertools.product(*[range(i) for i in self.shape])
            for i in itertools_product:
                yield i
        elif self.size <= self.threshold_for_replacement:
            candidates = list(itertools.product(*[range(i) for i in self.shape]))
            sample = random.sample(candidates, self.threshold_for_sampling)
            for i in sample:
                yield i
        else:
            for i in range(self.threshold_for_sampling):
                yield tuple([random.randint(0, j - 1) for j in self.shape])


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)

    One of lon1/lat1 or lon2/lat2 must be a scalar and the other a numpy array.
    """
    # convert decimal degrees to radians
    lon1_radians = np.radians(lon1)
    lat1_radians = np.radians(lat1)
    lon2_radians = np.radians(lon2)
    lat2_radians = np.radians(lat2)

    # haversine formula
    lon_difference = lon2_radians - lon1_radians
    lat_difference = lat2_radians - lat1_radians
    a = np.sin(lat_difference / 2) ** 2 + np.cos(lat1_radians) * np.cos(lat2_radians) * np.sin(lon_difference / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers.
    return c * r


def lon_lat_dims_decoding(dims=None, lon_dims=None, lat_dims=None):
    if dims is not None:
        if len(dims) == 1:
            lon_dims = dims
            lat_dims = dims
        elif len(dims) == 2:
            lon_dims = dims[0]
            lat_dims = dims[1]
        else:
            raise ValueError('dims must have one or two elements')
    return lon_dims, lat_dims


class LonLatExtent:
    def __init__(self, lon_min=None, lon_max=None, lat_min=None, lat_max=None, central_lon=None, central_lat=None,
                 lon_extent=None, lat_extent=None):
        if (lon_min is not None) and (lon_max is not None):
            self.lon_min = lon_min
            self.lon_max = lon_max
            self.central_lon = (lon_min + lon_max) / 2
            self.lon_extent = lon_max - lon_min
        elif (central_lon is not None) and (lon_extent is not None):
            self.lon_min = central_lon - lon_extent / 2
            self.lon_max = central_lon + lon_extent / 2
            self.central_lon = central_lon
            self.lon_extent = lon_extent
        else:
            raise ValueError('Either lon_min and lon_max or central_lon and lon_extent must be provided')
        if (lat_min is not None) and (lat_max is not None):
            self.lat_min = lat_min
            self.lat_max = lat_max
            self.central_lat = (lat_min + lat_max) / 2
            self.lat_extent = lat_max - lat_min
        elif (central_lat is not None) and (lat_extent is not None):
            self.lat_min = central_lat - lat_extent / 2
            self.lat_max = central_lat + lat_extent / 2
            self.central_lat = central_lat
            self.lat_extent = lat_extent
        else:
            raise ValueError('Either lat_min and lat_max or central_lat and lat_extent must be provided')


class PointsLonLat:
    def __init__(self, lon, lat, dims=('station',)):
        if len(lon.shape) != len(lat.shape):
            raise ValueError('lon and lat must have the same number of dimensions')
        self.lon = lon
        self.lat = lat
        self.lon_dims = dims
        self.lat_dims = dims
        self.num_dims = len(lon.shape)
        self.shape = lon.shape

    def bbox(self):
        return {'lon_min': self.lon.min(), 'lon_max': self.lon.max(),
                'lat_min': self.lat.min(), 'lat_max': self.lat.max()}

    def flatten_lon_lat(self):
        return self.lon, self.lat

    def subset(self, lon_lat_extent):
        lon_indices = np.where((self.lon >= lon_lat_extent.lon_min) & (self.lon <= lon_lat_extent.lon_max))[0]
        lat_indices = np.where((self.lat >= lon_lat_extent.lat_min) & (self.lat <= lon_lat_extent.lat_max))[0]
        lon = self.lon[lon_indices]
        lat = self.lat[lat_indices]
        return PointsLonLat(lon=lon, lat=lat, dims=self.lon_dims)

    def to_xarray(self, global_attributes=None, flatten=False):
        if global_attributes is None:
            global_attributes = {}
        cf_coordinates = {}
        if flatten:
            lon, lat = self.flatten_lon_lat()
            lon_dims = ('point',)
            lat_dims = ('point',)
            height_dims = ('point',)
            height_shape = [lon.shape[0]]
            point_var = xr.Variable(dims=('point',), data=np.arange(lon.shape[0]),
                                    attrs={'units': '1'}, encoding={'dtype': np.int8})
            cf_coordinates['point'] = point_var
        else:
            lon = self.lon
            lat = self.lat
            lon_dims = self.lon_dims
            lat_dims = self.lat_dims
            height_dims = ('lon', 'lat')
            height_shape = [self.lon.shape[0], self.lat.shape[-1]]
        lon_var = xr.Variable(dims=lon_dims, data=lon,
                              attrs={'units': 'degrees_east', 'standard_name': 'longitude', 'axis': 'X'},
                              encoding={'dtype': np.float32})
        lat_var = xr.Variable(dims=lat_dims, data=lat,
                              attrs={'units': 'degrees_north', 'standard_name': 'latitude', 'axis': 'Y'},
                              encoding={'dtype': np.float32})
        crs_data_array = xr.DataArray('', attrs={'grid_mapping_name': 'latitude_longitude',
                                                 'semi_major_axis': 6371000.0,
                                                 'inverse_flattening': 0.0})
        height_var = xr.Variable(dims=height_dims, data=np.zeros(height_shape, dtype=np.int8),
                                 attrs={'units': 'm', 'standard_name': 'height', 'grid_mapping': 'crs'},
                                 encoding={'dtype': np.int8})
        cf_coordinates['lon'] = lon_var
        cf_coordinates['lat'] = lat_var
        cf_attrs = {**{'Conventions': 'CF-1.11'}, **global_attributes}
        # 'zh' must be first in order to be selected in xarray scattermap
        ds = xr.Dataset(data_vars={'zh': height_var, 'crs': crs_data_array}, coords=cf_coordinates, attrs=cf_attrs)
        return ds

    def to_netcdf(self, path, global_attributes=None, flatten=False):
        self.to_xarray(global_attributes=global_attributes, flatten=flatten).to_netcdf(path, engine='h5netcdf')

    def boundary(self):
        # This is ambiguous. Convex Hull? Delaunay triangulation with area restriction?
        raise NotImplementedError()


class RegularGridLonLat(PointsLonLat):
    def __init__(self, lon, lat, lon_dims=('lon',), lat_dims=('lat',), shape_order=('lat', 'lon')):
        super().__init__(lon=lon, lat=lat, dims=None)
        self.lon_dims = lon_dims
        self.lat_dims = lat_dims
        if shape_order == ('lat', 'lon'):
            self.shape = (len(lat), len(lon))
        elif shape_order == ('lon', 'lat'):
            self.shape = (len(lon), len(lat))
        else:
            raise ValueError('shape_order must be either ("lat", "lon") or ("lon", "lat")')

    def flatten_lon_lat(self):
        mesh_lon, mesh_lat = np.meshgrid(self.lon, self.lat)
        return mesh_lon.flatten(), mesh_lat.flatten()

    def subset(self, lon_lat_extent):
        lon_indices = np.where((self.lon >= lon_lat_extent.lon_min) & (self.lon <= lon_lat_extent.lon_max))[0]
        lat_indices = np.where((self.lat >= lon_lat_extent.lat_min) & (self.lat <= lon_lat_extent.lat_max))[0]
        lon = self.lon[lon_indices]
        lat = self.lat[lat_indices]
        return RegularGridLonLat(lon=lon, lat=lat, lon_dims=self.lon_dims, lat_dims=self.lat_dims)

    def boundary(self):
        raise NotImplementedError()


class RectilinearGridLonLat(RegularGridLonLat):
    def resolution(self):
        lon_diff = np.absolute(np.diff(self.lon))
        lat_diff = np.absolute(np.diff(self.lat))
        nd_sample = NDimensionalSample((len(self.lon) - 1, len(self.lat) - 1),
                                       threshold_for_sampling=100000, threshold_for_replacement=1000000)
        neighbors_deg_distances = []
        neighbors_km_distances = []
        for (i, j) in nd_sample.iterator():
            neighbors_deg_distances.append(
                np.sqrt((self.lon[i] - self.lon[i + 1])**2 + (self.lat[j] - self.lat[j])**2))
            neighbors_deg_distances.append(
                np.sqrt((self.lon[i] - self.lon[i]) ** 2 + (self.lat[j] - self.lat[j + 1]) ** 2))
            neighbors_km_distances.append(
                haversine(self.lon[i], self.lat[j], self.lon[i + 1], self.lat[j]))
            neighbors_km_distances.append(
                haversine(self.lon[i], self.lat[j], self.lon[i], self.lat[j + 1]))
        return {'lon_avg_diff': np.mean(lon_diff),
                'lon_min_diff': np.min(lon_diff),
                'lon_max_diff': np.max(lon_diff),
                'lat_avg_diff': np.mean(lat_diff),
                'lat_min_diff': np.min(lat_diff),
                'lat_max_diff': np.max(lat_diff),
                'avg_deg_distance': np.mean(neighbors_deg_distances),
                'avg_km_distance': np.mean(neighbors_km_distances),
                'avg_distance_mode': nd_sample.mode}

    def sample(self, threshold_for_sampling=100000, threshold_for_replacement=1000000):
        nd_sample = NDimensionalSample((len(self.lon) - 1, len(self.lat) - 1),
                                       threshold_for_sampling=threshold_for_sampling,
                                       threshold_for_replacement=threshold_for_replacement)
        lon_sample = []
        lat_sample = []
        for (i, j) in nd_sample.iterator():
            lon_sample.append(self.lon[i])
            lat_sample.append(self.lat[j])
        return PointsLonLat(lon=np.array(lon_sample), lat=np.array(lat_sample), dims=('sample_point',))

    def subset(self, lon_lat_extent):
        lon_indices = np.where((self.lon >= lon_lat_extent.lon_min) & (self.lon <= lon_lat_extent.lon_max))[0]
        lat_indices = np.where((self.lat >= lon_lat_extent.lat_min) & (self.lat <= lon_lat_extent.lat_max))[0]
        lon = self.lon[lon_indices]
        lat = self.lat[lat_indices]
        return RectilinearGridLonLat(lon=lon, lat=lat, lon_dims=self.lon_dims, lat_dims=self.lat_dims)


class StructuredGridLonLat(PointsLonLat):
    def __init__(self, lon, lat, lon_dims=('yc', 'xc'), lat_dims=('yc', 'xc')):
        super().__init__(lon=lon, lat=lat, dims=None)
        self.lon_dims = lon_dims
        self.lat_dims = lat_dims
        self.shape = lon.shape

    def flatten_lon_lat(self):
        return self.lon.flatten(), self.lat.flatten()

    def resolution(self):
        nd_sample = NDimensionalSample((self.lon.shape[0] - 1, self.lon.shape[1] - 1),
                                       threshold_for_sampling=100000, threshold_for_replacement=1000000)
        lon_diffs = []
        lat_diffs = []
        neighbors_deg_distances = []
        neighbors_km_distances = []
        for (i, j) in nd_sample.iterator():
            lon_diffs.append(max(abs(self.lon[i, j + 1] - self.lon[i, j]),
                                 abs(self.lon[i + 1, j] - self.lon[i, j])))
            lat_diffs.append(max(abs(self.lat[i, j + 1] - self.lat[i, j]),
                                 abs(self.lat[i + 1, j] - self.lat[i, j])))
            neighbors_deg_distances.append(
                np.sqrt(
                    (self.lon[i, j] - self.lon[i + 1, j]) ** 2 + (self.lat[i, j] - self.lat[i + 1, j]) ** 2))
            neighbors_deg_distances.append(
                np.sqrt(
                    (self.lon[i, j] - self.lon[i, j + 1]) ** 2 + (self.lat[i, j] - self.lat[i, j + 1]) ** 2))
            neighbors_km_distances.append(
                haversine(self.lon[i, j], self.lat[i, j], self.lon[i + 1, j], self.lat[i + 1, j]))
            neighbors_km_distances.append(
                haversine(self.lon[i, j], self.lat[i, j], self.lon[i, j + 1], self.lat[i, j + 1]))
        return {'lon_avg_diff': np.mean(lon_diffs),
                'lon_min_diff': np.min(lon_diffs),
                'lon_max_diff': np.max(lon_diffs),
                'lat_avg_diff': np.mean(lat_diffs),
                'lat_min_diff': np.min(lat_diffs),
                'lat_max_diff': np.max(lat_diffs),
                'avg_deg_distance': np.mean(neighbors_deg_distances),
                'avg_km_distance': np.mean(neighbors_km_distances),
                'avg_distance_mode': nd_sample.mode}

    def sample(self, threshold_for_sampling=100000, threshold_for_replacement=1000000):
        nd_sample = NDimensionalSample((self.lon.shape[0] - 1, self.lon.shape[1] - 1),
                                       threshold_for_sampling=threshold_for_sampling,
                                       threshold_for_replacement=threshold_for_replacement)
        lon_sample = []
        lat_sample = []
        for (i, j) in nd_sample.iterator():
            lon_sample.append(self.lon[i, j])
            lat_sample.append(self.lat[i, j])
        return PointsLonLat(lon=np.array(lon_sample), lat=np.array(lat_sample), dims=('sample_point',))

    def nearest_idx(self, lon, lat, max_distance_km=None):
        distances = haversine(lon, lat, self.lon, self.lat)
        min_idx = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
        if distances[min_idx] > max_distance_km:
            raise ValueError('No grid point found within the specified maximum distance')
        return min_idx

    # ToDo: allowing single nearest point should be an explicit option?
    def subset_idx(self, lon_lat_extent):
        lon_lat_indices = np.where(
            (self.lon >= lon_lat_extent.lon_min) & (self.lon <= lon_lat_extent.lon_max) &
            (self.lat >= lon_lat_extent.lat_min) & (self.lat <= lon_lat_extent.lat_max))
        if lon_lat_indices[0].size == 0:
            d0_min_idx, d1_min_idx = self.nearest_idx(lon_lat_extent.central_lon, lon_lat_extent.central_lat,
                                                     max_distance_km=self.resolution()['avg_km_distance'])
            d0_max_idx = d0_min_idx
            d1_max_idx = d1_min_idx
        else:
            d0_min_idx = lon_lat_indices[0].min()
            d0_max_idx = lon_lat_indices[0].max()
            d1_min_idx = lon_lat_indices[1].min()
            d1_max_idx = lon_lat_indices[1].max()
        return d0_min_idx, d0_max_idx, d1_min_idx, d1_max_idx

    def subset(self, lon_lat_extent):
        d0_min_idx, d0_max_idx, d1_min_idx, d1_max_idx = self.subset_idx(lon_lat_extent)
        d0_max_idx += 1  # Include the last index
        d1_max_idx += 1  # Include the last index
        lon = self.lon[d0_min_idx:d0_max_idx, d1_min_idx:d1_max_idx]
        lat = self.lat[d0_min_idx:d0_max_idx, d1_min_idx:d1_max_idx]
        return StructuredGridLonLat(lon=lon, lat=lat, lon_dims=self.lon_dims, lat_dims=self.lat_dims)

    def boundary(self):
        polygon_coords = []
        for i in range(self.lon.shape[0] - 1):
            polygon_coords.append((self.lon[i, 0], self.lat[i, 0]))
        for j in range(self.lon.shape[1] - 1):
            polygon_coords.append((self.lon[-1, j], self.lat[-1, j]))
        for i in range(self.lon.shape[0] - 1, -1, -1):
            polygon_coords.append((self.lon[i, -1], self.lat[i, -1]))
        for j in range(self.lon.shape[1] - 2, 0, -1):
            polygon_coords.append((self.lon[0, j], self.lat[0, j]))
        return Polygon(polygon_coords)


def rectilinear_grid_lon_lat_from_shape_center_resolution(lon_size, lat_size, center_lon, center_lat, deg_resolution):
    half_lon = (lon_size - 1) * deg_resolution / 2
    half_lat = (lat_size - 1) * deg_resolution / 2
    lon1d = np.linspace(center_lon - half_lon, center_lon + half_lon, lon_size)
    lat1d = np.linspace(center_lat - half_lat, center_lat + half_lat, lat_size)
    return RectilinearGridLonLat(lon=lon1d, lat=lat1d)


def xarray_to_grid(grid_xarray, threshold_for_sampling=None, threshold_for_replacement=None,
                   lon_lat_extent=None):
    if len(grid_xarray['lon'].shape) == 2:
        grid = StructuredGridLonLat(lon=grid_xarray['lon'].values, lat=grid_xarray['lat'].values,
                                    lon_dims=grid_xarray['lon'].dims, lat_dims=grid_xarray['lat'].dims)
    elif grid_xarray['lon'].dims != grid_xarray['lat'].dims:
        grid = RectilinearGridLonLat(lon=grid_xarray['lon'].values, lat=grid_xarray['lat'].values,
                                     lon_dims=grid_xarray['lon'].dims, lat_dims=grid_xarray['lat'].dims)
    else:
        grid = PointsLonLat(lon=grid_xarray['lon'].values, lat=grid_xarray['lat'].values, dims=grid_xarray['lon'].dims)
    if lon_lat_extent is not None:
        grid = grid.subset(lon_lat_extent)
    if (threshold_for_sampling is not None) and (threshold_for_replacement is not None):
        return grid.sample(threshold_for_sampling=threshold_for_sampling,
                           threshold_for_replacement=threshold_for_replacement)
    else:
        return grid


def xarray_find_zoom_indices(ds, lon_min, lon_max, lat_min, lat_max):
    if len(ds['lon'].shape) == 2:
        indices = np.where((ds['lon'] >= lon_min) & (ds['lon'] <= lon_max) &
                           (ds['lat'] >= lat_min) & (ds['lat'] <= lat_max))
        return min(indices[0]), max(indices[0]), min(indices[1]), max(indices[1])
    elif ds['lon'].dims != ds['lat'].dims:
        raise NotImplementedError()
    else:
        raise NotImplementedError()


def xarray_spatial_subset_using_bounding_box_isel_kwargs(ds, lon_min, lon_max, lat_min, lat_max):
    # ToDo: manipulating longitude range should be done elsewhere, and with a more general method
    if ds['lon'].min() >= 0:
        if lon_min < 0:
            lon_min += 360
        if lon_max < 0:
            lon_max += 360
    if len(ds['lon'].shape) == 2:
        indices = np.where((ds['lon'] >= lon_min) & (ds['lon'] <= lon_max) &
                           (ds['lat'] >= lat_min) & (ds['lat'] <= lat_max))
        if 'rlat' not in ds.dims or 'rlon' not in ds.dims:
            # Currently a hack because everything is specific to rlon rlat grid
            raise NotImplementedError()
        return {'rlat': slice(min(indices[0]), max(indices[0]) + 1),
                'rlon': slice(min(indices[1]), max(indices[1]) + 1)}
    elif ds['lon'].dims != ds['lat'].dims:
        raise NotImplementedError()
    else:
        raise NotImplementedError()


def regridding_mask(grid_input, grid_regrid):
    boundary_polygon = grid_input.boundary()

    points_regrid = np.column_stack(grid_regrid.flatten_lon_lat())

    mask = shapely.covers(boundary_polygon, [Point(x, y) for x, y in points_regrid])

    # Reshape the mask to the original shape of lon1 and lat1
    mask = ~mask.reshape(grid_regrid.shape)

    return mask
