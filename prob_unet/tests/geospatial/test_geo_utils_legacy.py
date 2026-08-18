import tempfile

import numpy as np
import xarray as xr

import prob_unet.geospatial.geo_utils as geo_utils


def test_rectilinear_grid_lon_lat_resolution():
    grid = geo_utils.RectilinearGridLonLat(lon=np.arange(-160, -30, 1), lat=np.arange(20, 80, 1))
    grid_resolution = grid.resolution()
    assert grid_resolution['lon_avg_diff'] == 1.0
    assert grid_resolution['lat_avg_diff'] == 1.0
    assert grid_resolution['avg_deg_distance'] == 1.0
    assert 90 < grid_resolution['avg_km_distance'] < 91
    assert grid_resolution['avg_distance_mode'] == 'complete'

    grid = geo_utils.RectilinearGridLonLat(lon=np.arange(-160, -30, 0.1), lat=np.arange(20, 80, 0.1))
    grid_resolution = grid.resolution()
    assert 0.09 <grid_resolution['lon_avg_diff'] < 1.01
    assert 0.09 <grid_resolution['lat_avg_diff'] < 1.01
    assert 0.09 < grid_resolution['avg_deg_distance'] < 1.01
    assert 8.9 < grid_resolution['avg_km_distance'] < 9.1
    assert grid_resolution['avg_distance_mode'] == 'sample without replacement'

    grid = geo_utils.RectilinearGridLonLat(lon=np.arange(-160, -30, 0.01), lat=np.arange(20, 80, 0.01))
    grid_resolution = grid.resolution()
    assert 0.009 < grid_resolution['lon_avg_diff'] < 1.001
    assert 0.009 < grid_resolution['lat_avg_diff'] < 1.001
    assert 0.009 < grid_resolution['avg_deg_distance'] < 1.001
    assert 0.89 < grid_resolution['avg_km_distance'] < 0.91
    assert grid_resolution['avg_distance_mode'] == 'sample with replacement'


def test_structured_grid_lon_lat_resolution():
    (lon2d, lat2d) = np.meshgrid(np.arange(-160, -30, 1), np.arange(20, 80, 1))
    grid = geo_utils.StructuredGridLonLat(lon=lon2d, lat=lat2d)
    grid_resolution = grid.resolution()
    assert grid_resolution['lon_avg_diff'] == 1.0
    assert grid_resolution['lat_avg_diff'] == 1.0
    assert grid_resolution['avg_deg_distance'] == 1.0
    assert 90 < grid_resolution['avg_km_distance'] < 91
    assert grid_resolution['avg_distance_mode'] == 'complete'

    (lon2d, lat2d) = np.meshgrid(np.arange(-160, -30, 0.1), np.arange(20, 80, 0.1))
    grid = geo_utils.StructuredGridLonLat(lon=lon2d, lat=lat2d)
    grid_resolution = grid.resolution()
    assert 0.09 < grid_resolution['lon_avg_diff'] < 1.01
    assert 0.09 < grid_resolution['lat_avg_diff'] < 1.01
    assert 0.09 < grid_resolution['avg_deg_distance'] < 1.01
    assert 8.9 < grid_resolution['avg_km_distance'] < 9.1
    assert grid_resolution['avg_distance_mode'] == 'sample without replacement'

    (lon2d, lat2d) = np.meshgrid(np.arange(-160, -30, 0.01), np.arange(20, 80, 0.01))
    grid = geo_utils.StructuredGridLonLat(lon=lon2d, lat=lat2d)
    grid_resolution = grid.resolution()
    assert 0.009 < grid_resolution['lon_avg_diff'] < 1.001
    assert 0.009 < grid_resolution['lat_avg_diff'] < 1.001
    assert 0.009 < grid_resolution['avg_deg_distance'] < 1.001
    assert 0.89 < grid_resolution['avg_km_distance'] < 0.91
    assert grid_resolution['avg_distance_mode'] == 'sample with replacement'


def test_rectilinear_grid_lon_lat_from_shape_center_resolution():
    grid = geo_utils.rectilinear_grid_lon_lat_from_shape_center_resolution(10, 10, 0, 0, 1)
    assert len(grid.lat) == 10
    assert len(grid.lon) == 10

    # HRDPS ML grid
    grid = geo_utils.rectilinear_grid_lon_lat_from_shape_center_resolution(
        lon_size=2048, lat_size=1024, center_lon=-96.73, center_lat=48.95, deg_resolution=0.06)
    assert len(grid.lat) == 1024
    assert len(grid.lon) == 2048


def test_grid_to_netcdf():
    with tempfile.NamedTemporaryFile(suffix='.nc') as tmp_nc:
        grid = geo_utils.rectilinear_grid_lon_lat_from_shape_center_resolution(
            lon_size=2048, lat_size=1024, center_lon=-96.73, center_lat=48.95, deg_resolution=0.06)
        grid.to_netcdf(tmp_nc.name, global_attributes={'title': '2km North America ML grid'})
        grid2 = xr.open_dataset(tmp_nc.name)
        assert np.allclose(grid.lon, grid2.lon)
        assert np.allclose(grid.lat, grid2.lat)


def test_xarray_to_grid():
    grid = geo_utils.rectilinear_grid_lon_lat_from_shape_center_resolution(
        lon_size=2048, lat_size=1024, center_lon=-96.73, center_lat=48.95, deg_resolution=0.06)
    grid_xarray = grid.to_xarray(global_attributes={'title': '2km North America ML grid'})
    grid2 = geo_utils.xarray_to_grid(grid_xarray)
    assert grid.lon_dims == grid2.lon_dims
    assert grid.lat_dims == grid2.lat_dims
    assert np.allclose(grid.lon, grid2.lon)
    assert np.allclose(grid.lat, grid2.lat)


def test_structured_grid_boundary():
    lon = np.array([[1, 3, 5, 7], [1, 3, 5, 8], [1, 3, 5, 9], [2, 4, 6, 8]])
    lat = np.array([[1, 2, 3, 2], [3, 4, 5, 4], [5, 6, 7, 5], [8, 9, 9, 8]])
    grid = geo_utils.StructuredGridLonLat(lon, lat)
    boundary_polygon = grid.boundary()
    assert len(boundary_polygon.exterior.coords) == 13


def test_structured_grid_mask():
    lon = np.array([[1, 3, 5, 7], [1, 3, 5, 8], [1, 3, 5, 9], [2, 4, 6, 8]])
    lat = np.array([[1, 2, 3, 2], [3, 4, 5, 4], [5, 6, 7, 5], [8, 9, 9, 8]])
    grid_input = geo_utils.StructuredGridLonLat(lon, lat)
    grid_output = geo_utils.RegularGridLonLat(lon=np.arange(0, 10, 1), lat=np.arange(0, 10, 1))
    mask = geo_utils.regridding_mask(grid_input, grid_output)
    assert mask.sum() == 47
