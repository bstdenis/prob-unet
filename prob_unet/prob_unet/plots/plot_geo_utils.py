import cartopy.crs as ccrs
import numpy as np


def get_projection_object(projection=None):
    if projection is None:
        return ccrs.PlateCarree()
    if isinstance(projection, str):
        if projection == 'PlateCarree':
            return ccrs.PlateCarree()
        else:
            raise NotImplementedError()
    return projection


def add_projection_to_ax(fig, ax, projection):
    rect = ax.get_position()
    ax.remove()
    return fig.add_axes(rect.bounds, projection=projection)


def lon_lat_ticks(lon, lat, n_divisions=5):
    d_lon = lon.ptp() / n_divisions
    lon_ticks = np.arange(lon.min() + d_lon / 2.0, lon.max(), d_lon)
    d_lat = lat.ptp() / n_divisions
    lat_ticks = np.arange(lat.min() + d_lat / 2.0, lat.max(), d_lat)
    return lon_ticks, lat_ticks
