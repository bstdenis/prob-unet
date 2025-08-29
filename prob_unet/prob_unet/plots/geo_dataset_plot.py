import matplotlib
matplotlib.use('Agg')
import numpy as np

from prob_unet.plots.geo_dataset_plot_data import GeoDatasetPlotData
from prob_unet.plots.plot_geo_utils import add_projection_to_ax, get_projection_object
from prob_unet.plots.plot_utils import CustomFigure

try:
    import figanos.matplotlib as fg
except (ImportError, ModuleNotFoundError):
    fg = None

default_features = {"rivers": {"edgecolor": "black"},
                    "lakes": {"edgecolor": "black", "facecolor": "none"},
                    "coastline": {"edgecolor": "black"}, }


def geo_dataset_ax_plot(ax, geo_dataset_ax_data, ax_projection=None, original_data_projection=None,
                        ax_lon_min=None, ax_lon_max=None, ax_lat_min=None, ax_lat_max=None, vmin=None, vmax=None,
                        cbar_padding=0.05, title=None):
    ax.yaxis.tick_right()
    if ax_lon_min is None:
        ax_lon_min = geo_dataset_ax_data.lon.min()
    if ax_lon_max is None:
        ax_lon_max = geo_dataset_ax_data.lon.max()
    if ax_lat_min is None:
        ax_lat_min = geo_dataset_ax_data.lat.min()
    if ax_lat_max is None:
        ax_lat_max = geo_dataset_ax_data.lat.max()
    d_lat = (ax_lat_max - ax_lat_min) / 5
    lat_ticks = np.arange(ax_lat_min + (d_lat / 2.0), ax_lat_max, d_lat)
    d_lon = (ax_lon_max - ax_lon_min) / 5
    lon_ticks = np.arange(ax_lon_min + (d_lon / 2.0), ax_lon_max, d_lon)
    ax.set_xticks(lon_ticks, crs=ax_projection)
    ax.set_yticks(lat_ticks, crs=ax_projection)
    ax.set_xlim(ax_lon_min, ax_lon_max)
    ax.set_ylim(ax_lat_min, ax_lat_max)
    ax.set_aspect('auto')
    fg.gridmap(ax=ax, data=geo_dataset_ax_data.gridmap_data, projection=ax_projection,
               transform=original_data_projection, features=default_features, frame=True, show_time=False,
               plot_kw={'vmin': vmin, 'vmax': vmax, 'cbar_kwargs': {'fraction': 0.046, 'pad': cbar_padding}})
    if title is None:
        ax.set_title(f'{geo_dataset_ax_data.var_name} at time index {geo_dataset_ax_data.time_index}')
    else:
        ax.set_title(title)


def geo_dataset_fig_plot(fig, geo_dataset_plot_data, ax_projection=None, title=None):
    ax_projection = get_projection_object(ax_projection)
    ax = fig.add_subplot(111, projection=ax_projection)
    geo_dataset_ax_plot(ax, geo_dataset_plot_data, ax_projection=ax_projection, title=title)


def geo_dataset_ax_plot_2(ax, multi_geo_dataset_plot_data, key, projection=None):
    ax.yaxis.tick_right()
    d_lat = (multi_geo_dataset_plot_data.lat_max - multi_geo_dataset_plot_data.lat_min) / 5
    lat_ticks = np.arange(multi_geo_dataset_plot_data.lat_min + d_lat / 2.0, multi_geo_dataset_plot_data.lat_max, d_lat)
    d_lon = (multi_geo_dataset_plot_data.lon_max - multi_geo_dataset_plot_data.lon_min) / 5
    lon_ticks = np.arange(multi_geo_dataset_plot_data.lon_min + d_lon / 2.0, multi_geo_dataset_plot_data.lon_max, d_lon)
    ax.set_xticks(lon_ticks, crs=projection)
    ax.set_yticks(lat_ticks, crs=projection)
    ax.set_xlim(multi_geo_dataset_plot_data.lon_min, multi_geo_dataset_plot_data.lon_max)
    ax.set_ylim(multi_geo_dataset_plot_data.lat_min, multi_geo_dataset_plot_data.lat_max)
    ax.set_aspect('auto')
    # ToDo: vmin vmax could occur in plot_kw
    fg.gridmap(ax=ax, data=multi_geo_dataset_plot_data.gridmap_data[key],
               transform=multi_geo_dataset_plot_data.data_projections[key],
               features=default_features, frame=True, show_time=False,
               plot_kw={'cbar_kwargs': {'fraction': 0.046, 'pad': 0.04},
                        'vmin': multi_geo_dataset_plot_data.vmin,
                        'vmax': multi_geo_dataset_plot_data.vmax})
    ax.set_title(f'{multi_geo_dataset_plot_data.var_names[key]} '
                 f'at time index {multi_geo_dataset_plot_data.time_indices[key]}')


def multi_geo_dataset_fig_plot(fig, multi_geo_dataset_plot_data, projection=None):
    projection = get_projection_object(projection)
    n_cols = len(multi_geo_dataset_plot_data.gridmap_data)
    i = 1
    for key in multi_geo_dataset_plot_data.gridmap_data.keys():
        ax1 = fig.add_subplot(1, n_cols, i)
        ax1 = add_projection_to_ax(fig, ax1, projection)
        geo_dataset_ax_plot_2(ax1, multi_geo_dataset_plot_data, key, projection=projection)
        i += 1


def dataset_file_plot(dataset_file, save_file, bbox_inches='tight', title=None):
    custom_figure = CustomFigure(plot_fn=geo_dataset_fig_plot, set_default_figure=True)
    geo_dataset_plot_data = GeoDatasetPlotData(nc_file=dataset_file)
    custom_figure.plot(geo_dataset_plot_data, title=title)
    custom_figure.savefig(save_file, bbox_inches=bbox_inches)
