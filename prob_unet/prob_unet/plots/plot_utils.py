import copy
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class CustomFigure:
    def __init__(self, plot_fn=None, templates=None, template_name='custom_plot', templates_substitutes=None,
                 set_default_figure=False):
        # plot_fn must be a callable that takes target figure as its first argument
        # in cases where the figure size will be determined by the plot function, this CustomFigure can be 1st argument
        # it can also be a class and its set_default_templates_substitutes method will be called
        self.plot_fn = plot_fn
        self.templates = copy.copy(templates)
        if templates is not None:
            if hasattr(plot_fn, 'set_default_templates_substitutes'):
                plot_fn.set_default_templates_substitutes(self.templates)
            if templates_substitutes is not None:
                self.templates.add_substitutes(**templates_substitutes)
        self.template_name = template_name
        self.figure = None
        if set_default_figure:
            _ = self.create_figure(figsize=(16, 12), dpi=100.0, edgecolor='black')

    def create_figure(self, *args, **kwargs):
        self.figure = plt.figure(*args, **kwargs)
        return self.figure

    def plot(self, *args, **kwargs):
        if self.figure is None:
            self.plot_fn(self, *args, **kwargs)
        else:
            self.plot_fn(self.figure, *args, **kwargs)
        return self

    def savefig(self, figure_file=None, close_figure=True, pad=None, **kwargs):
        if figure_file is None:
            try:
                figure_file = self.templates.complete(self.template_name)
            except (ValueError, AttributeError, KeyError):
                logger.error('No plot file template found.')
                return None
        Path(figure_file).parent.mkdir(parents=True, exist_ok=True)
        if pad is not None:
            self.figure.tight_layout(pad=pad)
        self.figure.savefig(figure_file, **kwargs)
        if close_figure:
            plt.close(self.figure)
        return figure_file


def marker_size_as_fn_of_nb_of_pixels(fig, scale_power=1):
    fig_width_pixels = fig.get_figwidth() * fig.dpi
    fig_height_pixels = fig.get_figheight() * fig.dpi
    return (fig_width_pixels * fig_height_pixels) ** scale_power


def show_spine(ax, edge_color='black', line_width=2):
    for spine in ax.spines.values():
        spine.set_edgecolor(edge_color)
        spine.set_linewidth(line_width)


def ax_rect_from_position_and_aspect_ratio(position, aspect_ratio):
    position_aspect_ratio = (position.x1 - position.x0) / (position.y1 - position.y0)
    if aspect_ratio < position_aspect_ratio:
        height = position.y1 - position.y0
        width = height * aspect_ratio
    else:
        width = position.x1 - position.x0
        height = width / aspect_ratio
    left = (position.x0 + position.x1) / 2 - width / 2
    bottom = (position.y0 + position.y1) / 2 - height / 2
    return left, bottom, width, height


def truncate_colormap(cmap, min_value=0.0, max_value=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=min_value, b=max_value),
        cmap(np.linspace(min_value, max_value, n)))
    return new_cmap
