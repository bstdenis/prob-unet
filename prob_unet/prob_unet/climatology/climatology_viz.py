from pathlib import Path

from prob_unet.plots.geo_dataset_plot import dataset_file_plot


def plots_for_animation(climatology_files, path_figures, prefix=''):
    for i, climatology_file in enumerate(climatology_files):
        title = None
        if hasattr(climatology_file, 'title'):
            title = climatology_file.title
        if hasattr(climatology_file, 'path_climatology_file'):
            climatology_file = climatology_file.path_climatology_file
        dataset_file_plot(dataset_file=climatology_file,
                          save_file=Path(path_figures, f'{prefix}animation_frame_{i:08d}.png'),
                          bbox_inches=None, title=title)
