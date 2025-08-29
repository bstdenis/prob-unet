import tempfile
from pathlib import Path

from prob_unet.benchmarking.debug_netcdf import dummy_climatology_files_on_disk
from prob_unet.climatology import climatology_viz


def test_plots_for_animation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        climatology_files = dummy_climatology_files_on_disk(tmp_dir, num_files=4)
        climatology_viz.plots_for_animation(climatology_files, tmp_dir, prefix='test_')
        assert Path(tmp_dir, 'test_animation_frame_00000000.png').is_file()
        assert Path(tmp_dir, 'test_animation_frame_00000003.png').is_file()
