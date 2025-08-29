from prob_unet.climatology import climatology_io


def test_daily_climatology_file():
    climatology_file = climatology_io.DailyClimatologyFile(path_climatology_file='/path/to/test_tas_01-15.nc',
                                                           prefix='test_')
    assert str(climatology_file.path_climatology_file) == '/path/to/test_tas_01-15.nc'
    assert climatology_file.variable_name == 'tas'
    assert climatology_file.month == 1
    assert climatology_file.day == 15


def test_daily_climatology_file_from_metadata():
    climatology_file = climatology_io.DailyClimatologyFile(climatology_directory='/path/to', variable_name='tas',
                                                           month=1, day=15, prefix='test_')
    assert str(climatology_file.path_climatology_file) == '/path/to/test_tas_01-15.nc'
    assert climatology_file.variable_name == 'tas'
    assert climatology_file.month == 1
    assert climatology_file.day == 15
