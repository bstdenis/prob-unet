import re
from pathlib import Path


class DailyClimatologyFile:
    # File name pattern: {prefix}{variable_name}_{month:02d}-{day:02d}.nc
    # e.g., era5_tas_01-15.nc
    def __init__(self, path_climatology_file=None, climatology_directory=None, prefix='', variable_name=None,
                 month=None, day=None):
        if path_climatology_file is None:
            if (climatology_directory is None) or (variable_name is None) or (month is None) or (day is None):
                raise ValueError("If path_nc_file is None, all other parameters must be provided.")
            self.path_climatology_file = Path(climatology_directory,
                                              f'{prefix}{variable_name}_{month:02d}-{day:02d}.nc')
            self.variable_name = variable_name
            self.month = month
            self.day = day
        else:
            self.path_climatology_file = Path(path_climatology_file)
            pattern = rf'{prefix}(?P<variable_name>.+?)_(?P<month>\d{{2}})-(?P<day>\d{{2}})\.nc'
            match = re.match(pattern, self.path_climatology_file.name)
            if match is None:
                raise ValueError(f"File name {self.path_climatology_file.name} does not match expected pattern.")
            self.variable_name, self.month, self.day = match.groups()
            self.month = int(self.month)
            self.day = int(self.day)

    @property
    def title(self):
        return f'{self.variable_name} daily climatology @ {self.month:02d}-{self.day:02d}'


class HourlyClimatologyFile:
    def __init__(self, path_climatology_file=None, climatology_directory=None, prefix='', variable_name=None,
                 month=None, day=None, hour=None):
        if path_climatology_file is None:
            if (climatology_directory is None) or (variable_name is None) or (month is None) or (day is None) or (hour is None):
                raise ValueError("If path_nc_file is None, all other parameters must be provided.")
            self.path_climatology_file = Path(climatology_directory,
                                              f'{prefix}{variable_name}_{month:02d}-{day:02d}T{hour:02d}.nc')
            self.variable_name = variable_name
            self.month = month
            self.day = day
            self.hour = hour
        else:
            self.path_climatology_file = Path(path_climatology_file)
            pattern = rf'{prefix}(?P<var_name>.+?)_(?P<month>\d{{2}})-(?P<day>\d{{2}})T(?P<hour>\d{{2}})\.nc'
            match = re.match(pattern, self.path_climatology_file.name)
            if match is None:
                raise ValueError(f"File name {self.path_climatology_file.name} does not match expected pattern.")
            self.var_name, self.month, self.day, self.hour = match.groups()
            self.month = int(self.month)
            self.day = int(self.day)
            self.hour = int(self.hour)

    @property
    def title(self):
        return f'{self.var_name} hourly climatology @ {self.month:02d}-{self.day:02d}T{self.hour:02d}'
