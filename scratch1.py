import cdsapi
import xarray as xr
import pandas as pd
from urllib.request import urlopen
import netCDF4
c = cdsapi.Client()
# dataset to read
dataset = 'reanalysis-era5-pressure-levels'
# flag to download data
download_flag = False
# api parameters
params = {
    'format': 'netcdf',
    'product_type': 'reanalysis',
    'variable': ['u_component_of_wind', 'v_component_of_wind'],
    'pressure_level':'1000',
    # 'year':['2020'],
    # 'month':['{:02d}'.format(idx) for idx in range(1, 13)],
    # 'day': ['{:02d}'.format(idx) for idx in range(1, 32)],
    # 'time': ['{:02d}:00'.format(idx) for idx in range(24)],
    # 'date':'2020-01-01 12:00/2020-03-01 12:00',
    # 'date': '2020-01-01 12:00/2020-03-01 12:00',
    # 'date': list(pd.date_range('2020–01–01','2020–03–01', freq='h').strftime('%Y-%m-%d %H:%M')),
    # 'date': list(pd.date_range('2020-01-01', '2020-03-01', freq='h').strftime('%Y-%m-%d %H:%M')),
    'date': ['2020-01-01 00:00', '2020-01-01 01:00', '2020-01-01 02:00'],
    'grid': [0.25, 0.25],
    'area': [49.38, -124.67, 25.84, -66.95],
    }
# retrieves the path to the file
fl = c.retrieve(dataset, params)
# download the file
if download_flag:
    fl.download("./output.nc")
# load into memory
with urlopen(fl.location) as f:
    ds = xr.open_dataset(f.read())
    # file2read = netCDF4.Dataset(f.read())
    # ds = f.read()
df = ds.to_dataframe()
print(df.head())
