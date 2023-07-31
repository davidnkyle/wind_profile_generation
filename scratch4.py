import boto3
import botocore
import datetime
import matplotlib.pyplot as plt
import os.path
import xarray as xr

monthly_raw = xr.load_dataset('app/monthly_wind_speed.nc')
lat = 33
lon = 250
ds2 = monthly_raw.sel(lon=lon, lat=lat, method='nearest')



