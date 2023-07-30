import boto3
import botocore
import datetime
import matplotlib.pyplot as plt
import os.path
import xarray as xr

monthly_raw = xr.load_dataset('data/monthly_wind_speed.nc')


