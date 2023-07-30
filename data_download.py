import boto3
import botocore
import datetime
import pandas as pd
import xarray as xr
import numpy as np

era5_bucket = 'era5-pds'

# AWS access / secret keys required
# s3 = boto3.resource('s3')
# bucket = s3.Bucket(era5_bucket)

# No AWS keys required
client = boto3.client('s3', config=botocore.client.Config(signature_version=botocore.UNSIGNED))

# select date and variable of interest
date = datetime.date(2017,8,1)
var_east = 'eastward_wind_at_100_metres'
var_north = 'northward_wind_at_100_metres'

# file path patterns for remote S3 objects and corresponding local file
s3_data_ptrn = '{year}/{month}/data/{var}.nc'
data_file_east = 'data/{var}.nc'.format(var=var_east)
data_file_north = 'data/{var}.nc'.format(var=var_north)

xarray_list = []

for date in pd.date_range('2008-01-01', '2018-12-01', freq='MS'):
    year = date.strftime('%Y')
    month = date.strftime('%m')
    for var, data_file in zip([var_east, var_north], [data_file_east, data_file_north]):
        s3_data_key = s3_data_ptrn.format(year=year, month=month, var=var)
        print("Downloading %s from S3..." % s3_data_key)
        client.download_file(era5_bucket, s3_data_key, data_file)
    ds0 = xr.load_dataset(data_file_east)
    ds1 = xr.load_dataset(data_file_north)

    ds0_filtered = ds0.where((ds0['lon'] > 235) & (ds0['lon'] < 295) & (ds0['lat'] > 25) & (ds0['lat'] < 50), drop=True)
    ds1_filtered = ds1.where((ds1['lon'] > 235) & (ds1['lon'] < 295) & (ds1['lat'] > 25) & (ds1['lat'] < 50), drop=True)

    ds0_renamed = ds0_filtered.rename({var_east: 'wind_speed'})
    ds1_renamed = ds1_filtered.rename({var_north: 'wind_speed'})
    ds = np.sqrt(ds0_renamed ** 2 + ds1_renamed ** 2)

    ds_mean = ds.mean(dim='time0')

    xarray_list.append(ds_mean)

ds_comb = xr.concat(xarray_list, dim='time0')
ds_comb.to_netcdf('data/monthly_wind_speed.nc')







