import boto3
import botocore
import datetime
import matplotlib.pyplot as plt
import os.path
import xarray as xr

era5_bucket = 'era5-pds'

# AWS access / secret keys required
# s3 = boto3.resource('s3')
# bucket = s3.Bucket(era5_bucket)

# No AWS keys required
client = boto3.client('s3', config=botocore.client.Config(signature_version=botocore.UNSIGNED))

# select date and variable of interest
date = datetime.date(2017,9,1)
var = 'air_temperature_at_2_metres'

# file path patterns for remote S3 objects and corresponding local file
s3_data_ptrn = '{year}/{month}/data/{var}.nc'
data_file_ptrn = '{year}{month}_{var}.nc'

year = date.strftime('%Y')
month = date.strftime('%m')
s3_data_key = s3_data_ptrn.format(year=year, month=month, var=var)
data_file = data_file_ptrn.format(year=year, month=month, var=var)

if not os.path.isfile(data_file): # check if file already exists
    print("Downloading %s from S3..." % s3_data_key)
    client.download_file(era5_bucket, s3_data_key, data_file)

ds = xr.open_dataset(data_file)
print(ds.info)
print('hello')

# location coordinates
locs = [
    {'name': 'santa_monica', 'lon': -118.496245, 'lat': 34.010341},
    {'name': 'tallinn', 'lon': 24.753574, 'lat': 59.436962},
    {'name': 'honolulu', 'lon': -157.835938, 'lat': 21.290014},
    {'name': 'cape_town', 'lon': 18.423300, 'lat': -33.918861},
    {'name': 'dubai', 'lon': 55.316666, 'lat': 25.266666},
]

# convert westward longitudes to degrees east
for l in locs:
    if l['lon'] < 0:
        l['lon'] = 360 + l['lon']
print(locs)

ds_locs = xr.Dataset()

# interate through the locations and create a dataset
# containing the temperature values for each location
for l in locs:
    name = l['name']
    lon = l['lon']
    lat = l['lat']
    var_name = name

    ds2 = ds.sel(lon=lon, lat=lat, method='nearest')

    lon_attr = '%s_lon' % name
    lat_attr = '%s_lat' % name

    ds2.attrs[lon_attr] = ds2.lon.values.tolist()
    ds2.attrs[lat_attr] = ds2.lat.values.tolist()
    ds2 = ds2.rename({var: var_name}).drop(('lat', 'lon'))

    ds_locs = xr.merge([ds_locs, ds2])

print(ds_locs.data_vars)

def kelvin_to_celcius(t):
    return t - 273.15

def kelvin_to_fahrenheit(t):
    return t * 9/5 - 459.67

ds_locs_f = ds_locs.apply(kelvin_to_fahrenheit)

df_f = ds_locs_f.to_dataframe()
print(df_f.describe())
