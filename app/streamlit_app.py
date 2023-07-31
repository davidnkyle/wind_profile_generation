import streamlit as st
from sklearn import linear_model
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xarray as xr
# from google.cloud import storage_client


# def read_static_file(bucket_name, file_name):
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(file_name)
#
#     # Download the file's content as a string
#     file_content = blob.download_as_text()
#     return file_content
#
#
# def read_netcdf_file(bucket_name, file_name):
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(file_name)
#
#     # Download the NetCDF file to a temporary file on the local filesystem
#     temp_file_path = "/tmp/" + file_name
#     blob.download_to_filename(temp_file_path)
#
#     # Load the NetCDF file using xarray
#     ds = xr.load_dataset(temp_file_path)
#
#     # You can now work with the xarray dataset as needed
#     # For example, you can access variables, attributes, etc.
#     # Example:
#     var_data = ds['wind_speed'].values
#
#     return var_data


# Main Streamlit app
def main():
    st.title('Wind Data Modeling App')
    lat = st.number_input('Latitude', min_value=25, max_value=50, value=33)
    lon_raw = st.number_input('Longitude', min_value=-125, max_value=-65, value=-100)
    lon = lon_raw + 360

    st.map(data=pd.DataFrame({'lat': [lat], 'lon': [lon_raw]}), zoom=3, size=50000)
    ds = xr.load_dataset(r'app/monthly_wind_speed.nc')
    ds2 = ds.sel(lon=lon, lat=lat, method='nearest')
    df = ds2.to_dataframe()['wind_speed']
    df.index = pd.date_range('2008-01-01', '2008-02-01', freq='MS')
    st.dataframe(df)


if __name__ == '__main__':
    main()


