# Wind Profile Generation

See this app in action on GCP: link here.

The app simulates wind speed data based on the characteristics of ERA5 
data from anywhere in the US. 

This repository contains the files necessary to create a Docker container 
for deployment on Google Coud Run.

Data to run the app is collected through the `data_download.py` file. 
Using `boto3`, data from AWS's registry of open data is aggregated and downloaded. 
See more information on this data source here: [registry.opendata.aws/ecmwf-era5](https://registry.opendata.aws/ecmwf-era5/). 

The app using streamlit and all the files needed for the container are 
in the `app/` directory including the `Dockerfile`.

## Sources

Thanks to these sources for helping me create this repository:
 * [Deploy Streamlit using Docker](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker)
 * [era5-s3-via-boto.ipynb](https://github.com/planet-os/notebooks/blob/master/aws/era5-s3-via-boto.ipynb)
