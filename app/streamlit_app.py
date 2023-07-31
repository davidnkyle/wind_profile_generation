import math

import streamlit as st
from sklearn import linear_model
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xarray as xr
import plotly.express as px
from analysis import build_entire_model, generate_time_series, fourier, DEGREE

@st.cache_data
def convert_df(df):
   return df.to_csv(index=True).encode('utf-8')

BIN_WIDTH = 0.5

major_us_cities = {
    'None': (40.0, -100.0),
    'Chicago, IL': (41.84, -87.68),
    'Columbus, OH': (39.99, -82.99),
    'Charlotte, NC': (35.21, -80.83),
    'Denver, CO': (39.76, -104.88),
    'Houston, TX': (29.79, -95.39),
    'Los Angeles, CA': (34.02, -118.41),
    'Philadelphia, PA': (40.01, -75.13),
    'Phoenix, AZ': (33.57, -112.09),
    'New York, NY': (40.66, -73.94),
    'San Diego, CA': (32.81, -117.14),
    'Seattle, WA': (47.62, -122.35),
}

# Main Streamlit app
def main():
    st.set_page_config(layout="wide")

    st.title('Wind Data Modeling App')
    st.markdown('This app generates future simulated wind speeds for any location in the United States. '
                'GCP hosts the app throuh a Docker container with the python code found in this repository: '
                '[github.com/davidnkyle/wind_profile_generation](https://github.com/davidnkyle/wind_profile_generation)'
                '. Ten years of ERA5 data from a specified latitude and longitude are used to train an '
                'autoregressive model with seasonality. Then ten years of simulated future data is generated to match '
                'the characteristics of wind at your location.')
    with st.sidebar:
        st.markdown('# Select Location')
        st.markdown('Select a city from the dropdown or specify your latitude and longitude manually.')
        col_city, col_seed = st.columns(2)
        with col_city:
            city = st.selectbox('US City', major_us_cities, index=0)
            seed = st.number_input('Numpy Random Seed', min_value=1, value=876, step=1)
        with col_seed:
            pair = major_us_cities[city]
            lat = st.number_input('Latitude', min_value=25.0, max_value=50.0, value=pair[0], step=0.01)
            lon_raw = st.number_input('Longitude', min_value=-125.0, max_value=-65.0, value=pair[1], step=0.01)
            lon = lon_raw + 360

        st.map(data=pd.DataFrame({'lat': [lat], 'lon': [lon_raw]}), zoom=3, size=50000)

    ds = xr.load_dataset(r'monthly_wind_speed.nc')
    ds2 = ds.sel(lon=lon, lat=lat, method='nearest')
    df = ds2.to_dataframe()['wind_speed']
    df.index = pd.date_range('2008-01-01', '2018-12-01', freq='MS')
    rolling = pd.concat([df, df.rolling(12, 12, center=True).mean()], axis=1)
    rolling.columns = ['Monthly', 'Rolling Avg']

    # build model
    reg_models = build_entire_model(np.sqrt(df))

    reg_monthly = reg_models[0]
    single_year = pd.date_range(start='2020-01-01', end='2020-12-01', freq='d')
    julian_values = pd.Series(data=single_year.to_julian_date(), index=single_year)
    seasonal_fourier = fourier(julian_values, 365.25, degree=DEGREE).values
    month_index = (julian_values - julian_values[0]) / 365.25 * 12 + 0.99
    single_year_series = pd.Series(reg_monthly.predict(seasonal_fourier), index=month_index)**2
    monthly = pd.concat([df.groupby(df.index.month).mean(), single_year_series], axis=1)
    monthly.columns = ['ERA5 Monthly Profile', 'Fourier Model']

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('## Pull ERA5 Data')
        st.markdown('Monthly data values are from ERA5 Reanalysis dataset pulled from AWS. '
                    'See [registry.opendata.aws/ecmwf-era5](https://registry.opendata.aws/ecmwf-era5/) '
                    'for more details. The ten year dataset spans from 2008 to 2018.')

    with col2:
        st.markdown('## Generate Timeseries')
        st.markdown('An autoregressive integrated moving average model is used to simulate years 2025 to 2035 of monthly wind speed '
                    'values at the specified location. A fourier fit  captures the effects of seasonal wind speeds. '
                    'Try changing the seed on the left side panel to '
                    'sample different simulated instances.')


    gen_ts = generate_time_series(seed, *reg_models) ** 2
    gen_rolling = pd.concat([gen_ts, gen_ts.rolling(12, 12, center=True).mean()], axis=1)
    gen_rolling.columns = ['Monthly', 'Rolling Avg']
    gen_monthly = pd.concat([gen_ts.groupby(gen_ts.index.month).mean(), single_year_series], axis=1)
    gen_monthly.columns = ['Simulated Monthly Profile', 'Fourier Model']

    col3, col4 = st.columns(2)
    with col3:
        fig = px.line(rolling, range_y=(0, 10), title='ERA5 Time Series', labels={'index': 'Time', 'value': 'Wind Speed [m/s]'})
        st.plotly_chart(fig, use_container_width=True)
        nbins = math.ceil((rolling["Monthly"].max() - rolling["Monthly"].min()) / BIN_WIDTH)
        fig = px.histogram(rolling, x='Monthly', range_x=(0, 10), range_y=(0, 50), nbins=nbins, title='ERA5 Histogram', labels={'Monthly': 'Wind Speed [m/s]'})
        st.plotly_chart(fig, use_container_width=True)
        fig = px.line(monthly, range_y=(3, 8), labels={'index': 'Month', 'value': 'Wind Speed [m/s]'}, title='ERA5 Seasonal Profile')
        st.plotly_chart(fig, use_container_width=True)

        csv = convert_df(df)

        st.download_button(
            "Download ERA5 Data",
            csv,
            "era5_monthly_ws.csv",
            "text/csv",
            key='download-csv'
        )

    with col4:
        fig = px.line(gen_rolling, range_y=(0, 10), title='Simulated Time Series', labels={'index': 'Time', 'value': 'Wind Speed [m/s]'})
        st.plotly_chart(fig, use_container_width=True)
        nbins_gen = math.ceil((gen_rolling["Monthly"].max() - gen_rolling["Monthly"].min()) / BIN_WIDTH)
        fig = px.histogram(gen_rolling, x='Monthly', range_x=(0, 10), range_y=(0, 50), nbins=nbins_gen, title='Simulated Histogram', labels={'Monthly': 'Wind Speed [m/s]'})
        st.plotly_chart(fig, use_container_width=True)
        fig = px.line(gen_monthly, range_y=(3, 8), labels={'index': 'Month', 'value': 'Wind Speed [m/s]'}, title='Simulated Seasonal Profile')
        st.plotly_chart(fig, use_container_width=True)

        gen_csv = convert_df(gen_ts)

        st.download_button(
            "Download Simulated Data",
            gen_csv,
            "generated_monthly_ws.csv",
            "text/csv",
            key='download-csv-gen'
        )

    st.markdown('Download the csv files for the ERA5 or simulated wind speed data using the buttons above.')

if __name__ == '__main__':
    main()


