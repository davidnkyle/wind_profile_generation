from sklearn import linear_model
import pandas as pd
import numpy as np
import xarray as xr

DEGREE = 4
ALPHA = 10

def fourier(values, period=None, degree=3):
    r = values
    if period is not None:
        r = values/period * 2*np.pi
    args = pd.concat([(idx+1)*r for idx in range(degree)], axis=1)
    cols = ['{}sin'.format(idx+1) for idx in range(degree)] + ['{}cos'.format(idx+1) for idx in range(degree)]
    X_df = pd.concat([np.sin(args), np.cos(args)], axis=1)
    X_df.columns = cols
    return X_df.reindex(sorted(X_df.columns), axis=1)


def fit_seasonal_model(series, seasonal_fourier):
    reg_mu = linear_model.Ridge(alpha=ALPHA, fit_intercept=True)
    reg_mu.fit(seasonal_fourier, series.values)
    return reg_mu, reg_mu.predict(seasonal_fourier)


def fit_auto_regressive_model(resids, seasonal_fourier):
    x = resids.shift(1).iloc[1:].values[np.newaxis].T
    y = resids.iloc[1:].values
    reg_resid = linear_model.Ridge(alpha=ALPHA, fit_intercept=False)
    X = np.hstack([x, x*seasonal_fourier[1:,:]])
    reg_resid.fit(X, y)
    return reg_resid, reg_resid.predict(X)


def fit_error_model(error2, seasonal_fourier):
    reg_error = linear_model.Ridge(alpha=ALPHA, fit_intercept=True)
    reg_error.fit(seasonal_fourier, error2)
    return reg_error, reg_error.predict(seasonal_fourier)


def build_entire_model(series):
    julian_values = pd.Series(data=series.index.to_julian_date(), index=series.index)
    seasonal_fourier = fourier(julian_values, 365.25, degree=DEGREE).values
    reg_mu, series_mu = fit_seasonal_model(series, seasonal_fourier)
    resids = series - series_mu
    reg_resid, series_deltas = fit_auto_regressive_model(resids, seasonal_fourier)
    series_pred = series_mu[1:] + series_deltas
    error = series.iloc[1:] - series_pred
    reg_error, _ = fit_error_model(error**2, seasonal_fourier[1:,:])
    return reg_mu, reg_resid, reg_error


def generate_time_series(seed, reg_mu, reg_resid, reg_error, start='2025-01-01', end='2035-01-01'):
    np.random.seed(seed)
    new_index = pd.date_range(start=start, end=end, freq='MS')
    julian_gen = pd.Series(data=new_index.to_julian_date(), index=new_index)
    seasonal_fourier_gen = fourier(julian_gen, 365.25, degree=DEGREE).values
    mu = reg_mu.predict(seasonal_fourier_gen)
    sigma = np.sqrt(reg_error.predict(seasonal_fourier_gen))
    normals = np.random.normal(size=len(new_index))
    sigmas = normals * sigma
    resid_prev = 0
    resids = []
    for s, sfg in zip(sigmas, seasonal_fourier_gen):
        t6 = np.array([sfg])
        x_resid = np.hstack([np.array([[resid_prev]]), resid_prev*t6])
        pred_resid = reg_resid.predict(x_resid)[0]
        resid = pred_resid + s
        resids.append(resid)
        resid_prev = resid
    resids = np.array(resids)
    series = pd.Series(data=mu + resids, index=new_index, name='wind_speed')
    return series



