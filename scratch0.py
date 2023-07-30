from sklearn import linear_model
import pandas as pd
import numpy as np
import os

def rotate_clockwise(vx, vy, radians):
    sin = np.sin(radians)
    cos = np.cos(radians)
    new_x = vx*cos + vy*sin
    new_y = -vx*sin + vy*cos
    return np.vstack([new_x, new_y]).T

def fourier(values, period=None, degree=3):
    r = values
    if period is not None:
        r = values/period * 2*np.pi
    args = pd.concat([(idx+1)*r for idx in range(degree)], axis=1)
    cols = ['{}sin'.format(idx+1) for idx in range(degree)] + ['{}cos'.format(idx+1) for idx in range(degree)]
    X_df = pd.concat([np.sin(args), np.cos(args)], axis=1)
    X_df.columns = cols
    return X_df.reindex(sorted(X_df.columns), axis=1)

def interaction(X, Y):
    Z = np.zeros((X.shape[0], X.shape[1]*Y.shape[1]))
    for idx in range(X.shape[1]):
        i = idx*Y.shape[1]
        for jdx in range(Y.shape[1]):
            Z[:,i+jdx] = X[:,idx]*Y[:,jdx]
    return Z

df = pd.read_csv(os.path.expanduser(r'~/Downloads/ERA_original.txt'), index_col='Date_Time', sep='\t')[['Wind_Spd', 'Wind_Dir']]
df.index = pd.to_datetime(df.index)

dir_radians = df.Wind_Dir/180*np.pi
df.loc[:, 'east'] = np.sin(dir_radians)*df['Wind_Spd']
df.loc[:, 'north'] = np.cos(dir_radians)*df['Wind_Spd']

julian_values = pd.Series(data=df.index.to_julian_date(), index=df.index)
diurnal_fourier = fourier(julian_values, 1, degree=6).values
seasonal_fourier = fourier(julian_values, 365.25, degree=6).values
T6 = np.hstack([diurnal_fourier, seasonal_fourier, interaction(diurnal_fourier, seasonal_fourier)])
T5 = np.hstack([diurnal_fourier[:,0:10], seasonal_fourier[:,0:10], interaction(diurnal_fourier[:,0:10], seasonal_fourier[:,0:10])])
T4 = np.hstack([diurnal_fourier[:,0:8], seasonal_fourier[:,0:8], interaction(diurnal_fourier[:,0:8], seasonal_fourier[:,0:8])])
T3 = np.hstack([diurnal_fourier[:,0:6], seasonal_fourier[:,0:6], interaction(diurnal_fourier[:,0:6], seasonal_fourier[:,0:6])])
T2 = np.hstack([diurnal_fourier[:,0:4], seasonal_fourier[:,0:4], interaction(diurnal_fourier[:,0:4], seasonal_fourier[:,0:4])])

y = df[['east', 'north']].values
X_mu = T6
reg_mu = linear_model.Ridge(alpha=100, fit_intercept=True)
reg_mu.fit(X_mu, y)

df[['east_mu', 'north_mu']] = reg_mu.predict(T6)
df[['east_resid', 'north_resid']] = df[['east', 'north']].values - df[['east_mu', 'north_mu']].values

df['east_resid_prev'] = df.east_resid.shift(1)
df['north_resid_prev'] = df.north_resid.shift(1)

df1 = df.dropna()
df1 = df1.loc[~df1.index.hour.isin([7, 19])]

julian_values = pd.Series(data=df1.index.to_julian_date(), index=df1.index)
diurnal_fourier = fourier(julian_values, 1, degree=6).values
seasonal_fourier = fourier(julian_values, 365.25, degree=6).values
T6 = np.hstack([diurnal_fourier, seasonal_fourier, interaction(diurnal_fourier, seasonal_fourier)])

T4 = np.hstack([diurnal_fourier[:,0:8], seasonal_fourier[:,0:8], interaction(diurnal_fourier[:,0:8], seasonal_fourier[:,0:8])])
T3 = np.hstack([diurnal_fourier[:,0:6], seasonal_fourier[:,0:6], interaction(diurnal_fourier[:,0:6], seasonal_fourier[:,0:6])])
T2 = np.hstack([diurnal_fourier[:,0:4], seasonal_fourier[:,0:4], interaction(diurnal_fourier[:,0:4], seasonal_fourier[:,0:4])])

linear = df1[['east_resid_prev', 'north_resid_prev']].values

y = df1[['east_resid', 'north_resid']].values
X_resid = np.hstack([linear, interaction(linear, T6)])
reg_resid = linear_model.Ridge(alpha=1000, fit_intercept=False)
reg_resid.fit(X_resid, y)

east_north_deltas = reg_resid.predict(X_resid)

df1[['east_pred', 'north_pred']] = df1[['east_mu', 'north_mu']].values + east_north_deltas
df1['east_error'] = df1.east - df1.east_pred
df1['north_error'] = df1.north - df1.north_pred
df1['sq_error'] = df1.east_error**2 + df1.north_error**2

df1['pred_angle'] = np.arctan2(df1.east_pred, df1.north_pred)

df1[['dir_error2', 'spd_error2']] = rotate_clockwise(df1.east_error, df1.north_error, -df1.pred_angle)**2
sigma = np.sqrt(df1.sq_error.mean())

y = df1[['dir_error2', 'spd_error2']].values
# y = df1.sq_error.values
# direction_fourier = fourier(df1.pred_angle, degree=3).values
# X_error = np.hstack([direction_fourier, T4, interaction(direction_fourier, T2)])
X_error = T6
reg_error = linear_model.Ridge(alpha=1000, fit_intercept=True)
reg_error.fit(X_error, y)

# generate results

np.random.seed(991)
# reproduce the time series
# julian_values = pd.Series(data=df1.index.to_julian_date(), index=df1.index)
# diurnal_fourier = fourier(julian_values, 1, degree=3).values
# seasonal_fourier = fourier(julian_values, 365.25, degree=3).values
# T3 = np.hstack([diurnal_fourier, seasonal_fourier, interaction(diurnal_fourier, seasonal_fourier)])
# T2 = np.hstack([diurnal_fourier[:,0:4], seasonal_fourier[:,0:4], interaction(diurnal_fourier[:,0:4], seasonal_fourier[:,0:4])])
new_index = pd.date_range(start='2001-01-01', end='2001-12-31 23:00:00', freq='h')
df_new = pd.DataFrame(index=new_index)
julian_gen = pd.Series(data=new_index.to_julian_date(), index=new_index)
diurnal_fourier_gen = fourier(julian_gen, 1, 6).values
seasonal_fourier_gen = fourier(julian_gen, 365.25, degree=6).values
T6_gen = np.hstack([diurnal_fourier_gen, seasonal_fourier_gen, interaction(diurnal_fourier_gen, seasonal_fourier_gen)])
T4_gen = np.hstack([diurnal_fourier_gen[:,0:8], seasonal_fourier_gen[:,0:8], interaction(diurnal_fourier_gen[:,0:8], seasonal_fourier_gen[:,0:8])])
T3_gen = np.hstack([diurnal_fourier_gen[:,0:6], seasonal_fourier_gen[:,0:6], interaction(diurnal_fourier_gen[:,0:6], seasonal_fourier_gen[:,0:6])])
T2_gen = np.hstack([diurnal_fourier_gen[:,0:4], seasonal_fourier_gen[:,0:4], interaction(diurnal_fourier_gen[:,0:4], seasonal_fourier_gen[:,0:4])])
df_new[['east_mu', 'north_mu']] = reg_mu.predict(T6_gen)
df_new[['dir_sigma', 'spd_sigma']] = np.sqrt(reg_error.predict(T6_gen))

normals = np.random.normal(size=(df_new.shape[0], 2))
sigmas = normals * df_new[['dir_sigma', 'spd_sigma']].values

resid_prev = np.array([0, 0])
resids = []
for nds, t6, mu in zip(sigmas, T6_gen, df_new[['east_mu', 'north_mu']].values):
    t6 = np.array([t6])
    linear = np.array([resid_prev])
    x_resid = np.hstack([linear, interaction(linear, t6)])
    pred_resid = reg_resid.predict(x_resid)[0]
    pred_vec = mu + pred_resid
    pred_angle = np.arctan2(*pred_vec)
    nen = rotate_clockwise(nds[0], nds[1], pred_angle)[0]
    resid = pred_resid + nen
    resids.append(resid)
    resid_prev = resid
resids = np.array(resids)
df_new.loc[:, ['east', 'north']] = df_new[['east_mu', 'north_mu']].values + resids

df_new['Wind_Spd'] = np.sqrt(df_new.east**2 + df_new.north**2)
df_new['Wind_Dir'] = ((np.arctan2(df_new.east, df_new.north)/np.pi) % 2)*180

df_export = df_new[['Wind_Spd', 'Wind_Dir']].copy()
# df_export = df_export.loc['2000-01-01 07:00:00':'2019-06-01 06:00:00']
df_export['Wind_Dir'] = df_export['Wind_Dir'].map(lambda x: '%.0f' % x)
df_export.index.name = 'Date_Time'
df_export.to_csv('ERA_modeled_2001.txt', sep='\t', float_format='%.3f')



