# SEPP KDE Autoexcitatorio Espacio-Temporal Mejorado

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.neighbors import KernelDensity
from datetime import timedelta
from pyproj import CRS, Transformer
import matplotlib.pyplot as plt


def estimate_mu(events_df, bandwidth_space):
    coords = events_df[['x', 'y']].values
    kde_mu = KernelDensity(bandwidth=bandwidth_space, kernel='epanechnikov')
    kde_mu.fit(coords)
    return kde_mu


def estimate_nu(events_df, bandwidth_time):
    t = events_df['t'].values[:, None]
    kde_nu = KernelDensity(bandwidth=bandwidth_time, kernel='epanechnikov')
    kde_nu.fit(t)
    return kde_nu


def estimate_g(events_df, bandwidth_space, bandwidth_time):
    coords = events_df[['x', 'y', 't']].values
    kde_g = KernelDensity(bandwidth=1.0, kernel='epanechnikov')
    bandwidths = [bandwidth_space, bandwidth_space, bandwidth_time]
    kde_g.fit(coords / bandwidths)
    return kde_g, bandwidths


def simulate_sepp(events_df,
                  bbox,
                  crs='EPSG:4326',
                  cell_size_m=200,
                  time_step_hours=6,
                  horizon_days=7,
                  bandwidth_space=300,
                  bandwidth_time=24):
    """
    Modelo SEPP tipo KDE espacio-temporal con autoexcitación mejorado:
    λ(x,y,t) = ν(t)μ(x,y) + ∑ g(t-ti, x-xi, y-yi)
    """
    events_df = events_df.to_crs('EPSG:3857')
    events_df['timestamp'] = pd.to_datetime(events_df['fecha_crimen'])
    events_df['x'] = events_df.geometry.x
    events_df['y'] = events_df.geometry.y

    t0 = events_df['timestamp'].min()
    t_max = events_df['timestamp'].max()
    events_df['t'] = (events_df['timestamp'] - t0).dt.total_seconds() / 3600

    # Grilla espacial
    transformer = Transformer.from_crs(crs, 'EPSG:3857', always_xy=True)
    x_min, y_min = transformer.transform(bbox['lon_min'], bbox['lat_min'])
    x_max, y_max = transformer.transform(bbox['lon_max'], bbox['lat_max'])
    x_grid = np.arange(x_min, x_max, cell_size_m)
    y_grid = np.arange(y_min, y_max, cell_size_m)

    # Grilla temporal: comienza justo después del último evento
    t_start = (t_max - t0).total_seconds() / 3600
    t_grid = np.arange(t_start, t_start + horizon_days * 24, time_step_hours)

    xx, yy, tt = np.meshgrid(x_grid, y_grid, t_grid, indexing='ij')
    grid_points = np.column_stack([xx.ravel(), yy.ravel(), tt.ravel()])

    # Estimar μ(x, y)
    xx_mu, yy_mu = np.meshgrid(x_grid, y_grid, indexing='ij')
    xy_mu = np.column_stack([xx_mu.ravel(), yy_mu.ravel()])
    kde_mu = estimate_mu(events_df, bandwidth_space)
    log_mu = kde_mu.score_samples(xy_mu)
    mu_vals = np.exp(log_mu).reshape(xx_mu.shape)

    # Estimar ν(t)
    kde_nu = estimate_nu(events_df, bandwidth_time)
    log_nu = kde_nu.score_samples(tt.ravel()[:, None])
    nu_vals = np.exp(log_nu).reshape(tt.shape)

    # Calcular background = ν(t) * μ(x,y)
    background = mu_vals[:, :, None] * nu_vals

    # Estimar g(x,y,t)
    kde_g, bandwidths = estimate_g(events_df, bandwidth_space, bandwidth_time)
    g_vals = np.exp(kde_g.score_samples(grid_points / bandwidths)).reshape(xx.shape)

    # Intensidad total
    lambda_grid = background + g_vals

    # Salidas
    predicted_events = None
    hotspot_map = lambda_grid.max(axis=2) > np.percentile(lambda_grid, 98)
    metrics = {'note': 'Evaluation metrics to be implemented'}

    return lambda_grid, predicted_events, hotspot_map, metrics
