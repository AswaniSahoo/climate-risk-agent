"""Tests for the ERA5 data feed (tools/era5.py).

Only the pure `annual_maxima` function is tested — on a tiny in-memory xarray
Dataset, so it's fast and offline. The `open_era5` loader is the network edge
(reads WeatherBench2 from GCS) and is exercised manually, not in unit tests.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tools.era5 import annual_maxima, precip_annual_maxima_mm, wind_speed_annual_maxima


def _uniform_two_years() -> xr.Dataset:
    """Same field at every cell, with a clear annual max of 40 in 2000, 45 in 2001."""
    times = pd.date_range("2000-01-01", "2001-12-31", freq="D")
    lats = np.array([10.0, 20.0])
    lons = np.array([80.0, 90.0])
    base = np.full(times.size, 20.0)
    base[(times.year == 2000) & (times.dayofyear == 200)] = 40.0
    base[(times.year == 2001) & (times.dayofyear == 200)] = 45.0
    temp = np.broadcast_to(base[:, None, None], (times.size, 2, 2)).copy()
    return xr.Dataset(
        {"2m_temperature": (("time", "latitude", "longitude"), temp)},
        coords={"time": times, "latitude": lats, "longitude": lons},
    )


def test_annual_maxima_returns_yearly_peaks():
    ds = _uniform_two_years()
    maxima = annual_maxima(ds, "2m_temperature", latitude=10.0, longitude=80.0)
    assert maxima == [40.0, 45.0]


def test_annual_maxima_selects_nearest_grid_cell():
    times = pd.date_range("2000-01-01", "2000-12-31", freq="D")
    lats = np.array([10.0, 20.0])
    lons = np.array([80.0, 90.0])
    temp = np.zeros((times.size, 2, 2))
    temp[:, 1, 1] = 50.0  # only cell (lat=20, lon=90) is hot
    ds = xr.Dataset(
        {"t": (("time", "latitude", "longitude"), temp)},
        coords={"time": times, "latitude": lats, "longitude": lons},
    )
    assert annual_maxima(ds, "t", latitude=19.0, longitude=88.0) == [50.0]
    assert annual_maxima(ds, "t", latitude=11.0, longitude=81.0) == [0.0]


def test_wind_speed_annual_maxima_combines_uv_and_takes_yearly_max():
    times = pd.date_range("2000-01-01", "2001-12-31", freq="D")
    lats = np.array([10.0, 20.0])
    lons = np.array([80.0, 90.0])
    u = np.zeros((times.size, 2, 2))
    v = np.zeros((times.size, 2, 2))
    # cell (10,80): (u=3,v=4)->speed 5 in 2000; (u=6,v=8)->speed 10 in 2001
    mask2000 = (times.year == 2000) & (times.dayofyear == 100)
    mask2001 = (times.year == 2001) & (times.dayofyear == 100)
    u[mask2000, 0, 0], v[mask2000, 0, 0] = 3.0, 4.0
    u[mask2001, 0, 0], v[mask2001, 0, 0] = 6.0, 8.0
    ds = xr.Dataset(
        {
            "u10": (("time", "latitude", "longitude"), u),
            "v10": (("time", "latitude", "longitude"), v),
        },
        coords={"time": times, "latitude": lats, "longitude": lons},
    )
    maxima = wind_speed_annual_maxima(
        ds, latitude=10.0, longitude=80.0, u_var="u10", v_var="v10"
    )
    assert maxima == [5.0, 10.0]


def test_precip_annual_maxima_converts_metres_to_mm_and_skips_nan():
    times = pd.date_range("2000-01-01", "2001-12-31", freq="D")
    lats = np.array([10.0, 20.0])
    lons = np.array([80.0, 90.0])
    p = np.zeros((times.size, 2, 2))
    p[(times.year == 2000) & (times.dayofyear == 100), 0, 0] = 0.05  # 50 mm
    p[(times.year == 2001) & (times.dayofyear == 100), 0, 0] = 0.07  # 70 mm
    p[0, 0, 0] = np.nan  # record spin-up, like the real ERA5 24h accumulation
    ds = xr.Dataset(
        {"total_precipitation_24hr": (("time", "latitude", "longitude"), p)},
        coords={"time": times, "latitude": lats, "longitude": lons},
    )

    maxima = precip_annual_maxima_mm(ds, latitude=10.0, longitude=80.0)

    assert maxima == pytest.approx([50.0, 70.0])
