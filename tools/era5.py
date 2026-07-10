"""ERA5 data feed for the hazard-stats moat.

Two layers, split like get_forecast's HTTP boundary:
- `annual_maxima` — pure: from an xarray Dataset, pick the nearest grid cell to a
  lat/lon and reduce to one maximum per year (the block-maxima the GEV needs).
  Unit-tested offline on a tiny in-memory Dataset.
- `open_era5` — the network edge: open the WeatherBench2 ERA5 zarr from GCS
  (anonymous). Reuses the proven weather-transformer-scratch access pattern.
  Not unit-tested (it hits the cloud).

Pipe: open_era5() -> annual_maxima(...) -> tools.hazard_stats.return_level/period.
"""
from __future__ import annotations

from collections.abc import Sequence

import xarray as xr

from tools.hazard_stats import HazardStat, hazard_stat

# WeatherBench2 ERA5 subset (~5.625°, 64x32, 6-hourly) — same as weather-transformer-scratch.
WB2_PATH = "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr"


def annual_maxima(
    ds: xr.Dataset, var: str, latitude: float, longitude: float
) -> list[float]:
    """Yearly maximum of `var` at the grid cell nearest (latitude, longitude)."""
    cell = ds[var].sel(latitude=latitude, longitude=longitude, method="nearest")
    yearly = cell.groupby("time.year").max()
    return [float(v) for v in yearly.values]


def open_era5(path: str = WB2_PATH) -> xr.Dataset:
    """Open the WeatherBench2 ERA5 zarr from GCS (anonymous, lazy)."""
    import gcsfs

    fs = gcsfs.GCSFileSystem(token="anon")
    return xr.open_zarr(fs.get_mapper(path), consolidated=False)


def era5_hazard_stat(
    latitude: float,
    longitude: float,
    variable: str = "2m_temperature",
    path: str = WB2_PATH,
    return_periods: Sequence[int] = (10, 50, 100),
) -> HazardStat:
    """Live moat call: open WB2 ERA5 → annual maxima at (lat, lon) → GEV → HazardStat.

    Network edge (hits GCS); not unit-tested. The pieces it composes
    (annual_maxima, hazard_stat) are each tested offline.
    """
    ds = open_era5(path)
    maxima = annual_maxima(ds, variable, latitude, longitude)
    return hazard_stat(
        maxima,
        variable=variable,
        latitude=latitude,
        longitude=longitude,
        return_periods=return_periods,
    )
