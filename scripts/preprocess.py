import os
import sys
import numpy as np
import xarray as xr
import stackstac
import dask
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

os.environ["GDAL_HTTP_MAX_RETRY"] = "2"
os.environ["GDAL_HTTP_RETRY_DELAY"] = "1"
dask.config.set(scheduler="threads")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from scripts.api_access import get_stac_data

def get_10day_composites(area_name, bbox, bands, year=2021, epsg=32611):
    datetime_range = f"{year}-06-01/{year}-09-30" # Reduced date range for SPEED
    items = get_stac_data(area_name, config.S2_COLLECTION, bbox, datetime_range)
    items = [i for i in items if i.properties.get("eo:cloud_cover", 100) < 15][:10] # limit items
    if not items: raise ValueError(f"No clear images for {area_name}")
    stack = stackstac.stack(items, assets=bands, bounds_latlon=bbox, epsg=epsg, resolution=60)
    resampled = stack.resample(time="3D").mean(dim="time", skipna=True) # faster mean than median
    da = resampled.isel(time=slice(0, 36))
    da = da.assign_coords(time=np.arange(len(da.time)))
    return da

def sample_points(da_s2, da_cdl, num_points=200):
    cdl_data = da_cdl.values.squeeze()
    valid_y, valid_x = np.where(cdl_data > 0)
    if len(valid_y) == 0: raise ValueError("No valid CDL pixels")
    idx = np.random.choice(len(valid_y), min(num_points, len(valid_y)), replace=False)
    xs, ys = da_cdl.coords['x'].values[valid_x[idx]], da_cdl.coords['y'].values[valid_y[idx]]
    ts = da_s2.sel(x=xr.DataArray(xs, dims='point'), y=xr.DataArray(ys, dims='point'), method='nearest').values
    ts = np.transpose(ts, (2, 0, 1))
    labels = cdl_data[valid_y[idx], valid_x[idx]]
    return ts, labels

def main():
    areas = [('Arkansas', 32615), ('California', 32611)]
    for area_name, epsg in areas:
        print(f"\n--- FAST Processing {area_name} ---")
        try:
            bbox = config.STUDY_AREAS[area_name]['bbox']
            s2 = get_10day_composites(area_name, bbox, ['B02','B03','B04','B08'], year=2021, epsg=epsg).compute()
            cdl_items = get_stac_data(area_name, config.CDL_COLLECTION, bbox, "2021-01-01/2021-12-31")
            cdl = stackstac.stack(cdl_items, assets=['cropland'], bounds_latlon=bbox, epsg=epsg, resolution=60).isel(time=0).compute()
            X, y = sample_points(s2, cdl, num_points=200)
            
            # NDVI
            ndvi = (X[:,:,3] - X[:,:,2]) / (X[:,:,3] + X[:,:,2] + 1e-8)
            X = np.concatenate([X, ndvi[:,:,np.newaxis]], axis=2)
            mask = (~np.isnan(X)).all(axis=2).astype(np.float32)
            X = np.nan_to_num(X, 0.0)

            # Split (Non-stratified if needed to avoid error)
            X_tr, X_val_te, y_tr, y_val_te, m_tr, m_val_te = train_test_split(X, y, mask, test_size=0.3, random_state=42)
            X_va, X_te, y_va, y_te, m_va, m_te = train_test_split(X_val_te, y_val_te, m_val_te, test_size=0.5, random_state=42)

            # Normalization
            mu, std = X_tr.mean(axis=(0,1), keepdims=True), X_tr.std(axis=(0,1), keepdims=True)
            X_tr, X_va, X_te = (X_tr-mu)/(std+1e-6), (X_va-mu)/(std+1e-6), (X_te-mu)/(std+1e-6)

            for split, (xs, ys, ms) in [('train', (X_tr, y_tr, m_tr)), ('val', (X_va, y_va, m_va)), ('test', (X_te, y_te, m_te))]:
                np.save(f'X_{split}_{area_name}.npy', xs); np.save(f'y_{split}_{area_name}.npy', ys); np.save(f'mask_{split}_{area_name}.npy', ms)
            print(f"Done {area_name}")
        except Exception as e:
            print(f"Error {area_name}: {e}")

if __name__ == '__main__':
    main()
