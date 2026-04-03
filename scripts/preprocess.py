import os
import sys
import numpy as np
import xarray as xr
import stackstac
import pystac_client
import planetary_computer
from sklearn.model_selection import StratifiedShuffleSplit

# Ensure config and scripts can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from scripts.api_access import get_stac_data, mask_clouds

# ==============================
# CONFIG
# ==============================
os.environ["GDAL_HTTP_MAX_RETRY"] = "5"
os.environ["GDAL_HTTP_RETRY_DELAY"] = "2"

# ----------------------------------------------------------------------
# Temporal Composites (10-day intervals, 36 steps)
# ----------------------------------------------------------------------
def get_10day_composites(area_name, bbox, bands, year=2021, epsg=None):
    """
    Returns a DataArray (time=36, band, y, x) with 10-day median composites.
    """
    datetime_range = f"{year}-01-01/{year}-12-31"

    # For Sentinel-2, we need SCL for cloud masking
    search_bands = bands + ["SCL"] if "SCL" not in bands else bands
    
    items = get_stac_data(area_name, config.S2_COLLECTION, bbox, datetime_range)
    # Filter by cloud cover first to reduce stack size
    items = [i for i in items if i.properties["eo:cloud_cover"] < config.CLOUD_PERCENT]

    if not items:
        raise ValueError(f"No Sentinel-2 items found for {area_name}")

    # Limit to avoid memory issues (e.g. top 50 images)
    items = sorted(items, key=lambda x: x.properties["eo:cloud_cover"])[:50]

    stack = stackstac.stack(
        items, 
        assets=search_bands, 
        bounds_latlon=bbox, 
        epsg=epsg, 
        resolution=10,
        chunksize={"time": 1, "x": 256, "y": 256}
    ).astype("float32")

    # Cloud masking
    stack = mask_clouds(stack)
    
    times = stack.time.values
    start = np.datetime64(f"{year}-01-01")
    
    composites = []
    for i in range(36):
        t0 = start + np.timedelta64(i * 10, 'D')
        t1 = start + np.timedelta64((i + 1) * 10, 'D')
        
        mask_t = (times >= t0) & (times < t1)
        if mask_t.sum() > 0:
            comp = stack.isel(time=mask_t).median(dim='time', skipna=True)
        else:
            # If no data for this 10-day window, use NaN
            ref = stack.isel(time=0)
            comp = xr.full_like(ref, np.nan)
        
        # Add time coordinate
        comp = comp.assign_coords(time=i)
        composites.append(comp)

    # Concat along a NEW time dimension
    da = xr.concat(composites, dim='time')
    
    # Keep only requested bands (remove SCL if added)
    da = da.sel(band=bands)
    return da

# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------
def sample_points(da_s2, da_cdl, num_points=2000, seed=42):
    """
    Extract random points from the S2 cube and CDL labels.
    """
    np.random.seed(seed)
    
    # CDL might have multiple bands or just one
    if "band" in da_cdl.dims:
        cdl_data = da_cdl.sel(band='cropland').values
    else:
        cdl_data = da_cdl.values
        
    valid_y, valid_x = np.where(cdl_data > 0)
    
    if len(valid_y) == 0:
        raise ValueError("No valid CDL pixels found in the area")

    n_samples = min(num_points, len(valid_y))
    idx = np.random.choice(len(valid_y), n_samples, replace=False)
    
    y_idx = valid_y[idx]
    x_idx = valid_x[idx]

    x_coords = da_cdl.coords['x'].values
    y_coords = da_cdl.coords['y'].values
    xs = x_coords[x_idx]
    ys = y_coords[y_idx]

    print(f"Extracting {n_samples} points...")
    
    # Vectorized extraction
    try:
        ts = da_s2.sel(
            x=xr.DataArray(xs, dims='point'),
            y=xr.DataArray(ys, dims='point'),
            method='nearest'
        ).compute() # Result shape: (time=36, band=N, point=n_samples)
        
        # Transpose to (point, time, band)
        ts = ts.transpose('point', 'time', 'band').values
        labels = cdl_data[y_idx, x_idx]
        
    except Exception as e:
        print(f"Vectorized extraction failed: {e}. Using loop (slow)...")
        ts_list = []
        labels_list = []
        for i in range(len(xs)):
            try:
                p = da_s2.sel(x=xs[i], y=ys[i], method='nearest').compute().values
                ts_list.append(p)
                labels_list.append(cdl_data[y_idx[i], x_idx[i]])
            except:
                continue
        ts = np.stack(ts_list, axis=0) # (point, time, band)
        labels = np.array(labels_list)

    # Handle NaNs: replace with 0 and create mask
    mask = (~np.isnan(ts).any(axis=2)).astype(np.float32) # (point, time)
    ts = np.nan_to_num(ts, nan=0.0)
    
    # Remove points that are completely empty (all time steps are NaN/0)
    valid_mask = mask.sum(axis=1) > 0
    ts = ts[valid_mask]
    labels = labels[valid_mask]
    mask = mask[valid_mask]

    return ts, labels, mask

# ----------------------------------------------------------------------
# Main Processing
# ----------------------------------------------------------------------
def main():
    areas = [('California', 32611), ('Arkansas', 32615)]
    bands = ['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12']
    
    for area_name, epsg in areas:
        print(f"\n=== Processing {area_name} (2021) ===")
        bbox = config.STUDY_AREAS[area_name]['bbox']
        
        # 1. Get S2 Cube
        try:
            s2_cube = get_10day_composites(area_name, bbox, bands, year=2021, epsg=epsg)
            print(f"S2 Cube shape: {s2_cube.shape}")
        except Exception as e:
            print(f"Error processing S2 for {area_name}: {e}")
            continue

        # 2. Get CDL
        try:
            cdl_items = get_stac_data(area_name, config.CDL_COLLECTION, bbox, "2021-01-01/2021-12-31")
            
            # Fallback if config.CDL_COLLECTION fails
            if not cdl_items:
                print(f"[{area_name}] CDL with {config.CDL_COLLECTION} failed. Trying 'usda-cdl'...")
                cdl_items = get_stac_data(area_name, "usda-cdl", bbox, "2021-01-01/2021-12-31")
                
            if not cdl_items:
                raise ValueError("No CDL items found")
                
            cdl_stack = stackstac.stack(cdl_items, assets=['cropland'], bounds_latlon=bbox, epsg=epsg, resolution=10)
            cdl_cube = cdl_stack.isel(time=0).compute()
        except Exception as e:
            print(f"Error processing CDL for {area_name}: {e}")
            continue

        # 3. Sample Points
        try:
            X, y, mask = sample_points(s2_cube, cdl_cube, num_points=2000)
            print(f"Samples: X={X.shape}, y={y.shape}, mask={mask.shape}")
        except Exception as e:
            print(f"Error sampling for {area_name}: {e}")
            continue

        # ------------------------------------------------------------------
        # 4. Stratified Split with safety for small classes
        # ------------------------------------------------------------------
        # Count occurrences
        unique_classes, counts = np.unique(y, return_counts=True)
        # Keep only classes with at least 5 samples (for 80/20 train/val and then test split)
        valid_classes = unique_classes[counts >= 5]
        
        if len(valid_classes) < 2:
            print(f"Not enough classes with samples for stratified split in {area_name}")
            continue
            
        filter_mask = np.isin(y, valid_classes)
        X = X[filter_mask]
        y = y[filter_mask]
        mask = mask[filter_mask]
        
        print(f"Samples after filtering minority classes: {len(y)}")

        try:
            # First split: Train+Val vs Test
            sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_val_idx, test_idx = next(sss1.split(X, y))
            
            X_train_val, y_train_val, mask_train_val = X[train_val_idx], y[train_val_idx], mask[train_val_idx]
            X_test, y_test, mask_test = X[test_idx], y[test_idx], mask[test_idx]
            
            # Second split: Train vs Val
            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, val_idx = next(sss2.split(X_train_val, y_train_val))
            
            X_train, y_train, mask_train = X_train_val[train_idx], y_train_val[train_idx], mask_train_val[train_idx]
            X_val, y_val, mask_val = X_train_val[val_idx], y_train_val[val_idx], mask_train_val[val_idx]
            
            # 5. Normalization (using training mean/std)
            # Normalize over (point, time) but keep bands separate
            mean = X_train.mean(axis=(0, 1), keepdims=True)
            std = X_train.std(axis=(0, 1), keepdims=True)
            
            X_train_norm = (X_train - mean) / (std + 1e-6)
            X_val_norm = (X_val - mean) / (std + 1e-6)
            X_test_norm = (X_test - mean) / (std + 1e-6)
            
            # 6. Save
            data_dir = os.path.dirname(config.DATA_DIR) if config.DATA_DIR else "."
            os.makedirs(data_dir, exist_ok=True)
            
            np.save(os.path.join(data_dir, f"X_train_{area_name}.npy"), X_train_norm)
            np.save(os.path.join(data_dir, f"y_train_{area_name}.npy"), y_train)
            np.save(os.path.join(data_dir, f"mask_train_{area_name}.npy"), mask_train)
            
            np.save(os.path.join(data_dir, f"X_val_{area_name}.npy"), X_val_norm)
            np.save(os.path.join(data_dir, f"y_val_{area_name}.npy"), y_val)
            np.save(os.path.join(data_dir, f"mask_val_{area_name}.npy"), mask_val)
            
            np.save(os.path.join(data_dir, f"X_test_{area_name}.npy"), X_test_norm)
            np.save(os.path.join(data_dir, f"y_test_{area_name}.npy"), y_test)
            np.save(os.path.join(data_dir, f"mask_test_{area_name}.npy"), mask_test)
            
            print(f"Successfully saved data for {area_name}")
            
        except Exception as e:
            print(f"Error during splitting/saving for {area_name}: {e}")

if __name__ == "__main__":
    main()