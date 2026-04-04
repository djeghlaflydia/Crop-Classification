"""
preprocess.py - Data preprocessing pipeline
"""

import os
import sys
import ee
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from scripts.api_access import init_gee, get_s2_collection_with_indices, get_cdl_image

def extract_pixel_time_series(s2_collection, point, scale=20):
    """Extract time series for a single pixel"""
    time_series = []
    
    # Get all images
    image_list = s2_collection.toList(s2_collection.size())
    n_images = s2_collection.size().getInfo()
    
    for i in range(n_images):
        image = ee.Image(image_list.get(i))
        
        # Extract bands
        values = []
        for band in config.S2_BANDS + ['NDVI', 'EVI', 'NDWI']:
            value = image.select(band).sample(
                region=point,
                scale=scale,
                geometries=False
            ).first()
            
            try:
                val = value.getInfo()[band]
                values.append(val)
            except:
                values.append(np.nan)
        
        time_series.append(values)
    
    return np.array(time_series)

def sample_points_from_cdl(area_name, bbox, num_points=500):
    """Sample random points from CDL data"""
    geometry = ee.Geometry.BBox(*bbox)
    
    # Get CDL
    cdl = get_cdl_image(bbox, config.YEAR)
    
    # Get valid pixels (non-zero)
    valid_mask = cdl.neq(0)
    
    # Stratified sampling by class
    sample_points = valid_mask.stratifiedSample(
        numPoints=num_points,
        classBand='cropland',
        scale=config.RESOLUTION_METERS,
        geometries=True,
        seed=42,
        dropNulls=True
    )
    
    points_list = sample_points.getInfo()['features']
    
    # Extract coordinates and labels
    points = []
    labels = []
    
    for point in points_list:
        coords = point['geometry']['coordinates']
        label = point['properties']['cropland']
        points.append(ee.Geometry.Point(coords))
        labels.append(label)
    
    return points, labels

def create_timeseries_dataset(area_name, bbox, num_points=500):
    """Create complete time-series dataset"""
    print(f"\n  Creating dataset for {area_name}...")
    
    # Get S2 collection with indices
    s2_collection = get_s2_collection_with_indices(
        bbox, config.START_DATE, config.END_DATE, config.CLOUD_PERCENT
    )
    
    # Sample points from CDL
    points, labels = sample_points_from_cdl(area_name, bbox, num_points)
    
    # Extract time series for each point
    X = []
    y = []
    masks = []
    
    print(f"  Extracting time series for {len(points)} points...")
    
    for i, (point, label) in enumerate(tqdm(zip(points, labels), total=len(points))):
        try:
            ts = extract_pixel_time_series(s2_collection, point)
            
            # Create mask for valid values
            mask = (~np.isnan(ts)).all(axis=1).astype(np.float32)
            mask = np.expand_dims(mask, axis=-1)
            
            # Fill NaN with 0
            ts = np.nan_to_num(ts, 0)
            
            X.append(ts)
            y.append(label)
            masks.append(mask)
            
        except Exception as e:
            print(f"    Warning: Failed for point {i}: {e}")
            continue
    
    X = np.array(X)
    y = np.array(y)
    masks = np.array(masks)
    
    print(f"  Dataset shape: X={X.shape}, y={y.shape}, mask={masks.shape}")
    
    return X, y, masks

def normalize_data(X_train, X_val, X_test):
    """Normalize data using training statistics"""
    # Reshape to (samples * time, features)
    n_samples, n_time, n_features = X_train.shape
    
    # Compute mean and std across samples and time
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std == 0, 1, std)
    
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    X_test_norm = (X_test - mean) / std
    
    # Save normalization parameters
    norm_params = {'mean': mean.squeeze(), 'std': std.squeeze()}
    
    return X_train_norm, X_val_norm, X_test_norm, norm_params

def interpolate_time_series(X, method='linear'):
    """Interpolate missing values in time series"""
    from scipy import interpolate
    
    n_samples, n_time, n_features = X.shape
    X_interp = X.copy()
    
    for i in range(n_samples):
        for j in range(n_features):
            ts = X[i, :, j]
            valid_mask = ~np.isnan(ts)
            
            if valid_mask.sum() > 1:
                x_valid = np.arange(n_time)[valid_mask]
                y_valid = ts[valid_mask]
                
                # Linear interpolation
                f = interpolate.interp1d(x_valid, y_valid, kind=method, fill_value='extrapolate')
                X_interp[i, :, j] = f(np.arange(n_time))
    
    return X_interp

def main():
    """Main preprocessing function"""
    print("\n" + "="*70)
    print("DATA PREPROCESSING - Part 1")
    print("="*70)
    
    # Initialize GEE
    if not init_gee(config.GEE_PROJECT):
        print("Failed to initialize GEE")
        return
    
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    all_class_info = {}
    
    for area_name, settings in config.STUDY_AREAS.items():
        print(f"\n{'='*50}")
        print(f"Processing {area_name}")
        print(f"{'='*50}")
        
        bbox = settings['bbox']
        
        # Extract data
        X, y, mask = create_timeseries_dataset(
            area_name, bbox, num_points=config.MAX_SAMPLES_PER_AREA
        )
        
        # Interpolate missing values
        print("\n  Interpolating missing values...")
        X = interpolate_time_series(X, method='linear')
        
        # Split data
        X_train, X_temp, y_train, y_temp, m_train, m_temp = train_test_split(
            X, y, mask, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test, m_val, m_test = train_test_split(
            X_temp, y_temp, m_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"\n  Data split:")
        print(f"    Train: {X_train.shape}")
        print(f"    Val: {X_val.shape}")
        print(f"    Test: {X_test.shape}")
        
        # Encode labels
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_val_enc = le.transform(y_val)
        y_test_enc = le.transform(y_test)
        
        class_info = {
            'classes': le.classes_.tolist(),
            'n_classes': len(le.classes_),
            'class_names': [f"Class_{c}" for c in le.classes_]
        }
        all_class_info[area_name] = class_info
        
        print(f"\n  Classes: {class_info['n_classes']}")
        print(f"    {class_info['classes']}")
        
        # Normalize
        X_train_norm, X_val_norm, X_test_norm, norm_params = normalize_data(
            X_train, X_val, X_test
        )
        
        # Save data
        print("\n  Saving data...")
        np.save(f'{config.DATA_DIR}/X_train_{area_name}.npy', X_train_norm)
        np.save(f'{config.DATA_DIR}/X_val_{area_name}.npy', X_val_norm)
        np.save(f'{config.DATA_DIR}/X_test_{area_name}.npy', X_test_norm)
        
        np.save(f'{config.DATA_DIR}/y_train_{area_name}.npy', y_train_enc)
        np.save(f'{config.DATA_DIR}/y_val_{area_name}.npy', y_val_enc)
        np.save(f'{config.DATA_DIR}/y_test_{area_name}.npy', y_test_enc)
        
        np.save(f'{config.DATA_DIR}/mask_train_{area_name}.npy', m_train)
        np.save(f'{config.DATA_DIR}/mask_val_{area_name}.npy', m_val)
        np.save(f'{config.DATA_DIR}/mask_test_{area_name}.npy', m_test)
        
        # Save metadata
        import json
        with open(f'{config.DATA_DIR}/class_info_{area_name}.json', 'w') as f:
            json.dump(class_info, f, indent=2)
        
        with open(f'{config.DATA_DIR}/norm_params_{area_name}.json', 'w') as f:
            json.dump({k: v.tolist() for k, v in norm_params.items()}, f, indent=2)
        
        print(f"  ✅ Saved data for {area_name}")
    
    print("\n" + "="*70)
    print("✅ PREPROCESSING COMPLETE!")
    print(f"Data saved to '{config.DATA_DIR}/'")
    print("="*70)

if __name__ == "__main__":
    main()