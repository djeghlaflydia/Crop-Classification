"""
preprocess.py - Data preprocessing (VERSION ACCÉLÉRÉE)
"""

import os
import sys
import ee
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from scripts.api_access import init_gee, get_s2_collection, get_cdl_image, compute_ndvi, compute_evi, compute_ndwi

def sample_points_by_class_fast(area_name, bbox, points_per_class=30):
    """
    Échantillonnage rapide par classe
    """
    print(f"\n  Processing {area_name}...")
    
    geometry = ee.Geometry.BBox(*bbox)
    cdl = get_cdl_image(bbox, config.YEAR)
    
    # Obtenir toutes les classes
    hist = cdl.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=geometry,
        scale=30,
        maxPixels=1e6
    ).getInfo()
    
    class_counts = hist.get('cropland', {})
    available_classes = [int(k) for k in class_counts.keys() if int(k) > 0]
    print(f"  Classes disponibles: {len(available_classes)}")
    
    all_points = []
    all_labels = []
    
    for class_code in tqdm(available_classes, desc="  Échantillonnage classes"):
        class_mask = cdl.eq(class_code)
        
        samples = class_mask.stratifiedSample(
            numPoints=points_per_class,
            classBand='cropland',
            scale=30,
            geometries=True,
            seed=42,
            dropNulls=True
        )
        
        try:
            samples_list = samples.getInfo()['features']
            for feat in samples_list:
                coords = feat['geometry']['coordinates']
                all_points.append(ee.Geometry.Point(coords))
                all_labels.append(class_code)
        except:
            continue
    
    print(f"  Total points: {len(all_points)}")
    print(f"  Classes uniques: {len(np.unique(all_labels))}")
    
    return all_points, all_labels

def extract_time_series_batch(area_name, bbox, points, labels, batch_size=50):
    """
    Extraction par lots - BEAUCOUP PLUS RAPIDE
    """
    print(f"\n  Extraction des séries temporelles...")
    
    # Get S2 collection
    s2_collection = get_s2_collection(
        bbox, config.START_DATE, config.END_DATE, config.CLOUD_PERCENT
    )
    s2_collection = s2_collection.map(compute_ndvi)
    s2_collection = s2_collection.map(compute_evi)
    s2_collection = s2_collection.map(compute_ndwi)
    
    n_images = s2_collection.size().getInfo()
    bands = config.S2_BANDS + ['NDVI', 'EVI', 'NDWI']
    
    print(f"  Images: {n_images}, Bands: {len(bands)}")
    print(f"  Points: {len(points)}, Lots de {batch_size}")
    
    all_X = []
    all_y = []
    all_masks = []
    
    # Traiter par lots
    n_batches = (len(points) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(n_batches), desc="  Lots"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(points))
        
        batch_points = points[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        
        # Créer FeatureCollection pour ce lot
        features = []
        for i, (point, label) in enumerate(zip(batch_points, batch_labels)):
            features.append(ee.Feature(point, {'label': label, 'id': i}))
        
        points_fc = ee.FeatureCollection(features)
        
        # Créer le stack d'images pour ce lot
        image_list = s2_collection.toList(n_images)
        all_bands = []
        for i in range(n_images):
            image = ee.Image(image_list.get(i))
            for band in bands:
                all_bands.append(image.select(band).rename(f'{band}_{i}'))
        
        stack = ee.Image.cat(all_bands)
        
        # Échantillonner
        try:
            sampled = stack.sampleRegions(
                collection=points_fc,
                scale=20,
                geometries=False
            ).getInfo()
            
            # Reconstruire pour ce lot
            for feat in sampled['features']:
                props = feat['properties']
                label = props.get('label')
                point_id = props.get('id')
                
                if label is None:
                    continue
                
                all_y.append(label)
                
                series = []
                for i in range(n_images):
                    row = []
                    for j, band in enumerate(bands):
                        val = props.get(f'{band}_{i}', np.nan)
                        row.append(val if val is not None else np.nan)
                    series.append(row)
                
                ts = np.array(series, dtype=np.float32)
                mask = (~np.isnan(ts)).all(axis=1).astype(np.float32)
                mask = np.expand_dims(mask, axis=-1)
                ts = np.nan_to_num(ts, 0)
                
                all_X.append(ts)
                all_masks.append(mask)
                
        except Exception as e:
            print(f"\n    Lot {batch_idx} échoué: {e}")
            continue
    
    X = np.array(all_X)
    y = np.array(all_y)
    masks = np.array(all_masks)
    
    print(f"\n  ✅ Dataset: X={X.shape}, y={y.shape}")
    print(f"  Classes uniques: {np.unique(y)}")
    
    return X, y, masks

def main():
    print("\n" + "="*70)
    print("DATA PREPROCESSING - VERSION ACCÉLÉRÉE")
    print("="*70)
    
    if not init_gee(config.GEE_PROJECT):
        print("Failed to initialize GEE")
        return
    
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    for area_name, settings in config.STUDY_AREAS.items():
        print(f"\n{'='*50}")
        print(f"📊 {area_name}")
        print(f"{'='*50}")
        
        bbox = settings['bbox']
        
        # Échantillonner (30 points par classe)
        points, labels = sample_points_by_class_fast(area_name, bbox, points_per_class=30)
        
        if len(points) == 0:
            print(f"  ❌ Aucun point")
            continue
        
        # Extraire les séries par lots
        X, y, mask = extract_time_series_batch(area_name, bbox, points, labels, batch_size=50)
        
        if len(X) == 0:
            print(f"  ❌ Aucune donnée")
            continue
        
        # Split
        print("\n  Division des données...")
        X_train, X_temp, y_train, y_temp, m_train, m_temp = train_test_split(
            X, y, mask, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test, m_val, m_test = train_test_split(
            X_temp, y_temp, m_temp, test_size=0.5, random_state=42
        )
        
        print(f"    Train: {X_train.shape}")
        print(f"    Val:   {X_val.shape}")
        print(f"    Test:  {X_test.shape}")
        
        # Encodage
        print("\n  Encodage...")
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_val_enc = le.transform(y_val)
        y_test_enc = le.transform(y_test)
        
        class_info = {
            'classes': le.classes_.tolist(),
            'n_classes': len(le.classes_)
        }
        print(f"    Classes: {class_info['n_classes']}")
        
        # Normalisation
        print("\n  Normalisation...")
        mean = X_train.mean(axis=(0, 1), keepdims=True)
        std = X_train.std(axis=(0, 1), keepdims=True)
        std = np.where(std == 0, 1, std)
        
        X_train_norm = (X_train - mean) / std
        X_val_norm = (X_val - mean) / std
        X_test_norm = (X_test - mean) / std
        
        # Sauvegarde
        print("\n  Sauvegarde...")
        np.save(f'{config.DATA_DIR}/X_train_{area_name}.npy', X_train_norm)
        np.save(f'{config.DATA_DIR}/X_val_{area_name}.npy', X_val_norm)
        np.save(f'{config.DATA_DIR}/X_test_{area_name}.npy', X_test_norm)
        
        np.save(f'{config.DATA_DIR}/y_train_{area_name}.npy', y_train_enc)
        np.save(f'{config.DATA_DIR}/y_val_{area_name}.npy', y_val_enc)
        np.save(f'{config.DATA_DIR}/y_test_{area_name}.npy', y_test_enc)
        
        np.save(f'{config.DATA_DIR}/mask_train_{area_name}.npy', m_train)
        np.save(f'{config.DATA_DIR}/mask_val_{area_name}.npy', m_val)
        np.save(f'{config.DATA_DIR}/mask_test_{area_name}.npy', m_test)
        
        import json
        with open(f'{config.DATA_DIR}/class_info_{area_name}.json', 'w') as f:
            json.dump(class_info, f, indent=2)
        
        norm_params = {'mean': mean.squeeze().tolist(), 'std': std.squeeze().tolist()}
        with open(f'{config.DATA_DIR}/norm_params_{area_name}.json', 'w') as f:
            json.dump(norm_params, f, indent=2)
        
        print(f"\n  ✅ {area_name} terminé !")
    
    print("\n" + "="*70)
    print("✅ PREPROCESSING COMPLET")
    print("="*70)

if __name__ == "__main__":
    main()