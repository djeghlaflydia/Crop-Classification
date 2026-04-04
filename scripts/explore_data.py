"""
explore_data.py - Data exploration and visualization
"""

import os
import sys
import ee
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from typing import List

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from scripts.api_access import init_gee, get_s2_collection, get_cdl_image, compute_ndvi

# Crop type mapping for CDL
CDL_MAP = {
    1: 'Corn', 5: 'Soybeans', 24: 'Winter Wheat', 26: 'Durum Wheat',
    28: 'Spring Wheat', 36: 'Alfalfa', 42: 'Dry Beans', 46: 'Sunflower',
    51: 'Safflower', 61: 'Fallow/Idle', 66: 'Rye', 68: 'Sorghum',
    71: 'Oats', 75: 'Wheat', 111: 'Open Water', 121: 'Developed', 
    190: 'Woody Wetlands', 195: 'Herbaceous Wetlands', 204: 'Pasture',
    69: 'Cotton', 205: 'Triticale'
}

def get_cdl_class_distribution(area_name: str, bbox: List[float], year: int = 2021):
    """Analyze CDL class distribution"""
    geometry = ee.Geometry.BBox(*bbox)
    cdl = get_cdl_image(bbox, year)
    
    histogram = cdl.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=geometry,
        scale=30,
        maxPixels=1e6
    )
    
    hist_dict = histogram.getInfo()
    class_counts = hist_dict.get('cropland', {})
    
    df = pd.DataFrame([
        {'class_code': int(k), 'count': v, 'class_name': CDL_MAP.get(int(k), f'Class_{k}')}
        for k, v in class_counts.items() if int(k) > 0
    ])
    df = df.sort_values('count', ascending=False)
    
    return df

def plot_class_distribution(area_name: str, df: pd.DataFrame):
    """Plot crop class distribution"""
    plt.figure(figsize=(14, 6))
    
    top15 = df.head(15)
    bars = plt.bar(range(len(top15)), top15['count'], color='steelblue', alpha=0.8)
    plt.title(f'Crop Class Distribution - {area_name} (2021)', fontsize=16, fontweight='bold')
    plt.xlabel('Crop Type', fontsize=12)
    plt.ylabel('Number of Pixels', fontsize=12)
    plt.xticks(range(len(top15)), top15['class_name'], rotation=45, ha='right', fontsize=10)
    
    for bar, count in zip(bars, top15['count']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.02,
                f'{count:.0f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'results/class_distribution_{area_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: results/class_distribution_{area_name}.png")
    
    print(f"\n  Class Distribution Summary ({area_name}):")
    print(f"  Total classes: {len(df)}")
    print(f"  Top 5 classes:")
    for i in range(min(5, len(df))):
        print(f"    {i+1}. {df.iloc[i]['class_name']}: {df.iloc[i]['count']:.0f} pixels")

def analyze_temporal_patterns(area_name: str, bbox: List[float]):
    """Analyze NDVI temporal patterns with progress bar"""
    geometry = ee.Geometry.BBox(*bbox)
    
    # Get CDL
    cdl = get_cdl_image(bbox, config.YEAR)
    
    # Get major crop types (top 5 by area)
    dist_df = get_cdl_class_distribution(area_name, bbox, config.YEAR)
    top_classes = dist_df.head(5)['class_code'].tolist()
    top_class_names = [CDL_MAP.get(c, f'Class_{c}') for c in top_classes]
    
    # Get S2 collection with NDVI
    s2_collection = get_s2_collection(bbox, config.START_DATE, config.END_DATE, config.CLOUD_PERCENT)
    s2_collection = s2_collection.map(compute_ndvi)
    
    # Get image list
    image_list = s2_collection.toList(s2_collection.size())
    n_images = s2_collection.size().getInfo()
    
    print(f"  Found {n_images} images for temporal analysis")
    
    if n_images == 0:
        print("  No images available")
        return
    
    # Extract dates and NDVI values
    dates = []
    all_ndvi_data = {class_name: [] for class_name in top_class_names}
    
    print(f"  Processing {n_images} images...")
    
    # Get all images data with progress indicator
    for i in range(n_images):
        # Afficher la progression
        progress = (i + 1) / n_images * 100
        bar_length = 40
        filled = int(bar_length * (i + 1) // n_images)
        bar = '=' * filled + '>' + '.' * (bar_length - filled - 1)
        print(f"    Progress: [{bar:40s}] {progress:.1f}%", end='\r')
        
        image = ee.Image(image_list.get(i))
        
        # Get date
        date_prop = image.get('system:time_start')
        try:
            date_val = date_prop.getInfo()
            if date_val:
                date_str = datetime.fromtimestamp(date_val/1000).strftime('%Y-%m-%d')
                dates.append(date_str)
            else:
                dates.append(f'Image_{i}')
        except Exception:
            dates.append(f'Image_{i}')
        
        # Get NDVI for each class
        for class_code, class_name in zip(top_classes, top_class_names):
            class_mask = cdl.eq(class_code)
            ndvi = image.select('NDVI')
            
            mean_ndvi = ndvi.updateMask(class_mask).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=30,
                maxPixels=1e6,
                bestEffort=True
            ).get('NDVI')
            
            try:
                value = mean_ndvi.getInfo()
                if value is not None and value > -900:
                    all_ndvi_data[class_name].append(value)
                else:
                    all_ndvi_data[class_name].append(None)
            except:
                all_ndvi_data[class_name].append(None)
    
    print(f"\n    ✅ Completed!")
    
    # Create plot
    plt.figure(figsize=(14, 7))
    colors = plt.cm.Set2(np.linspace(0, 1, len(top_class_names)))
    
    for class_name, color in zip(top_class_names, colors):
        ndvi_values = all_ndvi_data[class_name]
        valid_pairs = [(dates[i], ndvi_values[i]) for i in range(len(ndvi_values)) 
                      if ndvi_values[i] is not None]
        
        if valid_pairs:
            valid_dates = [p[0] for p in valid_pairs]
            valid_ndvi = [p[1] for p in valid_pairs]
            plt.plot(valid_dates, valid_ndvi, marker='o', label=class_name, 
                    linewidth=2, markersize=4, color=color)
            print(f"    {class_name}: {len(valid_ndvi)} valid observations")
    
    if any(any(v is not None for v in all_ndvi_data[name]) for name in top_class_names):
        plt.title(f'NDVI Temporal Patterns by Crop Type - {area_name} (2021)', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('NDVI', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.ylim(-0.2, 1.0)
        plt.tight_layout()
        plt.savefig(f'results/temporal_patterns_{area_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: results/temporal_patterns_{area_name}.png")
    else:
        print("  No valid NDVI data available for plotting")

def analyze_data_quality(area_name: str, bbox: List[float]):
    """Analyze data quality - cloud cover and missing values (PNG only)"""
    
    collection = get_s2_collection(bbox, config.START_DATE, config.END_DATE, 100)
    
    # Get cloud cover distribution
    try:
        cloud_list = collection.aggregate_array('CLOUDY_PIXEL_PERCENTAGE').getInfo()
        cloud_cover = [float(c) for c in cloud_list if c is not None]
    except Exception as e:
        print(f"   Error getting cloud cover: {e}")
        cloud_cover = []
    
    # Monthly availability
    months = ['June', 'July', 'August', 'September']
    monthly_counts = []
    
    for month_idx in range(6, 10):
        try:
            start_month = f"{config.YEAR}-{month_idx:02d}-01"
            if month_idx == 9:
                end_month = f"{config.YEAR}-10-01"
            else:
                end_month = f"{config.YEAR}-{month_idx+1:02d}-01"
            
            filtered = collection.filterDate(start_month, end_month)
            count = filtered.size().getInfo()
            monthly_counts.append(count)
        except Exception as e:
            monthly_counts.append(0)
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    if cloud_cover:
        axes[0].hist(cloud_cover, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(x=config.CLOUD_PERCENT, color='red', linestyle='--', linewidth=2, 
                       label=f'Threshold: {config.CLOUD_PERCENT}%')
        axes[0].set_xlabel('Cloud Cover (%)', fontsize=12)
        axes[0].set_ylabel('Number of Images', fontsize=12)
        axes[0].set_title(f'Cloud Cover Distribution - {area_name}', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        print(f"  Total images: {len(cloud_cover)}")
        print(f"  Images with <{config.CLOUD_PERCENT}% cloud: {sum(c < config.CLOUD_PERCENT for c in cloud_cover)}")
    else:
        axes[0].text(0.5, 0.5, 'No cloud data available', ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title(f'Cloud Cover Distribution - {area_name}', fontsize=14, fontweight='bold')
    
    axes[1].bar(months, monthly_counts, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Month', fontsize=12)
    axes[1].set_ylabel('Number of Images', fontsize=12)
    axes[1].set_title(f'Monthly Data Availability - {area_name}', fontsize=14, fontweight='bold')
    
    for i, count in enumerate(monthly_counts):
        axes[1].text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=10)
    
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'results/data_quality_{area_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: results/data_quality_{area_name}.png")

def create_composite_stats(area_name: str, bbox: List[float]):
    """Create false-color composite statistics and save as PNG only"""
    try:
        # Get median composite
        collection = get_s2_collection(bbox, config.START_DATE, config.END_DATE, config.CLOUD_PERCENT)
        composite = collection.median()
        
        # Calculate statistics for each band
        bands = ['B2', 'B3', 'B4', 'B8']
        band_names = ['Blue', 'Green', 'Red', 'NIR']
        stats = {}
        
        for band, name in zip(bands, band_names):
            band_stats = composite.select(band).reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    reducer2=ee.Reducer.stdDev(), sharedInput=True
                ),
                geometry=ee.Geometry.BBox(*bbox),
                scale=30,
                maxPixels=1e6
            ).getInfo()
            
            stats[name] = {
                'mean': band_stats.get('mean', 0),
                'std': band_stats.get('stdDev', 0)
            }
        
        # Calculate NDVI statistics
        ndvi = composite.normalizedDifference(['B8', 'B4']).rename('NDVI')
        
        ndvi_stats = ndvi.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                reducer2=ee.Reducer.stdDev(), sharedInput=True
            ),
            geometry=ee.Geometry.BBox(*bbox),
            scale=30,
            maxPixels=1e6
        ).getInfo()
        
        ndvi_mean = ndvi_stats.get('mean', 0)
        ndvi_std = ndvi_stats.get('stdDev', 0)
        
        # Create bar chart of band means
        band_means = [stats[band]['mean'] for band in band_names]
        band_stds = [stats[band]['std'] for band in band_names]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Band reflectance plot
        bars = axes[0].bar(band_names, band_means, yerr=band_stds, 
                          color=['blue', 'green', 'red', 'darkgreen'], 
                          alpha=0.7, capsize=5)
        axes[0].set_title(f'Sentinel-2 Band Reflectance - {area_name}', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Band', fontsize=12)
        axes[0].set_ylabel('Mean Reflectance', fontsize=12)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        for bar, mean in zip(bars, band_means):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
        
        # NDVI summary plot
        axes[1].barh(['NDVI'], [ndvi_mean], xerr=ndvi_std, color='green', alpha=0.7, 
                    error_kw={'ecolor': 'black', 'capsize': 10})
        axes[1].set_xlim(-0.2, 1.0)
        axes[1].axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
        axes[1].axvline(x=0.33, color='orange', linestyle='--', alpha=0.5, label='Low vegetation')
        axes[1].axvline(x=0.66, color='darkgreen', linestyle='--', alpha=0.5, label='Dense vegetation')
        axes[1].set_title(f'Mean NDVI - {area_name} (2021)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('NDVI Value', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(f'results/band_ndvi_{area_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: results/band_ndvi_{area_name}.png")
        
    except Exception as e:
        print(f"  ⚠️ Could not create composite stats: {e}")

def main():
    """Main exploration function"""
    print("\n" + "="*70)
    print("DATA EXPLORATION - Part 1")
    print("="*70)
    
    # Initialize GEE
    if not init_gee(config.GEE_PROJECT):
        print("Failed to initialize GEE")
        return
    
    os.makedirs('results', exist_ok=True)
    
    for area_name, settings in config.STUDY_AREAS.items():
        print(f"\n{'='*50}")
        print(f"📊 Analyzing {area_name}")
        print(f"{'='*50}")
        
        bbox = settings['bbox']
        
        # 1. Class distribution
        print("\n1. Class Distribution Analysis...")
        try:
            dist_df = get_cdl_class_distribution(area_name, bbox, config.YEAR)
            plot_class_distribution(area_name, dist_df)
        except Exception as e:
            print(f"   Error: {e}")
        
        # 2. Temporal patterns
        print("\n2. Temporal Patterns Analysis...")
        try:
            analyze_temporal_patterns(area_name, bbox)
        except Exception as e:
            print(f"   Error: {e}")
        
        # 3. Data quality (PNG only)
        print("\n3. Data Quality Analysis...")
        try:
            analyze_data_quality(area_name, bbox)
        except Exception as e:
            print(f"   Error: {e}")
        
        # 4. Composite statistics (PNG only)
        print("\n4. Creating Composite Statistics...")
        try:
            create_composite_stats(area_name, bbox)
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n" + "="*70)
    print("✅ EXPLORATION COMPLETE! Check the 'results/' folder.")
    print("\n📸 PNG files generated:")
    print("  • class_distribution_{area}.png")
    print("  • temporal_patterns_{area}.png")
    print("  • data_quality_{area}.png")
    print("  • band_ndvi_{area}.png")
    print("="*70)

if __name__ == "__main__":
    main()