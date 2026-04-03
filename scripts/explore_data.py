import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import stackstac

# Parent directory for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from scripts.api_access import get_stac_data

CDL_MAP = {
    1: 'Corn',
    5: 'Soybeans',
    21: 'Barley',
    24: 'Winter Wheat',
    42: 'Dry Beans',
    61: 'Fallow/Idle Cropland',
    111: 'Open Water',
    121: 'Developed/Open Space',
    190: 'Woody Wetlands',
    195: 'Herbaceous Wetlands'
}

def analyze_area(area_name, epsg):
    print(f"\n--- Analyzing {area_name} ---")
    bbox = config.STUDY_AREAS[area_name]["bbox"]
    
    # Small patch for visualization
    lon, lat = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
    patch_bbox = [lon-0.01, lat-0.01, lon+0.01, lat+0.01]

    # Fetch Data
    s2_items = get_stac_data(area_name, config.S2_COLLECTION, patch_bbox, config.DATE_RANGE)
    s2_items = [i for i in s2_items if i.properties.get("eo:cloud_cover", 100) < 15]
    
    s2_stack = stackstac.stack(s2_items, assets=["B04", "B08"], bounds_latlon=patch_bbox, epsg=epsg, resolution=10)
    res_s2 = s2_stack.resample(time="10D", origin="2021-01-01").median(dim="time").compute()

    cdl_items = get_stac_data(area_name, config.CDL_COLLECTION, patch_bbox, "2021-01-01/2021-12-31")
    cdl_stack = stackstac.stack(cdl_items, assets=["cropland"], bounds_latlon=patch_bbox, epsg=epsg, resolution=10)
    res_cdl = cdl_stack.isel(time=0).compute()

    # NDVI
    red, nir = res_s2.sel(band="B04"), res_s2.sel(band="B08")
    ndvi = (nir - red) / (nir + red + 1e-8)

    # 1. Plot Class Distribution
    plt.figure(figsize=(10, 6))
    vals, counts = np.unique(res_cdl.values, return_counts=True)
    mask = vals > 0
    vals, counts = vals[mask], counts[mask]
    labels = [CDL_MAP.get(v, f"ID {v}") for v in vals]
    
    plt.bar(labels, counts, color='teal')
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Crop Class Distribution - {area_name} (2021)")
    plt.tight_layout()
    plt.savefig(f"results/class_dist_{area_name}.png")
    print(f"Saved class distribution for {area_name}")

    # 2. Plot Temporal Patterns per Class
    plt.figure(figsize=(12, 6))
    for val in vals:
        class_mask = (res_cdl == val).squeeze()
        # Mean NDVI for pixels of this class
        class_ndvi = ndvi.where(class_mask).mean(dim=["x", "y"])
        label = CDL_MAP.get(val, f"ID {val}")
        plt.plot(class_ndvi.time, class_ndvi.values, label=label, marker='.')

    plt.title(f"Phenology (Mean NDVI) by Crop Type - {area_name}")
    plt.xlabel("Date")
    plt.ylabel("NDVI")
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"results/phenology_{area_name}.png")
    print(f"Saved phenology plot for {area_name}")

    # 3. Missing Value Analysis
    plt.figure(figsize=(10, 4))
    # Count pixels that are NaN
    missing_ratio = (np.isnan(ndvi).sum(dim=["x", "y"]) / (ndvi.shape[1] * ndvi.shape[2])).values
    plt.bar(ndvi.time.values, missing_ratio, color='brown', alpha=0.6)
    plt.title(f"Data Gaps Over Time (NaN Ratio) - {area_name}")
    plt.ylabel("Ratio of missing pixels")
    plt.tight_layout()
    plt.savefig(f"results/missing_data_{area_name}.png")
    print(f"Saved gap analysis for {area_name}")

def main():
    if not os.path.exists("results"): 
        os.makedirs("results")
    
    analyze_area("Arkansas", 32615)
    analyze_area("California", 32611)

if __name__ == "__main__":
    main()
