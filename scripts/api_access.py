import pystac_client
import planetary_computer
import stackstac
import xarray as xr
import os
import sys
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Ensure config can be imported from parent dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


def get_stac_data(area_name, collection, bbox, datetime):
    """Fetch STAC items and return a signed collection."""
    catalog = pystac_client.Client.open(
        config.STAC_API_URL,
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=datetime,
    )
    
    items = search.item_collection()
    print(f"[{area_name}] Found {len(items)} items for {collection}")
    return items

def main():
    for area_name, settings in config.STUDY_AREAS.items():
        print(f"\n--- Processing {area_name} ---")
        bbox = settings["bbox"]
        # Determine UTM EPSG based on longitude
        # CA is roughly UTM 11N (32611), AR is roughly UTM 15N (32615)
        epsg = 32611 if area_name == "California" else 32615
        
        # 1. Fetch Sentinel-2 Data
        s2_items = get_stac_data(
            area_name, 
            config.S2_COLLECTION, 
            bbox, 
            config.DATE_RANGE
        )
        
        # Filter for cloud cover
        s2_items = [item for item in s2_items if item.properties["eo:cloud_cover"] < config.CLOUD_PERCENT]
        print(f"[{area_name}] {len(s2_items)} items after cloud filtering (<{config.CLOUD_PERCENT}%)")

        if not s2_items:
            print(f"No items found for {area_name}")
            continue

        # 2. Fetch CDL Data
        cdl_items = get_stac_data(
            area_name,
            config.CDL_COLLECTION,
            bbox,
            "2020-01-01/2020-12-31"
        )

        # 3. Create DataCubes using stackstac
        # Sentinel-2
        s2_stack = stackstac.stack(
            s2_items, 
            assets=["B04", "B03", "B02", "B08"], 
            bounds_latlon=bbox,
            epsg=epsg,
            resolution=10
        )
        
        # CDL
        cdl_stack = stackstac.stack(
            cdl_items,
            assets=["cropland"],
            bounds_latlon=bbox,
            epsg=epsg,
            resolution=10
        )

        print(f"[{area_name}] Data Cubes Created (Lazy Loaded):")
        print(f"  Sentinel-2 Stack Shape: {s2_stack.shape} (Time, Band, Y, X)")
        print(f"  CDL Stack Shape: {cdl_stack.shape}")
        
        # Example Calculation (NDVI)
        red = s2_stack.sel(band="B04")
        nir = s2_stack.sel(band="B08")
        ndvi = (nir - red) / (nir + red)
        print(f"  NDVI Cube Shape: {ndvi.shape}")

    print("\nDirect API access successful for all study areas.")

if __name__ == "__main__":
    main()
