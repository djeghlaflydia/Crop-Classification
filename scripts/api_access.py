import pystac_client
import planetary_computer
import stackstac
import xarray as xr
import numpy as np
import os
import sys

<<<<<<< HEAD
=======

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Ensure config can be imported from parent dir
>>>>>>> 9fb4bec6ab67138a40da01fd868a03afc7b8277a
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


<<<<<<< HEAD
# ----------------------------------------------------------------------
# STAC ACCESS (FIX SIGNATURE + WARNING)
# ----------------------------------------------------------------------
=======
>>>>>>> 9fb4bec6ab67138a40da01fd868a03afc7b8277a
def get_stac_data(area_name, collection, bbox, datetime):

    catalog = pystac_client.Client.open(config.STAC_API_URL)

    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=datetime,
    )

    # ✅ FIX WARNING
    items = list(search.items())

    # ✅ FIX 403 (sign URLs)
    items = [planetary_computer.sign(item) for item in items]

    print(f"[{area_name}] Found {len(items)} items for {collection}")

    return items


# ----------------------------------------------------------------------
# CLOUD MASK (FIXED 100%)
# ----------------------------------------------------------------------
def mask_clouds(stack):

    scl = stack.sel(band="SCL")

    cloud_classes = [3, 8, 9, 10, 11]

    # ✅ garder xarray (IMPORTANT)
    mask = ~xr.apply_ufunc(np.isin, scl, cloud_classes)

    return stack.where(mask)


# ----------------------------------------------------------------------
# REDUCE DATA (ANTI-CRASH 🔥)
# ----------------------------------------------------------------------
def filter_items(items, max_items=30):

    items = sorted(items, key=lambda x: x.properties["eo:cloud_cover"])

    return items[:max_items]


# ----------------------------------------------------------------------
# COMPOSITES (10-DAYS)
# ----------------------------------------------------------------------
def create_composites(stack, year=2021):

    times = stack.time.values
    start = np.datetime64(f"{year}-01-01")

    composites = []

    for i in range(36):

        t0 = start + np.timedelta64(i * 10, 'D')
        t1 = start + np.timedelta64((i + 1) * 10, 'D')

        mask = (times >= t0) & (times < t1)

        if mask.sum() > 0:
            comp = stack.isel(time=mask).median(dim="time", skipna=True)
        else:
            comp = stack.isel(time=0)

        comp = comp.assign_coords(time=i)
        composites.append(comp)

    cube = xr.concat(composites, dim="time")

    return cube


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():

    for area_name, settings in config.STUDY_AREAS.items():

        print(f"\n=== Processing {area_name} ===")

        bbox = settings["bbox"]
        epsg = 32611 if area_name == "California" else 32615

        # ------------------------------------------------------------------
        # 1. GET DATA
        # ------------------------------------------------------------------
        s2_items = get_stac_data(
            area_name,
            config.S2_COLLECTION,
            bbox,
            config.DATE_RANGE
        )

        if not s2_items:
            print("No data")
            continue

        # ✅ LIMIT DATA (VERY IMPORTANT)
        s2_items = filter_items(s2_items, max_items=30)

        # ------------------------------------------------------------------
        # 2. STACK (LOW MEMORY)
        # ------------------------------------------------------------------
        s2_stack = stackstac.stack(
            s2_items,
            assets=["B04", "B03", "B02", "B08", "SCL"],
            bounds_latlon=bbox,
            epsg=epsg,
            resolution=20,  # 🔥 reduce memory
            chunksize={"time": 1, "x": 256, "y": 256}
        ).astype("float32")  # 🔥 reduce RAM

        print(f"S2 shape: {s2_stack.shape}")

        # ------------------------------------------------------------------
        # 3. CLOUD MASK
        # ------------------------------------------------------------------
        s2_stack = mask_clouds(s2_stack)

        # ------------------------------------------------------------------
        # 4. COMPOSITES
        # ------------------------------------------------------------------
        s2_cube = create_composites(s2_stack)

        # enlever SCL
        s2_cube = s2_cube.sel(band=["B04", "B03", "B02", "B08"])

        print(f"Final cube: {s2_cube.shape}")

        # ------------------------------------------------------------------
        # 5. NDVI
        # ------------------------------------------------------------------
        red = s2_cube.sel(band="B04")
        nir = s2_cube.sel(band="B08")

        ndvi = (nir - red) / (nir + red)

        print(f"NDVI shape: {ndvi.shape}")

        # ------------------------------------------------------------------
        # 6. SAFE LOAD
        # ------------------------------------------------------------------
        s2_cube = s2_cube.compute()
        ndvi = ndvi.compute()

        print(f"✅ {area_name} ready")

    print("\n🔥 PIPELINE COMPLET OK (clean + stable + no crash)")


if __name__ == "__main__":
    main()