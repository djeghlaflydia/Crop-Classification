import os
import sys
import numpy as np
import stackstac
import dask
from sklearn.model_selection import train_test_split

# ==============================
# CONFIG
# ==============================
os.environ["GDAL_HTTP_MAX_RETRY"] = "5"
os.environ["GDAL_HTTP_RETRY_DELAY"] = "2"
dask.config.set(scheduler="threads")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from scripts.api_access import get_stac_data


# ==============================
# FILTER IMAGES
# ==============================
def filter_items(items, max_items=30):
    return sorted(items, key=lambda x: x.properties["eo:cloud_cover"])[:max_items]


# ==============================
# CLOUD MASK
# ==============================
def mask_clouds(stack):
    scl = stack.sel(band="SCL")

    cloud_classes = [3, 8, 9, 10, 11]

    mask = None
    for c in cloud_classes:
        cond = (scl == c)
        mask = cond if mask is None else (mask | cond)

    return stack.where(~mask)


# ==============================
# COMPOSITE S2
# ==============================
def create_composite(stack):
    stack = stack.chunk({"time": -1})
    return stack.median(dim="time", skipna=True)


# ==============================
# ALIGN CDL → S2 GRID
# ==============================
def align_cdl_to_s2(s2_cube, cdl_cube):
    ref = s2_cube.isel(band=0)
    return cdl_cube.interp_like(ref, method="nearest")


# ==============================
# EXTRACTION X + Y
# ==============================
def extract_pixels_with_labels(s2_cube, cdl_cube, num_samples=2000):

    X_data = s2_cube.values  # (band, y, x)
    Y_data = cdl_cube.values  # (y, x)

    bands, h, w = X_data.shape

    X_pixels = X_data.reshape(bands, -1).T
    Y_pixels = Y_data.reshape(-1)

    # enlever NaN dans X
    mask = ~np.isnan(X_pixels).any(axis=1)

    X_pixels = X_pixels[mask]
    Y_pixels = Y_pixels[mask]

    if len(X_pixels) == 0:
        raise ValueError("No valid pixels")

    idx = np.random.choice(len(X_pixels), min(num_samples, len(X_pixels)), replace=False)

    return X_pixels[idx], Y_pixels[idx]


# ==============================
# NORMALISATION
# ==============================
def normalize(X):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    return (X - mean) / (std + 1e-6)


# ==============================
# MAIN
# ==============================
def main():

    areas = [
        ("California", 32611),
        ("Arkansas", 32615)
    ]

    bands = ["B02", "B03", "B04", "B08"]

    for area, epsg in areas:

        print(f"\n=== Processing {area} ===")

        bbox = config.STUDY_AREAS[area]['bbox']

        # =========================
        # 1. LOAD SENTINEL-2
        # =========================
        items = get_stac_data(area, config.S2_COLLECTION, bbox, config.DATE_RANGE)
        items = filter_items(items)

        print(f"S2 images utilisées: {len(items)}")

        s2_stack = stackstac.stack(
            items,
            assets=bands + ["SCL"],
            bounds_latlon=bbox,
            epsg=epsg,
            resolution=20,
            chunksize={"time": 1, "x": 128, "y": 128}
        ).astype("float32")

        # =========================
        # 2. CLOUD MASK
        # =========================
        print("Masking clouds...")
        s2_stack = mask_clouds(s2_stack)
        s2_stack = s2_stack.persist()

        # =========================
        # 3. COMPOSITE S2
        # =========================
        print("Creating S2 composite...")
        s2_cube = create_composite(s2_stack)
        s2_cube = s2_cube.sel(band=bands).compute()

        print("S2 shape:", s2_cube.shape)

        # =========================
        # 4. LOAD CDL
        # =========================
        print("Loading CDL labels...")

        cdl_items = get_stac_data(area, config.CDL_COLLECTION, bbox, config.DATE_RANGE)

        if len(cdl_items) == 0:
            raise ValueError("No CDL data found")

        cdl_stack = stackstac.stack(
            cdl_items,
            bounds_latlon=bbox,
            epsg=epsg,
            resolution=30
        )

        # composite CDL
        cdl_cube = cdl_stack.median(dim="time").compute()

        # 🔥 FIX IMPORTANT : garder UNE seule bande
        if "band" in cdl_cube.dims:
            cdl_cube = cdl_cube.isel(band=0)

        cdl_cube = cdl_cube.squeeze()

        # 🔥 ALIGNEMENT FINAL
        cdl_cube = align_cdl_to_s2(s2_cube, cdl_cube)

        print("CDL aligned shape:", cdl_cube.shape)

        # =========================
        # 5. EXTRACTION X + Y
        # =========================
        print("Extracting pixels + labels...")
        X, Y = extract_pixels_with_labels(s2_cube, cdl_cube)

        print("X:", X.shape)
        print("Y:", Y.shape)

        # =========================
        # 6. NORMALISATION
        # =========================
        X = normalize(X)

        # =========================
        # 7. SPLIT
        # =========================
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )

        # =========================
        # 8. SAVE
        # =========================
        np.save(f"X_train_{area}.npy", X_train)
        np.save(f"X_test_{area}.npy", X_test)
        np.save(f"Y_train_{area}.npy", Y_train)
        np.save(f"Y_test_{area}.npy", Y_test)

        print(f" DONE - {area}")


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    main()