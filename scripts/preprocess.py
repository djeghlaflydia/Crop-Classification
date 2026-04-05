import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import pystac_client
import planetary_computer
import sys
import os

# 🔥 Fix import config
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config


# ================================
# STAC SEARCH
# ================================
def get_items(collection, bbox):
    catalog = pystac_client.Client.open(config.STAC_API_URL)

    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=config.DATE_RANGE
    )

    return list(search.items())   # ✅ FIX WARNING


# ================================
# LOAD SENTINEL-2 + CLOUD MASK
# ================================
def load_s2(items, max_images=30):
    images = []

    print("Loading S2 images...")

    for item in items[:max_images]:
        try:
            item = planetary_computer.sign(item)

            with rasterio.open(item.assets["B04"].href) as red:
                r = red.read(1)

            with rasterio.open(item.assets["B03"].href) as green:
                g = green.read(1)

            with rasterio.open(item.assets["B02"].href) as blue:
                b = blue.read(1)

            with rasterio.open(item.assets["B08"].href) as nir:
                n = nir.read(1)

            img = np.stack([r, g, b, n], axis=0)

            # ⚠️ FIX IMPORTANT
            if np.all(img == 0):
                continue

            mask = (img > 0).all(axis=0)
            img[:, ~mask] = np.nan

            images.append(img)

        except Exception as e:
            print("Skipped image:", e)
            continue

    if len(images) == 0:
        raise ValueError("❌ Aucune image chargée (check internet ou bbox)")

    print(f"S2 images utilisées: {len(images)}")

    return np.nanmedian(images, axis=0)


# ================================
# LOAD CDL + ALIGN
# ================================
def load_cdl(items, target_shape, target_transform):
    item = planetary_computer.sign(items[0])

    with rasterio.open(item.assets["data"].href) as src:
        data = src.read(1)

        dst = np.zeros(target_shape, dtype=data.dtype)

        reproject(
            source=data,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=src.crs,
            resampling=Resampling.nearest
        )

    return dst


# ================================
# EXTRACT PIXELS
# ================================
def extract_pixels(s2, cdl, n_samples=2000):
    print("Extracting pixels + labels...")

    X = s2.reshape(s2.shape[0], -1).T
    Y = cdl.flatten()

    # Mask valid pixels
    mask = (~np.isnan(X).any(axis=1)) & (Y > 0)

    X = X[mask]
    Y = Y[mask]

    print(f"Pixels valides: {len(X)}")

    # Sample
    idx = np.random.choice(len(X), n_samples, replace=False)

    return X[idx], Y[idx]


# ================================
# MAIN
# ================================
def main():
    for name, area in config.STUDY_AREAS.items():
        print(f"\n=== Processing {name} ===")

        # Sentinel-2
        s2_items = get_items(config.S2_COLLECTION, area["bbox"])
        print(f"[{name}] Found {len(s2_items)} items for S2")

        s2 = load_s2(s2_items)
        print("S2 shape:", s2.shape)

        # CDL
        cdl_items = get_items(config.CDL_COLLECTION, area["bbox"])
        print(f"[{name}] Found {len(cdl_items)} items for CDL")

        ref = rasterio.open(
            planetary_computer.sign(cdl_items[0]).assets["data"].href
        )

        cdl = load_cdl(cdl_items, s2.shape[1:], ref.transform)
        print("CDL aligned shape:", cdl.shape)

        # Dataset
        X, Y = extract_pixels(s2, cdl)

        print("X:", X.shape)
        print("Y:", Y.shape)

        # Split
        n = len(X)
        train = int(0.7 * n)
        val = int(0.15 * n)

        np.save(f"X_train_{name}.npy", X[:train])
        np.save(f"y_train_{name}.npy", Y[:train])

        np.save(f"X_val_{name}.npy", X[train:train+val])
        np.save(f"y_val_{name}.npy", Y[train:train+val])

        np.save(f"X_test_{name}.npy", X[train+val:])
        np.save(f"y_test_{name}.npy", Y[train+val:])

        print(f"✅ DONE - {name}")


if __name__ == "__main__":
    main()