import os
import sys
import numpy as np
import xarray as xr
import stackstac
import pystac_client
import planetary_computer
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from scripts.api_access import get_stac_data

# ----------------------------------------------------------------------
# Fonction pour créer les composites décadaires (médiane sur 10 jours)
# ----------------------------------------------------------------------
def get_10day_composites(area_name, bbox, bands, year=2021, epsg=32611):
    """
    Retourne un DataArray (time=36, band, y, x) avec composites médians sur 10 jours.
    """
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    datetime_range = f"{start_date}/{end_date}"

    items = get_stac_data(area_name, config.S2_COLLECTION, bbox, datetime_range)
    items = [i for i in items if i.properties["eo:cloud_cover"] < config.CLOUD_PERCENT]

    if not items:
        raise ValueError(f"Aucune image Sentinel-2 sans nuage pour {area_name}")

    stack = stackstac.stack(items, assets=bands, bounds_latlon=bbox, epsg=epsg, resolution=10)
    times = stack.time.values

    start = np.datetime64(f"{year}-01-01")
    composites = []
    for i in range(36):
        t0 = start + np.timedelta64(i*10, 'D')
        t1 = start + np.timedelta64((i+1)*10, 'D')
        mask = (times >= t0) & (times < t1)
        if mask.sum() > 0:
            comp = stack.isel(time=mask).median(dim='time')
            comp = comp.assign_coords(time=i)
        else:
            ref = stack.isel(time=0)
            comp = xr.full_like(ref, np.nan).expand_dims(time=[i])
        composites.append(comp)

    da = xr.concat(composites, dim='time')
    return da

# ----------------------------------------------------------------------
# Échantillonnage vectorisé (rapide)
# ----------------------------------------------------------------------
def sample_points(da_s2, da_cdl, num_points=2000, seed=42):
    """
    Extrait des points aléatoires de manière vectorisée.
    """
    np.random.seed(seed)
    cdl_data = da_cdl.sel(band='cropland').values
    valid_y, valid_x = np.where(cdl_data > 0)
    if len(valid_y) == 0:
        raise ValueError("Aucun pixel CDL valide dans la zone")

    n_to_try = min(int(num_points * 1.2), len(valid_y))
    idx = np.random.choice(len(valid_y), n_to_try, replace=False)
    y_idx = valid_y[idx]
    x_idx = valid_x[idx]

    x_coords = da_cdl.coords['x'].values
    y_coords = da_cdl.coords['y'].values
    xs = x_coords[x_idx]
    ys = y_coords[y_idx]

    # Extraction vectorisée (une seule requête)
    try:
        ts = da_s2.sel(
            x=xr.DataArray(xs, dims='point'),
            y=xr.DataArray(ys, dims='point'),
            method='nearest'
        ).values  # shape (n_points, 36, n_bands)
    except Exception as e:
        print(f"Erreur extraction vectorisée : {e}. Fallback à la boucle.")
        # Fallback en boucle si ça échoue
        ts_list = []
        for xi, yi in zip(xs, ys):
            try:
                ts_list.append(da_s2.sel(x=xi, y=yi, method='nearest').values)
            except:
                continue
        if not ts_list:
            raise ValueError("Aucun point valide")
        ts = np.stack(ts_list, axis=0)
        # Labels correspondants
        labels = cdl_data[y_idx[:len(ts_list)], x_idx[:len(ts_list)]]
    else:
        labels = cdl_data[y_idx, x_idx]

    # Éliminer les points contenant des NaN
    nan_mask = np.isnan(ts).any(axis=(1,2))
    ts = ts[~nan_mask]
    labels = labels[~nan_mask]

    if len(ts) == 0:
        raise ValueError("Aucun point valide après filtrage des NaN")
    if len(ts) < num_points:
        print(f"Attention : seulement {len(ts)} points valides sur {num_points} demandés.")
    elif len(ts) > num_points:
        ts = ts[:num_points]
        labels = labels[:num_points]

    return ts, labels

# ----------------------------------------------------------------------
# Fonction principale
# ----------------------------------------------------------------------
def main():
    for area_name, epsg in [('California', 32611), ('Arkansas', 32615)]:
        print(f"\n=== Traitement de {area_name} (2021) ===")
        bbox = config.STUDY_AREAS[area_name]['bbox']
        bands = ['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12']

        # 1. Données Sentinel-2 (année 2021)
        try:
            s2_cube = get_10day_composites(area_name, bbox, bands, year=2021, epsg=epsg)
            print(f"{area_name} S2 cube shape: {s2_cube.shape}")
        except Exception as e:
            print(f"Erreur Sentinel-2 pour {area_name} : {e}")
            continue

        # 2. Données CDL (année 2021)
        cdl_items = get_stac_data(area_name, config.CDL_COLLECTION, bbox, "2021-01-01/2021-12-31")
        if not cdl_items:
            print(f"Aucune donnée CDL pour {area_name} avec {config.CDL_COLLECTION}")
            cdl_items = get_stac_data(area_name, "usda-cdl", bbox, "2021-01-01/2021-12-31")
            if not cdl_items:
                print(f"Échec total pour {area_name}. Passer à la suivante.")
                continue

        cdl_cube = stackstac.stack(cdl_items, assets=['cropland'], bounds_latlon=bbox, epsg=epsg, resolution=30)
        cdl_cube = cdl_cube.isel(time=0)  # une seule date

        # 3. Échantillonnage
        X, y = sample_points(s2_cube, cdl_cube, num_points=2000)
        print(f"Échantillons extraits : X shape {X.shape}, y shape {y.shape}")

        # Remplacer les NaN par 0 (comme dans l'article)
        X[np.isnan(X)] = 0.0

        # Créer le masque de présence (1 si toutes les bandes à une date sont non nulles)
        mask = (X != 0).all(axis=2).astype(np.float32)  # (n, 36)

        # 4. Split stratifié train/val/test
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_val_idx, test_idx = next(sss.split(X, y))

        X_train_val = X[train_val_idx]
        y_train_val = y[train_val_idx]
        mask_train_val = mask[train_val_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]
        mask_test = mask[test_idx]

        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(sss2.split(X_train_val, y_train_val))

        X_train = X_train_val[train_idx]
        y_train = y_train_val[train_idx]
        mask_train = mask_train_val[train_idx]

        X_val = X_train_val[val_idx]
        y_val = y_train_val[val_idx]
        mask_val = mask_train_val[val_idx]

        # 5. Normalisation
        mean = X_train.mean(axis=(0, 1), keepdims=True)
        std = X_train.std(axis=(0, 1), keepdims=True)
        X_train_norm = (X_train - mean) / (std + 1e-6)
        X_val_norm   = (X_val   - mean) / (std + 1e-6)
        X_test_norm  = (X_test  - mean) / (std + 1e-6)

        # 6. Sauvegarde
        np.save(f'X_train_{area_name}.npy', X_train_norm)
        np.save(f'y_train_{area_name}.npy', y_train)
        np.save(f'mask_train_{area_name}.npy', mask_train)

        np.save(f'X_val_{area_name}.npy', X_val_norm)
        np.save(f'y_val_{area_name}.npy', y_val)
        np.save(f'mask_val_{area_name}.npy', mask_val)

        np.save(f'X_test_{area_name}.npy', X_test_norm)
        np.save(f'y_test_{area_name}.npy', y_test)
        np.save(f'mask_test_{area_name}.npy', mask_test)

        print(f"Données sauvegardées pour {area_name}")

if __name__ == '__main__':
    main()