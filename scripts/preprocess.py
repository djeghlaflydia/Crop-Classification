import os
import sys
import numpy as np
import xarray as xr
import stackstac
from sklearn.model_selection import StratifiedShuffleSplit
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from scripts.api_access import get_stac_data

# ----------------------------------------------------------------------
# Fonction pour créer les composites décadaires (médiane sur 10 jours)
# ----------------------------------------------------------------------
def get_10day_composites(area_name, bbox, bands, year=2020, epsg=32611):
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
# Échantillonnage de points aléatoires à partir du CDL (avec gestion d'erreurs)
# ----------------------------------------------------------------------
def sample_points_vectorized(da_s2, da_cdl, num_points=10000, seed=42):
    """Version vectorisée plus rapide mais peut échouer sur certaines tuiles."""
    np.random.seed(seed)
    cdl_data = da_cdl.sel(band='cropland').values
    valid_y, valid_x = np.where(cdl_data > 0)
    if len(valid_y) == 0:
        raise ValueError("Aucun pixel CDL valide dans la zone")

    n = min(num_points, len(valid_y))
    idx = np.random.choice(len(valid_y), n, replace=False)
    y_idx, x_idx = valid_y[idx], valid_x[idx]
    labels = cdl_data[y_idx, x_idx]

    x_coords = da_cdl.coords['x'].values
    y_coords = da_cdl.coords['y'].values

    # Récupérer les coordonnées géographiques
    x_vals = x_coords[x_idx]
    y_vals = y_coords[y_idx]

    # Extraction vectorisée (peut échouer si les données ne sont pas alignées)
    X = da_s2.sel(x=xr.DataArray(x_vals, dims='point'),
                  y=xr.DataArray(y_vals, dims='point'),
                  method='nearest').values
    # X shape: (point, time, band)
    X = np.transpose(X, (1, 0, 2))   # remettre en (point, time, band)
    return X, labels

def sample_points_loop(da_s2, da_cdl, num_points=10000, seed=42, max_attempts=None):
    """Version en boucle, plus lente mais plus robuste."""
    np.random.seed(seed)
    cdl_data = da_cdl.sel(band='cropland').values
    valid_y, valid_x = np.where(cdl_data > 0)
    if len(valid_y) == 0:
        raise ValueError("Aucun pixel CDL valide dans la zone")

    # On tire 2 fois plus de candidats pour compenser les échecs
    n_to_try = min(int(num_points * 2), len(valid_y))
    idx = np.random.choice(len(valid_y), n_to_try, replace=False)
    y_idx = valid_y[idx]
    x_idx = valid_x[idx]
    labels = cdl_data[y_idx, x_idx]

    x_coords = da_cdl.coords['x'].values
    y_coords = da_cdl.coords['y'].values

    X_list = []
    y_list = []
    for i in range(len(y_idx)):
        x_val = x_coords[x_idx[i]]
        y_val = y_coords[y_idx[i]]
        try:
            ts = da_s2.sel(x=x_val, y=y_val, method='nearest').values
            X_list.append(ts)
            y_list.append(labels[i])
        except Exception as e:
            # Silently skip
            continue
        if len(X_list) >= num_points:
            break
        if max_attempts and i >= max_attempts:
            break

    if len(X_list) == 0:
        raise ValueError("Aucun point valide extrait après plusieurs tentatives.")
    if len(X_list) < num_points:
        print(f"Attention : seulement {len(X_list)} points valides sur {num_points} demandés.")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    return X, y

def sample_points(da_s2, da_cdl, num_points=10000, seed=42):
    """Tente d'abord la version vectorisée, puis bascule en boucle en cas d'erreur."""
    try:
        X, y = sample_points_vectorized(da_s2, da_cdl, num_points, seed)
        print("Extraction vectorisée réussie.")
        return X, y
    except Exception as e:
        print(f"Erreur extraction vectorisée : {e}. Fallback à la boucle.")
        return sample_points_loop(da_s2, da_cdl, num_points, seed)

# ----------------------------------------------------------------------
# Fonction principale
# ----------------------------------------------------------------------
def main():
    for area_name, epsg in [('California', 32611), ('Arkansas', 32615)]:
        print(f"\n=== Traitement de {area_name} ===")
        bbox = config.STUDY_AREAS[area_name]['bbox']
        bands = ['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12']

        # 1. Données Sentinel-2
        try:
            s2_cube = get_10day_composites(area_name, bbox, bands, year=2020, epsg=epsg)
            print(f"{area_name} S2 cube shape: {s2_cube.shape}")
        except Exception as e:
            print(f"Erreur Sentinel-2 pour {area_name} : {e}")
            continue

        # 2. Données CDL
        cdl_items = get_stac_data(area_name, config.CDL_COLLECTION, bbox, "2020-01-01/2020-12-31")
        if not cdl_items:
            print(f"Aucune donnée CDL pour {area_name} avec {config.CDL_COLLECTION}")
            cdl_items = get_stac_data(area_name, "usda-cdl", bbox, "2020-01-01/2020-12-31")
            if not cdl_items:
                print(f"Échec total pour {area_name}. Passer à la suivante.")
                continue

        cdl_cube = stackstac.stack(cdl_items, assets=['cropland'], bounds_latlon=bbox, epsg=epsg, resolution=30)
        cdl_cube = cdl_cube.isel(time=0)  # une seule date

        # 3. Échantillonnage avec 1000 points (rapide pour test)
        X, y = sample_points(s2_cube, cdl_cube, num_points=1000)   # <--- MODIFIÉ : 1000 points
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