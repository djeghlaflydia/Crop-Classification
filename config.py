# config.py
DATA_DIR = "data"

<<<<<<< HEAD
# Bounding boxes très petites (~5 km x 5 km) pour des tests rapides
STUDY_AREAS = {
    "California": {
        "bbox": [-120.175, 36.725, -120.125, 36.775],   # ~5 km x 5 km
        "point": [-120.15, 36.75]
    },
    "Arkansas": {
        "bbox": [-91.475, 34.825, -91.425, 34.875],     # ~5 km x 5 km
        "point": [-91.45, 34.85]
    }
}

# Période (peut être 2021 si vous utilisez l’année 2021)
DATE_RANGE = "2021-01-01/2021-12-31"   # ← modifié pour 2021 (en accord avec l’article)
CLOUD_PERCENT = 20

# STAC API Settings
STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"
CDL_COLLECTION = "usda-cdl"
=======
# Study areas defined as [min_lon, min_lat, max_lon, max_lat]
STUDY_AREAS = {
    "California": {
        "bbox": [-120.5, 36.5, -119.5, 37.5],  # Near Fresno
        "point": [-120.0, 37.0]
    },
    "Arkansas": {
        "bbox": [-91.8, 34.5, -90.8, 35.5],    # Near Grand Prairie
        "point": [-91.3, 35.0]
    }
}

DATE_RANGE = "2021-01-01/2021-12-31"
CLOUD_PERCENT = 20

STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"
CDL_COLLECTION = "usda-cdl"
>>>>>>> 9fb4bec6ab67138a40da01fd868a03afc7b8277a
