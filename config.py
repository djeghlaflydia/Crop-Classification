# config.py
DATA_DIR = "data"

# Study areas defined as [min_lon, min_lat, max_lon, max_lat]
STUDY_AREAS = {
    "California": {
        "bbox": [-120.5, 36.5, -119.5, 37.5],  # Near Fresno
        "point": [-120.0, 37.0]
    },
    "Arkansas": {
        "bbox": [-91.8, 34.5, -90.8, 35.5],  # Near Grand Prairie
        "point": [-91.3, 35.0]
    }
}

DATE_RANGE = "2020-01-01/2020-12-31"
CLOUD_PERCENT = 20

# STAC API Settings
STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"
CDL_COLLECTION = "usda-cdl"