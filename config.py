# config.py
import os

# Data directories
DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Study areas with their bounding boxes and UTM zones
STUDY_AREAS = {
    "California": {
        "bbox": [-120.175, 36.725, -120.125, 36.775],  # ~5 km x 5 km
        "point": [-120.15, 36.75],
        "epsg": 32611,
        "state": "California",
        "name": "California"
    },
    "Arkansas": {
        "bbox": [-91.475, 34.825, -91.425, 34.875],    # ~5 km x 5 km
        "point": [-91.45, 34.85],
        "epsg": 32615,
        "state": "Arkansas",
        "name": "Arkansas"
    }
}

# Time parameters
YEAR = 2021
START_DATE = f"{YEAR}-06-01"  # Growing season start
END_DATE = f"{YEAR}-09-30"    # Growing season end
CLOUD_PERCENT = 20  # Max cloud cover percentage

# GEE Settings
GEE_PROJECT = "crop-classification-2021"  # Your GEE project

# Collection names
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CDL_COLLECTION = "USDA/NASS/CDL"

# Bands to extract from Sentinel-2
S2_BANDS = ["B2", "B3", "B4", "B8"]  # Blue, Green, Red, NIR
S2_BAND_NAMES = ["blue", "green", "red", "nir"]

# Model configuration (AMÉLIORÉ)
MODEL_CONFIG = {
    "d_model": 128,      # Augmenté: 64 → 128
    "n_stages": 4,       # Augmenté: 3 → 4
    "nhead": 8,          # Augmenté: 4 → 8
    "kernel_size": 5,    # Augmenté: 3 → 5
    "dropout": 0.2
}

# Training configuration (AUGMENTÉ)
BATCH_SIZE = 32
EPOCHS = 200           # Changé: 50 → 200
LEARNING_RATE = 0.001
PATIENCE = 20          # Changé: 10 → 20
NUM_WORKERS = 0

# Data configuration
MAX_SAMPLES_PER_AREA = 500
TEMPORAL_RESOLUTION = 36
RESOLUTION_METERS = 20