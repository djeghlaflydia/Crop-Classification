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

# Model configuration (from the paper)
MODEL_CONFIG = {
    "d_model": 64,
    "n_stages": 3,
    "nhead": 4,
    "kernel_size": 3,
    "dropout": 0.1
}

# Training configuration
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 10  # Early stopping
NUM_WORKERS = 4

# Data configuration
MAX_SAMPLES_PER_AREA = 500  # Number of sample points per area
TEMPORAL_RESOLUTION = 36  # 36 x 10-day composites = 360 days
RESOLUTION_METERS = 20  # Spatial resolution for sampling