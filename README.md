# Crop Classification using Multi-Source Satellite Data

This project implements a lightweight CNN-Transformer framework for pixel-based crop mapping using time-series Sentinel-2 imagery and USDA Cropland Data Layer (CDL) labels. It focuses on large-scale monitoring of crop types (wheat, rice, soybean, etc.) in the United States, specifically in **California** and **Arkansas**.

## Key Features

- **Direct API Access**: Unlike traditional workflows that require downloading massive `.SAFE` files, this project uses the **Microsoft Planetary Computer STAC API** to fetch data on-the-fly.
- **Lazy Loading**: Leverages `stackstac` and `xarray` for efficient memory management. Data is only loaded into memory when required for computation or visualization.
- **Multi-Source Integration**: Aligns Sentinel-2 Level-2A surface reflectance with land cover ground truth (USDA CDL).
- **Time-Series Analysis**: Captures the phenological development of vegetation throughout the growing season (2020).

## Project Structure

```text
Crop-Classification/
├── config.py              # Central configuration (BBOX, Collections, Dates)
├── requirements.txt       # Project dependencies
├── data/                  # Placeholder for any local artifacts
├── notebooks/             # Primary presentation and exploration
│   └── exploration.ipynb  # Main Jupyter notebook for the final report
├── results/               
│   └── plots/             # Generated NDVI and class distribution plots
└── scripts/               # Core logic and reproducible scripts
    ├── api_access.py      # STAC query and DataCube initialization
    └── explore_data.py    # Automated visualization and analysis
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/djeghlaflydia/Crop-Classification.git
   cd Crop-Classification
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Exploration
To generate the NDVI time-series and class distribution plots for both study areas:
```bash
python scripts/explore_data.py
```
Outputs will be saved in the `results/plots/` directory.

### 2. Interactive Analysis
Open the Jupyter notebook for a detailed step-by-step walkthrough of the data acquisition and phenology analysis:
```bash
jupyter notebook notebooks/exploration.ipynb
```

## Methodology

### Study Areas
- **California**: Focused on the Central Valley region (near Fresno), characterized by high crop diversity (almonds, grapes, etc.) (378 time steps, 4 bands, ~11k x 9k pixels).
- **Arkansas**: Focused on the Grand Prairie region, known for extensive rice and soybean cultivation (254 time steps, 4 bands, ~11k x 9k pixels).

The NDVI time-series show clear vegetation growth cycles, and the CDL distribution captures a diverse range of crop types, providing a solid foundation for the classification model.

### Data Processing
The project utilizes **Sentinel-2 Level-2A** data (Atmospherically corrected). We compute the **NDVI (Normalized Difference Vegetation Index)** using Red (B04) and NIR (B08) bands to monitor crop growth cycles. Label data is sourced from the **USDA NASS Cropland Data Layer (CDL)** at 30m resolution, resampled to match Sentinel-2's 10m grid.
