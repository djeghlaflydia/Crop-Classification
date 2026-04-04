# Crop Classification using Multi-Source Satellite Data

A lightweight CNN-Transformer framework for pixel-based crop mapping using time-series Sentinel-2 imagery and USDA Cropland Data Layer (CDL) labels.

## 📋 Overview

This project implements and reproduces the methodology described in the paper *"A lightweight CNN-Transformer network for pixel-based crop mapping using time-series Sentinel-2 imagery"*. The goal is to monitor crop types (wheat, rice, soybean, corn, etc.) over large agricultural areas in the United States, specifically in **California** and **Arkansas**.

Unlike traditional workflows that require downloading massive `.SAFE` files (hundreds of GB), this project uses **Google Earth Engine (GEE)** API to fetch and process data directly in the cloud, enabling efficient large-scale analysis.

## 🎯 Project Objectives (Part 1)

1. **Literature Review**: Analysis of the paper's methodology
2. **Dataset Acquisition**: Access Sentinel-2 and CDL data via GEE
3. **Data Exploration**: Visualize NDVI time-series, class distributions, temporal patterns
4. **Data Preprocessing**: Cloud filtering, interpolation, normalization, index extraction
5. **Model Implementation**: CNN-Transformer architecture training and evaluation

## 📊 Data Sources

| Data Type | Source | Access Method |
|-----------|--------|---------------|
| **Sentinel-2 L2A** | European Space Agency (ESA) | Google Earth Engine API |
| **Cropland Data Layer (CDL)** | USDA NASS | Google Earth Engine API |

### Study Areas

| Area | Bounding Box | UTM Zone | Key Crops |
|------|--------------|----------|-----------|
| **California** | `[-120.175, 36.725, -120.125, 36.775]` | 32611 | Wheat, Cotton, Alfalfa, Corn, Pasture |
| **Arkansas** | `[-91.475, 34.825, -91.425, 34.875]` | 32615 | Woody Wetlands, Soybeans, Rice, Cotton |

### Time Period

- **Growing Season**: June 1, 2021 - September 30, 2021
- **Cloud Filter**: Maximum 20% cloud cover

## 🏗️ Project Structure

```text
Crop-Classification/
├── config.py                 # Configuration (paths, bbox, model params)
├── requirements.txt          # Python dependencies
├── data/                     # Preprocessed training data (.npy files)
│   ├── X_train_California.npy
│   ├── X_val_California.npy
│   ├── X_test_California.npy
│   ├── y_train_California.npy
│   ├── class_info_California.json
│   └── ... (same for Arkansas)
├── results/                  # All outputs
│   ├── explore/              # Exploration PNG files
│   │   ├── California/
│   │   │   ├── class_distribution_California.png
│   │   │   ├── temporal_patterns_California.png
│   │   │   ├── data_quality_California.png
│   │   │   └── band_ndvi_California.png
│   │   └── Arkansas/
│   │       └── ...
│   └── train/                # Training outputs
│       ├── California/
│       │   ├── best_model.pth
│       │   ├── training_history.json
│       │   └── training_curves.png
│       └── Arkansas/
│           └── ...
└── scripts/                  # Core Python modules
    ├── api_access.py         # GEE connection & data retrieval
    ├── explore_data.py       # Data exploration & visualization
    ├── preprocess.py         # Data cleaning & preparation
    ├── model.py              # MCTNet architecture
    ├── train.py              # Model training with early stopping
    └── evaluate.py           # Final evaluation & metrics

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

3. **Authenticate with Google Earth Engine**:
   ```bash
   earthengine authenticate
   ```
   Follow the browser instructions to sign in with your Google account.

4. **Set your GEE project**:
   ```bash
   earthengine set_project your-project-id
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


Part 1 requires:
1. Literature Review:
   - Analysis of the paper methodology:
     • Data preparation
     • Temporal sampling strategy
     • Model architecture
     • Training procedure
     • Evaluation metrics

2. Dataset Acquisition:
   - Sentinel-2 data
   - Cropland Data Layer (CDL) data
   - Study areas: California and Arkansas

3. Data Exploration:
   - Visualization of time-series vegetation indices
   - Class distribution analysis
   - Temporal patterns of different crops
   - Detection of missing values and noise

4. Data Preprocessing:
   - Cloud filtering
   - Time-series interpolation
   - Normalization
   - Vegetation indices extraction
   - Alignment with crop labels

5. Model Implementation:
   - Reimplementation of the CNN-Transformer architecture from the paper
   - Model training
   - Evaluation of results
   - Comparison with the results reported in the paper




Où vont vraiment les données :
Donnée	Source réelle	Comment on y accède
Sentinel-2	Agence Spatiale Européenne (ESA)	Via Google Earth Engine
CDL (Cropland Data Layer)	USDA (Département Agriculture US)	Via Google Earth Engine