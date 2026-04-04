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
├── results/
│  ├── explore/
│  │   ├── California/
│  │   │   ├── class_distribution_California.png
│  │   │   ├── temporal_patterns_California.png
│  │   │   ├── data_quality_California.png
│  │   │   └── band_ndvi_California.png
│  │   └── Arkansas/
│  │       └── ...
│  ├── train/
│  │   ├── California/
│  │   │   ├── best_model.pth
│  │   │   ├── training_history.json
│  │   │   ├── training_curves.png
│  │   │   ├── X_test.npy
│  │   │   ├── y_test.npy
│  │   │   └── mask_test.npy
│  │   └── Arkansas/
│  │       └── ...
│  └── evaluate/                    ← MANQUE DANS TON README
│  │   ├── California/
│  │   │   ├── confusion_matrix.png
│  │   │   ├── per_class_metrics.png
│  │   │   ├── evaluation_results.json
│  │   │   └── summary.txt
│  │   ├── Arkansas/
│  │   │   └── ...
│  │   ├── comparison_with_paper.json
│  │   └── comparison_summary.txt
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

## 📈 Pipeline Execution

### Step 1: Data Exploration

```bash
python scripts/explore_data.py
```
Outputs will be saved in the `results/plots/` directory.

| File                            | Description                         |
|---------------------------------|-------------------------------------|
| `class_distribution_{area}.png` | Distribution of crop types from CDL |
| `temporal_patterns_{area}.png`  | NDVI time-series for top 5 crop classes |
| `data_quality_{area}.png`       | Cloud cover analysis & monthly availability |
| `band_ndvi_{area}.png`          | Sentinel-2 band reflectance & NDVI statistics |

**Output statistics**:

| Area       | Classes | Samples | Timesteps | Bands |
|------------|---------|---------|-----------|-------|
| California | 43      | 2,111   | 47        | 7     |
| Arkansas   | 26      | 664     | 18        | 7     |

---

### Step 2: Data Preprocessing
```bash
python scripts/preprocess.py
```
## What It Does
- Samples points from CDL (proportional to class distribution)
- Extracts Sentinel-2 time-series for each point
- Calculates vegetation indices:
  - NDVI
  - EVI
  - NDWI
- Interpolates missing values (linear interpolation)
- Normalizes data (mean / std standardization)
- Splits data into:
  - Train (70%)
  - Validation (15%)
  - Test (15%)
- Saves processed data as `.npy` files in `data/`

---

### Step 3: Model Training
```bash
python scripts/train.py
```
### 🧠 Model Architecture (MCTNet)

- **Input**: Time-series (47 / 18 timesteps × 7 bands)
- **Positional Encoding**: ALPE (Adaptive Learned Positional Encoding)
- **Feature Extraction**: 4 stages of CNN-Transformer fusion
- **Classification**: MLP with dropout

---

### ⚙️ Training Configuration

| Parameter             | Value            |
|-----------------------|------------------|
| Epochs                | 200 |
| Batch size            | 32  |
| Learning rate         | 0.001 (Cosine annealing) |
| Optimizer             | AdamW with weight decay |
| Loss                  | Cross-entropy with class weighting |
| Early stopping        | 20 epochs patience |
| d_model               | 128 |
| n_stages              | 4   |
| nhead                 | 8   |
| kernel_size           | 5   |
| dropout               | 0.2 |

| Area | Classes | Train Samples | Val Samples | Best Val Acc | Best Balanced Acc |
|------|---------|---------------|-------------|--------------|-------------------|
| California | 43 | 1,477 | 317 | 14.83% | 11.37% |
| Arkansas | 26 | 464 | 100 | 15.00% | 11.70% |

### 📁 Outputs (`results/train/{area}/`)

- `best_model.pth` → Best model weights  
- `training_history.json` → Loss and accuracy per epoch  
- `training_curves.png` → 4-panel training visualization  

---

### Step 4: Model Evaluation
```bash
python scripts/evaluate.py
```
### Metrics Computed
- Overall Accuracy (OA)
- Cohen’s Kappa
- F1-Score (Macro & Weighted)
- Per-class Precision, Recall, F1
- Confusion Matrix (counts & normalized)

### Outputs (`results/evaluate/{area}/`)

- `confusion_matrix.png` → Confusion matrix (counts & normalized)  
- `per_class_metrics.png` → Per-class precision, recall, F1  
- `evaluation_results.json` → Detailed evaluation metrics  
- `summary.txt` → Text summary of results 

---





## 📄 Comparison with Paper

### 📊 Reported Results

| Area | Paper OA | Our OA |
|------|---------|--------|
| Arkansas | ~89.2% | ~17.0% |
| California | ~87.6% | ~17.4% |

---

### 🔍 Performance Gap Analysis
Our lower performance can be attributed to:

- Limited samples (≈30 points per class vs. full pixel coverage)
- CPU-only training (no GPU acceleration)
- Small study areas (5km × 5km vs. larger regions)
- Conservative hyperparameters (to prevent overfitting)

---

##  Methodology Details

###  1. Literature Review

Key contributions from the paper:

- **Data Preparation**: Cloud filtering using QA60 band, 10-day composites  
- **Temporal Sampling**: 36 timesteps (10-day intervals) over the growing season  
- **Model Architecture**: CNN (local features) + Transformer (global dependencies) + ECA attention  
- **Training Strategy**: Cross-entropy loss, Adam optimizer, early stopping  
- **Evaluation Metrics**: OA, Kappa, F1-score, confusion matrix  

###  2. Data Acquisition via GEE

```bash
# Sentinel-2 access
s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(geometry)
    .filterDate(start_date, end_date)
    .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 20))

# CDL access
cdl = ee.Image(f'USDA/NASS/CDL/{year}')
```

###  3. Data Preprocessing Pipeline
```bash
Raw S2 → Cloud masking → NDVI/EVI/NDWI → Interpolation → Normalization → Train/Val/Test split
Raw CDL → Class extraction → Stratified sampling → Alignment with S2 pixels
```
###  4. Model Architecture (MCTNet)
```bash
Input (B, T, 7)
    ↓
Linear Projection (→ d_model=128)
    ↓
ALPE (Positional Encoding + Mask)
    ↓
Stage 1: CTFusion (CNN + Transformer + ECA)
Stage 2: CTFusion
Stage 3: CTFusion
Stage 4: CTFusion
    ↓
Global Average Pooling
    ↓
Classifier (Linear → ReLU → Dropout → Linear)
    ↓
Output (B, num_classes)
```



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