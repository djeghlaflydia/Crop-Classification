# Literature Review: MCTNet for Crop Classification

## 1. Paper Overview
**Title**: MCTNet: A Multitemporal Crop Transformer Network for Large-scale Crop Mapping
**Objective**: To perform pixel-based crop mapping using multitemporal satellite imagery, addressing challenges like cloud cover and missing data while maintaining a lightweight architecture.

## 2. Methodology Analysis

### 2.1. Data Preparation
- **Source**: Sentinel-2 Level-2A surface reflectance data.
- **Labels**: USDA Cropland Data Layer (CDL).
- **Composite Strategy**: The paper typically uses 10-day or 15-day median composites to reduce noise and cloud influence.
- **Vegetation Indices**: Uses raw spectral bands (B02, B03, B04, B08) and often derived indices like NDVI to capture phenological cycles.

### 2.2. Temporal Sampling Strategy
- The model handles fixed-length time series (e.g., 36 timesteps for a full year at 10-day intervals).
- **Missing Data Handling**: Instead of simple interpolation, it uses a **Learnable Positional Encoding (ALPE)** that incorporates a mask indicating valid vs. invalid observations.

### 2.3. Model Architecture (MCTNet)
MCTNet is a hybrid CNN-Transformer architecture designed for efficiency:
- **Input Projection**: High-dimensional spectral data is projected into a latent space.
- **ALPE (Attention-based Learned Positional Encoding)**: Incorporates temporal context and data quality (masks) into the feature representation.
- **CTFusion (CNN-Transformer Fusion) Blocks**:
    - **CNN Branch**: Uses 1D convolutions to capture local temporal dependencies (e.g., rapid growth phases).
    - **Transformer Branch**: Uses self-attention to capture long-range temporal dependencies (e.g., seasonal patterns).
    - **ECA (Efficient Channel Attention)**: Refines features across channels by focusing on the most informative spectral-temporal components.
- **Global Pooling & Classifier**: Aggregates temporal features for final crop class prediction.

### 2.4. Training Procedure
- **Loss Function**: Weighted Cross-Entropy (to handle class imbalance).
- **Optimizer**: Adam with learning rate scheduling.
- **Evaluation**: OA (Overall Accuracy), Kappa Coefficient, and F1-score.

### 2.5. Evaluation Metrics
- **OA**: Total correct pixels / total pixels.
- **Kappa**: Measures agreement beyond chance.
- **F1-score (Macro)**: Balanced measure for multi-class tasks, especially with imbalanced labels.

## 3. Comparison with Baseline Models
MCTNet aims to outperform standard GRUs, LSTMs, and standalone Transformers by combining local and global temporal feature extraction while being more robust to missing data.
