# 🌦️ Paper Weather: Deep Learning for Weather Nowcasting

**Paper Weather** is a research-oriented deep learning project designed to perform short-term weather forecasting (nowcasting) using **ConvLSTM** networks. It leverages high-resolution **ERA5-Land** reanalysis data to predict future precipitation and temperature patterns based on historical sequences.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset Details](#-dataset-details)
- [Model Architecture](#-model-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
    - [1. Data Preparation](#1-data-preparation)
    - [2. preprocessing](#2-preprocessing)
    - [3. Training (Local & Cloud)](#3-training-local--cloud)
- [Configuration](#-configuration)

---

## 🔭 Project Overview

The goal of this project is to predict weather variables for the next **6 hours** given the past **24 hours** of data. It addresses challenges in processing large geospatial datasets by implementing memory-efficient batching and optimized data loading pipelines.

**Key Features:**
*   **ConvLSTM Core**: capturing spatiotemporal dependencies.
*   **Memory Efficiency**: Custom "batched" data loader to handle years of high-res NetCDF data on limited-RAM systems (like Colab).
*   **Performance**: Mixed Precision Training (AMP) and `cudnn.benchmark` integration for 2x faster training.
*   **Verification**: Comprehensive validation / testing splits including recent 2024-2025 data.

---

## 🌍 Dataset Details

We use European Centre for Medium-Range Weather Forecasts (ECMWF) **ERA5-Land** data.

*   **Source**: [Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu/)
*   **Variables**:
    *   `tp` (Total Precipitation)
    *   `t2m` (2m Temperature)
*   **Resolution**: Hourly data, gridded (31 x 41 spatial patch).
*   **Splits**:
    *   **Train**: 2015 - 2021 (7 years)
    *   **Validation**: 2022 - 2023 (2 years)
    *   **Test**: 2024 - 2025 (2 years)

---

## 🧠 Model Architecture

The `WeatherNowcaster` model is an Encoder-Decoder architecture:

1.  **Encoder**:
    *   Stack of **ConvLSTM Cells**.
    *   Processes the input sequence ($T_{in}=24$) to extract spatiotemporal features.
    *   Passes the hidden state ($H, C$) to the decoder.

2.  **Decoder**:
    *   Stack of **ConvLSTM Cells**.
    *   Autoregressively generates predictions for the future sequence ($T_{out}=6$).
    *   Uses the previous step's prediction as input for the next step.

3.  **Output Head**:
    *   1x1 Convolution mapping hidden features back to the 2 channel output (`tp`, `t2m`).

---

## 📂 Project Structure

```
paper-weather-nowcasting/
├── Dataset/                 # Raw NetCDF files (monthly/daily chunks)
├── data/
│   └── batched/             # Processed .npy batches (Generated)
│       ├── train/
│       ├── val/
│       ├── test/
│       ├── stats.npz        # Normalization mean/std
│       └── metadata.npz
├── notebooks/
│   ├── 01_setup_and_download.ipynb   # CDS API download script
│   ├── 02_preprocessing.ipynb        # Exploration & preprocessing logic
│   ├── 03_training.ipynb             # ⚡ COLAB/Cloud training (Recommended)
│   └── 03_local_training.ipynb       # Local GPU training
├── checkpoints/             # Saved models (.pth)
├── figures/                 # Training curves and visualizations
├── preprocess_batched.py    # CLI script for memory-efficient preprocessing
├── train_cpu.py             # CPU-only verification script ("Dry Run")
├── train_local.py           # Standard local training script
└── requirements.txt         # Dependencies
```

---

## ⚙️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/ui07xWizardOp/paper-weather-nowcasting.git
    cd paper-weather-nowcasting
    ```

2.  **Install dependencies**:
    ```bash
    pip install numpy torch matplotlib xarray netCDF4 tqdm pandas
    ```

    *Note: For GPU support, enable CUDA with PyTorch.*

---

## 🚀 Usage Guide

### 1. Data Preparation
Place your raw `.nc` (NetCDF) files in the `Dataset/` directory.
*(Ensure filenames or metadata cover the 2015-2025 range for the full split)*.

### 2. Preprocessing
Run the batch generator to convert bulky NetCDF files into efficient `.npy` batches.

```bash
python preprocess_batched.py
```
*   **Output**: Creates `data/batched/` with train/val/test folders.
*   *Why?* Loading small `.npy` files during training is 10x faster and lighter than opening minimal NetCDF handles.

### 3. Training (Local / Cloud)

#### Option A: Google Colab (Recommended)
1.  Upload the generated `data/batched/` folder to your Google Drive (`MyDrive/WeatherPaper/data/batched`).
2.  Open `notebooks/03_training.ipynb` in Colab.
3.  Mount Drive and run. The notebook is optimised for T4 GPUs.

#### Option B: Local Training (GPU)
```bash
python train_local.py
```
*   Ensures massive speedups using Mixed Precision (AMP).

#### Option C: Verification (CPU Dry Run)
If you just want to verify the code works without a GPU:
```bash
python train_cpu.py
```
*   Runs a tiny model for 1 epoch to catch errors.

---

## 🔧 Configuration

You can adjust hyperparameters in `train_local.py` or the notebooks:

```python
BATCH_SIZE = 32      # Reduce to 16 if VRAM < 8GB
HIDDEN_DIM = 128     # Model capacity
T_IN = 24            # Input history steps
T_OUT = 6            # Prediction steps
LR = 1e-3            # Learning rate
```

---

## 📜 License

This project is open-source. Please check the LICENSE file for details.
