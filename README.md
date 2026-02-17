# Paper Weather: Deep Learning for Precipitation Nowcasting 🌧️

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

> **Code for the paper**: *"Deep Spatio-Temporal Nowcasting with Physically Grounded Loss Functions: A Case Study on the Indian Subcontinent"* (Under Review, IEEE GRSL).

## 🌍 Abstract
Standard deep learning models for weather forecasting often suffer from "blurriness"—predicting a safe average rather than capturing the extreme nature of cloudbursts. This project introduces a **Weighted Mean Squared Error (WMSE)** loss function grounded in physical precipitation thresholds ($>1 \text{mm/hr}$), trained on 11 years of high-resolution **ERA5-Land** data.

**Key Result**: We achieve a **Critical Success Index (CSI) of 0.65**, a 12% improvement over standard ConvLSTM baselines, using a single NVIDIA T4 GPU.

## 🚀 Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/travel-copilot/paper-weather.git
    cd paper-weather
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 📊 Usage

### Training
To train the ConvLSTM model with Weighted MSE:
```bash
python train.py --config configs/era5_land.yaml
```

### Reproducing Figures
To generate the qualitative and quantitative plots used in the paper (Figures 3 & 4):
```bash
python generate_paper_plots.py
```
Outputs will be saved to `figures/`.

## 📂 Project Structure
- `model.py`: PyTorch implementation of ConvLSTM, Encoder-Decoder, and WeightedMSELoss.
- `train.py`: Main training loop with Mixed Precision (AMP).
- `data_loader.py`: Efficient NetCDF/Zarr dataloading for ERA5-Land.
- `generate_paper_plots.py`: Publication-ready visualization script.

## 📝 Citation
If you use this code or dataset, please cite:
```bibtex
@article{PaperWeather2024,
  title={Deep Spatio-Temporal Nowcasting with Physically Grounded Loss Functions},
  author={Travel Copilot Team},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2024}
}
```
