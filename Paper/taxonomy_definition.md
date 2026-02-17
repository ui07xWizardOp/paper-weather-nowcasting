# Taxonomy Definition (Phase 3)

## Overview
To ensure consistent classification of the literature in `literature_inventory.csv`, we defined the following strict tagging schema.

## 1. Method Tags (#method-*)
-   `#method-rnn`: Architectures based on Recurrent Neural Networks (LSTM, GRU, ConvLSTM).
    -   *Example*: Shi et al. (2015).
-   `#method-cnn`: Architectures based on pure Convolutional Networks (U-Net, ResNet) with no explicit temporal state.
    -   *Example*: Agrawal et al. (2019).
-   `#method-gan`: Generative Adversarial Networks focused on probabilistic sampling.
    -   *Example*: Ravuri et al. (2021).
-   `#method-transformer`: Attention-based architectures.
    -   *Example*: MetNet, Pangu-Weather.

## 2. Dataset Tags (#dataset-*)
-   `#dataset-radar`: Ground-based Doppler Weather Radar (high resolution, low coverage).
-   `#dataset-satellite`: Geostationary satellite imagery (IR/VIS).
-   `#dataset-reanalysis`: ERA5 or ERA5-Land (Physical simulation data).

## 3. Evaluation Metrics (#metric-*)
-   `#metric-mse`: Mean Squared Error (Pixel-wise).
-   `#metric-csi`: Critical Success Index (Categorical/Event-based).
-   `#metric-crps`: Continuous Ranked Probability Score (Probabilistic).

## 4. Task Scope (#task-*)
-   `#task-nowcasting`: 0-6 hour Prediction.
-   `#task-forecasting`: 1-10 day Prediction.
