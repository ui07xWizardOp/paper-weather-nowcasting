# Hypothesis Generation Log (Phase 0)

## Project Initialization: Paper Weather
**Date**: Project Start
**Goal**: High-resolution precipitation nowcasting for India using ERA5-Land.

## Hypothesis 1: The "Blurriness" Trade-off
-   **Hypothesis**: Standard regression losses (MSE/MAE) applied to precipitation will result in unimodal, blurry predictions that minimize error by averaging possible futures.
-   **Risk**: The model will fail to predict high-intensity events (Cloudbursts), effectively acting as a "low-pass filter".
-   **Mitigation Strategy**: Implement a **Weighted Mean Squared Error (WMSE)** loss function that explicitly penalizes errors on values $> \tau$ (e.g., 1 mm/hr) more heavily than background zeros.

## Hypothesis 2: Reanalysis as Ground Truth
-   **Hypothesis**: ERA5-Land (9km) contains sufficient spatial feature definition to train a convolutional recurrent network, despite being a model output itself.
-   **Risk**: The "smoothening" inherent in reanalysis might limit the model's ability to learn sharp convective features compared to Radar data.
-   **Mitigation Strategy**: Use **Log-Transformed Z-Score Normalization** to expand the dynamic range of the data and approximate a normal distribution, making the "tails" significantly more visible to the optimizer.

## Hypothesis 3: Regional Generalization
-   **Hypothesis**: A model trained on a fixed $31 \times 41$ grid over India can generalize across seasons (Monsoon vs. Winter).
-   **Risk**: Overfitting to the dominant Southwest Monsoon patterns.
-   **Mitigation Strategy**: Strict **Temporal Splitting** (Train: 2015-2021, Val: 2022-2023, Test: 2024-2025) ensures evaluations are performed on "unseen" climate years, reducing temporal leakage.
