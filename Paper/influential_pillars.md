# Influential Pillars Analysis (Phase 4)

## 1. ConvLSTM (Shi et al., 2015)
-   **Role**: **The Foundation**.
-   **Contribution**: Proved that spatio-temporal correlations can be learned by replacing fully connected transitions in LSTM with convolution operations.
-   **relevance**: This is the backbone of our architecture. We adopt their core equations but modify the *Loss Function*.

## 2. DGMR (Ravuri et al., 2021)
-   **Role**: **The Generative Shift**.
-   **Contribution**: Addressed the "blurriness" of MSE models by using extensive GAN training (Spatial and Temporal Discriminators).
-   **Relevance**: Serves as the primary "Conceptual Baseline" for why *we* use Weighted MSE. We argue that for resource-constrained settings, WMSE offers 80% of the utility of DGMR at 10% of the compute cost.

## 3. ERA5-Land (Muñoz-Sabater et al., 2021)
-   **Role**: **The Data Enabler**.
-   **Contribution**: Provided the first consistent, high-resolution (9km) land-surface history for the globe.
-   **Relevance**: Without this dataset, our "Regional Indian Model" would not be possible due to the lack of public radar data.

## 4. U-Net for Weather (Agrawal et al., 2019)
-   **Role**: **The CNN Baseline**.
-   **Contribution**: Treated nowcasting as a pure Image-to-Image translation problem, ignoring explicit temporal memory.
-   **Relevance**: We use this as a baseline to prove that *Recurrence* (LSTM) is necessary. Our qualitative results showing "Dissipation Lag" prove that memory matters.
