# Snowballing Protocol Log (Phase 1)

## Strategy
We employed a "Seed + Snowball" strategy to ensure comprehensive coverage of the *Deep Learning for Precipitation Nowcasting* domain, specifically targeting the transition from Radar-based to Reanalysis-based methods.

## Round 1: Seed Selection
**Criteria**: Papers that defined the modern field (>1000 citations) or introduced key datasets.
1.  **Shi et al. (2015)**: *Convolutional LSTM Network* (The Backbone).
2.  **Ravuri et al. (2021)**: *Deep Generative Models of Radar* (The Generative Shift).
3.  **Muñoz-Sabater et al. (2021)**: *ERA5-Land* (The Dataset).

## Round 2: Backward Snowballing (Ancestry)
*Tracing references INSIDE the seed papers to find foundations.*
-   **From Shi et al. (2015)**:
    -   Found: *Hochreiter (1997)* - Original LSTM.
    -   Found: *Wang et al. (2004)* - SSIM Metric (Crucial for loss function).
-   **From Ravuri et al. (2021)**:
    -   Found: *Agrawal et al. (2019)* - U-Net baselines (Google Research).

## Round 3: Forward Snowballing (Descendants)
*Tracing papers that CITE the seed papers (using Google Scholar / Semantic Scholar).*
-   **Citing Shi et al. (2015)**:
    -   Found: *Sonderby et al. (2020)* - MetNet (Attention-based).
    -   Found: *Bi et al. (2023)* - Pangu-Weather (Foundation Models).
-   **Citing Muñoz-Sabater (2021)**:
    -   Found: Regional validation studies (mostly Hydrology journals, filtered for ML focus).

## Round 4: Filtering & Inclusion
**Total Candidates**: 150+
**Inclusion Criteria**:
-   Must focus on *Spatio-Temporal* forecasting (not just time-series).
-   Must use *Grid Data* (Radar, Satellite, or Reanalysis).
-   Must report CSI/POD metrics (for comparability).

**Final Selection**: ~30 Core Papers included in `references.bib`.
