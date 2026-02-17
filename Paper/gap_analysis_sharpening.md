# Gap Analysis Sharpening (Phase 11)

## 1. Why hasn't this gap been addressed before?
**Technical Barrier**: The lack of high-resolution, consistent historical data for the Indian subcontinent.
-   **Radar**: India's radar network (DWR) coverage is patchy and data is often not public.
-   **Satellites**: IMERG/GPM are too coarse (10km or 0.5 deg) or have high latency.
-   **Reanalysis**: ERA5 (0.25 deg) was too smooth. **ERA5-Land (0.1 deg, released 2021)** finally provided the "Golden Dataset" required for pixel-level deep learning.

## 2. What would addressing this gap enable?
**Impact**: Hyper-local flood warning for "Ungauged Basins".
-   NWP models take 6 hours to spin up. Flash floods happen in 2 hours.
-   A successful nowcast model provides a **0-6 hour "Intelligence Bridge"** that activates disaster response teams *before* the NWP model even finishes its run.

## 3. What makes this the right time? ("Why Now?")
**Enabling Convergence**:
-   **Data**: ERA5-Land (2021) provides 70 years of consistent Hourly/9km data.
-   **Hardware**: T4/V100 GPUs are now "Colab-accessible", allowing researchers outside the "Big Tech" sphere to train ConvLSTMs.
-   **Validation**: The failure of pure GraphCast on local scales (seen in 2023) has renewed interest in "Regional Expert" models.

## 4. What are the key challenges?
**Technical Obstacles**:
-   **Zero-Inflation**: 90% of the pixels are dry. Standard MSE optimizes for "0 everywhere".
-   **Orographic Locking**: The Himalayas create static rain bands that models overfit to as "background noise" rather than dynamic weather.
