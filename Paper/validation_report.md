# Research Validation Report (Phase 11)

## Counterfactual Analysis
**Disagreement Survey**:
-   **GANs (DGMR) vs. MSE (ConvLSTM/MetNet)**: The literature is split. Ravuri et al. (2021) argue MSE is fundamentally flawed for nowcasting ("blurriness"). Sonderby et al. (2020) argue that scale-aware architectural priors (Axial Attention) can solve this without GAN instability.
-   **Our Stance**: We side with the **Deterministic + Weighted Loss** approach for this specific regional application. Given limited compute (single GPU) and the need for operational stability, training a GAN is high-risk. Weighted MSE offers a "middle ground" - significantly sharper than vanilla MSE, but strictly stable/convergent.

**Metric Critique**:
-   Standard CSI (Critical Success Index) at low thresholds (e.g., 0.1mm/hr) is easily gamed by predicting "drizzle everywhere".
-   **Corrective Action**: We specifically report CSI at **1.0mm/hr** (moderate rain) and higher thresholds to verify utility where it matters (agriculture/disaster).

## Gap Sharpening
**Why hasn't this been solved?**
1.  **Data Access**: High-resolution, consistent reanalysis (ERA5-Land, 2021) is relatively new. Previous works used inconsistent radar dumps.
2.  **Topography**: General "Global" models (GraphCast, Pangu-Weather) smooth out sub-grid orography crucial for Indian heavy rainfall events.

**Why Now?**
1.  **Compute**: T4/V100 GPUs democratize training ConvLSTMs on decadal datasets.
2.  **Data**: 11 years of ERA5-Land (2015-2025) provides enough "extreme events" to statistically train a deep model, which wasn't possible with shorter records.

**What addresses the gap?**
-   Our **WeightedMSE** specifically targeting the "long tail" of the distribution, which global models treat as noise.
