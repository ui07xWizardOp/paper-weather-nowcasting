# Verification & Triangulation Report (Phase 11)

## Claim 1: "MSE Loss leads to blurry predictions."
-   **Source A (Primary)**: Mathieu et al. (2015) - "Deep multi-scale video prediction beyond mean squared error." Introduces Gradient Difference Loss to combat blur.
-   **Source B (Secondary)**: Ravuri et al. (2021) - "Skilful precipitation nowcasting..." Explicitly states MSE yields "smooth, unimodal" forecasts.
-   **Verdict**: **Confirmed**. The averaging property of MSE in stochastic domains is a settled mathematical fact.

## Claim 2: "NWP Spin-up issues affect early forecast hours."
-   **Source A (Primary)**: Sun et al. (2014) - "Use of NWP for Nowcasting..." Discusses the 0-6h "spin-up" gap where radar extrapolation wins.
-   **Source B (Secondary)**: Hersbach et al. (2020) - "The ERA5 global reanalysis." Notes that even reanalysis has spin-up periods (though mitigated by 4D-Var).
-   **Verdict**: **Confirmed**. This validates our choice to use Deep Learning for the 0-6h window.

## Claim 3: "ConvLSTM captures spatio-temporal correlations."
-   **Source A (Primary)**: Shi et al. (2015) - Original ConvLSTM paper.
-   **Source B (Secondary)**: Wang et al. (2017) - "PredRNN." Acknowledges ConvLSTM as the baseline for ST-modeling, though proposing improvements.
-   **Verdict**: **Confirmed**. It is the standard "workhorse" architecture.

## Claim 4: "Imbalanced Data (Zero-Inflation) hinders training."
-   **Source A (Primary)**: Shi et al. (2017) - "Deep Learning for Precipitation Nowcasting..." Introduces B-MSE (Balanced MSE) to handle zeros.
-   **Source B (Secondary)**: Lin et al. (2017) - "Focal Loss." Generalizes the problem of class imbalance in dense detectors (analogous to rain pixels).
-   **Verdict**: **Confirmed**. Justifies our "Weighted MSE".

## Claim 5: "ERA5-Land is suitable for hydrological modeling."
-   **Source A (Primary)**: Muñoz-Sabater et al. (2021) - "ERA5-Land: A state-of-the-art global reanalysis..." Explicitly designed for land surface variables.
-   **Source B (Secondary)**: ECMWF Documentation.
-   **Verdict**: **Confirmed**. It is the gold standard for historical land-surface consistency.
