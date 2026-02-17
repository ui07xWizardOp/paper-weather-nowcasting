# Contradiction Matrix (Phase 5)

| Theme | Claim A | Claim B | Our Stance |
| :--- | :--- | :--- | :--- |
| **Blurriness** | "Blurry forecasts are a failure mode of MSE loss." (Ravuri 2021) | "Blurry forecasts maximize CSI scores by hedging bets." (Shi 2017) | **Synthesis**: Blurriness is bad for *visualization* but acceptable for *warning*, provided the high-intensity threshold is met. |
| **Architecture** | "Recurrence (RNN) is essential for motion." (Shi 2015) | "Attention (Transformers) handles long-range dependencies better." (Sonderby 2020) | **Synthesis**: RNNs are sufficient for short-term (0-6h) nowcasting where motion is continuous. Transformers are overkill for small data. |
| **Data Source** | "Only Radar is fast enough for Nowcasting." (Traditional Met) | "Reanalysis can simulate radar-like fields." (Hersbach 2020) | **Synthesis**: In data-sparse regions, Reanalysis is the *only* option. We must adapt our models to its latency. |
| **Model Scale** | "Bigger is Better (Foundation Models)." (Bi 2023) | "Domain-Specific Architectures beat generalists." (This Paper) | **Synthesis**: Global models smooth out local extremes. Small, tuned models win on local tasks. |
