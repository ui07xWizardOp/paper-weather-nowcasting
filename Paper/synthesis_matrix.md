# Synthesis Matrix (Phase 5)

| Paper | Year | Architecture | Loss Function | Data Source | Prediction Horizon | Key Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Shi et al.** | 2015 | ConvLSTM | MSE | Radar Echo | 0-90 mins | Blurry predictions |
| **Shi et al.** | 2017 | TrajGRU | Balanced MSE | Radar Echo | 0-120 mins | High compute cost |
| **Agrawal et al.** | 2019 | U-Net | Cross-Entropy | Radar | 0-60 mins | No temporal memory |
| **Sonderby et al.** | 2020 | MetNet | Cross-Entropy | Radar + Sat | 0-8 hours | Heavy memory usage |
| **Ravuri et al.** | 2021 | DGMR (GAN) | Hinge + Grid | Radar | 0-90 mins | Unstable training |
| **Bi et al.** | 2023 | Pangu-Weather | Transformer | MAE | ERA5 (Global) | 1-7 days | Smooths local extremes |
| **Ours** | 2024 | ConvLSTM | **Weighted MSE** | **ERA5-Land** | **0-6 hours** | Deterministic (no ensemble) |
