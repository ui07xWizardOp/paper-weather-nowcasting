# External Reproducibility Audit (Phase 11)

## Overview
We surveyed the top 10 referenced papers in our study to assess the "Reproducibility Crisis" in the field.

| Paper | Model | Code Available? | Data Available? | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Shi et al. (2015)** | ConvLSTM | **Yes** | **Yes** | Moving MNIST is public. |
| **Shi et al. (2017)** | TrajGRU | **Yes** | **Yes** | HKO-7 is public. |
| **Agrawal et al. (2019)** | U-Net | No | No | Google internal dataset. |
| **Sonderby et al. (2020)** | MetNet | No | No | Google internal dataset. |
| **Ravuri et al. (2021)** | DGMR | **Yes** | No | UK Met Office data is restricted. |
| **Bi et al. (2023)** | Pangu-Weather | **Yes** | **Yes** | Inference code only. Training code closed. |
| **Ours (2024)** | **Paper Weather** | **YES** | **YES** | **Full Training Pipeline + ERA5-Land Downloader.** |

## Conclusion
While major labs (DeepMind, Google) publish architectures, they often withhold training code or datasets. Our contribution is significant because we provide an **End-to-End Open Source Pipeline** for the *specific* task of regional nowcasting on public data.
