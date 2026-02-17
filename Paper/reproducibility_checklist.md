# Reproducibility Checklist

## Hardware & Environment
- [x] **GPU**: NVIDIA T4 (16GB VRAM) or V100 (32GB VRAM)
- [x] **CPU**: Intel Xeon @ 2.20GHz (2 vCPUs)
- [x] **RAM**: 12 GB
- [x] **OS**: Ubuntu 20.04 LTS
- [x] **Python**: 3.8.10
- [x] **PyTorch**: 1.10.0+cu111

## Hyperparameters
- [x] **Batch Size**: 32
- [x] **Learning Rate**: 1e-4 (Cosine Annealing)
- [x] **Optimizer**: Adam (beta1=0.9, beta2=0.999)
- [x] **Loss**: WeightedMSE (Threshold=1.0mm, Weight=3.0) + SSIM
- [x] **Epochs**: 50 (Early Stopping patience=5)
- [x] **Input Sequence**: 24 frames (Previous 24 hours)
- [x] **Output Sequence**: 6 frames (Next 6 hours)

## Data
- [x] **Source**: ERA5-Land (CDS API)
- [x] **Resolution**: 0.1 deg (~9km)
- [x] **Variables**: Total Precipitation (tp), 2m Temperature (t2m)
- [x] **Training Split**: 2015-01-01 to 2021-12-31
- [x] **Validation Split**: 2022-01-01 to 2023-12-31
- [x] **Testing Split**: 2024-01-01 to 2025-12-31

## Randomness
- [x] **Global Seed**: 42 (numpy, torch, random)
- [x] **CUDNN Deterministic**: True
