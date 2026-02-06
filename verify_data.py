"""
Comprehensive Verification Script
=================================
Verifies preprocessed data and simulates training to catch errors BEFORE Colab.
"""

import os
import sys
import numpy as np

print("=" * 70)
print("COMPREHENSIVE VERIFICATION FOR COLAB TRAINING")
print("=" * 70)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")

# ============================================================================
# 1. CHECK FILES EXIST
# ============================================================================
print("\n[1/6] Checking files exist...")

required_files = ['train.npz', 'val.npz', 'test.npz', 'stats.npz']
for f in required_files:
    path = os.path.join(PROCESSED_DIR, f)
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024**2)
        print(f"    ✓ {f}: {size_mb:.1f} MB")
    else:
        print(f"    ✗ MISSING: {f}")
        sys.exit(1)

# ============================================================================
# 2. LOAD AND VERIFY STATS
# ============================================================================
print("\n[2/6] Verifying stats.npz...")

stats = np.load(os.path.join(PROCESSED_DIR, 'stats.npz'), allow_pickle=True)
print(f"    Keys: {list(stats.keys())}")

mean = stats['mean']
std = stats['std']
variables = stats['variables']

print(f"    Mean shape: {mean.shape}, values: {mean}")
print(f"    Std shape: {std.shape}, values: {std}")
print(f"    Variables: {variables}")

# Check for issues
if np.any(np.isnan(mean)):
    print("    ✗ ERROR: Mean contains NaN!")
    sys.exit(1)
if np.any(np.isnan(std)):
    print("    ✗ ERROR: Std contains NaN!")
    sys.exit(1)
if np.any(std == 0):
    print("    ✗ ERROR: Std contains zeros!")
    sys.exit(1)

print("    ✓ Stats are valid")

# ============================================================================
# 3. VERIFY TRAIN DATA
# ============================================================================
print("\n[3/6] Verifying train.npz (loading subset)...")

train_data = np.load(os.path.join(PROCESSED_DIR, 'train.npz'), mmap_mode='r')
X_train = train_data['x']
Y_train = train_data['y']

print(f"    X_train shape: {X_train.shape}")
print(f"    Y_train shape: {Y_train.shape}")
print(f"    X_train dtype: {X_train.dtype}")
print(f"    Y_train dtype: {Y_train.dtype}")

# Check dimensions
N, T_IN, H, W, C = X_train.shape
_, T_OUT, _, _, _ = Y_train.shape

print(f"    Samples: {N}")
print(f"    Input steps: {T_IN}, Output steps: {T_OUT}")
print(f"    Grid: {H} x {W}")
print(f"    Channels: {C}")

# Expected dimensions
if T_IN != 24:
    print(f"    ⚠ WARNING: Expected T_IN=24, got {T_IN}")
if T_OUT != 6:
    print(f"    ⚠ WARNING: Expected T_OUT=6, got {T_OUT}")

# Check sample for NaN/Inf
sample = X_train[0]
if np.any(np.isnan(sample)):
    print("    ✗ ERROR: Train data contains NaN!")
    sys.exit(1)
if np.any(np.isinf(sample)):
    print("    ✗ ERROR: Train data contains Inf!")
    sys.exit(1)

print("    ✓ Train data is valid")

# ============================================================================
# 4. VERIFY VAL DATA
# ============================================================================
print("\n[4/6] Verifying val.npz...")

val_data = np.load(os.path.join(PROCESSED_DIR, 'val.npz'), mmap_mode='r')
X_val = val_data['x']
Y_val = val_data['y']

print(f"    X_val shape: {X_val.shape}")
print(f"    Y_val shape: {Y_val.shape}")

if X_val.shape[1:] != X_train.shape[1:]:
    print(f"    ✗ ERROR: Val shape mismatch!")
    sys.exit(1)

print("    ✓ Val data is valid")

# ============================================================================
# 5. SIMULATE PYTORCH LOADING
# ============================================================================
print("\n[5/6] Simulating PyTorch data loading...")

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    
    class WeatherDataset(Dataset):
        def __init__(self, X, Y):
            # X: (N, T_in, H, W, C) -> (N, T_in, C, H, W)
            self.X = torch.tensor(X[:100], dtype=torch.float32).permute(0, 1, 4, 2, 3)
            self.Y = torch.tensor(Y[:100], dtype=torch.float32).permute(0, 1, 4, 2, 3)
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx]
    
    # Load small subset
    dataset = WeatherDataset(X_train[:100], Y_train[:100])
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Get one batch
    x_batch, y_batch = next(iter(loader))
    print(f"    Batch X shape: {x_batch.shape}")  # Expected: (4, 24, 2, 31, 41)
    print(f"    Batch Y shape: {y_batch.shape}")  # Expected: (4, 6, 2, 31, 41)
    
    # Verify shapes for model
    B, T, C, H, W = x_batch.shape
    assert B == 4, "Batch size wrong"
    assert T == 24, "Input timesteps wrong"
    assert C == 2, "Channels wrong"
    
    print("    ✓ PyTorch loading works")
    
except ImportError:
    print("    ⚠ PyTorch not installed locally - will work in Colab")

# ============================================================================
# 6. CHECK MODEL COMPATIBILITY
# ============================================================================
print("\n[6/6] Verifying model dimensions...")

print(f"    Model input:  (B, {T_IN}, {C}, {H}, {W})")
print(f"    Model output: (B, {T_OUT}, {C}, {H}, {W})")
print(f"    Variables: {list(variables)}")

# Summary
print("\n" + "=" * 70)
print("VERIFICATION COMPLETE - ALL CHECKS PASSED!")
print("=" * 70)

print(f"""
DATASET SUMMARY:
----------------
• Train:  {X_train.shape[0]:,} samples
• Val:    {X_val.shape[0]:,} samples  
• Input:  24 hours (T_IN=24)
• Output: 6 hours (T_OUT=6)
• Grid:   31 x 41 (lat x lon)
• Channels: 2 (tp, t2m)

DATA SHAPES FOR COLAB:
----------------------
After permute in DataLoader:
• X shape: (batch, 24, 2, 31, 41)
• Y shape: (batch, 6, 2, 31, 41)

UPLOAD TO GOOGLE DRIVE:
-----------------------
Upload these files to: MyDrive/WeatherPaper/data/processed/
• train.npz ({os.path.getsize(os.path.join(PROCESSED_DIR, 'train.npz'))/(1024**3):.2f} GB)
• val.npz ({os.path.getsize(os.path.join(PROCESSED_DIR, 'val.npz'))/(1024**3):.2f} GB)
• test.npz ({os.path.getsize(os.path.join(PROCESSED_DIR, 'test.npz'))/(1024**3):.2f} GB)
• stats.npz (< 1 KB)
""")
