"""
Local Preprocessing Script for Weather Nowcasting (FINAL)
==========================================================
Run this locally to preprocess the dataset, then upload output to Colab for training.

FIXED: 
- Generates sequences WITHIN each month to avoid cross-month gaps
- Fills ocean NaN values (ERA5-Land has ~25% NaN for ocean pixels)

Usage:
    python preprocess_local.py

Output:
    data/processed/train.npz
    data/processed/val.npz
    data/processed/test.npz
    data/processed/stats.npz
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

try:
    import xarray as xr
except ImportError:
    print("Installing required packages...")
    os.system("pip install xarray netCDF4 numpy pandas")
    import xarray as xr

# === CONFIGURATION ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "Dataset")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data", "processed")

# Time split configuration  
TRAIN_YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
VAL_YEARS = [2022, 2023]
TEST_YEARS = [2024, 2025]

# Sequence parameters
T_IN = 24   # 24 hours of input (past)
T_OUT = 6   # 6 hours of output (future)


def load_single_file(filepath):
    """Load a single NetCDF file and standardize it."""
    ds = xr.open_dataset(filepath, engine='netcdf4')
    
    # Rename valid_time -> time
    if 'valid_time' in ds.coords and 'time' not in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
    
    # Handle expver
    if 'expver' in ds.dims:
        ds = ds.isel(expver=0, drop=True)
    elif 'expver' in ds.coords:
        ds = ds.drop_vars('expver', errors='ignore')
    
    # Drop 'number'
    if 'number' in ds.coords:
        ds = ds.drop_vars('number', errors='ignore')
    
    return ds


def create_sequences_from_chunk(data, t_in, t_out):
    """Create sequences from a continuous chunk of data (single month)."""
    X_list, Y_list = [], []
    
    n_timesteps = data.shape[0]
    
    if n_timesteps < t_in + t_out:
        return [], []
    
    for i in range(t_in, n_timesteps - t_out + 1):
        x_seq = data[i - t_in:i]
        y_seq = data[i:i + t_out]
        X_list.append(x_seq)
        Y_list.append(y_seq)
    
    return X_list, Y_list


def main():
    print("=" * 70)
    print("LOCAL PREPROCESSING FOR WEATHER NOWCASTING (FINAL)")
    print("=" * 70)
    
    # Step 1: Find NetCDF files
    print("\n[1/5] Finding NetCDF files...")
    all_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.nc")))
    
    if not all_files:
        print(f"ERROR: No NetCDF files found in {DATASET_DIR}")
        sys.exit(1)
    
    print(f"    Found {len(all_files)} files")
    
    # Get variable order from first file
    first_ds = load_single_file(all_files[0])
    available_vars = list(first_ds.data_vars)
    ordered_vars = [v for v in ['tp', 't2m', 'msl'] if v in available_vars]
    if not ordered_vars:
        ordered_vars = available_vars
    print(f"    Variables: {ordered_vars}")
    
    # Create land mask from first file (where data is NOT NaN)
    land_mask = ~np.isnan(first_ds[ordered_vars[0]].values[0])  # (lat, lon)
    print(f"    Land pixels: {land_mask.sum()} / {land_mask.size} ({land_mask.sum()/land_mask.size*100:.1f}%)")
    first_ds.close()
    
    # Step 2: First pass - collect training data for normalization
    print("\n[2/5] Computing normalization statistics from training data...")
    
    train_data_chunks = []
    
    for i, filepath in enumerate(all_files):
        ds = load_single_file(filepath)
        year = pd.to_datetime(ds.time.values[0]).year
        
        if year in TRAIN_YEARS:
            data_arrays = [ds[var].values for var in ordered_vars]
            data = np.stack(data_arrays, axis=-1).astype(np.float32)
            
            # Apply land mask - only take land pixels for stats
            # But keep spatial structure for sequences
            train_data_chunks.append(data)
        
        ds.close()
        
        if (i + 1) % 20 == 0:
            print(f"    Scanning file {i + 1}/{len(all_files)}...")
    
    # Compute stats only from land pixels
    train_all = np.concatenate(train_data_chunks, axis=0)
    
    # Mask ocean pixels when computing stats
    train_masked = train_all[:, land_mask, :]  # (time, land_pixels, channels)
    mean = np.nanmean(train_masked, axis=(0, 1), keepdims=False)
    std = np.nanstd(train_masked, axis=(0, 1), keepdims=False)
    std = np.where(std < 1e-6, 1.0, std)
    
    print(f"    Train samples: {train_all.shape[0]} timesteps")
    print(f"    Mean: {mean}")
    print(f"    Std: {std}")
    
    del train_data_chunks, train_all, train_masked
    
    # Step 3: Second pass - generate sequences
    print("\n[3/5] Generating sequences (per-file processing)...")
    
    X_train, Y_train = [], []
    X_val, Y_val = [], []
    X_test, Y_test = [], []
    
    for i, filepath in enumerate(all_files):
        ds = load_single_file(filepath)
        year = pd.to_datetime(ds.time.values[0]).year
        
        # Load and process data
        data_arrays = [ds[var].values for var in ordered_vars]
        data = np.stack(data_arrays, axis=-1).astype(np.float32)
        
        # Fill NaN with 0 (ocean pixels will be masked anyway or use 0)
        data = np.nan_to_num(data, nan=0.0)
        
        # Normalize
        data_norm = (data - mean) / std
        
        ds.close()
        
        # Generate sequences
        X_seqs, Y_seqs = create_sequences_from_chunk(data_norm, T_IN, T_OUT)
        
        # Add to appropriate split
        if year in TRAIN_YEARS:
            X_train.extend(X_seqs)
            Y_train.extend(Y_seqs)
        elif year in VAL_YEARS:
            X_val.extend(X_seqs)
            Y_val.extend(Y_seqs)
        elif year in TEST_YEARS:
            X_test.extend(X_seqs)
            Y_test.extend(Y_seqs)
        
        if (i + 1) % 20 == 0:
            print(f"    Processed {i + 1}/{len(all_files)} files - "
                  f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Convert to numpy arrays
    print("\n[4/5] Converting to arrays...")
    
    X_train = np.array(X_train, dtype='float32')
    Y_train = np.array(Y_train, dtype='float32')
    X_val = np.array(X_val, dtype='float32')
    Y_val = np.array(Y_val, dtype='float32')
    X_test = np.array(X_test, dtype='float32')
    Y_test = np.array(Y_test, dtype='float32')
    
    print(f"    Train: X={X_train.shape}, Y={Y_train.shape}")
    print(f"    Val:   X={X_val.shape}, Y={Y_val.shape}")
    print(f"    Test:  X={X_test.shape}, Y={Y_test.shape}")
    
    # Step 5: Save output
    print("\n[5/5] Saving processed data...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    np.savez_compressed(os.path.join(OUTPUT_DIR, 'train.npz'), x=X_train, y=Y_train)
    np.savez_compressed(os.path.join(OUTPUT_DIR, 'val.npz'), x=X_val, y=Y_val)
    np.savez_compressed(os.path.join(OUTPUT_DIR, 'test.npz'), x=X_test, y=Y_test)
    np.savez_compressed(os.path.join(OUTPUT_DIR, 'stats.npz'), 
                        mean=mean, std=std, variables=ordered_vars, land_mask=land_mask)
    
    # Calculate sizes
    train_size = os.path.getsize(os.path.join(OUTPUT_DIR, 'train.npz')) / (1024**2)
    val_size = os.path.getsize(os.path.join(OUTPUT_DIR, 'val.npz')) / (1024**2)
    test_size = os.path.getsize(os.path.join(OUTPUT_DIR, 'test.npz')) / (1024**2)
    
    print(f"\n" + "=" * 70)
    print("PREPROCESSING COMPLETE!")
    print("=" * 70)
    print(f"\nOutput saved to: {OUTPUT_DIR}")
    print(f"  - train.npz: {len(X_train)} samples ({train_size:.1f} MB)")
    print(f"  - val.npz:   {len(X_val)} samples ({val_size:.1f} MB)")
    print(f"  - test.npz:  {len(X_test)} samples ({test_size:.1f} MB)")
    print(f"  - stats.npz: normalization parameters + land mask")
    
    print(f"\n>>> NEXT: Upload 'data/processed' folder to Colab for training")


if __name__ == "__main__":
    main()
