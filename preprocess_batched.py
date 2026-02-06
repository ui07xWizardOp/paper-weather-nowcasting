"""
Memory-Efficient Preprocessing for Weather Nowcasting
=====================================================
Saves data in batched format to avoid memory issues in Colab.

Instead of one giant file, creates multiple smaller batch files
that can be loaded one at a time during training.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

try:
    import xarray as xr
except ImportError:
    os.system("pip install xarray netCDF4")
    import xarray as xr

# === CONFIGURATION ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "Dataset")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data", "batched")

# Time split
TRAIN_YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
VAL_YEARS = [2022, 2023]
TEST_YEARS = [2024, 2025]

T_IN = 24
T_OUT = 6
BATCH_SIZE = 500  # Samples per batch file


def load_single_file(filepath):
    ds = xr.open_dataset(filepath, engine='netcdf4')
    if 'valid_time' in ds.coords and 'time' not in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
    if 'expver' in ds.dims:
        ds = ds.isel(expver=0, drop=True)
    elif 'expver' in ds.coords:
        ds = ds.drop_vars('expver', errors='ignore')
    if 'number' in ds.coords:
        ds = ds.drop_vars('number', errors='ignore')
    return ds


def create_sequences_from_chunk(data, t_in, t_out):
    X_list, Y_list = [], []
    n_timesteps = data.shape[0]
    if n_timesteps < t_in + t_out:
        return [], []
    for i in range(t_in, n_timesteps - t_out + 1):
        X_list.append(data[i - t_in:i])
        Y_list.append(data[i:i + t_out])
    return X_list, Y_list


def save_batches(X_list, Y_list, split_name, output_dir):
    """Save sequences in batches."""
    if not X_list:
        return 0
    
    os.makedirs(os.path.join(output_dir, split_name), exist_ok=True)
    
    n_samples = len(X_list)
    n_batches = (n_samples + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, n_samples)
        
        batch_x = np.array(X_list[start:end], dtype='float32')
        batch_y = np.array(Y_list[start:end], dtype='float32')
        
        np.save(os.path.join(output_dir, split_name, f'X_batch_{batch_idx:04d}.npy'), batch_x)
        np.save(os.path.join(output_dir, split_name, f'Y_batch_{batch_idx:04d}.npy'), batch_y)
    
    # Save metadata
    np.savez(os.path.join(output_dir, split_name, 'metadata.npz'),
             n_samples=n_samples, n_batches=n_batches, batch_size=BATCH_SIZE)
    
    return n_batches


def main():
    print("=" * 60)
    print("MEMORY-EFFICIENT PREPROCESSING")
    print("=" * 60)
    
    all_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.nc")))
    print(f"\nFound {len(all_files)} files")
    
    # Get variables
    first_ds = load_single_file(all_files[0])
    ordered_vars = [v for v in ['tp', 't2m'] if v in first_ds.data_vars]
    land_mask = ~np.isnan(first_ds[ordered_vars[0]].values[0])
    first_ds.close()
    print(f"Variables: {ordered_vars}")
    
    # First pass: compute stats from training data
    print("\n[1/3] Computing normalization stats...")
    train_values = []
    
    for i, filepath in enumerate(all_files):
        ds = load_single_file(filepath)
        year = pd.to_datetime(ds.time.values[0]).year
        
        if year in TRAIN_YEARS:
            data_arrays = [ds[var].values for var in ordered_vars]
            data = np.stack(data_arrays, axis=-1).astype(np.float32)
            # Only use land pixels for stats
            train_values.append(data[:, land_mask, :])
        ds.close()
        
        if (i + 1) % 30 == 0:
            print(f"    Scanned {i + 1}/{len(all_files)} files")
    
    train_all = np.concatenate(train_values, axis=0)
    mean = np.nanmean(train_all, axis=(0, 1))
    std = np.nanstd(train_all, axis=(0, 1))
    std = np.where(std < 1e-6, 1.0, std)
    
    print(f"    Mean: {mean}")
    print(f"    Std: {std}")
    
    del train_values, train_all
    
    # Save stats
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUTPUT_DIR, 'stats.npz'), 
             mean=mean, std=std, variables=ordered_vars, land_mask=land_mask)
    
    # Second pass: generate sequences and save in batches
    print("\n[2/3] Generating sequences per file...")
    
    X_train, Y_train = [], []
    X_val, Y_val = [], []
    X_test, Y_test = [], []
    
    for i, filepath in enumerate(all_files):
        ds = load_single_file(filepath)
        year = pd.to_datetime(ds.time.values[0]).year
        
        data_arrays = [ds[var].values for var in ordered_vars]
        data = np.stack(data_arrays, axis=-1).astype(np.float32)
        data = np.nan_to_num(data, nan=0.0)
        data_norm = (data - mean) / std
        
        ds.close()
        
        X_seqs, Y_seqs = create_sequences_from_chunk(data_norm, T_IN, T_OUT)
        
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
            print(f"    Processed {i + 1}/{len(all_files)} - "
                  f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Save in batches
    print("\n[3/3] Saving batched data...")
    
    n_train = save_batches(X_train, Y_train, 'train', OUTPUT_DIR)
    print(f"    Train: {len(X_train)} samples in {n_train} batches")
    del X_train, Y_train
    
    n_val = save_batches(X_val, Y_val, 'val', OUTPUT_DIR)
    print(f"    Val: {len(X_val)} samples in {n_val} batches")
    del X_val, Y_val
    
    n_test = save_batches(X_test, Y_test, 'test', OUTPUT_DIR)
    print(f"    Test: {len(X_test)} samples in {n_test} batches")
    del X_test, Y_test
    
    # Summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    
    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"""
Files to upload to Google Drive (MyDrive/WeatherPaper/data/batched/):
  stats.npz
  train/ (folder with {n_train} batch files)
  val/ (folder with {n_val} batch files)  
  test/ (folder with {n_test} batch files)

Each batch file is ~{BATCH_SIZE * 24 * 31 * 41 * 2 * 4 / 1e6:.0f} MB (manageable for Colab RAM)
""")


if __name__ == "__main__":
    main()
