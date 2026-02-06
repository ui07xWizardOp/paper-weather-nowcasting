"""
Convert npz to memory-mappable npy format
=========================================
Colab crashes because np.load() on compressed npz loads everything into RAM.
This script converts to uncompressed .npy files that can be memory-mapped.

Run this locally, then upload the new files to Google Drive.
"""

import os
import numpy as np

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")

print("=" * 60)
print("CONVERTING TO MEMORY-MAPPABLE FORMAT")
print("=" * 60)

# Convert train data
print("\n[1/3] Converting train.npz...")
train = np.load(os.path.join(PROCESSED_DIR, 'train.npz'))
np.save(os.path.join(PROCESSED_DIR, 'X_train.npy'), train['x'])
np.save(os.path.join(PROCESSED_DIR, 'Y_train.npy'), train['y'])
print(f"    X_train.npy: {train['x'].shape}")
print(f"    Y_train.npy: {train['y'].shape}")
del train

# Convert val data
print("\n[2/3] Converting val.npz...")
val = np.load(os.path.join(PROCESSED_DIR, 'val.npz'))
np.save(os.path.join(PROCESSED_DIR, 'X_val.npy'), val['x'])
np.save(os.path.join(PROCESSED_DIR, 'Y_val.npy'), val['y'])
print(f"    X_val.npy: {val['x'].shape}")
print(f"    Y_val.npy: {val['y'].shape}")
del val

# Convert test data
print("\n[3/3] Converting test.npz...")
test = np.load(os.path.join(PROCESSED_DIR, 'test.npz'))
np.save(os.path.join(PROCESSED_DIR, 'X_test.npy'), test['x'])
np.save(os.path.join(PROCESSED_DIR, 'Y_test.npy'), test['y'])
print(f"    X_test.npy: {test['x'].shape}")
print(f"    Y_test.npy: {test['y'].shape}")
del test

# List new files
print("\n" + "=" * 60)
print("CONVERSION COMPLETE")
print("=" * 60)

print("\nNew files to upload:")
for f in sorted(os.listdir(PROCESSED_DIR)):
    if f.endswith('.npy') or f == 'stats.npz':
        size = os.path.getsize(os.path.join(PROCESSED_DIR, f)) / (1024**3)
        print(f"  {f}: {size:.2f} GB")

print("""
UPLOAD TO GOOGLE DRIVE:
  X_train.npy, Y_train.npy
  X_val.npy, Y_val.npy
  X_test.npy, Y_test.npy
  stats.npz

These files support memory-mapping and won't crash Colab!
""")
