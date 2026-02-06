
import numpy as np
import os

try:
    meta = np.load('data/batched/test/metadata.npz')
    print(f"Test samples: {meta['n_samples']}")
    print(f"Test batches: {meta['n_batches']}")
    
    # Expected: 2 years * 365 days * 24 hours ~= 17520
    # Actually 2024 is leap year (366) + 2025 (365) = 731 days * 24 = 17544
    n_expected = 17544
    print(f"Expected approx: {n_expected}")
    
    if abs(meta['n_samples'] - n_expected) < 500: # Allow some margin for missing hours/files
        print("MATCHES_EXPECTED_RANGE")
    else:
        print("SAMPLE_COUNT_MISMATCH")

except Exception as e:
    print(f"Error: {e}")
