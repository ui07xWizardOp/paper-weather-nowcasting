"""
Detailed Dataset Analysis for Colab Compatibility
"""
import xarray as xr
import os
import numpy as np

path = 'Dataset'
files = sorted([f for f in os.listdir(path) if f.endswith('.nc')])

print('='*70)
print('DETAILED DATASET ANALYSIS FOR COLAB COMPATIBILITY')
print('='*70)

# Check first few files in detail
issues = []
time_coords_used = set()
expver_issues = []

for i, f in enumerate(files[:10]):
    ds = xr.open_dataset(os.path.join(path, f))
    
    # Check time coord name
    if 'time' in ds.coords:
        time_coords_used.add('time')
    if 'valid_time' in ds.coords:
        time_coords_used.add('valid_time')
    
    # Check expver (common CDS issue)
    if 'expver' in ds.coords or 'expver' in ds.data_vars:
        if 'expver' in ds.coords:
            expver_vals = np.unique(ds.expver.values)
            if len(expver_vals) > 1:
                expver_issues.append(f'{f}: multiple expver values {expver_vals}')
    
    # Check for NaN values
    for var in ds.data_vars:
        nan_count = np.sum(np.isnan(ds[var].values.flatten()))
        total = ds[var].values.size
        if nan_count > 0:
            pct = nan_count / total * 100
            if pct > 50:
                issues.append(f'{f}: {var} has {pct:.1f}% NaN values')
    
    ds.close()

print(f'\nTime Coordinate Names Used: {time_coords_used}')
if 'valid_time' in time_coords_used and 'time' not in time_coords_used:
    print('[!] ISSUE: Files use valid_time instead of time')
    print('    Most older code expects time as the coordinate name')
    print('    Solution: Rename coordinate in Colab using:')
    print('    ds = ds.rename({"valid_time": "time"})')

if expver_issues:
    print(f'\nExpver Issues Found ({len(expver_issues)}):')
    for e in expver_issues[:5]:
        print(f'  {e}')
    print('  [!] Multiple expver values require merging before analysis')
else:
    print('\nNo expver issues found (single expver in files checked)')

if issues:
    print(f'\nNaN Issues ({len(issues)}):')
    for i in issues:
        print(f'  {i}')
else:
    print('\nNo significant NaN issues in checked files')

# Check time continuity
print('\n' + '='*70)
print('TIME RANGE ANALYSIS')
print('='*70)
all_times = []
for f in files[:20]:
    ds = xr.open_dataset(os.path.join(path, f))
    time_coord = 'valid_time' if 'valid_time' in ds.coords else 'time'
    times = ds[time_coord].values
    all_times.extend(times)
    print(f'{f}: {str(times[0])[:19]} to {str(times[-1])[:19]} ({len(times)} steps)')
    ds.close()
