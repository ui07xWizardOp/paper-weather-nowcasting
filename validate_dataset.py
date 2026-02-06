"""
Dataset Validation Script for NetCDF files
Checks for file integrity, structure consistency, and potential issues
"""

import os
import sys

# Check if required libraries are available
try:
    import xarray as xr
    import numpy as np
except ImportError as e:
    print(f"Missing library: {e}")
    print("Please install required libraries: pip install xarray netCDF4 numpy")
    sys.exit(1)

def validate_dataset(dataset_path):
    """Validate all NetCDF files in the dataset directory"""
    
    # Get all .nc files
    files = sorted([f for f in os.listdir(dataset_path) if f.endswith('.nc')])
    print(f"=" * 60)
    print(f"Dataset Validation Report")
    print(f"=" * 60)
    print(f"Dataset Path: {dataset_path}")
    print(f"Total NetCDF files found: {len(files)}")
    print(f"=" * 60)
    
    valid_files = []
    corrupted_files = []
    file_info = []
    
    # Check each file
    for i, filename in enumerate(files):
        filepath = os.path.join(dataset_path, filename)
        file_size = os.path.getsize(filepath)
        
        try:
            ds = xr.open_dataset(filepath)
            
            info = {
                'filename': filename,
                'size_bytes': file_size,
                'dimensions': dict(ds.dims),
                'data_vars': list(ds.data_vars),
                'coords': list(ds.coords),
                'status': 'valid'
            }
            
            # Check for time coordinate
            if 'time' in ds.coords:
                info['time_range'] = (str(ds.time.values[0]), str(ds.time.values[-1]))
                info['time_steps'] = len(ds.time)
            elif 'valid_time' in ds.coords:
                info['time_range'] = (str(ds.valid_time.values[0]), str(ds.valid_time.values[-1]))
                info['time_steps'] = len(ds.valid_time)
            else:
                info['time_range'] = None
                info['time_steps'] = 0
            
            file_info.append(info)
            valid_files.append(filename)
            ds.close()
            
            # Progress indicator
            if (i + 1) % 20 == 0:
                print(f"Processed {i + 1}/{len(files)} files...")
                
        except Exception as e:
            corrupted_files.append({
                'filename': filename,
                'size_bytes': file_size,
                'error': str(e)
            })
    
    # Print summary
    print(f"\n" + "=" * 60)
    print(f"VALIDATION SUMMARY")
    print(f"=" * 60)
    print(f"✓ Valid files: {len(valid_files)}")
    print(f"✗ Corrupted/Invalid files: {len(corrupted_files)}")
    
    # Print first valid file details as reference
    if file_info:
        print(f"\n" + "-" * 60)
        print(f"SAMPLE FILE STRUCTURE (first valid file)")
        print(f"-" * 60)
        ref = file_info[0]
        print(f"Filename: {ref['filename']}")
        print(f"Size: {ref['size_bytes'] / 1024:.2f} KB")
        print(f"Dimensions: {ref['dimensions']}")
        print(f"Data Variables: {ref['data_vars']}")
        print(f"Coordinates: {ref['coords']}")
        if ref['time_range']:
            print(f"Time Range: {ref['time_range'][0]} to {ref['time_range'][1]}")
            print(f"Time Steps: {ref['time_steps']}")
    
    # Check for inconsistencies across files
    if len(file_info) > 1:
        print(f"\n" + "-" * 60)
        print(f"CONSISTENCY CHECK")
        print(f"-" * 60)
        
        ref_dims = file_info[0]['dimensions']
        ref_vars = set(file_info[0]['data_vars'])
        
        inconsistent_dims = []
        inconsistent_vars = []
        
        for info in file_info[1:]:
            # Check for different dimensions (excluding time)
            dims_match = True
            for key in ['latitude', 'longitude', 'lat', 'lon']:
                if key in ref_dims and key in info['dimensions']:
                    if ref_dims[key] != info['dimensions'][key]:
                        dims_match = False
                        
            if not dims_match:
                inconsistent_dims.append(info['filename'])
            
            if set(info['data_vars']) != ref_vars:
                inconsistent_vars.append(info['filename'])
        
        if inconsistent_dims:
            print(f"⚠ Files with different spatial dimensions: {len(inconsistent_dims)}")
            for f in inconsistent_dims[:5]:
                print(f"  - {f}")
        else:
            print(f"✓ All files have consistent spatial dimensions")
            
        if inconsistent_vars:
            print(f"⚠ Files with different variables: {len(inconsistent_vars)}")
            for f in inconsistent_vars[:5]:
                print(f"  - {f}")
        else:
            print(f"✓ All files have consistent variables")
    
    # List corrupted files
    if corrupted_files:
        print(f"\n" + "-" * 60)
        print(f"CORRUPTED FILES DETAILS")
        print(f"-" * 60)
        for cf in corrupted_files:
            print(f"\nFilename: {cf['filename']}")
            print(f"Size: {cf['size_bytes'] / 1024:.2f} KB")
            print(f"Error: {cf['error']}")
    
    # Size statistics
    print(f"\n" + "-" * 60)
    print(f"FILE SIZE STATISTICS")
    print(f"-" * 60)
    sizes = [info['size_bytes'] for info in file_info]
    if sizes:
        print(f"Min size: {min(sizes) / 1024:.2f} KB")
        print(f"Max size: {max(sizes) / 1024:.2f} KB")
        print(f"Avg size: {sum(sizes) / len(sizes) / 1024:.2f} KB")
        print(f"Total size: {sum(sizes) / (1024*1024):.2f} MB")
    
    return {
        'valid': valid_files,
        'corrupted': corrupted_files,
        'file_info': file_info
    }

if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(__file__), "Dataset")
    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}")
        sys.exit(1)
    
    result = validate_dataset(dataset_path)
    print(f"\n" + "=" * 60)
    print(f"Validation complete!")
    print(f"=" * 60)
