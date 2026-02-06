"""
Download 2025 ERA5-Land Data from Copernicus CDS
================================================
This script downloads ERA5-Land hourly data for 2025 to complete your test dataset.

Prerequisites:
    1. Create account at: https://cds.climate.copernicus.eu/
    2. Get your API key from: https://cds.climate.copernicus.eu/api-how-to
    3. Create file ~/.cdsapirc with your credentials:
       url: https://cds.climate.copernicus.eu/api/v2
       key: YOUR_UID:YOUR_API_KEY

Usage:
    python download_2025_data.py

Note: Each month takes ~10-15 minutes to download. Total ~2-3 hours for full year.
"""

import os
import sys

try:
    import cdsapi
except ImportError:
    print("Installing cdsapi...")
    os.system("pip install cdsapi")
    import cdsapi

# Configuration - same region as your existing dataset
DATASET_DIR = os.path.join(os.path.dirname(__file__), "Dataset")
YEAR = 2025

# Bounding box from your existing dataset
# Lat: 17°N to 20°N, Lon: 81°E to 85°E (Odisha coastal region)
# Grid: 31 lat x 41 lon at 0.1° resolution
AREA = [20, 81, 17, 85]  # North, West, South, East

# Variables to download (same as existing data)
VARIABLES = ['total_precipitation', '2m_temperature']

# All hours
HOURS = [f'{h:02d}:00' for h in range(24)]

# All days
DAYS = [f'{d:02d}' for d in range(1, 32)]

# Months to download
MONTHS = [f'{m:02d}' for m in range(1, 13)]


def download_month(client, year, month):
    """Download one month of ERA5-Land data."""
    filename = f"era5land_{year}{month}.nc"
    filepath = os.path.join(DATASET_DIR, filename)
    
    if os.path.exists(filepath):
        print(f"  {filename} already exists, skipping...")
        return True
    
    print(f"  Downloading {filename}...")
    
    try:
        client.retrieve(
            'reanalysis-era5-land',
            {
                'product_type': 'reanalysis',
                'variable': VARIABLES,
                'year': str(year),
                'month': month,
                'day': DAYS,
                'time': HOURS,
                'area': AREA,
                'format': 'netcdf',
            },
            filepath
        )
        
        size_mb = os.path.getsize(filepath) / (1024**2)
        print(f"    Downloaded: {size_mb:.1f} MB")
        return True
        
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def main():
    print("=" * 60)
    print("DOWNLOADING 2025 ERA5-LAND DATA")
    print("=" * 60)
    
    # Check credentials
    cdsapirc = os.path.expanduser("~/.cdsapirc")
    if not os.path.exists(cdsapirc):
        print(f"""
ERROR: CDS API credentials not found!

Please create {cdsapirc} with:
    url: https://cds.climate.copernicus.eu/api/v2
    key: YOUR_UID:YOUR_API_KEY

Get your API key from: https://cds.climate.copernicus.eu/api-how-to
""")
        sys.exit(1)
    
    print(f"\n✓ Found CDS credentials at {cdsapirc}")
    
    # Create output directory
    os.makedirs(DATASET_DIR, exist_ok=True)
    print(f"Output directory: {DATASET_DIR}")
    
    # Initialize client
    print("\nConnecting to CDS API...")
    client = cdsapi.Client()
    
    # Download each month
    print(f"\nDownloading {YEAR} data (12 months)...")
    print("-" * 40)
    
    successful = 0
    failed = 0
    
    for month in MONTHS:
        if download_month(client, YEAR, month):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"\nSuccessful: {successful} months")
    print(f"Failed: {failed} months")
    
    if failed == 0:
        print(f"""
✓ All 2025 data downloaded successfully!

NEXT STEPS:
1. Re-run preprocessing to include 2025 in test set:
   python preprocess_batched.py

2. The test set will now include 2024-2025 (2 years)
""")
    else:
        print("\n⚠ Some downloads failed. Check your API credentials and try again.")


if __name__ == "__main__":
    main()
