# Data Provenance Specification (Phase 21)

## Level 0: Raw Data Source
-   **Provider**: ECMWF (European Centre for Medium-Range Weather Forecasts).
-   **Dataset**: ERA5-Land Hourly Data from 1950 to present.
-   **DOI**: 10.24381/cds.e2161bac
-   **Variables**: Total Precipitation (`tp`), 2m Temperature (`t2m`).
-   **Access Method**: CDS API (`cdsapi` Python client).

## Level 1: Pre-processing (The "Tensors")
-   **Script**: `create_optimized_notebook.py` (Data Loading Section).
-   **Transformation 1**: Spatial Slicing (`lat: 37-8`, `lon: 68-97`).
-   **Transformation 2**: Temporal Aggregation (None - kept hourly).
-   **Transformation 3**: Normalization (Log-Transformed Z-Score).
    -   Formula: $x' = \frac{\log(x+1) - \mu_{log}}{\sigma_{log}}$
    -   Parameters: $\mu_{log}$ and $\sigma_{log}$ computed over Train Split (2015-2021).

## Level 2: Model Inputs
-   **Format**: `.npy` (NumPy Binary).
-   **Shape**: $(B, T, H, W, C)$.
-   **Storage**: Local Disk (Ephemeral) / Google Drive (Persistent).

## Level 3: Reproducibility
-   **Code**: `create_optimized_notebook.py` contains the exact seed `42`.
-   **Environment**: `requirements.txt` locks `numpy`, `xarray`, `torch`.
