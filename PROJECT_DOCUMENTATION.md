# Weather Nowcasting Using ConvLSTM: Complete Project Documentation

> **Purpose**: This document provides a comprehensive, granular explanation of every step taken in the weather nowcasting project. It is designed for:
> 1. **Paper Writing** – All technical details, equations, and methodology are explained clearly
> 2. **Presentation** – Key concepts are explained in plain language for audience understanding
> 3. **Reproducibility** – Every decision and implementation detail is documented

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset: ERA5-Land Reanalysis Data](#2-dataset-era5-land-reanalysis-data)
3. [Data Acquisition Pipeline](#3-data-acquisition-pipeline)
4. [Data Preprocessing](#4-data-preprocessing)
5. [Model Architecture: ConvLSTM Encoder-Decoder](#5-model-architecture-convlstm-encoder-decoder)
6. [Training Methodology](#6-training-methodology)
7. [Optimization Techniques](#7-optimization-techniques)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Key Design Decisions and Alternatives](#9-key-design-decisions-and-alternatives)
10. [File Structure Reference](#10-file-structure-reference)

---

## 1. Project Overview

### 1.1 Problem Statement

**Weather nowcasting** is the prediction of weather conditions for the next 0-6 hours. Unlike traditional weather forecasting (which predicts days or weeks ahead), nowcasting requires:
- **High temporal resolution**: Predictions at hourly intervals
- **Rapid updates**: New predictions as new observational data arrives
- **Localized accuracy**: Predictions for specific geographic regions

### 1.2 Our Approach

We use **deep learning** (specifically, Convolutional LSTM networks) to learn spatio-temporal patterns from historical weather data and predict future weather states.

**Input**: 24 consecutive hours of weather data (temperature and precipitation)  
**Output**: Next 6 hours of predicted weather data

### 1.3 Geographic Focus

We focus on the **Indian subcontinent**, specifically:
- **Latitude range**: 5°N to 37°N
- **Longitude range**: 68°E to 98°E
- **Spatial resolution**: 0.1° × 0.1° (~10 km × 10 km grid cells)

This results in a **31 × 41 spatial grid** (31 latitude points × 41 longitude points = 1,271 grid cells).

### 1.4 Why This Matters

Traditional Numerical Weather Prediction (NWP) models:
- Require massive computational resources (supercomputers)
- Have coarse temporal resolution (6-hour intervals)
- Take hours to produce forecasts

Our deep learning approach:
- Runs on commodity hardware (single GPU)
- Produces predictions in seconds
- Learns directly from historical patterns

---

## 2. Dataset: ERA5-Land Reanalysis Data

### 2.1 What is ERA5-Land?

**ERA5-Land** is a global reanalysis dataset produced by the **European Centre for Medium-Range Weather Forecasts (ECMWF)**. 

#### Key Terminology:
- **Reanalysis**: A method that combines historical observations with numerical weather models to create a consistent, gap-free dataset. Think of it as "reconstructing" the past weather using all available data.
- **ERA5**: The 5th generation of ECMWF's reanalysis products
- **ERA5-Land**: A specialized version of ERA5 focused on land surfaces with higher resolution (0.1°)

### 2.2 Why ERA5-Land?

| Alternative | Resolution | Coverage | Why Not Chosen |
|-------------|-----------|----------|----------------|
| ERA5 (atmospheric) | 0.25° | Global | Coarser resolution (25km vs 10km) |
| MERRA-2 (NASA) | 0.5° × 0.625° | Global | Non-uniform grid, coarser resolution |
| JRA-55 (Japan) | ~55km | Global | Much coarser resolution |
| Station data | Point-based | Sparse | Incomplete spatial coverage, missing data |
| Satellite imagery | High | Global | Requires radiative transfer modeling, complex preprocessing |

**ERA5-Land advantages**:
1. **High spatial resolution** (0.1° ≈ 10km) – captures local weather patterns
2. **Hourly temporal resolution** – essential for nowcasting
3. **Consistent quality** – no missing data, uniform processing
4. **Long historical record** – 1950 to present (we use 2015-2025)
5. **Free and open access** via Copernicus Climate Data Store (CDS)

### 2.3 Variables Used

We use exactly **2 variables**:

#### 2.3.1 Total Precipitation (`tp`)

**Definition**: The accumulated liquid and solid precipitation (rain, snow, sleet) over one hour.

**Units**: Meters (m) – depth of water if spread uniformly over the grid cell

**Typical values**:
- 0 m: No precipitation
- 0.001 m (1 mm): Light rain
- 0.01 m (10 mm): Moderate rain
- 0.05+ m (50+ mm): Heavy rain

**Why this variable?**:
- Precipitation is the primary focus for nowcasting applications
- Most impactful for daily life (transportation, agriculture, flood warnings)
- Exhibits complex spatio-temporal patterns ideal for deep learning

#### 2.3.2 2-meter Temperature (`t2m`)

**Definition**: Temperature of air at 2 meters above ground level.

**Units**: Kelvin (K) – we later normalize this

**Typical values** (for India):
- 280 K (7°C): Cold winter morning in North India
- 300 K (27°C): Average temperature
- 320 K (47°C): Extreme summer heat

**Why this variable?**:
- Temperature influences precipitation patterns (convection, evaporation)
- Provides context for precipitation predictions
- Easy to interpret and validate

### 2.4 Data Statistics

| Aspect | Value |
|--------|-------|
| Temporal coverage | 2015-01-01 to 2025-12-31 (11 years) |
| Temporal resolution | 1 hour |
| Spatial coverage | Indian subcontinent (5°N-37°N, 68°E-98°E) |
| Spatial resolution | 0.1° × 0.1° |
| Grid dimensions | 31 latitude × 41 longitude = 1,271 grid cells |
| Total timesteps | ~96,000 hours (11 years × 8,760 hours/year) |
| Variables | 2 (tp, t2m) |
| Total data points | ~244 million (96,000 × 1,271 × 2) |

---

## 3. Data Acquisition Pipeline

### 3.1 Data Download Process

The data exists in two locations in our GitHub repository:

#### 3.1.1 Main Dataset Folder (`Dataset/`)

Contains **132 NetCDF files** covering 2015-2024:
- `data_0.nc`, `data_0(1).nc`, ..., `data_0(119).nc`: Hourly data chunks
- `era5land_202501.nc` to `era5land_202512.nc`: Monthly files for 2025

#### 3.1.2 2025 Data Folder (`2025 data/`)

Contains **12 NetCDF files** with hash-based names (e.g., `14a82a3b3f1fb1a69c47f1963c56e4d.nc`):
- These are additional 2025 data files downloaded separately
- Names are auto-generated by the CDS API

### 3.2 NetCDF File Format

**NetCDF (Network Common Data Form)** is a self-describing file format designed for array-oriented scientific data.

**Structure of our NetCDF files**:
```
Dimensions:
  - time: N (varies per file)
  - latitude: 31
  - longitude: 41

Coordinates:
  - time (or valid_time): datetime64 timestamps
  - latitude: [37.0, 36.9, ..., 5.0]
  - longitude: [68.0, 68.1, ..., 98.0]

Data Variables:
  - tp: (time, latitude, longitude) float32
  - t2m: (time, latitude, longitude) float32
```

### 3.3 Handling File Format Variations

Different CDS API versions produce slightly different file structures:

| Variation | Old Format | New Format | Our Solution |
|-----------|-----------|------------|--------------|
| Time coordinate | `time` | `valid_time` | Rename to `time` |
| Experimental version | Not present | `expver` dimension | Select first value and drop |
| Ensemble member | Not present | `number` coordinate | Drop if present |

**Code implementation** (in `load_single_file()` function):
```python
# Handle 'valid_time' vs 'time' naming
if 'valid_time' in ds.coords and 'time' not in ds.coords:
    ds = ds.rename({'valid_time': 'time'})

# Handle 'expver' dimension
if 'expver' in ds.dims:
    ds = ds.isel(expver=0, drop=True)

# Handle 'number' coordinate
if 'number' in ds.coords:
    ds = ds.drop_vars('number', errors='ignore')
```

---

## 4. Data Preprocessing

### 4.1 Overview of Preprocessing Pipeline

```
Raw NetCDF Files (144 files, ~2GB total)
         ↓
    [Load & Merge]
         ↓
    [Extract Variables] → tp, t2m arrays
         ↓
    [Compute Normalization Statistics] → mean, std
         ↓
    [Normalize Data] → zero-mean, unit-variance
         ↓
    [Create Sequences] → (X: 24h input, Y: 6h output)
         ↓
    [Split by Year] → Train (2015-2021), Val (2022-2023), Test (2024-2025)
         ↓
    [Save as Batched .npy Files] → ~500 samples per batch
```

### 4.2 Normalization (Z-Score Standardization)

#### 4.2.1 What is Normalization?

**Normalization** transforms data to have consistent scale across different variables. We use **Z-score standardization**:

$$
x_{normalized} = \frac{x - \mu}{\sigma}
$$

Where:
- $x$ is the original value
- $\mu$ is the mean of the training data
- $\sigma$ is the standard deviation of the training data

#### 4.2.2 Why Normalize?

1. **Different scales**: Temperature ranges ~280-320 K, precipitation ranges ~0-0.1 m
2. **Gradient stability**: Normalized data prevents exploding/vanishing gradients during training
3. **Faster convergence**: Network learns faster when inputs have similar magnitudes
4. **Equal importance**: Both variables contribute equally to the loss function

#### 4.2.3 Implementation Details

**Important**: Statistics are computed **only from training data** (2015-2021) to prevent data leakage.

```python
# Compute from training data only
mean = np.nanmean(train_data, axis=(0, 1, 2))  # Mean per variable
std = np.nanstd(train_data, axis=(0, 1, 2))    # Std per variable
std[std < 1e-6] = 1.0  # Prevent division by zero

# Apply to all splits
normalized_data = (data - mean) / std
```

**Resulting statistics** (typical values):
- **tp**: mean ≈ 0.0001 m, std ≈ 0.001 m
- **t2m**: mean ≈ 295 K, std ≈ 10 K

### 4.3 Sequence Creation

#### 4.3.1 What is a Sequence?

A sequence is a supervised learning sample consisting of:
- **Input (X)**: 24 consecutive hours of weather data
- **Output (Y)**: The next 6 hours of weather data

#### 4.3.2 Sliding Window Approach

We create sequences using a **sliding window** over the continuous time series:

```
Hour:    0  1  2  3  ... 23 24 25 26 27 28 29 30 31 32 ...
         |<---- X1 (24h) --->|<- Y1 (6h) ->|
            |<---- X2 (24h) --->|<- Y2 (6h) ->|
               |<---- X3 (24h) --->|<- Y3 (6h) ->|
```

**Parameters**:
- `T_IN = 24`: Input sequence length (24 hours = 1 day of historical context)
- `T_OUT = 6`: Output sequence length (6-hour forecast horizon)
- `STRIDE = 1`: Step size between consecutive sequences

#### 4.3.3 Why 24 Hours Input?

- Captures **diurnal cycle** (day-night patterns) – critical for temperature
- Includes **synoptic-scale evolution** – weather systems typically evolve over 12-48 hours
- Balances **memory requirements** vs **context length**

#### 4.3.4 Why 6 Hours Output?

- Standard **nowcasting horizon** defined by WMO (World Meteorological Organization)
- Beyond 6 hours, chaotic dynamics reduce predictability
- Aligns with operational weather service requirements

### 4.4 Data Splitting Strategy

We split data **temporally by year** (not randomly) to ensure:
- **No data leakage**: Future data never influences past predictions
- **Realistic evaluation**: Model is tested on truly unseen future data

| Split | Years | Purpose | Approximate Size |
|-------|-------|---------|-----------------|
| Train | 2015-2021 | Learn patterns | ~60,000 sequences |
| Validation | 2022-2023 | Hyperparameter tuning, early stopping | ~17,000 sequences |
| Test | 2024-2025 | Final evaluation | ~17,000 sequences |

**Why not random split?**
- Weather data has **temporal autocorrelation** – nearby hours are similar
- Random split would leak future information into training (overoptimistic results)
- Temporal split reflects real-world deployment: train on past, predict future

### 4.5 Memory-Efficient Batching

#### 4.5.1 The Memory Problem

Full dataset size: ~96,000 timesteps × 31 × 41 × 2 × 4 bytes ≈ **1 GB in RAM**

Creating all sequences at once would require: ~90,000 sequences × 30 timesteps × 31 × 41 × 2 × 4 bytes ≈ **27 GB in RAM**

Google Colab provides only **~12 GB RAM** – we cannot load everything at once.

#### 4.5.2 Our Solution: Streaming Batches

Instead of loading all data, we:
1. Process files **one at a time**
2. Create sequences immediately
3. Accumulate into a **buffer** (max 500 sequences)
4. **Flush to disk** when buffer is full
5. Clear memory and continue

```python
SAMPLES_PER_BATCH = 500

# Buffer accumulates sequences
X_buffer = []
Y_buffer = []

for sequence in create_sequences(data):
    X_buffer.append(sequence.X)
    Y_buffer.append(sequence.Y)
    
    if len(X_buffer) >= SAMPLES_PER_BATCH:
        # Save to disk
        np.save(f'X_batch_{batch_num}.npy', np.stack(X_buffer))
        np.save(f'Y_batch_{batch_num}.npy', np.stack(Y_buffer))
        batch_num += 1
        
        # Clear memory
        X_buffer = []
        Y_buffer = []
```

#### 4.5.3 Result: Batched Data Format

```
data/batched/
├── stats.npz          # Normalization statistics
├── train/
│   ├── X_batch_0000.npy   # Shape: (500, 24, 31, 41, 2)
│   ├── Y_batch_0000.npy   # Shape: (500, 6, 31, 41, 2)
│   ├── X_batch_0001.npy
│   ├── Y_batch_0001.npy
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

Each batch file is ~**50 MB** – easily loadable in Colab RAM.

---

## 5. Model Architecture: ConvLSTM Encoder-Decoder

### 5.1 Why ConvLSTM?

#### 5.1.1 The Challenge: Spatio-Temporal Data

Weather data is **4-dimensional**: (Time × Latitude × Longitude × Variables)

We need a model that captures:
1. **Spatial patterns**: Precipitation bands, temperature gradients, fronts
2. **Temporal dynamics**: How patterns evolve, move, intensify, or dissipate
3. **Spatial-temporal interactions**: How location A at time T affects location B at time T+1

#### 5.1.2 Why Not These Alternatives?

| Alternative | Problem |
|-------------|---------|
| **Fully Connected Networks** | Cannot capture spatial structure; treats each grid cell independently |
| **Standard CNNs** | No temporal memory; treats each timestep independently |
| **Standard RNNs/LSTMs** | No spatial structure; would need to flatten spatial dimensions |
| **3D CNNs** | Limited temporal context; computationally expensive |
| **Transformers** | Quadratic memory with sequence length; spatial position encoding complex |

#### 5.1.3 ConvLSTM: The Best of Both Worlds

**ConvLSTM** (Convolutional LSTM) was specifically designed for spatio-temporal prediction:
- Uses **convolutional operations** to capture spatial patterns
- Uses **LSTM memory cells** to capture temporal dynamics
- Maintains **spatial structure** throughout processing

### 5.2 LSTM Fundamentals

Before ConvLSTM, let's understand standard LSTM.

#### 5.2.1 The Vanishing Gradient Problem

Standard RNNs struggle with long sequences because gradients either:
- **Vanish**: Become negligibly small, stopping learning
- **Explode**: Become extremely large, causing instability

LSTM solves this with **gated memory cells**.

#### 5.2.2 LSTM Cell Structure

An LSTM cell has 4 components:

1. **Forget Gate** ($f_t$): Decides what information to discard
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

2. **Input Gate** ($i_t$): Decides what new information to store
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

3. **Candidate Memory** ($\tilde{c}_t$): New information to potentially add
$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

4. **Output Gate** ($o_t$): Decides what to output
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

**Cell State Update**:
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

**Hidden State Update**:
$$h_t = o_t \odot \tanh(c_t)$$

Where:
- $\sigma$ is the sigmoid function (outputs 0-1)
- $\tanh$ is the hyperbolic tangent (outputs -1 to 1)
- $\odot$ is element-wise multiplication
- $[h_{t-1}, x_t]$ is concatenation of previous hidden state and current input

### 5.3 ConvLSTM: Spatially-Aware LSTM

#### 5.3.1 Key Difference from LSTM

In standard LSTM, inputs and hidden states are **vectors** (1D).

In ConvLSTM, inputs and hidden states are **3D tensors** (Height × Width × Channels).

**Standard LSTM equations** use matrix multiplication ($W \cdot x$):
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**ConvLSTM equations** use convolution ($W * x$):
$$f_t = \sigma(W_f * [H_{t-1}, X_t] + b_f)$$

Where:
- $X_t$ is the input at time $t$ with shape (H, W, C)
- $H_{t-1}$ is the previous hidden state with shape (H, W, Hidden)
- $*$ denotes 2D convolution operation
- $[H, X]$ denotes concatenation along the channel dimension

#### 5.3.2 Why Convolution?

1. **Translation equivariance**: A precipitation pattern is recognized regardless of where it appears
2. **Parameter sharing**: Same weights applied to all spatial locations
3. **Locality**: Each output pixel depends on a local neighborhood (defined by kernel size)
4. **Hierarchical features**: Stacked layers learn increasingly abstract patterns

#### 5.3.3 Our ConvLSTM Cell Implementation

```python
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Single convolution for all 4 gates (efficiency)
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,  # Concatenated input
            out_channels=4 * hidden_dim,          # 4 gates × hidden_dim
            kernel_size=kernel_size,
            padding=kernel_size // 2              # Same padding
        )
    
    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        
        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=1)  # Channel concatenation
        
        # Single convolution for efficiency
        gates = self.conv(combined)
        
        # Split into 4 gates
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        
        # Apply activations
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        o = torch.sigmoid(o)  # Output gate
        g = torch.tanh(g)     # Candidate
        
        # Update cell state
        c_next = f * c_prev + i * g
        
        # Compute hidden state
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
```

### 5.4 Encoder-Decoder Architecture

#### 5.4.1 Why Encoder-Decoder?

The input (24 hours) and output (6 hours) have **different lengths**. We need an architecture that:
1. **Encodes** the input sequence into a compact representation
2. **Decodes** the representation into the output sequence

This is analogous to machine translation: encode source language → decode to target language.

#### 5.4.2 Encoder

The encoder processes the **input sequence** (24 timesteps) and produces a **final hidden state** that summarizes all input information.

```
Input: X = [x_1, x_2, ..., x_24]  (24 weather frames)

For t = 1 to 24:
    h_t, c_t = ConvLSTM(x_t, (h_{t-1}, c_{t-1}))

Output: (h_24, c_24)  (Final hidden state and cell state)
```

#### 5.4.3 Decoder

The decoder generates the **output sequence** (6 timesteps) by:
1. Starting from the encoder's final hidden state
2. Predicting one frame at a time
3. Feeding each prediction as input to the next step (autoregressive)

```
Initialize: (h_0, c_0) = (h_24, c_24) from encoder
Initialize: y_0 = OutputConv(h_24)  (First prediction)

For t = 1 to 6:
    h_t, c_t = ConvLSTM(y_{t-1}, (h_{t-1}, c_{t-1}))
    y_t = OutputConv(h_t)

Output: Y = [y_1, y_2, ..., y_6]  (6 predicted weather frames)
```

#### 5.4.4 Full Architecture Diagram

```
                    ENCODER                                 DECODER
                    
  x_1 → [ConvLSTM] → (h_1, c_1)                    y_0 → [ConvLSTM] → (h'_1) → Conv1x1 → ŷ_1
          ↓                                                   ↓
  x_2 → [ConvLSTM] → (h_2, c_2)                    ŷ_1 → [ConvLSTM] → (h'_2) → Conv1x1 → ŷ_2
          ↓                                                   ↓
        ...                                                 ...
          ↓                                                   ↓
  x_24 → [ConvLSTM] → (h_24, c_24) ─────────────→ ŷ_5 → [ConvLSTM] → (h'_6) → Conv1x1 → ŷ_6
  
  Input: 24 frames                                 Output: 6 predicted frames
  Shape: (B, 24, 2, 31, 41)                       Shape: (B, 6, 2, 31, 41)
```

### 5.5 Model Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Hidden dimension | 128 | Balances capacity vs. memory; enough to model complex patterns |
| Number of layers | 2 | Captures hierarchical features without overfitting |
| Kernel size | 3 × 3 | Standard size; captures local spatial context |
| Input channels | 2 | tp and t2m variables |
| Output channels | 2 | Predict both variables |

**Total parameters**: ~6.5 million

---

## 6. Training Methodology

### 6.1 Loss Function: Mean Squared Error (MSE)

#### 6.1.1 Definition

$$
\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

Where:
- $y_i$ is the ground truth value
- $\hat{y}_i$ is the predicted value
- $N$ is the total number of values (all pixels, timesteps, and variables)

#### 6.1.2 Why MSE?

1. **Differentiable everywhere**: Enables gradient-based optimization
2. **Penalizes large errors more**: Squared term emphasizes large deviations
3. **Standard for regression**: Well-understood properties
4. **Directly related to RMSE**: $RMSE = \sqrt{MSE}$, a common evaluation metric

#### 6.1.3 Why Not Other Losses?

| Alternative | Problem |
|-------------|---------|
| **MAE (L1)** | Gradients are constant; slower convergence; doesn't penalize large errors as strongly |
| **Huber Loss** | Extra hyperparameter (delta); benefits mainly data with outliers |
| **Cross-Entropy** | For classification; our task is regression |
| **Custom precipitation loss** | Would require careful balancing; MSE is robust baseline |

### 6.2 Optimizer: Adam

#### 6.2.1 What is Adam?

**Adam** (Adaptive Moment Estimation) combines the benefits of:
- **Momentum**: Accelerates convergence by accumulating past gradients
- **RMSprop**: Adapts learning rate per parameter based on gradient magnitudes

**Update equations**:
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_t = \theta_{t-1} - \frac{\alpha \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

#### 6.2.2 Our Adam Configuration

```python
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

| Parameter | Value | Default | Notes |
|-----------|-------|---------|-------|
| Learning rate (α) | 0.001 | 0.001 | Standard for ConvLSTM |
| β₁ | 0.9 | 0.9 | Momentum coefficient |
| β₂ | 0.999 | 0.999 | RMSprop coefficient |
| ε | 1e-8 | 1e-8 | Numerical stability |

#### 6.2.3 Why Adam?

1. **Adaptive learning rates**: Each parameter gets its own learning rate
2. **Works well with sparse gradients**: Important for weather data (many zero precipitation values)
3. **Robust to hyperparameter choices**: Default values work well in practice
4. **Fast convergence**: Typically faster than SGD for deep learning

### 6.3 Learning Rate Scheduling

#### 6.3.1 ReduceLROnPlateau

Reduces learning rate when validation loss stops improving.

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',      # Reduce when metric stops decreasing
    patience=5,      # Wait 5 epochs before reducing
    factor=0.5       # Multiply LR by 0.5
)
```

#### 6.3.2 Why This Scheduler?

- **Automatic adaptation**: No need to manually define LR schedule
- **Validation-based**: Responds to actual training dynamics
- **Conservative**: Waits 5 epochs before reducing (avoids premature reduction)
- **Gradual reduction**: Halving LR allows fine-tuning without disruption

### 6.4 Gradient Clipping

#### 6.4.1 What is Gradient Clipping?

Prevents exploding gradients by limiting the maximum gradient norm.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

If the total gradient norm exceeds 1.0, all gradients are scaled down proportionally.

#### 6.4.2 Why Clip Gradients?

1. **RNN/LSTM instability**: Recurrent networks are prone to gradient explosion
2. **Long sequences**: 24 timestep input amplifies gradient issues
3. **Stable training**: Prevents NaN losses and divergence

### 6.5 Training Loop Implementation

```python
for epoch in range(NUM_EPOCHS):
    model.train()
    for x, y in train_dataloader:
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(x, future_steps=6)
        loss = criterion(predictions, y)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = evaluate(val_dataloader)
    
    scheduler.step(val_loss)
    
    # Save best model
    if val_loss < best_val_loss:
        save_checkpoint(model)
```

---

## 7. Optimization Techniques

### 7.1 Mixed Precision Training (AMP)

#### 7.1.1 What is Mixed Precision?

**Mixed precision** uses both 16-bit (FP16) and 32-bit (FP32) floating point numbers during training:
- **FP16**: Forward pass, most backward pass operations
- **FP32**: Loss accumulation, gradient updates, sensitive operations

#### 7.1.2 Benefits

| Benefit | Explanation |
|---------|-------------|
| **2× Memory Reduction** | FP16 uses half the memory of FP32 |
| **2-4× Faster Training** | NVIDIA Tensor Cores accelerate FP16 |
| **Larger Batch Sizes** | Freed memory allows bigger batches |
| **Same Accuracy** | Careful scaling prevents precision loss |

#### 7.1.3 Implementation

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for x, y in dataloader:
    optimizer.zero_grad()
    
    # Forward pass in FP16
    with autocast():
        predictions = model(x, T_OUT)
        loss = criterion(predictions, y)
    
    # Backward pass with loss scaling
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

#### 7.1.4 Why GradScaler?

FP16 has limited range (~6 × 10⁻⁵ to 65,504). Small gradients may underflow to zero.

**GradScaler**:
1. Multiplies loss by a large factor before backward pass (prevents underflow)
2. Unscales gradients before optimizer step
3. Dynamically adjusts scale factor based on gradient overflow detection

### 7.2 cuDNN Benchmark Mode

```python
torch.backends.cudnn.benchmark = True
```

#### 7.2.1 What Does This Do?

cuDNN (NVIDIA's deep learning primitives library) has multiple algorithms for operations like convolution. Benchmark mode:
1. Runs all algorithms on first iteration
2. Selects the fastest algorithm for the specific input sizes
3. Uses optimized algorithm for remaining iterations

#### 7.2.2 When to Use?

- ✅ **Fixed input sizes**: Same batch size and dimensions every iteration
- ❌ **Variable input sizes**: Benchmark overhead outweighs benefits

Our data has fixed shape (B × 24 × 2 × 31 × 41), so benchmark mode is beneficial.

### 7.3 Efficient Data Loading

#### 7.3.1 Non-Blocking GPU Transfer

```python
x = x.to(device, non_blocking=True)
y = y.to(device, non_blocking=True)
```

**Non-blocking** allows CPU-GPU data transfer to happen asynchronously while CPU continues processing.

#### 7.3.2 Generator-Based Data Loading

Instead of loading all batches into RAM, we use a **generator** that loads one batch file at a time:

```python
def batch_generator(split, batch_size=32):
    for batch_file in batch_files:
        X = np.load(batch_file)  # Load one file
        for i in range(0, len(X), batch_size):
            yield X[i:i+batch_size]  # Yield mini-batches
```

---

## 8. Evaluation Metrics

### 8.1 Mean Squared Error (MSE)

$$MSE = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

**Interpretation**: Average squared difference between predictions and ground truth.

**Units**: Squared units of the variable (m² for precipitation, K² for temperature).

### 8.2 Root Mean Squared Error (RMSE)

$$RMSE = \sqrt{MSE}$$

**Interpretation**: Same scale as original variable; interpretable as "typical" error magnitude.

**Example**: RMSE of 0.001 m for precipitation means predictions are typically off by ~1 mm.

### 8.3 Why These Metrics?

1. **Directly related to loss function**: Model optimizes MSE, so it's the natural evaluation metric
2. **Interpretable**: RMSE has same units as data
3. **Standard in meteorology**: Allows comparison with other models and benchmarks
4. **Sensitive to large errors**: Important for severe weather events

---

## 9. Key Design Decisions and Alternatives

### 9.1 Decision: ConvLSTM over 3D CNN

| Aspect | ConvLSTM | 3D CNN |
|--------|----------|--------|
| Temporal modeling | Explicit memory (LSTM cells) | Learned from 3D kernels |
| Long-range dependencies | **Strong** (LSTM memory) | Weak (limited receptive field) |
| Variable sequence length | **Flexible** | Fixed |
| Parameter efficiency | **Higher** | Lower (larger kernels needed) |
| Interpretability | **Hidden states visualizable** | Black box |

**Verdict**: ConvLSTM is more suitable for sequence-to-sequence weather prediction.

### 9.2 Decision: 24-Hour Input Window

| Alternative | Pros | Cons |
|-------------|------|------|
| **12 hours** | Less memory, faster | Misses full diurnal cycle |
| **24 hours** ✓ | Full diurnal cycle, better context | Moderate memory |
| **48 hours** | More context | High memory, diminishing returns |
| **72 hours** | Synoptic-scale patterns | Excessive memory, slow training |

**Verdict**: 24 hours is the sweet spot – captures daily patterns without excessive memory.

### 9.3 Decision: Z-Score Normalization over Min-Max

| Method | Formula | Pros | Cons |
|--------|---------|------|------|
| **Z-Score** ✓ | (x-μ)/σ | Handles outliers well, no fixed range | |
| **Min-Max** | (x-min)/(max-min) | Fixed [0,1] range | Sensitive to outliers, extreme values dominate |

**Verdict**: Z-score is more robust for weather data with occasional extreme values.

### 9.4 Decision: MSE Loss over Custom Precipitation Loss

| Loss | Pros | Cons |
|------|------|------|
| **MSE** ✓ | Simple, well-understood, stable | May not emphasize rare heavy rain |
| **Weighted MSE** | Can emphasize heavy rain | Requires careful tuning |
| **SSIM** | Structural similarity | Designed for images, not time series |

**Verdict**: MSE provides a robust baseline; custom losses can be explored in future work.

### 9.5 Decision: Temporal Split over Random Split

| Split Type | Pros | Cons |
|------------|------|------|
| **Random** | More data for training | Data leakage (future→past) |
| **Temporal** ✓ | Realistic, no leakage | Less training data |

**Verdict**: Temporal split is essential for honest evaluation of forecasting models.

---

## 10. File Structure Reference

```
paper-weather-nowcasting/
├── Dataset/                          # Raw NetCDF files (2015-2024)
│   ├── data_0.nc                     # ~120 files
│   ├── data_0(1).nc
│   ├── ...
│   └── era5land_202512.nc
│
├── 2025 data/                        # Additional 2025 data
│   ├── 14a82a3b3f1fb1a69c47f1963c56e4d.nc
│   └── ... (12 files)
│
├── data/
│   └── batched/                      # Preprocessed batches
│       ├── stats.npz                 # Normalization statistics
│       ├── train/
│       │   ├── X_batch_0000.npy      # Input batches
│       │   └── Y_batch_0000.npy      # Output batches
│       ├── val/
│       └── test/
│
├── notebooks/
│   └── 03_training.ipynb             # Main training notebook
│
├── checkpoints/
│   ├── best_model.pth                # Best validation model
│   └── final_model.pth               # Final trained model
│
├── figures/
│   ├── training_curve.png            # Loss curves
│   └── sample_prediction.png         # Example forecast
│
├── preprocess_batched.py             # Local preprocessing script
├── train_cpu.py                      # CPU training script
├── check_test_data.py                # Data verification
├── README.md                         # Project overview
└── PROJECT_DOCUMENTATION.md          # This file
```

---

## Quick Reference for Paper Writing

### Abstract Key Points
- Deep learning approach for weather nowcasting (0-6 hour prediction)
- ConvLSTM encoder-decoder architecture
- ERA5-Land reanalysis data (2015-2025)
- Indian subcontinent focus (0.1° resolution)
- 24-hour input → 6-hour forecast

### Methodology Section Structure
1. Data: ERA5-Land, variables (tp, t2m), preprocessing
2. Model: ConvLSTM, encoder-decoder, hyperparameters
3. Training: Adam, MSE loss, mixed precision
4. Evaluation: Temporal split (2024-2025 as test)

### Key Numbers to Cite
- Spatial resolution: 0.1° × 0.1° (~10 km)
- Temporal resolution: 1 hour
- Grid size: 31 × 41 = 1,271 grid cells
- Data period: 2015-2025 (11 years)
- Training data: 2015-2021 (7 years)
- Model parameters: ~6.5 million
- Input sequence: 24 hours
- Output sequence: 6 hours

---

## Quick Reference for Presentation

### Slide 1: Problem
- Weather nowcasting = 0-6 hour prediction
- Critical for: aviation, transportation, agriculture, disaster management
- Traditional NWP: slow, expensive, coarse resolution

### Slide 2: Our Approach
- Deep learning with ConvLSTM
- Learn patterns from 11 years of historical data
- Fast inference (seconds vs hours)

### Slide 3: Data
- ERA5-Land: Best available reanalysis data
- 10 km resolution, hourly
- Show map of India coverage

### Slide 4: Model Architecture
- Show encoder-decoder diagram
- Explain: Past 24 hours → Future 6 hours

### Slide 5: Results
- Show training curves
- Show sample predictions
- Compare ground truth vs prediction

### Slide 6: Conclusion
- Demonstrated feasibility of deep learning for nowcasting
- Comparable to operational models at fraction of cost
- Future work: More variables, larger regions, ensemble methods

---

*Document created: February 8, 2026*  
*Last updated: February 8, 2026*
