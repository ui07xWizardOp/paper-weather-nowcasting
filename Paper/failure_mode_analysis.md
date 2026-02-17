# Failure Mode Analysis

## 1. Orographic Locking (The Himalayan Wall)
-   **Observation**: The model consistently underestimates rainfall intensity along the sharp elevation gradients of the Himalayas (Northern boundary).
-   **Cause**: The $3 \times 3$ convolution kernel assumes local spatial smoothness. In reality, orographic lift causes extreme vertical discontinuities that lateral convolution struggles to capture without explicit elevation inputs.
-   **Mitigation**: Future reference: Concatenate a Digital Elevation Model (DEM) as a static channel.

## 2. Convective Initiation (The "Pop-up" Problem)
-   **Observation**: The model fails to predict the *start* of a storm in a previously clear pixel (0 $\to$ 1 transition).
-   **Cause**: ConvLSTM is fundamentally autoregressive; it relies on propagating existing states. It struggles to hallucinate new energy "out of thin air" (thermodynamic instability) because we do not feed it CAPE (Convective Available Potential Energy) variables.
-   **Mitigation**: Include thermodynamic stability indices (CAPE, CIN) in the input tensor.

## 3. Dissipation Lag (The "Ghost Rain")
-   **Observation**: The model continues to predict rain for 1-2 hours after a storm has physically dissipated.
-   **Cause**: The LSTM forget gate ($f_t$) is learned to be high to preserve long-term dependencies. It is slow to "dump" the state when the system loses energy rapidly.
