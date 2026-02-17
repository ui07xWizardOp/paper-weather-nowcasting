# Visual Logic Map: Figure 1 (Architecture)

## Goal
Transform Figure 1 from a generic "Box Diagram" into a "Data Flow Narrative."

## The "Story" of the Tensor
1.  **Input ($t-2 \to t$)**:
    -   Visual: Stack of 3 Grids (Blue).
    -   Annotation: "Historical Context"
2.  **Encoder (The Compression)**:
    -   Visual: Cones narrowing.
    -   Key Detail: Show the $Cell_{State}$ ($C_t$) passing *horizontally* between blocks to emphasize "Memory".
3.  **The Bottleneck**:
    -   Visual: Smallest, densest cube.
    -   Annotation: "Latent Spatio-Temporal Representation"
4.  **Decoder (The Expansion)**:
    -   Visual: Cones widening.
    -   Key Detail: Show the "Prediction" ($Y_{t+1}$) being fed back as "Input" for the next step ($Y_{t+2}$).
    -   Annotation: "Autoregressive Rollout"
5.  **Output ($t+1 \to t+6$)**:
    -   Visual: Stack of 6 Grids (Red/Orange).
    -   Annotation: "Future Forecast"

## Aesthetic Rules
-   **Color Coding**: Inputs (Blue), Operations (Grey), Memory (Green), Outputs (Red).
-   **Flow**: strictly Left-to-Right.
-   **Icons**: Use standard symbols for Convolution ($\otimes$) and Element-wise add ($\oplus$).
