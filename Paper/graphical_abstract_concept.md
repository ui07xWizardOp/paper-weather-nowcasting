# Graphical Abstract Concept

**Title**: Deep Spatio-Temporal Nowcasting for Indian Monsoon

**Layout**: Left-to-Right Flow

1.  **Input Panel (Left)**: 
    -   Stack of 24 frames of ERA5-Land maps (Spinning Globe icon -> India Focus).
    -   Label: "Input Sequence ($T=24h$)"

2.  **Model Core (Center)**:
    -   Schematic of ConvLSTM Cell (Convolution + Gates).
    -   Arrows showing "Encoder" compression and "Decoder" expansion.
    -   Highlight: "Weighted MSE Loss" impacting the gradients.

3.  **Output Panel (Right)**:
    -   Side-by-side comparison map:
        -   **Ours**: Sharp, intense red patches (Heavy Rain).
        -   **Baseline**: Blurry, blue smear (Light Rain).
    -   Metrics overlay: "CSI: 0.65 (+12%)".
