```mermaid
graph TD
    subgraph Input Processing
        Raw[Raw ERA5 Data (B, 24, 31, 41, 2)] --> Log[Log Transform]
        Log --> Norm[Z-Score Norm]
        Norm --> Tensor[Input Tensor X (B, 24, 2, 31, 41)]
    end

    subgraph Encoder-Decoder
        Tensor --> Enc1[ConvLSTM Enc 1 (128ch, 3x3)]
        Enc1 --> Enc2[ConvLSTM Enc 2 (128ch, 3x3)]
        Enc2 --> State[Latent State H, C]
        
        State --> Dec1[ConvLSTM Dec 1 (128ch, 3x3)]
        Dec1 --> Dec2[ConvLSTM Dec 2 (128ch, 3x3)]
        Dec2 --> Feat[Feature Map (B, 6, 128, 31, 41)]
    end

    subgraph Output Head
        Feat --> Conv1x1[Conv 1x1]
        Conv1x1 --> Pred[Prediction Y_hat (B, 6, 2, 31, 41)]
    end

    subgraph Loss Calculation
        Pred --> WMSE[Weighted MSE Loss]
        Pred --> SSIM[SSIM Loss]
        WMSE --> Total[Total Loss]
        SSIM --> Total
    end
```
