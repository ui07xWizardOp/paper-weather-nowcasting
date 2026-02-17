```mermaid
graph TD
    classDef foundation fill:#f9f,stroke:#333,stroke-width:2px;
    classDef method fill:#bbf,stroke:#333,stroke-width:2px;
    classDef data fill:#dfd,stroke:#333,stroke-width:2px;
    classDef ours fill:#f96,stroke:#333,stroke-width:4px;

    L[LSTM (Hochreiter 1997)]:::foundation --> CL[ConvLSTM (Shi 2015)]:::method
    CL --> TG[TrajGRU (Shi 2017)]:::method
    CL --> OURS[Ours (Weighted MSE)]:::ours
    
    UNet[U-Net (Ronneberger 2015)]:::foundation --> MetNet[MetNet (Sonderby 2020)]:::method
    
    ERA5[ERA5 (Hersbach 2020)]:::data --> ERAL[ERA5-Land (Munoz 2021)]:::data
    ERAL --> OURS
    
    SSIM[SSIM (Wang 2004)]:::foundation --> OURS
    
    DGMR[DGMR (Ravuri 2021)]:::method -.->|Alternative| OURS
```
