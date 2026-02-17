# Twitter/X Thread Abstract (Phase 15)

1/5 🧵
Startling Fact: Standard AI weather models "blur" extreme events to minimize error. Safe for stats, deadly for flood warning.
Our new paper "Deep Spatio-Temporal Nowcasting with Physically Grounded Loss" fixes this.
[Link to Paper] #AI #Meteorology #DeepLearning

2/5 🧠
The Problem: In India, "average" rain is useless. We need to predict the cloudburst.
We introduce a **Weighted MSE** loss that dynamically penalizes errors on heavy rain ($>1mm/hr$) 3x more than drizzle.
Result? Sharper storms, fewer misses.

3/5 🌍
Data: 11 Years of ERA5-Land.
Region: The Indian Subcontinent ($31 \times 41$ grid).
Resolution: 9km / Hourly.
This allows us to model complex monsoon dynamics often missed by global models.

4/5 📊
Performance:
CSI (Critical Success Index): **0.65** (+12% over baseline).
POD (Detection): **0.74**.
All on a single NVIDIA T4 GPU. High-end forecasting, low-end hardware.

5/5 🔗
Code is Open Source!
We believe in reproducible climate science.
Check out the repo and the `ConvLSTM` implementation here:
[GitHub Link]
#OpenScience #ClimateChange
