# Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting (Shi2015)
**Q**: How can we model the spatio-temporal dynamics of precipitation better than FC-LSTM or Optical Flow?
**M**: Introduces Convolutional LSTM (ConvLSTM) unit where input-to-state and state-to-state transitions use convolution operations to preserve spatial topology.
**E**: Achieves lower MSE and higher CSI compared to FC-LSTM and ROVER (Optical Flow) on Moving MNIST and Radar Echo datasets.
**L**: Blurry predictions due to MSE loss; limited horizon handling (decay over time).
**R**: Foundational architecture for our encoder-decoder backbone.

# Deep Learning for Precipitation Nowcasting: A Benchmark and a New Model (Shi2017)
**Q**: How to handle rotational and non-rigid motion in weather systems that standard convolution (location-invariant) struggles with?
**M**: Proposes Trajectory GRU (TrajGRU) with location-variant recurrent connections that can "warp" the state based on learned flow fields.
**E**: Outperforms ConvLSTM on HKO-7 dataset, specifically for rotational storm events.
**L**: High computational complexity; training instability; still suffers from blurriness.
**R**: Demonstrates the need for motion-aware modeling, which we address via simpler ConvLSTM + SSIM/WeightedMSE trade-off.

# Skilful precipitation nowcasting using deep generative models of radar (Ravuri2021/DGMR)
**Q**: How to generate sharp, realistic precipitation fields that preserve small-scale variability and verify utility for forecasters?
**M**: Uses a Deep Generative Model (GAN) with spatial and temporal discriminators to enforce realism, optimizing a mix of adversarial and grid losses.
**E**: Ranked 1st by expert meteorologists for realism and utility compared to PySTEPS and UNet, despite worse standard metrics (CSI/MSE).
**L**: Computationally equivalent to training a massive GAN; extremely hard to effectively train; probabilistic output requires ensemble sampling for value.
**R**: Highlights the "Blurriness vs. Realism" trade-off which our "Weighted MSE" attempts to balance deterministically.

# The ERA5 Global Reanalysis (Hersbach2020)
**Q**: How to produce a consistent, global, high-resolution historical record of the atmosphere?
**M**: 4D-Var data assimilation system (IFS Cycle 41r2) combining observations from satellites, stations, and planes into a coherent physics model.
**E**: Provides hourly data at 31km (ERA5) and 9km (ERA5-Land) from 1950-Present; significantly reduced error biases compared to ERA-Interim.
**L**: "Reanalysis" is not "Observation" - it smooths out extreme localized events; precipitation is a forecast product, not direct assimilation.
**R**: Our primary dataset source; understanding its "smoothness" motivates our aggressive Log-Norm + WeightedMSE pipeline.
