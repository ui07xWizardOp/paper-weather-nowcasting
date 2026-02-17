# Disagreement Survey (Phase 11 Counterfactuals)

## "Deep Learning Cannot Extrapolate"
-   **Source**: *Impossibility of Extrapolation in High-Dimensions* (Xu et al., 2021).
-   **Argument**: Neural Networks are interpolators. They cannot predict weather states they haven't seen in the training data (e.g., a "record-breaking" storm).
-   **Our Counter**: We acknowledge this. We are not predicting *unseen physics*, but *unseen combinations* of known physics. Our 11-year dataset covers enough "extreme tails" to bracket the likely future distribution.

## "ConvLSTM is Dead, Use Transforms"
-   **Source**: *ViT-22B* (Google, 2023) and *Swin-Unet* papers.
-   **Argument**: RNNs (LSTMs) are serial, slow, and forget long contexts. Transformers parallelize better and have infinite memory.
-   **Our Counter**: For *Time-Series Video* (which weather is), the "Forget Gate" is actually a feature, not a bug. Old weather (t-12 hours) *should* be forgotten to focus on the immediate kinematic trajectory. Transformers often over-attend to irrelevant distant pasts without heavy masking.

## "Station Data is the Only Truth"
-   **Source**: Traditional Meteorologists / Hydrologists.
-   **Argument**: Reanalysis (ERA5) is a simulation. It is not real rain. Only rain gauges are real.
-   **Our Counter**: Gauges are "Point Truth". They have 0 spatial dimension. You cannot train a CNN on points. ERA5-Land is the "Spatial Truth" (best guess). We trade *point accuracy* for *spatial consistency*, which is required for flood *extent* mapping.
