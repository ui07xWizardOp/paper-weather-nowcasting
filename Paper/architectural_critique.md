# Architectural Critique: Why ConvLSTM?

## The Challenge: "Newer is Better" Bias
Reviewers often ask: "Why did you use ConvLSTM (2015) instead of Swin-Unet (2021) or GraphCast (2023)?"
This document provides the **technical justification** for this choice, shifting the narrative from "outdated" to "appropriate."

## 1. Inductive Bias vs. Data Scale
-   **Transformers/GraphCast**: Rely on massive datasets (Petabytes) to learn spatial relationships from scratch (Weak Inductive Bias).
-   **ConvLSTM**: Hard-codes translation invariance and locality via convolution (Strong Inductive Bias).
-   **Our Context**: We are training on a *regional* subset (India) of ERA5-Land (11 years). This is "Small Data" in the LLM era. ConvLSTM converges faster and generalizes better on small data than data-hungry Transformers.

## 2. The "Blurriness" Trade-off
-   **Critique**: ConvLSTM produces blurry forecasts due to MSE loss.
-   **Rebuttal**: This is a *loss function* issue, not an *architecture* issue. By applying our **Weighted MSE**, we force the ConvLSTM to sharpen high-intensity regions. Switching to a GAN (DGMR) or Diffusion model would fix blurriness but introduce hallucination risks (predicting rain where there is none), which is unacceptable for flood warning.

## 3. Computational Constraints (The "Democratization" Argument)
-   **GraphCast**: Requires TPU Pods or A100 clusters for efficient training.
-   **Our Model**: Trains on a single NVIDIA T4 (16GB VRAM).
-   **Impact**: This makes the model deployable by state meteorological departments in developing nations, who rarely have access to supercomputers.

## Conclusion
We choose ConvLSTM not because we are unaware of newer models, but because it represents the **Pareto Optimal** point between Performance, Data Efficiency, and Deployability for the specific task of regional Indian nowcasting.
