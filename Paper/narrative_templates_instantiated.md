# Narrative Templates Instantiated (Phase 7)

## Template A: Evolution (Methodological History)
**Context**: Used in Section 2 (Related Work).

"Approaches to **Precipitation Nowcasting** have evolved from **Optical Flow extrapolation** to **Deep Learning tensor prediction**, with a primary focus on **overcoming the blurriness of deterministic forecasts**.
**Shi et al. (2015)** introduced the **ConvLSTM**, which **preserved spatial topology during temporal recurrence** by **replacing dense matrix multiplications with convolution operators**.
Building upon this foundation, **Shi et al. (2017)** extended the paradigm by **dynamic trajectory learning (TrajGRU)**, achieving **better handling of rotation** on **HKO-7 Radar Data**.
Empirically, these approaches have demonstrated **superior skill over NWP in the 0-2 hour window** across **standard benchmarks like Moving MNIST** [cite1, cite2].
However, these methods still suffer from **regressive mean convergence**, particularly when **predicting extreme events**, due to **the averaging property of the MSE loss function**.
Consequently, recent investigations have explored **Generative Adversarial Networks (GANs)** as a potential remedy."

## Template B: Comparative (Ours vs. SOTA)
**Context**: Used in Section 5 (Discussion).

"While **DGMR (Ravuri et al., 2021)** achieves **spectral consistency** through **adversarial training**, **our Weighted-MSE ConvLSTM** takes a fundamentally different approach by **physically anchoring the loss function**.
This divergence creates a critical trade-off: **DGMR** offers **realistic textures (sharpness)** but requires **unstable GAN training and 4x compute**, whereas **our method** provides **reliable intensity capture** at the cost of **fine-scale texture loss**.
Quantitatively, **DGMR** achieves **lower CRPS** on **UK Radar**, compared to **higher CSI** for **Weighted-MSE** on **ERA5-Land**, suggesting **our approach is better suited for 'Event Detection' (Did it rain?) rather than 'Texture Synthesis' (What did it look like?)**.
The choice between these paradigms thus depends critically on **the deployment constraints**, with **GANs** preferable when **visual realism is paramount** and **Weighted-MSE** more suitable for **operational flood warning on limited hardware**."

## Template C: Gap Statement (The "Why")
**Context**: Used in Section 1 (Introduction).

"Despite significant advances in **global AI weather forecasting**, **hyper-local predictions in data-sparse regions** remains a critical challenge due to **the resolution mismatch of foundation models**.
Existing methods [GraphCast, Pangu] have achieved **synoptic mastery** but consistently fail when **resolving mesoscale convection (<10km)**, as demonstrated by **the smoothing of extreme rainfall peaks in mountainous terrain**.
This limitation fundamentally constrains **disaster risk reduction**, preventing **AI benefits** from being realized in **vulnerable Global South geographies**.
Addressing this gap requires **a region-specific, physically-constrained architecture**, which presents an opportunity for **democratizing high-performance nowcasting**.
This observation motivates our investigation into **the ERA5-Land ConvLSTM**, which **trades global breadth for local depth**."
