# Conference Presentation Storyboard (15 Minutes)

## Slide 1: Title & Hook (1 min)
- **Visual**: Spinning globe zooming into India (GIF). Title: "Code Red: Nowcasting for the Indian Monsoon."
- **Script**: "Good morning. In July 2023, a flash flood in Himachal Pradesh caused \$500M in damage. The lead time? Zero. Today, I present a deep learning system that buys us 6 hours."

## Slide 2: The Problem - Why is this hard? (2 min)
- **Visual**: Split screen. Left: NWP (Spin-up gap). Right: Optical Flow (Blurry fade).
- **Script**: "NWP is too slow (`delta-t > 6h`). Optical flow is fast but assumes rain moves like a rigid object. We need the physics of NWP with the speed of AI."

## Slide 3: The Gap - Why ERA5-Land? (2 min)
- **Visual**: Bar chart comparing station density (USA vs. India).
- **Script**: "Most AI models train on dense implementation_plan radar. India doesn't have that. We leverage ERA5-Land reanalysis—a physically consistent 'digital twin' of the atmosphere."

## Slide 4: Methodology - The Weighted MSE (3 min)
- **Visual**: Equation of WMSE with a "Heatmap" showing where the weights activate (on the storm front).
- **Script**: "Standard MSE loves averages. It predicts a light drizzle everywhere to be 'safe'. Our Weighted MSE forces the model to care about the extremes—the cloudbursts."

## Slide 5: Results - Quantitative (2 min)
- **Visual**: Table comparing CSI scores. Big green arrow "+12%".
- **Script**: "The results are clear. We achieve a CSI of 0.65, a 12% jump over the ConvLSTM baseline. That's not just a number; that's 40 extra minutes of warning."

## Slide 6: Results - Qualitative (2 min)
- **Visual**: 3-panel video. Ground Truth vs. Ours vs. Baseline.
- **Script**: "Watch the storm evolution. The baseline (right) blurs out. Our model (center) maintains the convective core."

## Slide 7: Impact & Ethics (2 min)
- **Visual**: Low-cost GPU icon (T4) vs. Supercomputer.
- **Script**: "This runs on a single T4 GPU. We are democratizing high-res forecasting for regional agencies."

## Slide 8: Conclusion & Future Work (1 min)
- **Visual**: GitHub QR Code. Bullet points: GANs, Attention.
- **Script**: "We've bridged the gap between reanalysis and nowcasting. The code is open source. Thank you."

## Q&A Prep
- **Q**: "Why not GraphCast?"
- **A**: "GraphCast is 0.25-degree (25km). We are 0.1-degree (9km). We need local precision."
