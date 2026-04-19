Here i will push my experiments methodology.  I will run multiple different types of experiments on React 2024 Dataset.
# Experiment: Gemma-BeLFusion Bridge Fix & Baseline Evaluation (v2)

## Objective
Establish a true evaluation baseline for the Gemma4-integrated BeLFusion model by fixing the weight serialization bug in the conditioning bridge. 

## Architecture & Modifications
* **Bug Fix (`matchers.py`):** Corrected the `torch.load` mapping. In the unpatched version (v1), bridge weights were re-initializing upon load, creating an artificial noise-injection effect that falsely inflated diversity metrics.
* **Hyperparameters (`config/2_belfusion_ldm.yaml`):** Maintained `k=10` configuration. Learning rate decayed to 0.000 by epoch 78.
* **Conditioning Method:** Static pooling (1 embedding per 30s clip) mapped via an additive bridge to the 25-dimensional audio-emotion feature space.

## Evaluation Metrics (Test Split)

| Metric | Buggy v1 (val) | Patched v2 (test) | Paper Baseline |
| :--- | :--- | :--- | :--- |
| **FRC** (↑) | 0.1932 | **0.1175** | 0.1200 |
| **FRD** (↓) | 87.68 | **91.59** | 91.60 |
| **FRVar** (↑) | 0.0119 | 0.0082 | 0.0082 |
| **TLCC** (↓) | 46.61 | 45.23 | 44.87 |

## Root Cause Analysis
The patched v2 results align almost perfectly with the original paper's baseline. This indicates that the model is actively ignoring the Gemma signal. 

**The Bottleneck:** The current architecture projects a single, static `(2560,)` vector into `(25,)` and adds it to every frame across a 750-frame sequence. Because a single vector cannot provide meaningful temporal variance for micro-expressions, the network optimizes the bridge weights to zero to minimize loss, reverting to a standard audio-only BeLFusion model.

## Next Steps
1. **Bridge Redesign:** Move from an Additive bridge to a Concatenation bridge (`25 + 128 = 153` dims) to preserve the distinct semantic signal (Branch: `experiment/concat-bridge`).
2. **Ablation:** Test `k=1` to force the network to rely on the conditioning signal rather than diffusion diversity.
