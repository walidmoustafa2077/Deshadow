# LP-IOANet — Design Document

> **Source of truth:** NotebookLM notebook *"PyTorch: An Imperative and High-Performance Deep Learning Library"* (sources: LP-IOANet paper, LRA&LDRA paper, Coordinate Attention paper, LPTN paper, FastDepth paper).
>
> **Goal:** Build a real-time, high-resolution **document shadow removal** model that runs at 192×256 internally and upscales 4× to 768×1024, while preserving text detail.

---

## 1. Overview & Why This Design

**LP-IOANet** = **L**aplacian **P**yramid + **I**nput/**O**utput **A**ttention **Net**work.

The core problem: document shadow removal must be **accurate**, **real-time**, and produce **high-resolution** output. Running a heavy network directly on high-res images is too slow and uses too much memory (the prior SOTA, BEDSR, runs out of memory even on a 24GB GPU).

**The key idea:** Do the hard shadow-removal work at **low resolution (192×256)**, then efficiently **upsample 4×** using a lightweight Laplacian pyramid. This gives:
- ~84 FPS on desktop, ~20 FPS on mobile.
- **35% relative MAE improvement** over BEDSR.
- 10× faster, 30× less memory, ~2000× fewer FLOPs than BEDSR.

### Why a Laplacian pyramid for upsampling?
From the LPTN source: in photorealistic image-to-image tasks, **domain-specific attributes (illumination/color) live in the low-frequency component**, while **content detail (text) lives in the high-frequency components**. The high-frequency components of shadow vs. shadow-free images are nearly identical (MSE ~1/71 and ~1/65 of the low-frequency difference). So we only need to translate the low-frequency part with a network, and refine the high-frequency parts with cheap masks — preserving text while saving huge compute.

---

## 2. System Architecture

```
High-res input (768×1024)
        │
        ▼  downscale ×4 (bilinear)
Low-res input (192×256)
        │
        ▼
┌─────────────────────────────┐
│  IOANet (shadow removal)    │  ← Stage 1 (trained first, then frozen)
│  ┌───────────────────────┐  │
│  │ Input Attention (LRA) │  │
│  └──────────┬────────────┘  │
│             │               │
│  ┌──────────▼────────────┐  │
│  │ Encoder (MobileNetV2) │  │
│  │ Decoder (FB-Decoder)  │  │  ← skip connections
│  └──────────┬────────────┘  │
│             │               │
│  ┌──────────▼────────────┐  │
│  │ Output Attention(LDRA)│  │
│  └──────────┬────────────┘  │
│             │               │
│  out = LDRA(net(x)) + LRA(x)│  ← long residual connection
└─────────────┬───────────────┘
              │  low-res shadow-free (192×256)
              ▼
┌─────────────────────────────┐
│  Laplacian Pyramid Upsampler│  ← Stage 2 (trained with IOANet frozen)
│  Level 1: 192×256 → 384×512 │
│  Level 2: 384×512 → 768×1024│
└─────────────┬───────────────┘
              ▼
      High-res shadow-free (768×1024)
```

---

## 3. Component 1 — Encoder-Decoder Backbone (IOANet core)

### 3.1 Encoder: MobileNetV2
- **What:** Hierarchical feature extractor using MobileNetV2 layers.
- **Initialization:** **ImageNet-pretrained weights** — gives robust, generalized features from the start, accelerating convergence and improving in-the-wild generalization.
- **Why MobileNetV2 (not others):** The FastDepth source systematically compared ShuffleNetV2, MNasNet, MobileNetV2, MobileNetV3, EfficientNet variants, MixNet, GhostNet, FastDepth. EfficientNet had slightly higher raw accuracy, but **MobileNetV2 has the best complexity-to-accuracy trade-off** for edge execution. It uses **depthwise separable convolutions** and **inverted residuals** to minimize parameters while keeping strong spatial features.

### 3.2 Decoder: FB-Decoder
- **What:** Custom upsampling decoder built from **FBNet blocks**.
- **Why FBNet:** The FastDepth source compared FBNet-based decoders vs. standard NNConv5 decoders. **FBNet decoders consistently outperformed NNConv5** while having significantly fewer parameters. The final layer is replaced with a simple upsampling operation (nearest-neighbor ×2).

### 3.3 Skip Connections
- **What:** **5 skip connections** from encoder to decoder.
- **Why:** Preserve high-frequency spatial detail lost during downsampling. The decoder combines encoder features at each spatial resolution (U-Net style), which is essential for sharp text reconstruction.

### 3.4 Structure Requirements (from FastDepth)
- Encoder: MobileNetV2 feature layers (channels ramp up: 16 → 24 → 32 → 64 → 96 → 160 → 320 → 1280).
- Decoder: FBNet-style blocks with `1×1 conv + bilinear/nearest upsampling`, `DW conv 5×5`, `BN + ReLU`.
- 5 skip connections from encoder to decoder.
- Final layer: simple upsampling to output resolution.

---

## 4. Component 2 — Input/Output Attention (IOA)

### 4.1 Concept (from LRA&LDRA source)
Instead of reconstructing the whole image (wastes capacity on non-shadow regions) or predicting a plain residual (sub-optimal), IOA splits the task:

```
I_out = LDRA( R(I_shadow; θ_R); θ_LDRA ) + LRA( I_shadow; θ_LRA )
```

- **LRA (Input Attention):** applied to the **input** image.
- **LDRA (Output Attention):** applied to the **output** of the shadow removal network.
- **Long residual connection:** the two streams are **summed**.

### 4.2 Why this design
1. **Localizes model capacity:** non-shadow regions are copied straight through the long residual, so the network only focuses on shadow areas.
2. **Blending/color correction:** adds capacity for seamless blending at shadow boundaries with negligible overhead.
3. **Parallelizable:** input attention runs concurrently with the network.
4. **Efficient:** minimal computational overhead vs. a separate refinement network.

### 4.3 Implementation: Coordinate Attention (from Coordinate Attention source)
LRA and LDRA are implemented using **Coordinate Attention** (the paper's [23] reference), chosen because it has:
- Minimal overhead.
- Strong spatial component (global-pooling-free, direction-aware).
- Ability to capture cross-channel information.

**Coordinate Attention math:**
1. **Coordinate information embedding:** two 1D global poolings (instead of 2D global pooling which loses position):
   - Horizontal: `z_h(h) = (1/W) Σ_i x(h, i)`
   - Vertical: `z_w(w) = (1/H) Σ_j x(j, w)`
2. **Coordinate attention generation:**
   - Concatenate → shared `1×1 conv` → non-linear activation: `f = δ(F1([z_h, z_w]))`
   - Split back → two `1×1 convs` → sigmoid: `g_h = σ(F_h(f_h))`, `g_w = σ(F_w(f_w))`
   - Apply: `y_c(i,j) = x_c(i,j) × g_h_c(i) × g_w_c(j)`
   - Reduction ratio `r` (e.g., 32) controls block size.

### 4.4 Structure Requirements
- Two independent Coordinate Attention modules (LRA and LDRA) with **separate parameters** (`θ_LRA ≠ θ_LDRA`).
- LRA operates on the 3-channel input.
- LDRA operates on the 3-channel network output.
- Output = `LDRA(net(x)) + LRA(x)`.

---

## 5. Component 3 — Laplacian Pyramid Upsampling Module

### 5.1 The 2-Level Pyramid
Decompose the image into frequency bands (from LPTN / Burt-Adelson):
- **Level 0:** original high-res `I_0` at 768×1024.
- **Level 1:** low-pass + downsample → `I_1` at 384×512.
- **Level 2:** low-pass + downsample → `I_2` at 192×256 (the IOANet core resolution).

High-frequency residuals (for exact reconstruction):
- `h_0 = I_0 − EXPAND(I_1)` at 768×1024.
- `h_1 = I_1 − EXPAND(I_2)` at 384×512.

### 5.2 Residual Refinement Network
- Operates at **intermediate resolution (384×512)**.
- **Original LPTN:** 22.8 GFLOPs.
- **LPTN-lite:** uses **depthwise separable convolutions** → 3.82 GFLOPs.
- **LP-IOANet:** further reduces width → **1.47 GFLOPs**.

### 5.3 Reconstruction
- The low-frequency (shadow-removed) component is translated by IOANet.
- High-frequency components are refined via lightweight masks, upsampled and fine-tuned per level.
- All components are summed to reconstruct the high-res output.

### 5.4 Structure Requirements
- 2-level pyramid (192×256 → 384×512 → 768×1024).
- Nearest-neighbor ×2 upsampling between levels.
- Lightweight residual blocks using **depthwise separable convolutions** (DW conv + 1×1 conv).
- Leaky ReLU activations.
- Reconstruction loss applied at the final high-res output.

---

## 6. Training Pipeline (Two-Stage)

### Stage 1 — Train IOANet (low-res)
- Resolution: 192×256.
- **1000 epochs**, Adam optimizer.
- **Losses:** `L1 × 10 + LPIPS × 5`.
- **Mixed batch sampling:** 15 from A-BSDD, 15 from Doc3DS+, 2 from A-OSR.
- LPIPS loss improves results (justified in the paper's ablation).

### Stage 2 — Train Upsampler (high-res)
- **Freeze IOANet** weights.
- Train the Laplacian pyramid upsampler.
- **200 epochs** on A-BSDD (high-res).
- **Loss:** L1 only.

### Why two-stage?
- The upsampler needs high-res data; the core network can be trained on any low-res data.
- Two-stage lets us leverage low-res datasets for better generalization.

---

## 7. Datasets

| Dataset | Size | Unique Images | Resolution |
|---------|------|----------------|------------|
| A-BSDD | 24,082 | 1,328 | High |
| Doc3DS+ | 71,595 | 9,393 | Low |
| A-OSR | 1,410 | 23 | Low |

- **A-BSDD** = augmented BSDD (illumination + shadow color augmentation).
- **A-OSR** = augmented OSR.
- For a first implementation, the public **BSDD** dataset is a good starting point.

---

## 8. Evaluation Metrics
- **PSNR**, **SSIM**, **MAE** on the BSDD dataset.
- Report separately for: all regions, non-shadow regions, shadow regions.
- Compare against BEDSR.

---

## 9. Implementation Plan (File Structure)

```
Deshadow/
├── docs/
│   └── LP-IOANet_Design_Document.md   ← this file
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── coordinate_attention.py    ← Coordinate Attention (LRA/LDRA)
│   │   ├── ioanet.py                  ← Encoder-Decoder + IOA
│   │   ├── upsampler.py               ← Laplacian pyramid upsampler
│   │   └── lp_ioanet.py               ← Full pipeline
│   ├── losses/
│   │   └── losses.py                  ← L1 + LPIPS
│   ├── data/
│   │   └── dataset.py                 ← Dataset loaders
│   ├── train_stage1.py                ← Train IOANet
│   ├── train_stage2.py                ← Train upsampler
│   └── config.py                      ← Hyperparameters
└── requirements.txt
```

---

## 10. Requirements Checklist (before coding)

- [ ] PyTorch + torchvision (for pretrained MobileNetV2).
- [ ] LPIPS package (`pip install lpips`).
- [ ] Coordinate Attention module (LRA + LDRA, separate params).
- [ ] MobileNetV2 encoder (ImageNet-pretrained).
- [ ] FB-Decoder with 5 skip connections.
- [ ] Long residual connection: `out = LDRA(net(x)) + LRA(x)`.
- [ ] 2-level Laplacian pyramid upsampler with depthwise-separable residual blocks.
- [ ] Two-stage training loop (Stage 1: L1+LPIPS; Stage 2: L1, frozen core).
- [ ] Dataset loader for paired shadow/shadow-free images.
- [ ] Evaluation: PSNR, SSIM, MAE.

---

## 11. Key References (from notebook sources)
1. **LP-IOANet paper** — overall architecture, two-stage training, datasets.
2. **LRA & LDRA paper** — input/output attention formulation and rationale.
3. **Coordinate Attention paper** — the attention implementation used for LRA/LDRA.
4. **LPTN paper** — Laplacian pyramid translation network (upsampling module).
5. **FastDepth paper** — MobileNetV2 + FBNet encoder-decoder backbone and skip connections.

---

# Part B — Why Each Value & Formula

This section explains the *reasoning* behind every number and equation, grounded in the notebook sources.

## B.1 Why each hyperparameter value

### Resolutions: 192×256 → 384×512 → 768×1024
- **192×256 (low-res core):** chosen because it **closely matches the aspect ratio of document pages (A4)**. Running the heavy shadow-removal network here reduces memory by **30×** and speeds up processing by **10×** vs. running at full resolution.
- **768×1024 (high-res output):** the target document output. 768×1024 → 192×256 is a **perfect 4× downscale**, so the pyramid reconstructs cleanly. High-res output is needed for downstream **OCR** readability.
- **384×512 (intermediate):** the residual refinement network operates here — the sweet spot between latency and quality.

### 2-level Laplacian pyramid
- Naive interpolation loses sharp text outlines; heavy convolutional decoders are too slow.
- A **2-level pyramid** gives the **optimal trade-off between latency and reconstruction quality**. Deeper pyramids (3–4 levels) reduce GFLOPs further but **slightly degrade PSNR** (from LPTN ablation: PSNR drops 22.09 → 21.95 going from L=3 to L=5).

### 1000 epochs (Stage 1) + 200 epochs (Stage 2)
- **1000 epochs** for IOANet: enough to converge the low-res shadow-removal network on the mixed datasets.
- **200 epochs** for the upsampler: the upsampler is much lighter (only needs to learn residual refinement), so it converges faster. It's trained on high-res A-BSDD only.

### Loss weights: L1 × 10, LPIPS × 5
- **Empirically chosen** by the authors. L1 is weighted higher (10) because it's the primary pixel-accuracy term; LPIPS (5) adds perceptual quality.
- The paper's ablation shows **adding LPIPS improves results** vs. L1-only, justifying its inclusion.

### Batch sampling: 15 / 15 / 2 (A-BSDD / Doc3DS+ / A-OSR)
- **Proportional to dataset size and diversity.** A-BSDD (24,082) and Doc3DS+ (71,595) are large, so they get 15 each; A-OSR (1,410) is tiny, so only 2. This balances the mix so the model sees all three domains without over-weighting the small one.

### Adam optimizer
- From the Adam source: Adam **outperforms other optimizers** on deep CNNs and non-convex objectives, and is **5–10× faster per iteration** than quasi-Newton methods (SFO). It uses bias-corrected estimates of the 1st and 2nd gradient moments, giving fast, stable convergence — ideal for this task.

### Reduction ratio r = 32 (Coordinate Attention)
- Controls the bottleneck width (`C/r`). From the Coordinate Attention source: reducing the channel dimension in the bottleneck **avoids too much information loss** (unlike CBAM which squeezes to 1 channel). r=32 is the standard value that balances capacity vs. overhead.

### 5 skip connections
- From FastDepth/U-Net sources: skip connections **combine encoder features with decoder features at each spatial resolution**, preserving high-frequency spatial detail (text) lost during downsampling. 5 levels is the standard U-Net hierarchy.

### Depthwise separable convolutions (upsampler)
- From LPTN source: replacing standard convs with **depthwise separable convs** (DW conv + 1×1 conv) cuts the upsampler from **22.8 → 3.82 GFLOPs** (LPTN-lite), and reducing width further reaches **1.47 GFLOPs** (LP-IOANet). This is what enables real-time mobile operation.

---

## B.2 Why each formula

### B.2.1 The long residual connection
$$\mathbf{I}_{out} = \text{LDRA}\big(R(\mathbf{I}_{shadow}); \theta_{LDRA}\big) + \text{LRA}(\mathbf{I}_{shadow}; \theta_{LRA})$$

**Why this structure:**
- **Vanilla approach** `I_out = R(I_shadow)` reconstructs the *whole* image — wasting capacity on non-shadow regions (which should be unchanged).
- **Plain residual** `I_out = R(I_shadow) + I_shadow` focuses on shadow regions but is sub-optimal (no blending/color correction).
- **LRA + LDRA** extends the residual: `LRA(I_shadow)` copies non-shadow regions straight through (so `R` only works on shadows), and `LDRA(R(...))` adds blending/color-correction capacity. The **sum** merges both streams.
- Note: plain residual is a **special case** where LRA and LDRA are identity functions.

### B.2.2 Coordinate Attention (used for LRA & LDRA)
**Coordinate information embedding** (two 1D poolings instead of 2D global pooling):
$$z_h^c(h) = \frac{1}{W}\sum_{0 \le i < W} x_c(h, i), \qquad z_w^c(w) = \frac{1}{H}\sum_{0 \le j < H} x_c(j, w)$$

**Why:** 2D global pooling (SE attention) squeezes all spatial info into one vector, **losing positional information**. Splitting into horizontal + vertical 1D poolings preserves position along one axis while capturing long-range dependencies along the other.

**Coordinate attention generation:**
$$f = \delta\big(F_1([z_h, z_w])\big), \qquad g_h = \sigma(F_h(f_h)), \qquad g_w = \sigma(F_w(f_w))$$

**Why:** Concatenate the two direction-aware maps, compress with a shared `1×1 conv` (reduction r), then split and apply two separate `1×1 convs` + sigmoid to produce attention weights in [0,1].

**Final re-weighting:**
$$y_c(i,j) = x_c(i,j) \times g_h^c(i) \times g_w^c(j)$$

**Why:** Each pixel is scaled by both its row's horizontal attention and its column's vertical attention. This lets the model **precisely locate objects** (e.g., shadow boundaries) — better than SE (no position) or CBAM (local 7×7 conv, loses long-range).

### B.2.3 Laplacian pyramid residuals
$$h_0 = I_0 - \text{EXPAND}(I_1), \qquad h_1 = I_1 - \text{EXPAND}(I_2)$$

**Why:** The Laplacian pyramid is **invertible** — it records the *difference* between each level and the upsampled next level. This isolates **high-frequency detail (text)** into the residuals `h_0, h_1`, while the low-frequency `I_2` holds global illumination/color.

**Why this matters for shadow removal:** From LPTN source, the high-frequency components of shadow vs. shadow-free images are **nearly identical** (MSE ~1/71 and ~1/65 of the low-frequency difference). So:
- Only the **low-frequency** part needs the heavy network (IOANet).
- The **high-frequency residuals** are refined with cheap masks (`ĥ = h ⊗ M`), preserving text while saving huge compute.

**Reconstruction:** `I_0 = EXPAND(I_1) + h_0`, and recursively up the pyramid — the exact inverse of decomposition.

---

# Part C — How Blur Is Prevented & Why This Plan Could Fail

## C.1 How LP-IOANet prevents blur in shadow regions

Deep networks are notorious for producing **blurry outputs** and random artifacts. LP-IOANet prevents blur through **four layered defenses**:

### 1. LPIPS perceptual loss (Stage 1)
- The core network is trained with `L1 × 10 + LPIPS × 5`.
- **LPIPS measures distance in a deep feature space** (VGG features) and is **highly sensitive to blur** — much more than pixel-level L1.
- This forces the generator to keep **crisp document contrast** instead of producing smoothed-out regions.

### 2. Input/Output Attention + long residual connection
- Standard networks reconstruct the *whole* image from scratch, wasting capacity on non-shadow regions and introducing blur.
- The **long residual connection** copies non-shadow background regions **directly from the input** to the output.
- Clear areas **bypass the lossy encoder-decoder reconstruction entirely** — so they can never be blurred.

### 3. Laplacian pyramid preserves high-frequency text detail
- Text strokes are **high-frequency components**. Standard auto-encoders compress to a low-dimensional latent space and struggle to reconstruct fine detail at high resolution.
- The Laplacian pyramid uses **closed-form, reversible frequency separation**:
  - Low-frequency `I_2` (illumination/color) → translated by IOANet.
  - High-frequency residuals `h_0, h_1` (text) → **kept nearly unchanged**, only refined with cheap masks.
- Because the high-frequency components of shadow vs. shadow-free images are **nearly identical** (MSE ~1/71 and ~1/65 of the low-frequency difference), the text detail is preserved almost losslessly.

### 4. High-frequency refinement (the critical piece)
- From the LPTN ablation: **removing the high-frequency refinement modules causes blurring** and drops PSNR to 20.87 (from 22.03).
- The blur happens because the translated low-frequency component **mismatches** the unchanged high-frequency components.
- The mask-based refinement (`ĥ = h ⊗ M`) re-aligns them, preserving texture.

---

## C.2 Why this plan could fail — failure modes & risks

### 1. Laplacian pyramid / upsampling limitations
- **Halo artifacts:** the progressive masking strategy avoids convolving large high-frequency components directly (for efficiency), but this can introduce **halo artifacts at stark contrast boundaries** (e.g., deep shadow → light paper). The paper itself notes this as a major limitation.
- **Cannot generate novel content:** because the pyramid is a closed-form decomposition, it **cannot hallucinate text that was completely destroyed** by a very dark shadow. If the underlying characters are mathematically obliterated, LP-IOANet cannot recreate them (unlike heavy pixel-supervised generative models).
- **Level-count trade-off:** more pyramid levels (L=5) give speedups and memory savings but **slightly reduce PSNR**, risking fine text detail.

### 2. Low-resolution core (IOANet) risks
- IOANet runs at **192×256**. It must rely entirely on the upsampler to reintroduce high-frequency text detail.
- If the upsampler fails, **text contrast and readability degrade** — which directly hurts downstream **OCR** (low-res input is a known cause of poor OCR accuracy).

### 3. Shadow boundary / penumbra issues
- **Penumbra** (partially shadowed boundary) is the hardest region — illumination changes gradually.
- A **binary shadow mask cannot model this gradual change**, producing **visible boundary artifacts**.
- **Strong shadows** are especially hard: shadow strength is difficult to estimate, and shadow-boundary pixels look very similar to surrounding text.

### 4. Colored text under shadows
- From the OSR source: when **colored text is covered by strong shadows**, the output text tends to become **dark and lose color** (e.g., red/blue text turns black), causing visual inconsistency.

### 5. Dataset limitations
- **Doc3DShade** (source of Doc3DS+) has **few hard shadow examples** — mostly self-shadows of the document. So the model may not learn to handle strong, complex shadows well.
- **Synthetic datasets** (BSDD is Blender-synthetic) may not fully transfer to real-world, uncontrolled photos.
- **Entirely-shadowed documents** or **multiple-light complex shadows** are known failure cases.

### 6. Training / two-stage risks
- **Two-stage training** is a risk: Stage 1 trains IOANet on low-res, Stage 2 trains the upsampler with IOANet **frozen**. If Stage 1 produces a sub-optimal core, Stage 2 cannot fix it.
- **One-stage end-to-end** is possible but only on A-BSDD (needs high-res data), so it's not used.
- **LPIPS** requires a pretrained VGG network — adds dependency and compute.

### 7. Coordinate Attention limitations
- Coordinate Attention is **lightweight** (good for mobile) but is a **channel/spatial re-weighting** — it doesn't add the capacity of a full refinement network. For very complex shadows, the attention may not be enough.

### 8. MobileNetV2 encoder limitations
- MobileNetV2 was chosen for **efficiency**, not maximum accuracy (EfficientNet was more accurate but heavier).
- The **ImageNet-pretrained** features are for natural images, not documents — may need fine-tuning to adapt to document textures.

### 9. Interpretability / debugging
- Deep models sacrifice interpretability. If the model fails, it's **hard to diagnose** which component (attention, pyramid, upsampler) caused the failure.

---

## C.3 Summary: the biggest risks to watch

| Risk | Likelihood | Impact |
|------|-----------|--------|
| Halo artifacts at shadow boundaries | Medium | Visual quality |
| Cannot recover destroyed text (very dark shadows) | Medium | Readability |
| Colored text loses color | Medium | Visual consistency |
| Low-res core → upsampler bottleneck | Low-Med | Text sharpness |
| Dataset gap (synthetic → real) | Medium | Generalization |
| Two-stage training compounding errors | Low | Overall accuracy |

**Mitigations to consider:**
- Use **shadow matting** (soft masks) instead of binary masks for penumbra.
- Add **color-constancy** handling for colored text.
- Include **strong-shadow examples** in training data.
- Consider **sparse convolution** on high-frequency components to reduce halo artifacts (suggested in the LPTN source).
