# LLM Operator-Graph Density: Results & Interpretation

`results/` contains the exported FX-graph metrics and figures for the
`llm-op-evolution` study; `code/` the pipeline that produced them. The goal is to track how the *frontend complexity* of
open causal language models has changed from 2022 to 2026, measured through the
lens of PyTorch Core ATen operators.

## What we measure

For each model we:

1. Load the smallest publicly available checkpoint from a model family.
2. Export a single forward pass (`batch_size=1`, `seq_len=128`) with
   `torch.export` at the Core ATen IR level.
3. Count `call_function` nodes in the exported FX graph, split by a coarse
   operator category.
4. Normalize the count by the number of transformer layers to get **nodes per
   layer** — a proxy for how many distinct low-level operations a model's
   architecture expresses per layer.

**Important:** this is *not* total FLOPs, parameter count, or wall-clock time.
It is a structural measure of graph density / architectural complexity.

## Dataset

22 open models spanning April 2022 to May 2026 (see `results/summary.txt` and
`results/metrics.json`). The manifest is in `code/config/models.yaml`.

Notable recent additions:

* `deepseek-ai/deepseek-coder-v2-lite-base` (MoE code model, 2024)
* `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (reasoning distill, 2025)
* `Qwen/Qwen3-1.7B` and `Qwen/Qwen3.5-4B` (2025–2026)
* `google/gemma-4-E2B-it` (2026)
* `HuggingFaceTB/SmolLM3-3B` (2026)

## Key findings

### 1. The baseline got denser

Early 2022 dense models sit at roughly **60–77 Core ATen nodes per layer**
(OPT-125M ≈63, BLOOM-560M ≈60, Pythia-410M ≈77).

### 2. The Llama-style stack created a new plateau

From late 2023 through most of 2024, small dense models converge around
**90–105 nodes/layer**: TinyLlama (103.5), Phi-2 (90.3), OLMo 1B (90.4),
StableLM 2 (89.9), Qwen2 0.5B (103.2), DeepSeek Coder 1.3B (95.2), SmolLM
(102.4), Phi-3 Mini (103.7). The architectural recipe (RMSNorm / LayerNorm,
RoPE/GQA, SwiGLU/GeGLU, residual streams) is broadly shared and expresses as a
consistent operator-graph signature.

### 3. MoE and DeepSeek push the ceiling up

The first clear jump above the plateau comes from MoE architectures:

* **Qwen 1.5 MoE A2.7B**: 152.8 nodes/layer (47 unique ops)
* **DeepSeek Coder V2 Lite**: 180.6 nodes/layer (46 unique ops)

These models keep the base transformer recipe but add expert-routing and
gating logic, which shows up as extra scatter/gather, top-k selection, and
conditional routing operations in the Core ATen graph.

### 4. 2025 models refine rather than explode

The 2025 dense models we measured stay within the same broad band but lean
slightly higher:

* DeepSeek R1 Distill Qwen 1.5B: 109.9
* Qwen3 0.6B: 118.7
* Qwen3 1.7B: 129.9
* OLMo 2 1B: 114.6

The increase relative to 2024 is modest and appears to come from small
architectural refinements (e.g., updated normalization placement, additional
auxiliary projections) rather than a wholesale new paradigm.

### 5. 2026 is mixed: a return to the plateau, a denser dense model, and a
new architectural class

* **SmolLM3 3B**: 105.2 nodes/layer — essentially the same density as the
  2024 small-model plateau.
* **Gemma 4 E2B IT**: 165.6 nodes/layer (39 unique ops) — denser than the
  2024 plateau, but still within the dense-transformer family.
* **Qwen 3.5 4B**: **1293.0 nodes/layer** — an extreme outlier that dominates
  the linear-scale timeline.

The Qwen 3.5 4B number is **real, not an export bug**, but it reflects a
different architectural class *and* a **PyTorch-fallback implementation
artifact**. Its config (`Qwen3_5Config`) shows a **hybrid attention stack**:
24 of the 32 layers use **Gated DeltaNet** linear attention rather than
standard softmax attention, alternating with 8 full-attention layers
(`layer_types`: 3× `linear_attention` + 1× `full_attention`, repeated).

The catch is that the current environment does **not** have the optimized
`flash-linear-attention`/`causal-conv1d` kernels installed, so `transformers`
falls back to pure-Python implementations of the DeltaNet recurrence. Those
fallbacks contain explicit loops over the sequence length and chunk size,
plus many small reshape/pad/cumsum/where operations. When `torch.export`
traces them, every one of those operations becomes a Core ATen node.

We tried installing `flash-linear-attention` to see the kernelized graph, but
that made the export **fail**: FLA's optimized kernels are written in Triton,
and `torch.export` cannot trace into Triton kernels (it errors with "Cannot
access data pointer of Tensor"). So we cannot produce a kernel-normalized
Core ATen count for this model with the current pipeline. The ≈41,375 nodes
are the only `torch.export`-compatible view available.

So the value is a genuine signal that DeltaNet is structurally more complex
than softmax attention, but the *magnitude* is amplified by the missing
kernels and by the fact that any optimized kernel would be opaque to our
graph counter. It is **not directly comparable** to the Llama-style
softmax-attention trend line.

### 6. A second lens: architectural heterogeneity

Because ATen-node density is confounded by implementation details (missing
kernels, Triton opacity), we also scored each model on a hand-curated
**architectural mechanism inventory**. The inventory tracks which distinct
ideas each model combines: attention variants (full, linear, sliding-window,
GQA, MQA), routing (MoE), position encoding (RoPE, ALiBi, learned),
normalization (RMSNorm, LayerNorm), FFN gating, and model-level features
(MTP, multimodal, tied embeddings, KV sharing).

A **heterogeneity score** counts the number of mechanism categories touched,
plus a small bonus for having multiple mechanisms within one category (e.g.,
full + linear attention). The resulting timeline shows a cleaner throughline:

* 2022 models (Pythia, OPT, BLOOM): score **3.0–4.5** — homogeneous stacks
  with one attention type and one normalization type.
* 2023–2024 Llama-style dense models: score **4.0–5.5** — the standard recipe
  (RMSNorm, RoPE, gated MLP, sometimes GQA) plus the first MoE and sliding-window
  entries.
* 2025 models: score **4.0–5.5** — refinement, not a large jump.
* 2026: SmolLM3 stays at **5.5**, Gemma 4 jumps to **7.5**, and Qwen 3.5
  reaches **8.0**.

Crucially, Qwen 3.5 is only slightly more heterogeneous than Gemma 4 (8.0 vs
7.5) but its ATen-node density is **~8× higher**. That gap is the signature of
the Python-fallback / missing-kernel artifact, not of an 8× architectural leap.

The scatter of density vs. heterogeneity shows three clusters:

1. **Dense-transformer band** (lower-left): low density, low-to-moderate
   heterogeneity — the refinement regime.
2. **MoE band** (middle): moderate density, moderate heterogeneity — same
   attention recipe plus routing.
3. **Hybrid-attention / multimodal outliers** (right): high heterogeneity,
   with Qwen 3.5 pulled far upward in density by the fallback implementation.

## Figures

| File | What it shows |
|------|---------------|
| `timeline_density.png` | Linear-scale scatter of nodes/layer over time, plus total nodes in the forward pass. The Qwen 3.5 outlier compresses the rest. |
| `timeline_density_log.png` | Log-scale version of the timeline. Makes the 2022→2025 gradual rise and the 2026 outlier visible on the same axis. |
| `timeline_density_dense_only.png` | Linear-scale timeline with MoE and hybrid linear-attention models removed, so the dense-transformer trend (60→100→130) is readable. |
| `year_density.png` | Year-binned median/mean density with min–max whiskers. Shows the slow upward drift and the 2026 mean being pulled up by the outlier. |
| `category_composition.png` | Stacked area chart of operator categories over time. |
| `unique_ops.png` | Number of distinct ATen operators used by each model. |
| `mechanism_heatmap.png` | Binary heatmap of models × architectural mechanisms. |
| `heterogeneity_timeline.png` | Heterogeneity score over time. |
| `density_vs_heterogeneity.png` | ATen density vs. mechanism diversity; shows the implementation/ architecture split. |

## Caveats

* **Small variants only.** We export the smallest released checkpoint per
  family; larger variants may have different layer widths or routing depths.
* **Forward pass only.** No training graph, no KV-cache inference graph, no
  speculative decoding or advanced sampling.
* **Single configuration.** `seq_len=128`, `batch_size=1`. Longer sequences
  would add position- and attention-related nodes.
* **Dtype differences.** The default is float32; a few GPU-heavy checkpoints
  (e.g., Phi-2, Qwen 1.5 MoE, DeepSeek Coder V2 Lite) use float16. The metric
  is dominated by graph structure, not dtype.
* **Gated models excluded.** Gemma 3/4 smaller variants and some DeepSeek
  checkpoints were gated or unavailable, so the 2026 sample is not exhaustive.
* **Custom / Triton kernels are opaque.** Models that dispatch to fused Triton
  or custom CUDA/ROCm kernels (e.g., optimized DeltaNet/linear-attention
  implementations) will appear as far fewer ATen nodes, or may fail to export,
  because `torch.export` cannot see inside those kernels. The counts here are
  therefore biased toward what the *PyTorch frontend* can observe.
* **Mechanism inventory is hand-curated.** The heterogeneity score depends on
  which mechanisms we choose to count and how we categorize them. It is meant
  as a qualitative complement to ATen density, not an objective ground-truth
  complexity measure.
* **Outlier interpretation.** The Qwen 3.5 4B value should be treated as a
  signal to investigate rather than a settled architectural fact.

## Reproducing

From `code/` (see `code/README.md`):

```bash
# Export all models, analyze mechanisms, and regenerate figures
HF_HUB_ENABLE_HF_TRANSFER=0 python scripts/run_analysis.py --device cuda --output ../results
python scripts/analyze_mechanisms.py --output ../results
python scripts/run_analysis.py --plot-only --output ../results

# Regenerate figures from existing metrics.json
python scripts/analyze_mechanisms.py --output ../results
python scripts/run_analysis.py --plot-only --output ../results
```
