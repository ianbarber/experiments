# Lab notebook — Interpretable Chess Representation

Condensed execution log for work performed 2026-08-25/26 on the Ryzen AI Max+
395 machine. The standalone development repo used a more verbose append-only
notebook; this archive retains the decisions, failures, and measured milestones.

## 1. Framing and machine preparation

- Defined the target as FEN piece placement only. Five other FEN fields require
  game history and are not visible in a photograph.
- Chose a hard `8 x 8 x 13` semantic bottleneck plus a separate 64-value
  appearance VAE. Prohibited encoder-decoder skips and soft semantic values at
  the decoder boundary.
- Stopped and disabled `muse-glimmer.service` to free the APU.
- Installed AMD's ROCm 7.2 PyTorch 2.9.1 wheels in Python 3.12. A mixed-precision
  convolution/optimizer/checkpoint smoke test passed on `gfx1151`.

## 2. Data and invariants

- Acquired ChessReD annotations + ChessReD2K images and one ChessSight train
  shard plus its validation Parquet.
- Canonicalized grids as row-major `a8..h1`: empty 0, white `PNBRQK` 1–6,
  black `pnbrqk` 7–12. Added FEN round-trip tests.
- Built manifests recording source, group, license, FEN, grid, split, corners,
  and known capture metadata. Rectified images using coordinate-ordered board
  corners.
- Combined audit: 11,506 images, 11,360 unique positions, no duplicate/group
  errors. Twenty-eight validation images shared a train position and were
  removed from the strict evaluation set.

## 3. Semantic baseline

- Five-stage convolutional encoder, base width 32, 256px input, AdamW `3e-4`,
  FP16, 50/50 source sampler.
- After 200 exploratory updates: ChessReD validation 94.25% square, 9.70% exact
  board, 3.68 mean incorrect squares.
- An earlier checkpoint resume failed safely because `TorchVersion` was not
  accepted by PyTorch's weights-only loader. Stored version metadata as a plain
  string and restarted the main run.

## 4. Structured VAE failures and fixes

- First 50-step joint run produced a blurred checkerboard with no pieces. L1
  rewarded the board background before small high-frequency objects.
- Supplying ground-truth hard semantics for 50 decoder-warmup steps did not fix
  it. Architectural access to semantics did not force useful semantic rendering.
- Up-weighted pixels in occupied squares 5:1 and reported occupied/empty errors
  separately. After 100 more steps, coarse position-dependent marks emerged.
- Returned to predicted hard Gumbel-Softmax semantics for 50 steps. A semantic
  swap with fixed appearance changed edited squares 4.22x more than unchanged
  squares: the first causal substrate-use result.

## 5. Class-aware fine-tuning

- Froze the semantic baseline and passed reconstructions through it. Added its
  64-square CE with weight 0.1; gradients flowed to the decoder, not the frozen
  recognizer.
- Ran three 150-step replicas from one common checkpoint with seeds 20260826–28.
  These are warm-start sensitivity replicas, not independent from-scratch runs.
- Balanced position-disjoint means: 89.63% square, 6.38% exact, L1 0.1257,
  frozen-recognizer CE 0.459.
- Matched seed-20260828 control without the term: recognizer CE 0.678 vs 0.450
  with the term, while L1 changed 0.12536 -> 0.12600. This supports improved
  class-aware reconstruction, not improved encoder recognition.

## 6. Interventions, probes, and codec

- Final cross-position semantic intervention: 36 changed squares; output change
  0.0681 inside vs 0.0184 outside, locality ratio 3.69x.
- Linear board probe on `z`: 72.20% square accuracy vs 68.22% majority, 0% exact.
  Appearance-source probe: 100%. Conclusion: useful style channel with residual
  semantic leakage.
- Implemented `CRP1`: nibble-packed/zlib semantics and per-image-scaled int8 `z`,
  with versioned header and CRC32. Tests cover semantic prefix decode,
  quantization bounds, and corruption.
- Balanced validation: 128.16 bytes total, 53.16-byte base, 0.01565 bpp,
  quantization delta L1 0.000065. Decoder distortion dominates.
- Full strict validation exposed synthetic weakness: 84.22% square / 0.76%
  exact on ChessSight versus 95.46% / 11.25% on ChessReD. Public test remained
  untouched.

## Loose ends

- Render recognizable class-specific piece shapes rather than occupancy blobs.
- Penalize board information recoverable from the appearance code.
- Diagnose the synthetic-domain regression and run independent from-scratch
  seeds before public-test evaluation.
- Add a learned entropy model only after decoder fidelity improves.
