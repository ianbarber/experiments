# Interpretable Chess-State Substrate and Semantic Compression

**Date:** 2026-08-26 · **Machine:** AMD Ryzen AI Max+ 395 / Radeon 8060S
(`gfx1151`), ROCm 7.2, PyTorch 2.9.1

## Brief

Can a model compress high-dimensional chessboard photographs through a named,
editable state rather than an arbitrary VAE latent?

The model is forced through two channels: a hard `8 x 8 x 13` square grid `s`,
which serializes directly to piece-placement FEN, and a 64-value variational
appearance code `z`. The decoder receives only `(s, z)`; there are no skip
connections.

## Headline results

- The supervised encoder reached **94.25% square / 9.70% exact-board accuracy**
  on held-out ChessReD validation games. The gap is the point: square accuracy
  badly overstates usable whole-board recognition.
- Holding appearance fixed and swapping only the semantic grid changed output
  pixels **3.69x more inside edited squares** than elsewhere. The named state is
  causally active, not merely an auxiliary prediction.
- A frozen semantic recognizer used as a perceptual loss improved decoded-board
  recognizability by **33.7%** over an equal-step control, with +0.00064 L1.
- The first real bitstream averages **128.16 bytes**: 53.16-byte independently
  decodable semantic base plus a 75-byte int8 appearance enhancement.
  Quantization contributes negligible error.
- The model remains visibly crude: it reconstructs board layout, occupancy, and
  piece color, but not recognizable piece geometry. Appearance also leaks some
  board state, and synthetic exact-board accuracy is poor.

## Contents

| Path | What |
|---|---|
| `REPORT.md` | Architecture, quantitative results, qualitative panels, intervention, probes, codec, limitations |
| `LABNOTES.md` | Condensed chronological execution record, including failed approaches |
| `code/` | Source, tests, environment lock, acquisition/training/evaluation scripts |
| `results/` | Small JSON records for the reported replicas, probes, and codec evaluations |
| `images/` | Reconstruction and semantic-intervention panels |

The source datasets are non-commercial: ChessSight is CC BY-NC 4.0 and
ChessReD is CC BY-NC-SA 4.0. See `images/README.md` for panel attribution.
