# TRISEMBLE

A three-model ensemble tracker for the [VOTS2026](https://www.votchallenge.net/vots2026/) (Visual Object Tracking and Segmentation) challenge, combining SAM3, DAM4SAM, and Cutie to produce robust per-object segmentation masks across long, difficult video sequences.

**Final public leaderboard score: Q = 0.50**

## Overview

TRISEMBLE runs three independent tracking/segmentation models in parallel on each frame and fuses their outputs into a single mask per tracked object:

- **SAM3** — runs as a persistent inference daemon (kept resident in GPU memory across the sequence rather than reloaded per frame), providing strong general-purpose segmentation.
- **DAM4SAM** (DAM4SAM/SAM2-based) — handles temporally consistent mask propagation between frames.
- **Cutie** — a memory-based video object segmentation model, adding a third independent vote on object identity and extent.

An `ensemble_vot_wrapper.py` implements the VOT/TraX protocol and coordinates per-frame calls to all three components; `fusion_engine.py` combines their outputs into the final prediction.

## Development history

TRISEMBLE went through several named iterations before reaching its final score:

| Stage | Tracker | Configuration | Q |
|---|---|---|---|
| 1 | — | SAM3.1 + SAMBAMOTR | Abandoned (API/RoPE issues) |
| 2 | SAMBA-D4 | SAM3 + DAM4SAM (baseline) | 0.05 |
| 3 | TRIDENT | SAM2 (swapped) + DAM4SAM + Cutie | 0.09 |
| 4 | TRIDENT | SAM2 + DAM4SAM + Cutie (optimized) | 0.14 |
| 5 | TRISEMBLE | SAM2 + DAM4SAM + Cutie (renamed, optimized) | 0.15 |
| 6 | TRISEMBLE | Backbone reverted to SAM3 + foreground/background fix | **0.50** |

The backbone was swapped from SAM3 to SAM2 midway through development for better compatibility with DAM4SAM, then reverted back to SAM3 once it was found to outperform SAM2 after other bugs were fixed. The final jump in score came from combining that reversion with a fix to a mask-polarity bug where segmentation was occasionally landing on the background instead of the foreground object.

## Requirements

Evaluated using the [VOT toolkit](https://github.com/votchallenge/toolkit) against the VOTS2026 dataset. See `trackers.ini` for the exact command used to invoke the tracker.

## Acknowledgments

Developed as part of a summer research internship at the Indian Institute of Technology Tirupati, under the supervision of Prof. Rama Krishna Sai Gorthi.
