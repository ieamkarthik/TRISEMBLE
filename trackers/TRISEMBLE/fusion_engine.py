#!/usr/bin/env python3
"""
fusion_engine.py  –  Mask fusion for the Cutie + D4SM + SAM3 ensemble
Place at: /mnt/DATA1/INTERNSHIP/vots2026_workspace/trackers/ensemble/fusion_engine.py

Strategy
--------
For each object independently:
  1. Collect up to 3 binary masks (SAM3, Cutie, D4SM).
  2. Majority pixel vote: a pixel is ON if ≥ ceil(n_valid/2) models agree.
     - 3 models alive → pixel needs ≥ 2 votes
     - 2 models alive → pixel needs ≥ 2 votes (strict consensus; falls back to
                        single-model if pairwise IoU < CONSENSUS_IOU_THRESHOLD)
     - 1 model  alive → use that mask as-is
  3. Per-object confidence tracks recent IoU history so a model that is
     consistently drifting gets down-weighted automatically.
  4. Optional morphological cleanup (fill small holes, remove tiny blobs).

All public API returns / expects numpy bool arrays (HxW) or dicts {obj_id: mask}.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict, deque
import cv2

# ── tuneable knobs ─────────────────────────────────────────────────────────────
CONSENSUS_IOU_THRESHOLD = 0.25   # below this, two-model "consensus" is ignored
CONFIDENCE_WINDOW       = 10     # frames used to compute rolling confidence
MIN_MASK_AREA_PX        = 20     # blobs smaller than this are discarded
HOLE_FILL_AREA_PX       = 200    # holes smaller than this are filled
# ──────────────────────────────────────────────────────────────────────────────


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over Union for two bool masks."""
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def _cleanup(mask: np.ndarray) -> np.ndarray:
    """Remove tiny blobs and fill small holes."""
    m = mask.astype(np.uint8)

    # Fill small holes
    contours, _ = cv2.findContours(~m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < HOLE_FILL_AREA_PX:
            cv2.drawContours(m, [cnt], 0, 1, -1)

    # Remove small blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    clean = np.zeros_like(m)
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= MIN_MASK_AREA_PX:
            clean[labels == lbl] = 1

    return clean.astype(bool)


class FusionEngine:
    """
    Stateful per-sequence fusion engine.

    Usage
    -----
    engine = FusionEngine(model_names=['sam3', 'cutie', 'd4sm'])
    engine.reset()                                    # new sequence

    fused = engine.fuse(
        obj_id  = 1,
        masks   = {'sam3': mask_np, 'cutie': mask_np, 'd4sm': mask_np},
        # any key can be None if that model failed this frame
    )
    """

    MODEL_WEIGHTS_DEFAULT = {
        'sam3':  1.0,
        'cutie': 1.2,   # slightly higher: Cutie has memory → better temporal consistency
        'd4sm':  1.2,   # slightly higher: deformable attention → better shape accuracy
    }

    def __init__(self, model_names: List[str] = None):
        self.model_names = model_names or list(self.MODEL_WEIGHTS_DEFAULT.keys())
        self._conf: Dict[str, Dict[int, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=CONFIDENCE_WINDOW)))
        self._weights = {m: self.MODEL_WEIGHTS_DEFAULT.get(m, 1.0) for m in self.model_names}

    # ── public API ─────────────────────────────────────────────────────────────

    def reset(self):
        """Call at the start of every new sequence."""
        self._conf.clear()

    def fuse(
        self,
        obj_id: int,
        masks:  Dict[str, Optional[np.ndarray]],
    ) -> np.ndarray:
        """
        Parameters
        ----------
        obj_id : int   object label
        masks  : dict  model_name → HxW bool mask (or None if model failed)

        Returns
        -------
        HxW bool numpy array  –  fused mask for this object
        """
        valid: Dict[str, np.ndarray] = {
            k: v.astype(bool) for k, v in masks.items()
            if v is not None and v.any()          # skip empty / failed masks
        }

        # ── degenerate cases ──────────────────────────────────────────────────
        if not valid:
            # All models failed – return empty mask with same shape as first non-None
            for v in masks.values():
                if v is not None:
                    return np.zeros(v.shape, dtype=bool)
            return np.zeros((1, 1), dtype=bool)

        if len(valid) == 1:
            m = _cleanup(next(iter(valid.values())))
            self._update_conf(obj_id, valid, m)
            return m

        # ── weighted pixel vote ───────────────────────────────────────────────
        h, w = next(iter(valid.values())).shape
        vote_map = np.zeros((h, w), dtype=np.float32)
        total_w  = 0.0
        for name, mask in valid.items():
            w_i = self._get_weight(name, obj_id)
            vote_map += mask.astype(np.float32) * w_i
            total_w  += w_i

        # Threshold: majority of weighted votes
        threshold = total_w / 2.0
        fused = vote_map > threshold

        # ── two-model fallback: check consensus quality ───────────────────────
        if len(valid) == 2:
            names = list(valid.keys())
            iou_pair = _iou(valid[names[0]], valid[names[1]])
            if iou_pair < CONSENSUS_IOU_THRESHOLD:
                # Models disagree strongly – trust the higher-confidence one
                best = max(valid.keys(), key=lambda n: self._get_weight(n, obj_id))
                fused = valid[best]

        fused = _cleanup(fused)
        self._update_conf(obj_id, valid, fused)
        return fused

    def fuse_all(
        self,
        all_masks: Dict[str, Dict[int, Optional[np.ndarray]]],
    ) -> Dict[int, np.ndarray]:
        """
        Convenience: fuse all objects at once.

        Parameters
        ----------
        all_masks : {model_name: {obj_id: mask_or_None}}

        Returns
        -------
        {obj_id: fused_mask}
        """
        # Gather all object ids across all models
        obj_ids = set()
        for model_masks in all_masks.values():
            obj_ids.update(model_masks.keys())

        fused = {}
        for oid in obj_ids:
            per_model = {m: all_masks[m].get(oid) for m in all_masks}
            fused[oid] = self.fuse(oid, per_model)
        return fused

    # ── internal helpers ───────────────────────────────────────────────────────

    def _get_weight(self, model_name: str, obj_id: int) -> float:
        """Return adaptive weight = static_weight × rolling_confidence."""
        static = self._weights.get(model_name, 1.0)
        hist   = self._conf[model_name][obj_id]
        if len(hist) == 0:
            return static
        return static * (sum(hist) / len(hist))

    def _update_conf(
        self,
        obj_id: int,
        valid_masks: Dict[str, np.ndarray],
        fused_mask: np.ndarray,
    ):
        """
        For each model that contributed, record IoU(model_mask, fused_mask)
        as its confidence for this frame.
        """
        fused_any = fused_mask.any()
        for name, mask in valid_masks.items():
            score = _iou(mask, fused_mask) if fused_any else 0.0
            self._conf[name][obj_id].append(max(score, 0.1))  # floor at 0.1


# ── module-level convenience ───────────────────────────────────────────────────

_default_engine: Optional[FusionEngine] = None


def get_engine(model_names: List[str] = None) -> FusionEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = FusionEngine(model_names)
    return _default_engine
