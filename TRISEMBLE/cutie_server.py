#!/usr/bin/env python3
"""
cutie_server.py  –  Cutie VOS daemon for the ensemble tracker
Place at: /mnt/DATA1/INTERNSHIP/vots2026_workspace/trackers/ensemble/cutie_server.py

Listens on ZMQ REP port 5557.
Commands  (all sent as pickle dicts):
  {'cmd': 'ping'}
  {'cmd': 'init',  'frame': HxWx3 uint8, 'masks': {obj_id(int): HxW bool}}
  {'cmd': 'track', 'frame': HxWx3 uint8}
  {'cmd': 'reset'}
Replies are also pickle dicts with 'status'/'masks'/'error'.
"""

import sys, os, traceback

CUTIE_ROOT  = '/mnt/DATA1/INTERNSHIP/Cutie'
CHECKPOINT  = '/mnt/DATA1/INTERNSHIP/Cutie/checkpoints/cutie-base-mega.pth'
DEVICE      = 'cuda:0'
PORT        = 5557

sys.path.insert(0, CUTIE_ROOT)

import zmq, pickle
import numpy as np
import torch

# ── helpers ────────────────────────────────────────────────────────────────────

def frame_to_tensor(frame_np: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8  →  1x3xHxW float32 on GPU, normalised 0-1"""
    t = torch.from_numpy(frame_np.astype(np.float32) / 255.0)
    return t.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
def masks_dict_to_label_map(masks_dict: dict, h: int, w: int) -> torch.Tensor:
    """
    {obj_id: HxW bool/uint8}  →  HxW long tensor
    pixel value = obj_id  (0 = background)
    """
    lmap = np.zeros((h, w), dtype=np.int64)
    for obj_id, mask in masks_dict.items():
        lmap[mask.astype(bool)] = int(obj_id)
    return torch.from_numpy(lmap).to(DEVICE)


def prob_to_masks_dict(output_prob: torch.Tensor, obj_ids: list) -> dict:
    """
    Cutie returns  N x H x W  probability tensor (one channel per object).
    Convert to {obj_id: HxW bool numpy}.
    """
    result = {}
    probs = output_prob.cpu().float()   # N x H x W
    if probs.dim() == 3 and probs.shape[0] == len(obj_ids):
        for i, oid in enumerate(obj_ids):
            result[oid] = (probs[i].numpy() > 0.5)
    elif probs.dim() == 2:
        # single object – treat as confidence map
        result[obj_ids[0]] = (probs.numpy() > 0.5)
    return result
# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print('[Cutie] Loading model …')
    from omegaconf import OmegaConf
    from cutie.model.cutie import CUTIE
    from cutie.inference.inference_core import InferenceCore

    # ── build config ──────────────────────────────────────────────────────────
    eval_config_path = os.path.join(CUTIE_ROOT, 'cutie', 'config', 'eval_config.yaml')
    model_config_path = os.path.join(CUTIE_ROOT, 'cutie', 'config', 'model', 'base.yaml')

    if not os.path.exists(eval_config_path):
        raise FileNotFoundError(f"Missing eval config: {eval_config_path}")
    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"Missing model config: {model_config_path}")

    cfg = OmegaConf.load(eval_config_path)
    model_cfg = OmegaConf.load(model_config_path)
    cfg.model = model_cfg

    cfg.amp = cfg.get('amp', False)
    cfg.max_internal_size = cfg.get('max_internal_size', -1)
    cfg.flip_aug = cfg.get('flip_aug', False)

    print(f'[Cutie] Using eval config: {eval_config_path}')
    print('[Cutie] Model config merged.')

    # ── load weights ──────────────────────────────────────────────────────────
    cutie_model = CUTIE(cfg).to(DEVICE).eval()
    sd = torch.load(CHECKPOINT, map_location=DEVICE)
    # Some checkpoints wrap weights under a 'model' key
    if 'model' in sd:
        sd = sd['model']
    cutie_model.load_state_dict(sd, strict=False)
    print('[Cutie] Weights loaded OK ✓')

    # ── ZMQ socket ────────────────────────────────────────────────────────────
    context = zmq.Context()
    sock = context.socket(zmq.REP)
    sock.bind(f'tcp://127.0.0.1:{PORT}')
    print(f'[Cutie] Server ready on port {PORT}')

    processor = None
    obj_ids   = []

    while True:
        raw = sock.recv()
        try:
            msg = pickle.loads(raw)
        except Exception as e:
            sock.send(pickle.dumps({'error': f'Unpickle failed: {e}'}))
            continue

        cmd = msg.get('cmd', '')

        try:
            # ── ping ──────────────────────────────────────────────────────────
            if cmd == 'ping':
                sock.send(pickle.dumps({'status': 'ok'}))
            # ── init  (first frame + ground-truth masks) ─────────────────────
            elif cmd == 'init':
                frame_np   = msg['frame']    # HxWx3 uint8
                masks_dict = msg['masks']    # {int_id: HxW bool}
                obj_ids    = sorted(masks_dict.keys())

                h, w    = frame_np.shape[:2]
                frame_t = frame_to_tensor(frame_np)
                lmap    = masks_dict_to_label_map(masks_dict, h, w)

                processor = InferenceCore(cutie_model, cfg=cfg)
                processor.set_all_labels(obj_ids)

                with torch.inference_mode():
                    output_prob = processor.step(frame_t, lmap, objects=obj_ids)

                result_masks = prob_to_masks_dict(output_prob, obj_ids)
                sock.send(pickle.dumps({'status': 'ok', 'masks': result_masks}))

            # ── track  (subsequent frames, no prompt) ────────────────────────
            elif cmd == 'track':
                frame_np = msg['frame']
                frame_t  = frame_to_tensor(frame_np)

                with torch.inference_mode():
                    output_prob = processor.step(frame_t)

                result_masks = prob_to_masks_dict(output_prob, obj_ids)
                sock.send(pickle.dumps({'masks': result_masks}))
            # ── reset  (new sequence) ─────────────────────────────────────────
            elif cmd == 'reset':
                processor = None
                obj_ids   = []
                torch.cuda.empty_cache()
                sock.send(pickle.dumps({'status': 'ok'}))

            else:
                sock.send(pickle.dumps({'error': f'Unknown cmd: {cmd}'}))

        except Exception as e:
            sock.send(pickle.dumps({'error': str(e), 'traceback': traceback.format_exc()}))


if __name__ == '__main__':
    main()
