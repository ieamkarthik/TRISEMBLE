#!/usr/bin/env python3
"""D4SM server – tiny model, CPU offloading for low GPU memory."""
import os, sys, pickle, traceback
import numpy as np
from PIL import Image
import torch
import zmq

D4SM_ROOT = '/mnt/DATA1/INTERNSHIP/vots_project/d4sm'
CHECKPOINT_DIR = os.path.join(D4SM_ROOT, 'checkpoints')
sys.path.insert(0, D4SM_ROOT)

from tracking_wrapper_mot import DAM4SAMMOT

PORT = 5558

def make_full_size(mask, output_sz):
    w, h = output_sz
    if mask.shape[0] == h and mask.shape[1] == w:
        return mask
    pad_x = w - mask.shape[1]
    if pad_x < 0:
        mask = mask[:, :mask.shape[1] + pad_x]; pad_x = 0
    pad_y = h - mask.shape[0]
    if pad_y < 0:
        mask = mask[:mask.shape[0] + pad_y, :]; pad_y = 0
    return np.pad(mask, ((0, pad_y), (0, pad_x)), 'constant', constant_values=0)
def masks_dict_to_init_regions(masks_dict, width, height):
    regions = []
    for obj_id in sorted(masks_dict.keys()):
        m_full = make_full_size(masks_dict[obj_id].astype(np.uint8), (width, height))
        regions.append({'obj_id': obj_id, 'mask': m_full})
    return regions

def output_to_masks_dict(outputs, tracker):
    return {oid: m.astype(bool) for oid, m in zip(tracker.all_obj_ids, outputs['masks'])}

def main():
    print('[D4SM] Loading DAM4SAM tiny model …', flush=True)
    tracker = DAM4SAMMOT(model_size='tiny', checkpoint_dir=CHECKPOINT_DIR, offload_state_to_cpu=True)
    print('[D4SM] Model ready.', flush=True)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f'tcp://127.0.0.1:{PORT}')
    print(f'[D4SM] Server listening on port {PORT}', flush=True)

    initialized = False
    while True:
        raw = sock.recv()
        try:
            msg = pickle.loads(raw)
        except Exception as e:
            sock.send(pickle.dumps({'error': f'Unpickle failed: {e}'})); continue
        cmd = msg.get('cmd', '')
        try:
            if cmd == 'ping':
                sock.send(pickle.dumps({'status': 'ok'}))
            elif cmd == 'init':
                frame_np = msg['frame']
                masks_dict = msg['masks']
                image = Image.fromarray(frame_np)
                w, h = image.width, image.height
                init_regions = masks_dict_to_init_regions(masks_dict, w, h)
                _ = tracker.initialize(image, init_regions)
                initialized = True
                sock.send(pickle.dumps({'status': 'ok', 'masks': {k: v.astype(bool) for k, v in masks_dict.items()}}))
            elif cmd == 'track':
                if not initialized:
                    sock.send(pickle.dumps({'error': 'Not initialized'})); continue
                frame_np = msg['frame']
                image = Image.fromarray(frame_np)
                outputs = tracker.track(image)
                masks = output_to_masks_dict(outputs, tracker)
                sock.send(pickle.dumps({'masks': masks}))
            elif cmd == 'reset':
                initialized = False
                torch.cuda.empty_cache()
                sock.send(pickle.dumps({'status': 'ok'}))
            else:
                sock.send(pickle.dumps({'error': f'Unknown cmd: {cmd}'}))
        except Exception as e:
            sock.send(pickle.dumps({'error': str(e), 'traceback': traceback.format_exc()}))

if __name__ == '__main__':
    main()
