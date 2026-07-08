#!/usr/bin/env python3
"""
sam3_client.py  –  raw‑socket client for the SAM3 daemon (port 8765)
Place at: /mnt/DATA1/INTERNSHIP/vots2026_workspace/trackers/ensemble/sam3_client.py

Matches the protocol of sam3_server.py:
  - start_session(color_dir)
  - add_prompt(session_id, frame_idx=0, points, point_labels, obj_id)
  - propagate_start(session_id, start_frame_idx=0, max_frame_num_to_track)
  - propagate_next(session_id)   (call repeatedly)
  - close_session(session_id)
"""

import socket, pickle, struct, os
import numpy as np
from PIL import Image


class SAM3RawClient:
    def __init__(self, host='127.0.0.1', port=8765):
        self.host = host
        self.port = port
        self.sock = None
        self.session_id = None
        self.gen_started = False
        self.num_objects = 0
        self.width = 0
        self.height = 0

    def _connect(self):
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(30)
        self.sock.connect((self.host, self.port))

    def _send(self, obj):
        data = pickle.dumps(obj)
        self.sock.sendall(struct.pack('>I', len(data)) + data)

    def _recv(self):
        raw_len = self._recv_exact(4)
        if not raw_len:
            return None
        length = struct.unpack('>I', raw_len)[0]
        data = self._recv_exact(length)
        return pickle.loads(data) if data else None

    def _recv_exact(self, n):
        buf = b''
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _call(self, req):
        self._send(req)
        resp = self._recv()
        if resp is None:
            raise RuntimeError("SAM3 daemon closed connection")
        if not resp.get('ok'):
            raise RuntimeError(f"SAM3 daemon error: {resp.get('error')}")
        return resp

    def ping(self):
        """Trivial connection test — start a dummy session and close it."""
        try:
            self._connect()
            resp = self._call({'cmd': 'start_session', 'color_dir': '/tmp'})
            sid = resp['result']['session_id']
            self._call({'cmd': 'close_session', 'session_id': sid})
            self.sock.close()
            self.sock = None
            return True
        except Exception:
            return False

    def init(self, color_dir: str, masks: dict):
        """
        Start a SAM3 tracking session.
        color_dir : path to directory containing the sequence frames
        masks     : {obj_id: HxW bool}  ground‑truth frame‑0 masks
        """
        self._connect()

        # 1. start session
        resp = self._call({'cmd': 'start_session', 'color_dir': color_dir})
        self.session_id = resp['result']['session_id']

        # 2. determine frame count and image size
        frame_files = sorted(f for f in os.listdir(color_dir)
                             if f.lower().endswith(('.jpg', '.png')))
        self.total_frames = len(frame_files)
        if not frame_files:
            raise RuntimeError("No images in color_dir")
        img = Image.open(os.path.join(color_dir, frame_files[0]))
        self.width, self.height = img.size
        self.num_objects = len(masks)

        # 3. add prompt for each object (centroid point)
        for obj_id, mask in masks.items():
            mask_arr = np.asarray(mask).astype(bool)
            rows, cols = np.where(mask_arr)
            if rows.size == 0:
                pt = [0.5, 0.5]
            else:
                pt = [cols.mean() / self.width, rows.mean() / self.height]
            self._call({
                'cmd': 'add_prompt',
                'session_id': self.session_id,
                'frame_idx': 0,
                'points': [pt],
                'point_labels': [1],
                'obj_id': int(obj_id),
                'rel_coordinates': True,
            })

        # 4. start propagation
        self._call({
            'cmd': 'propagate_start',
            'session_id': self.session_id,
            'start_frame_idx': 0,
            'max_frame_num_to_track': self.total_frames,
        })
        self.gen_started = True

        # 5. consume frame‑0 result (the init frame)
        resp = self._call({'cmd': 'propagate_next', 'session_id': self.session_id})
        item0 = resp.get('item')
        if item0 is not None:
            return self._extract_masks(item0)
        return {int(k): v for k, v in masks.items()}  # fallback to gt

    def track(self):
        """Return masks for the next frame (dict obj_id → bool mask)."""
        if not self.gen_started:
            raise RuntimeError("SAM3 session not started")
        resp = self._call({'cmd': 'propagate_next', 'session_id': self.session_id})
        item = resp.get('item')
        if item is None or resp.get('stop'):
            self.gen_started = False
            return {}
        return self._extract_masks(item)

    def reset(self):
        """Close current session."""
        if self.session_id:
            try:
                self._call({'cmd': 'close_session', 'session_id': self.session_id})
            except:
                pass
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        self.sock = None
        self.session_id = None
        self.gen_started = False

    def _extract_masks(self, item):
        outs = item['outputs']
        out_ids = list(outs['out_obj_ids'])
        masks = outs['out_binary_masks']
        result = {}
        for obj_id in range(1, self.num_objects + 1):
            if obj_id in out_ids:
                idx = out_ids.index(obj_id)
                m = masks[idx].astype(bool)
                if m.sum() > 0:
                    result[obj_id] = m
        # fill missing objects with empty mask
        for obj_id in range(1, self.num_objects + 1):
            if obj_id not in result:
                result[obj_id] = np.zeros((self.height, self.width), dtype=bool)
        return result
