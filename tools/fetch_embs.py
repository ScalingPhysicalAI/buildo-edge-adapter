"""Fetch real guidance embeddings from the live VLA server for the three test
scenes; saved as .npy for quantization calibration and drift evaluation."""
import os
import io
import pickle
import socket
import struct
import numpy as np
from PIL import Image

HOME = os.path.expanduser("~/edgevla")
with open(f"{HOME}/pod_address.txt") as f:
    ip, _ssh, app = f.read().split()

def send_msg(conn, obj):
    data = pickle.dumps(obj, protocol=4)
    conn.sendall(struct.pack("!I", len(data)) + data)

def recv_msg(conn):
    hdr = b""
    while len(hdr) < 4:
        hdr += conn.recv(4 - len(hdr))
    (n,) = struct.unpack("!I", hdr)
    buf = b""
    while len(buf) < n:
        buf += conn.recv(min(65536, n - len(buf)))
    return pickle.loads(buf)

def jpeg(path):
    img = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=90)
    return b.getvalue()

def goal(x, y):
    th = np.arctan2(y, x)
    return [x, y, float(np.cos(th)), float(np.sin(th))]

jobs = [
    ("office_right", f"{HOME}/AsyncVLA/inference/past.png", goal(10, -100)),
    ("office_left", f"{HOME}/AsyncVLA/inference/past.png", goal(10, 100)),
    ("mirror_left", f"{HOME}/past_mirror.png", goal(10, 100)),
    ("open_straight", f"{HOME}/open_past.png", goal(100, 0)),
]
conn = socket.create_connection((ip, int(app)), timeout=30)
conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
for name, img, g in jobs:
    send_msg(conn, {"t": 0.0, "jpeg": jpeg(img), "goal_pose": g})
    msg = recv_msg(conn)
    np.save(f"{HOME}/emb_{name}.npy", msg["emb"].astype(np.float32))
    print(f"{name}: emb {msg['emb'].shape} std={msg['emb'].std():.3f}")
conn.close()
print("EMBS_OK")
