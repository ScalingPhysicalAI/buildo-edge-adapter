"""EdgeVLA client for the STM32MP2 (torch-free: onnxruntime + numpy + PIL).
Architecture follows Algorithm 1 of the AsyncVLA paper (arXiv:2602.13476).

Same structure as the laptop's edge_client.py:
  camera thread  : replays test frames into a timestamped buffer
  network thread : ships newest frame + goal pose to the cloud VLA server,
                   receives 8x1024 guidance embeddings, timestamp-matches
  edge loop      : runs the Edge Adapter (ONNX) on (current, delayed, guidance)
                   and converts pose chunks to velocity commands

Results print to the terminal and are saved as JSON for later plotting.
"""
import argparse
import io
import json
import pickle
import socket
import struct
import threading
import time

import os

HOME = "/usr/local/edgevla"
# Board's rootfs is full, so onnxruntime can't write its telemetry ID under
# /root and prints a startup warning; point HOME at the writable partition.
os.environ["HOME"] = HOME

import numpy as np
import onnxruntime as ort
from PIL import Image

parser = argparse.ArgumentParser(description="EdgeVLA MP2 edge client")
parser.add_argument("seconds", nargs="?", type=float, default=30.0)
parser.add_argument("--goal", nargs=2, type=float, default=[10.0, -100.0],
                    metavar=("X_FWD", "Y_LEFT"))
parser.add_argument("--img-past", default=f"{HOME}/past.png")
parser.add_argument("--img-cur", default=f"{HOME}/cur.png")
parser.add_argument("--tag", default="mp2_run")
parser.add_argument("--threads", type=int, default=2, help="onnxruntime intra-op threads")
parser.add_argument("--model", default=f"{HOME}/edge_adapter.onnx",
                    help="ONNX model file (default: original fp32)")
parser.add_argument("--bench", action="store_true",
                    help="only benchmark local inference, no network")
args = parser.parse_args()

_th = np.arctan2(args.goal[1], args.goal[0])
GOAL_POSE = [args.goal[0], args.goal[1], float(np.cos(_th)), float(np.sin(_th))]
RUN_S = args.seconds
CAM_SWITCH_S = 4.0
CAM_HZ = 3.0

with open(f"{HOME}/pod_address.txt") as f:
    _ip, _ssh_port, _app_port = f.read().split()
SERVER = (_ip, int(_app_port))

# ---- Edge Adapter (ONNX) ------------------------------------------------------
so = ort.SessionOptions()
so.intra_op_num_threads = args.threads
so.add_session_config_entry("session.intra_op.allow_spinning", "0")
sess = ort.InferenceSession(args.model, sess_options=so,
                            providers=["CPUExecutionProvider"])
print(f"Edge Adapter loaded ({args.model}, {args.threads} threads)")

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

def prep96(img_pil):
    a = np.asarray(img_pil.resize((96, 96), Image.BILINEAR), dtype=np.float32) / 255.0
    a = a.transpose(2, 0, 1)
    return ((a - MEAN) / STD)[None]

def run_adapter(img_cur96, img_past96, emb):
    return sess.run(None, {"img_cur": img_cur96, "img_past": img_past96,
                           "guidance": emb})[0]

def delta_to_pose(delta):
    dx, dy = delta[..., 0], delta[..., 1]
    dtheta = np.arctan2(delta[..., 3], delta[..., 2])
    x, y, theta = dx[:, 0], dy[:, 0], dtheta[:, 0]
    poses = [np.stack([x, y, np.cos(theta), np.sin(theta)], axis=-1)]
    for t in range(1, dx.shape[1]):
        ct, st = np.cos(theta), np.sin(theta)
        x = x + ct * dx[:, t] - st * dy[:, t]
        y = y + st * dx[:, t] + ct * dy[:, t]
        theta = theta + dtheta[:, t]
        poses.append(np.stack([x, y, np.cos(theta), np.sin(theta)], axis=-1))
    return np.stack(poses, axis=1)

def pd_controller(poses, spacing=0.1):
    wp = poses[0][4].copy()
    wp[:2] *= spacing
    dx, dy, hx, hy = wp
    DT = 1 / 3
    if abs(dx) < 1e-8:
        lin = 0.0
        ang = np.sign(dy) * np.pi / (2 * DT) if abs(dy) > 1e-8 else np.arctan2(hy, hx) / DT
    else:
        lin, ang = dx / DT, np.arctan(dy / dx) / DT
    return float(np.clip(lin, 0, 0.3)), float(np.clip(ang, -0.3, 0.3))

# ---- Frames -------------------------------------------------------------------
frame_past = Image.open(args.img_past).convert("RGB").resize((224, 224), Image.BILINEAR)
frame_cur = Image.open(args.img_cur).convert("RGB").resize((224, 224), Image.BILINEAR)
print(f"Goal: x_fwd={GOAL_POSE[0]} y_left={GOAL_POSE[1]}  frames: {args.img_past}, {args.img_cur}")

if args.bench:
    img = prep96(frame_cur)
    emb = np.random.randn(1, 8, 1024).astype(np.float32)
    run_adapter(img, img, emb)  # warmup
    n = 10
    t0 = time.perf_counter()
    for _ in range(n):
        run_adapter(img, img, emb)
    dt = (time.perf_counter() - t0) / n
    print(f"BENCH: {dt*1000:.1f} ms/pass -> {1/dt:.2f} Hz ({args.threads} threads)")
    raise SystemExit

start = time.time()

def camera_frame():
    return frame_past if (time.time() - start) < CAM_SWITCH_S else frame_cur

def jpeg_bytes(img):
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=90)
    return b.getvalue()

buffer = {}
guidance_lock = threading.Lock()
guidance = {"emb": None, "t": None, "img_past96": None, "server_infer_s": None}
stats = {"k": [], "edge_ms": [], "t_wall": [], "server_infer": []}
stop = threading.Event()

def send_msg(conn, obj):
    data = pickle.dumps(obj, protocol=4)
    conn.sendall(struct.pack("!I", len(data)) + data)

def recv_msg(conn):
    hdr = b""
    while len(hdr) < 4:
        part = conn.recv(4 - len(hdr))
        if not part:
            return None
        hdr += part
    (n,) = struct.unpack("!I", hdr)
    buf = b""
    while len(buf) < n:
        part = conn.recv(min(65536, n - len(buf)))
        if not part:
            return None
        buf += part
    return pickle.loads(buf)

def camera_loop():
    while not stop.is_set():
        t = time.time()
        buffer[round(t, 3)] = camera_frame()
        for k in [k for k in buffer if t - k > 30.0]:
            buffer.pop(k, None)
        time.sleep(1.0 / CAM_HZ)

def workstation_loop():
    conn = socket.create_connection(SERVER, timeout=30)
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print("Connected to cloud VLA server")
    while not stop.is_set():
        if not buffer:
            time.sleep(0.05)
            continue
        t_img = max(buffer)
        send_msg(conn, {"t": t_img, "jpeg": jpeg_bytes(buffer[t_img]), "goal_pose": GOAL_POSE})
        msg = recv_msg(conn)
        if msg is None:
            print("server closed connection")
            break
        t_k = msg["t"]
        matched = buffer.get(t_k) or min(buffer.items(), key=lambda kv: abs(kv[0] - t_k))[1]
        with guidance_lock:
            guidance["emb"] = msg["emb"].astype(np.float32)
            guidance["t"] = t_k
            guidance["img_past96"] = prep96(matched)
            guidance["server_infer_s"] = msg["infer_s"]
    conn.close()

threading.Thread(target=camera_loop, daemon=True).start()
threading.Thread(target=workstation_loop, daemon=True).start()

print("Waiting for first guidance from workstation...")
while guidance["emb"] is None:
    time.sleep(0.1)
print("First guidance received; edge loop running")

last_traj = None
n = 0
while time.time() - start < RUN_S:
    t0 = time.perf_counter()
    img_cur96 = prep96(camera_frame())
    with guidance_lock:
        emb, t_g, img_past96 = guidance["emb"], guidance["t"], guidance["img_past96"]
        srv_s = guidance["server_infer_s"]
    deltas = run_adapter(img_cur96, img_past96, emb)
    poses = delta_to_pose(deltas)
    lin, ang = pd_controller(poses)
    edge_ms = (time.perf_counter() - t0) * 1000
    k = time.time() - t_g
    stats["k"].append(k)
    stats["edge_ms"].append(edge_ms)
    stats["t_wall"].append(time.time() - start)
    stats["server_infer"].append(srv_s)
    last_traj = poses[0, :, :2]
    n += 1
    if n % 5 == 0:
        print(f"[{time.time()-start:5.1f}s] edge {edge_ms:6.1f}ms ({1000/edge_ms:4.1f} Hz) "
              f"staleness k={k:4.2f}s vel=({lin:.2f} m/s, {ang:+.2f} rad/s) "
              f"traj_end=({last_traj[-1][0]:.2f},{last_traj[-1][1]:.2f})")
stop.set()

k_arr, e_arr = np.array(stats["k"]), np.array(stats["edge_ms"])
print(f"\n==== RESULT over {RUN_S:.0f}s (STM32MP2) ====")
print(f"edge adapter passes: {n} avg {e_arr.mean():.1f}ms -> {1000/e_arr.mean():.2f} Hz")
print(f"guidance staleness k: mean {k_arr.mean():.2f}s min {k_arr.min():.2f}s max {k_arr.max():.2f}s")
print(f"server VLA inference: mean {np.mean(stats['server_infer'])*1000:.0f}ms")
print(f"final trajectory (normalized, x fwd / y left):\n{last_traj}")

out = {"tag": args.tag, "goal": GOAL_POSE, "stats": {k: list(map(float, v)) for k, v in stats.items()},
       "final_traj": last_traj.tolist()}
out_path = f"{HOME}/{args.tag}.json"
with open(out_path, "w") as f:
    json.dump(out, f)
print("Saved:", out_path)
print("MP2_RUN_OK")
