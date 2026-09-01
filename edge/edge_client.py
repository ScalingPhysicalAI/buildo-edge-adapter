"""EdgeVLA client (laptop variant; the STM32MP2 runs mp2_client.py).
Architecture follows Algorithm 1 of the AsyncVLA paper (arXiv:2602.13476).

Implements the onboard-controller loop of Algorithm 1 in the AsyncVLA paper:
  - "camera" loop: replays sample frames with timestamps into a buffer
  - sender loop: ships the latest frame + goal pose to the workstation server
  - on receiving guidance embeddings, retrieves the matching delayed frame
    from the buffer by timestamp
  - edge loop: runs the trained Edge Adapter at max rate on (current frame,
    delayed frame, guidance), converts poses to velocity commands via the
    repo's PD controller math

Camera schedule: past.png for the first CAM_SWITCH_S seconds, then cur.png.
This reproduces the single-machine demo's condition (VLA sees the stale
past.png while the adapter sees cur.png), so the resulting trajectory can be
compared directly against the verified single-machine output.
"""
import os
import argparse
import importlib.util
import io
import pickle
import socket
import struct
import sys
import threading
import time
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
# Avoid OpenMP spin-wait contention with the network/camera threads: with the
# default one-worker-per-core setting, the forward pass degrades ~11x (measured
# 653ms vs 87ms) as soon as any other thread needs CPU time.
torch.set_num_threads(1)
from PIL import Image
from torchvision.transforms import Normalize
from torchvision.transforms.functional import to_tensor, resize

HOME = os.path.expanduser("~/edgevla")
ASYNCVLA = f"{HOME}/AsyncVLA"
with open(f"{HOME}/pod_address.txt") as _f:
    _ip, _ssh_port, _app_port = _f.read().split()
SERVER = (_ip, int(_app_port))  # RunPod public IP : exposed TCP port

parser = argparse.ArgumentParser(description="EdgeVLA edge client")
parser.add_argument("seconds", nargs="?", type=float, default=60.0,
                    help="run duration in seconds (default 60)")
parser.add_argument("--goal", nargs=2, type=float, default=[10.0, -100.0],
                    metavar=("X_FWD", "Y_LEFT"),
                    help="normalized goal pose: x forward, y left (repo demo: 10 -100)")
parser.add_argument("--img-cur", default=None, help="path to custom current-view image")
parser.add_argument("--img-past", default=None, help="path to custom past-view image")
parser.add_argument("--tag", default="result", help="suffix for the output figure filename")
args = parser.parse_args()

# Goal heading points toward the goal (repo demo [10,-100,0,-1] follows this too:
# atan2(-100,10) = -84 deg, i.e. cos~0, sin~-1)
_th = np.arctan2(args.goal[1], args.goal[0])
GOAL_POSE = [args.goal[0], args.goal[1], float(np.cos(_th)), float(np.sin(_th))]
RUN_S = args.seconds
CAM_SWITCH_S = 4.0
CAM_HZ = 3.0
OUT_PNG = f"{HOME}/results/edgevla_split_{args.tag}.png"
os.makedirs(f"{HOME}/results", exist_ok=True)
print(f"Goal pose: x_fwd={GOAL_POSE[0]} y_left={GOAL_POSE[1]}  run={RUN_S:.0f}s")

# ---- Load Edge Adapter exactly as in the verified standalone test ----------
constants = types.ModuleType("prismatic.vla.constants")
constants.NUM_ACTIONS_CHUNK, constants.ACTION_DIM, constants.POSE_DIM = 8, 4, 4
constants.IGNORE_INDEX, constants.ACTION_TOKEN_BEGIN_IDX, constants.STOP_INDEX = -100, 31743, 2
sys.modules["prismatic"] = types.ModuleType("prismatic")
sys.modules["prismatic.vla"] = types.ModuleType("prismatic.vla")
sys.modules["prismatic.vla.constants"] = constants
sys.path.insert(0, f"{HOME}/Learning-to-Drive-Anywhere-with-MBRA/train")
spec = importlib.util.spec_from_file_location("small_head", f"{ASYNCVLA}/prismatic/models/small_head.py")
small_head = importlib.util.module_from_spec(spec)
spec.loader.exec_module(small_head)

model = small_head.Edge_adapter(obs_encoding_size=1024, mha_num_attention_heads=4,
                                mha_num_attention_layers=4, mha_ff_dim_factor=4)
state = torch.load(f"{HOME}/checkpoints/shead--750000_checkpoint.pt", map_location="cpu")
state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}
model.load_state_dict(state, strict=False)
model = model.float().eval()
print("Edge Adapter loaded (76M params)")

normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def prep96(img_pil):
    return normalize(resize(to_tensor(img_pil), (96, 96))).unsqueeze(0)

def delta_to_pose(delta):
    dx, dy = delta[..., 0], delta[..., 1]
    dtheta = torch.atan2(delta[..., 3], delta[..., 2])
    x, y, theta = dx[:, 0], dy[:, 0], dtheta[:, 0]
    poses = [torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=-1)]
    for t in range(1, dx.shape[1]):
        ct, st = torch.cos(theta), torch.sin(theta)
        x = x + ct * dx[:, t] - st * dy[:, t]
        y = y + st * dx[:, t] + ct * dy[:, t]
        theta = theta + dtheta[:, t]
        poses.append(torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=-1))
    return torch.stack(poses, dim=1)

def pd_controller(actions, spacing=0.1):
    wp = actions.float().numpy()[0][4].copy()
    wp[:2] *= spacing
    dx, dy, hx, hy = wp
    DT = 1 / 3
    if abs(dx) < 1e-8:
        lin, ang = 0.0, np.sign(dy) * np.pi / (2 * DT) if abs(dy) > 1e-8 else np.arctan2(hy, hx) / DT
    else:
        lin, ang = dx / DT, np.arctan(dy / dx) / DT
    return float(np.clip(lin, 0, 0.3)), float(np.clip(ang, -0.3, 0.3))

# ---- Frames -----------------------------------------------------------------
past_path = args.img_past or f"{ASYNCVLA}/inference/past.png"
cur_path = args.img_cur or f"{ASYNCVLA}/inference/cur.png"
frame_past = Image.open(past_path).convert("RGB").resize((224, 224), Image.BILINEAR)
frame_cur = Image.open(cur_path).convert("RGB").resize((224, 224), Image.BILINEAR)
print(f"Frames: past={past_path} cur={cur_path}")

def jpeg_bytes(img):
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=90)
    return b.getvalue()

start = time.time()

def camera_frame():
    return frame_past if (time.time() - start) < CAM_SWITCH_S else frame_cur

# ---- Shared state ------------------------------------------------------------
buffer = {}          # timestamp -> PIL frame (the paper's image buffer B)
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
        # keep buffer bounded
        for k in [k for k in buffer if t - k > 30.0]:
            buffer.pop(k, None)
        time.sleep(1.0 / CAM_HZ)

def workstation_loop():
    """Send newest frame to server, receive guidance, timestamp-match."""
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
            guidance["emb"] = torch.from_numpy(msg["emb"])
            guidance["t"] = t_k
            guidance["img_past96"] = prep96(matched)
            guidance["server_infer_s"] = msg["infer_s"]
    conn.close()

threading.Thread(target=camera_loop, daemon=True).start()
threading.Thread(target=workstation_loop, daemon=True).start()

# ---- Edge loop ----------------------------------------------------------------
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
    with torch.no_grad():
        deltas = model(img_cur96, img_past96, emb)
    poses = delta_to_pose(deltas)
    lin, ang = pd_controller(poses)
    edge_ms = (time.perf_counter() - t0) * 1000
    k = time.time() - t_g
    stats["k"].append(k)
    stats["edge_ms"].append(edge_ms)
    stats["t_wall"].append(time.time() - start)
    stats["server_infer"].append(srv_s)
    last_traj = poses[0, :, :2].numpy()
    n += 1
    if n % 20 == 0:
        print(f"[{time.time()-start:5.1f}s] edge {edge_ms:5.1f}ms ({1000/edge_ms:4.1f} Hz) "
              f"staleness k={k:4.2f}s vel=({lin:.2f} m/s, {ang:+.2f} rad/s) "
              f"traj_end=({last_traj[-1][0]:.2f},{last_traj[-1][1]:.2f})")
stop.set()

# ---- Report -------------------------------------------------------------------
k_arr, e_arr = np.array(stats["k"]), np.array(stats["edge_ms"])
print(f"\n==== RESULT over {RUN_S:.0f}s ====")
print(f"edge adapter passes: {n} avg {e_arr.mean():.1f}ms -> {1000/e_arr.mean():.1f} Hz")
print(f"guidance staleness k: mean {k_arr.mean():.2f}s min {k_arr.min():.2f}s max {k_arr.max():.2f}s")
print(f"server VLA inference: mean {np.mean(stats['server_infer'])*1000:.0f}ms")
print(f"final trajectory (normalized, x fwd / y left):\n{last_traj}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].plot([0.0, *(-last_traj[:, 1])], [0.0, *last_traj[:, 0]], "o-", color="blue",
             label="edge adapter (distributed)")
axes[0].plot(-GOAL_POSE[1] * 0.1 * 10, GOAL_POSE[0] * 0.1 * 10, "r*", markersize=15)
axes[0].set_title(f"Trajectory (goal: x_fwd={GOAL_POSE[0]:.0f}, y_left={GOAL_POSE[1]:.0f})")
axes[0].set_xlabel("y (left)"); axes[0].set_ylabel("x (forward)")
axes[0].axis("equal"); axes[0].legend()
axes[1].plot(stats["t_wall"], k_arr)
axes[1].set_title("Guidance staleness k (s) over time"); axes[1].set_xlabel("wall time (s)")
axes[2].hist(e_arr, bins=30)
axes[2].set_title(f"Edge Adapter latency (ms), mean {e_arr.mean():.0f}ms = {1000/e_arr.mean():.0f} Hz")
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=90)
print("Saved:", OUT_PNG)
print("SPLIT_RUN_OK")
