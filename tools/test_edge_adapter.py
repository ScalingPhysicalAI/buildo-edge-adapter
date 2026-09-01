"""Standalone Edge Adapter test (laptop / edge side).

Loads the trained 76M Edge Adapter from the AsyncVLA release checkpoint and runs
it on the repo's sample images with a DUMMY guidance tensor (the real 8x1024
guidance will come from OmniVLA on the RunPod server later). Verifies the model
loads, runs on CPU, and measures achievable inference rate.

Deliberately avoids importing the `prismatic` package (whose __init__ pulls in
transformers/accelerate); instead loads small_head.py directly and stubs the
constants module. The AsyncVLA repo itself is not modified.
"""
import os
import importlib.util
import sys
import time
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.transforms import Normalize
from torchvision.transforms.functional import to_tensor, resize

HOME = os.path.expanduser("~/edgevla")
ASYNCVLA = f"{HOME}/AsyncVLA"
MBRA_TRAIN = f"{HOME}/Learning-to-Drive-Anywhere-with-MBRA/train"
CKPT = f"{HOME}/checkpoints/shead--750000_checkpoint.pt"

# --- Stub prismatic.vla.constants (values copied from the repo) -------------
constants = types.ModuleType("prismatic.vla.constants")
constants.NUM_ACTIONS_CHUNK = 8
constants.ACTION_DIM = 4
constants.POSE_DIM = 4
constants.IGNORE_INDEX = -100
constants.ACTION_TOKEN_BEGIN_IDX = 31743
constants.STOP_INDEX = 2
pkg_prismatic = types.ModuleType("prismatic")
pkg_vla = types.ModuleType("prismatic.vla")
sys.modules["prismatic"] = pkg_prismatic
sys.modules["prismatic.vla"] = pkg_vla
sys.modules["prismatic.vla.constants"] = constants

# --- Load small_head.py directly from file ----------------------------------
sys.path.insert(0, MBRA_TRAIN)  # provides vint_train
spec = importlib.util.spec_from_file_location(
    "small_head", f"{ASYNCVLA}/prismatic/models/small_head.py"
)
small_head = importlib.util.module_from_spec(spec)
spec.loader.exec_module(small_head)

# --- Build model and load trained weights (config_nav/dataset_config.yaml) --
model = small_head.Edge_adapter(
    obs_encoding_size=1024,
    mha_num_attention_heads=4,
    mha_num_attention_layers=4,
    mha_ff_dim_factor=4,
)
n_params = sum(p.numel() for p in model.parameters())
print(f"Edge Adapter parameters: {n_params/1e6:.1f}M")

state = torch.load(CKPT, map_location="cpu")
state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
         for k, v in state.items()}
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"Missing keys: {missing}")
print(f"Unexpected keys: {unexpected}")
model = model.float().eval()

# --- Preprocess sample images exactly like inference/run_asyncvla.py --------
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def prep(path):
    img = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    return normalize(resize(to_tensor(img), (96, 96))).unsqueeze(0)

img_past = prep(f"{ASYNCVLA}/inference/past.png")  # delayed obs I_{t-k}
img_cur = prep(f"{ASYNCVLA}/inference/cur.png")    # current obs I_t

# Dummy stand-in for OmniVLA's projected action-token embeddings (8x1024)
guidance = torch.randn(1, 8, 1024)

# --- delta_to_pose (copied from inference/run_asyncvla.py) ------------------
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

# --- Single forward pass + sanity checks -------------------------------------
with torch.no_grad():
    deltas = model(img_cur, img_past, guidance)
poses = delta_to_pose(deltas)
print(f"Output delta chunk shape: {tuple(deltas.shape)} (expect (1, 8, 4))")
print(f"Pose chunk:\n{poses[0, :, :2]}")
assert deltas.shape == (1, 8, 4), "unexpected output shape"
assert torch.isfinite(deltas).all(), "non-finite outputs"

# --- Benchmark ----------------------------------------------------------------
for _ in range(3):  # warmup
    with torch.no_grad():
        model(img_cur, img_past, guidance)
n_iter = 20
t0 = time.perf_counter()
for _ in range(n_iter):
    with torch.no_grad():
        model(img_cur, img_past, guidance)
dt = (time.perf_counter() - t0) / n_iter
print(f"CPU inference: {dt*1000:.1f} ms/pass -> {1/dt:.1f} Hz "
      f"(paper's Jetson Orin target: 8 Hz)")

# --- Visualization ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(Image.open(f"{ASYNCVLA}/inference/past.png"))
axes[0].set_title("Delayed obs $I_{t-k}$ (past.png)")
axes[1].imshow(Image.open(f"{ASYNCVLA}/inference/cur.png"))
axes[1].set_title("Current obs $I_t$ (cur.png)")
xy = poses[0, :, :2].numpy()
axes[2].plot([0.0, *(-xy[:, 1])], [0.0, *xy[:, 0]], "o-", color="blue")
axes[2].set_title("Trajectory (DUMMY guidance - not meaningful yet)")
axes[2].set_xlabel("y (left) [normalized]")
axes[2].set_ylabel("x (forward) [normalized]")
axes[2].axis("equal")
for a in axes[:2]:
    a.axis("off")
plt.tight_layout()
out = f"{HOME}/results/edgevla_edge_test.png"
os.makedirs(f"{HOME}/results", exist_ok=True)
plt.savefig(out, dpi=90)
print(f"Saved visualization: {out}")
print("EDGE_ADAPTER_TEST_OK")
