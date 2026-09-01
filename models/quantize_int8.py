"""Build two INT8 variants of the Edge Adapter and measure trajectory drift vs
float32 using REAL guidance embeddings. Originals are never modified.
  A) dynamic quant, MatMul-only  (transformer int8, CNN stays fp32)
  B) static QDQ quant, per-channel, calibrated on real scenes+embeddings
"""
import glob
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader, QuantFormat, QuantType,
    quantize_dynamic, quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image

HOME = os.path.expanduser("~/edgevla")
FP32 = f"{HOME}/edge_adapter.onnx"
MM = f"{HOME}/edge_adapter_int8mm.onnx"
QDQ = f"{HOME}/edge_adapter_int8qdq.onnx"
PRE = f"{HOME}/_preproc_tmp.onnx"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

def prep96(path):
    img = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    a = np.asarray(img.resize((96, 96), Image.BILINEAR), dtype=np.float32) / 255.0
    return ((a.transpose(2, 0, 1) - MEAN) / STD)[None]

scenes = {
    "office": (prep96(f"{HOME}/AsyncVLA/inference/cur.png"), prep96(f"{HOME}/AsyncVLA/inference/past.png")),
    "mirror": (prep96(f"{HOME}/cur_mirror.png"), prep96(f"{HOME}/past_mirror.png")),
    "open": (prep96(f"{HOME}/open_cur.png"), prep96(f"{HOME}/open_past.png")),
}
embs = {p.split("emb_")[1][:-4]: np.load(p) for p in glob.glob(f"{HOME}/emb_*.npy")}
cases = [("office", "office_right"), ("office", "office_left"),
         ("mirror", "mirror_left"), ("open", "open_straight")]

# ---- A: dynamic, MatMul only ---------------------------------------------------
quantize_dynamic(FP32, MM, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])

# ---- B: static QDQ -------------------------------------------------------------
quant_pre_process(FP32, PRE, skip_symbolic_shape=True)

class Reader(CalibrationDataReader):
    def __init__(self):
        self.data = iter([
            {"img_cur": scenes[s][0], "img_past": scenes[s][1], "guidance": embs[e]}
            for s, e in cases
        ])
    def get_next(self):
        return next(self.data, None)

quantize_static(PRE, QDQ, Reader(), quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                per_channel=True)

import os
for p in (FP32, MM, QDQ):
    print(f"{os.path.basename(p)}: {os.path.getsize(p)/1e6:.0f} MB")

# ---- Evaluate drift on real inputs ---------------------------------------------
def delta_to_pose(delta):
    dx, dy = delta[..., 0], delta[..., 1]
    dtheta = np.arctan2(delta[..., 3], delta[..., 2])
    x, y, theta = dx[:, 0], dy[:, 0], dtheta[:, 0]
    poses = [np.stack([x, y], axis=-1)]
    for t in range(1, dx.shape[1]):
        ct, st = np.cos(theta), np.sin(theta)
        x = x + ct * dx[:, t] - st * dy[:, t]
        y = y + st * dx[:, t] + ct * dy[:, t]
        theta = theta + dtheta[:, t]
        poses.append(np.stack([x, y], axis=-1))
    return np.stack(poses, axis=1)

sessions = {n: ort.InferenceSession(p, providers=["CPUExecutionProvider"])
            for n, p in (("fp32", FP32), ("int8mm", MM), ("int8qdq", QDQ))}
for scene, emb_name in cases:
    feed = {"img_cur": scenes[scene][0], "img_past": scenes[scene][1], "guidance": embs[emb_name]}
    trajs = {n: delta_to_pose(s.run(None, feed)[0]) for n, s in sessions.items()}
    ref = trajs["fp32"]
    line = f"{emb_name:15s} fp32 end=({ref[0,-1,0]:6.2f},{ref[0,-1,1]:6.2f})"
    for n in ("int8mm", "int8qdq"):
        t = trajs[n]
        line += (f" | {n} end=({t[0,-1,0]:6.2f},{t[0,-1,1]:6.2f})"
                 f" drift={np.abs(t-ref).max():.3f}")
    print(line)
os.remove(PRE)
print("QUANT2_OK")
