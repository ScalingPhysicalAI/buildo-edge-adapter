"""Export the Edge Adapter to ONNX and verify output parity with PyTorch."""
import os
import importlib.util
import sys
import types

import numpy as np
import torch

HOME = os.path.expanduser("~/edgevla")
ASYNCVLA = f"{HOME}/AsyncVLA"
OUT = f"{HOME}/edge_adapter.onnx"

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

img_cur = torch.randn(1, 3, 96, 96)
img_past = torch.randn(1, 3, 96, 96)
emb = torch.randn(1, 8, 1024)

torch.onnx.export(
    model, (img_cur, img_past, emb), OUT,
    input_names=["img_cur", "img_past", "guidance"],
    output_names=["deltas"],
    opset_version=17,
    dynamo=False,
)
print("Exported:", OUT)

# Parity check
import onnxruntime as ort
sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
with torch.no_grad():
    ref = model(img_cur, img_past, emb).numpy()
out = sess.run(None, {"img_cur": img_cur.numpy(), "img_past": img_past.numpy(),
                      "guidance": emb.numpy()})[0]
diff = np.abs(ref - out).max()
print(f"max |torch - onnx| = {diff:.2e}")
assert diff < 1e-4, "parity check failed"
print("PARITY_OK")
