"""AsyncVLA workstation server (runs on the RunPod 4090).

Loads the base VLA (OmniVLA) + token projector using the repo's own code
(imported from inference/run_asyncvla.py, unmodified) and serves guidance:
  receive {timestamp, jpeg image, goal_pose}  ->  run base VLA
  reply   {timestamp, 8x1024 action-token embeddings}
Implements the workstation loop of Algorithm 1 in the AsyncVLA paper.
Run from /workspace/asyncvla/AsyncVLA.
"""
import importlib.util
import io
import pickle
import socket
import struct
import time

import numpy as np
import torch
from PIL import Image

PORT = 15001

# Import the repo's inference module (its __main__ block does not run)
spec = importlib.util.spec_from_file_location("run_asyncvla_mod", "./inference/run_asyncvla.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

print("Loading models...", flush=True)
cfg = mod.InferenceConfig()
(vla, action_head, pose_projector, shead, action_proj,
 device_id, NUM_PATCHES, action_tokenizer, processor) = mod.define_model(cfg)
vla.eval(); action_proj.eval(); pose_projector.eval()

goal_image = Image.open("./inference/goal.png").convert("RGB").resize((224, 224), Image.BILINEAR)
inf = mod.Inference(
    save_dir=".", lan_inst_prompt="xxxx", goal_utm=(0.0, 0.0), goal_compass=0.0,
    goal_image_PIL=goal_image, action_tokenizer=action_tokenizer, processor=processor,
)

MODALITY_POSE_ONLY = torch.as_tensor([4], dtype=torch.float32)  # 2D goal pose


def vla_guidance(image_pil, goal_pose_norm):
    """Base-VLA forward pass -> projected 8x1024 action-token embeddings."""
    batch = inf.data_transformer_asyncvla(
        image_pil, "xxxx", goal_image, np.asarray(goal_pose_norm, dtype=np.float64),
        action_tokenizer=action_tokenizer, processor=processor,
    )
    modality_id = MODALITY_POSE_ONLY
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            modality_id=modality_id.to(torch.bfloat16).to(device_id),
            labels=batch["labels"].to(device_id),
            output_hidden_states=True,
            proprio=batch["goal_pose"].to(torch.bfloat16).to(device_id),
            proprio_projector=pose_projector,
            noisy_actions=None, noisy_action_projector=None,
            diffusion_timestep_embeddings=None, use_film=False,
        )
    gt = batch["labels"][:, 1:].to(device_id)
    mask = mod.get_current_action_mask(gt) | mod.get_next_actions_mask(gt)
    hidden = output.hidden_states[-1][:, NUM_PATCHES:-1]
    act_hidden = hidden[mask].reshape(1, mod.NUM_ACTIONS_CHUNK * mod.ACTION_DIM, -1).to(torch.bfloat16)
    with torch.no_grad():
        proj = action_proj.predict_action(act_hidden.detach(), modality_id.to(torch.bfloat16).to(device_id))
    return proj.float().cpu().numpy()  # (1, 8, 1024)


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


def send_msg(conn, obj):
    data = pickle.dumps(obj, protocol=4)
    conn.sendall(struct.pack("!I", len(data)) + data)


# Warmup pass so first client request isn't slowed by CUDA init
warm = Image.open("./inference/past.png").convert("RGB").resize((224, 224), Image.BILINEAR)
t0 = time.perf_counter()
vla_guidance(warm, [10.0, -100.0, 0.0, -1.0])
print(f"Warmup inference: {time.perf_counter()-t0:.2f}s", flush=True)

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("0.0.0.0", PORT))
srv.listen(1)
print(f"LISTENING on {PORT}", flush=True)

while True:
    conn, addr = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print("Client connected:", addr, flush=True)
    try:
        while True:
            msg = recv_msg(conn)
            if msg is None:
                break
            img = Image.open(io.BytesIO(msg["jpeg"])).convert("RGB").resize((224, 224), Image.BILINEAR)
            t0 = time.perf_counter()
            emb = vla_guidance(img, msg["goal_pose"])
            dt = time.perf_counter() - t0
            send_msg(conn, {"t": msg["t"], "emb": emb.astype(np.float32), "infer_s": dt})
            print(f"served t={msg['t']:.3f} infer={dt*1000:.0f}ms", flush=True)
    except (ConnectionError, EOFError):
        # Normal end of a client run: the edge client exits and the socket resets.
        print("client session ended", flush=True)
    finally:
        conn.close()
