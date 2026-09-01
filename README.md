# EdgeVLA

Distributed Vision-Language-Action (VLA) robot control: a large VLA runs on a
remote GPU server while a lightweight Edge Adapter runs on the robot,
asynchronously fusing the server's (stale) guidance with the robot's current
camera view. Control stays fast and local; intelligence stays large and remote.

The architecture follows Algorithm 1 of the AsyncVLA paper
([arXiv:2602.13476](https://arxiv.org/abs/2602.13476)) using the authors'
released OmniVLA checkpoint and Edge Adapter weights
([NHirose/AsyncVLA_release](https://huggingface.co/NHirose/AsyncVLA_release)).
Everything else in this repo - the client/server split over the public
internet, the torch-free ONNX/INT8 port to an embedded board, the performance
tuning, and the validation tooling - was built by us on top of that baseline.

```
        CLOUD GPU (RTX 4090)                    ROBOT (STM32MP2, 2x Cortex-A35)
  ┌─────────────────────────────┐          ┌──────────────────────────────────┐
  │  OmniVLA 8.3B  (~150 ms)    │  8x1024  │  camera thread (3 Hz buffer)     │
  │  server/server.py           │──emb────▶│  Edge Adapter 76M INT8 (237 ms)  │
  │  TCP :15001                 │◀──jpeg───│  edge/mp2_client.py              │
  └─────────────────────────────┘  + goal  │  -> action chunk -> PD -> vel    │
       guidance every ~1.4 s               └──────────────────────────────────┘
                                              new action chunk every ~240 ms
```

## What we built on top of AsyncVLA

The paper demonstrates the algorithm with a workstation GPU and a laptop-class
edge device on a local network, both running PyTorch. Our target was much
harsher: a **cloud GPU reached over the public internet** and an
**STM32MP2 board (2x Cortex-A35 @ 1.2 GHz, 2 GB RAM)** - a device that cannot
even install PyTorch. Getting from "paper reference code" to "real-time on
embedded hardware" required every optimization below.

| Optimization | Before | After | Where |
|---|---|---|---|
| ONNX port of the Edge Adapter (drop PyTorch entirely on the robot) | won't fit / won't install | 3-package runtime (onnxruntime, numpy, pillow) | `models/export_onnx.py`, `edge/mp2_client.py` |
| INT8 static QDQ quantization, calibrated on **real** guidance embeddings | 565 ms/pass (1.8 Hz) | **237 ms/pass (4.2 Hz)**, worst-case drift 1.7 cm | `models/quantize_int8.py`, `tools/fetch_embs.py` |
| Model artifact size | 245 MB fp32 ONNX | 64 MB INT8 ONNX (3.8x smaller) | `models/quantize_int8.py` |
| Thread-contention fix (PyTorch reference client) | 653 ms/pass under load | 87 ms/pass (**~7.5x**) | `torch.set_num_threads(1)` in `edge/edge_client.py` |
| Thread-contention fix (robot) | intra-op spin-waiting starves camera/network threads | spinning disabled, stable 4.2 Hz | session config in `edge/mp2_client.py` |
| CPU frequency pinning on the board | ondemand governor ramps late | `performance` governor, consistent latency | setup step, documented below |
| Ops hardening for ephemeral cloud pods | manual SSH surgery after every pod restart | one-command recovery + idempotent server launch | `tools/recover_pod.py`, `tools/start_server.py` |

Two of these were non-obvious and are worth internalizing before touching the
code - see [Performance notes](#performance-notes-hard-won-do-not-regress).

## Measured results (this exact stack)

| Metric | Value |
|---|---|
| Server VLA inference (8.3B, RTX 4090) | 139–156 ms |
| Guidance staleness (internet round trip, mean) | 0.9–1.5 s |
| Edge Adapter on STM32MP2, fp32 ONNX | 565 ms/pass (1.8 Hz) |
| Edge Adapter on STM32MP2, INT8 ONNX | **237 ms/pass (4.2 Hz)** |
| Sensor-to-action (frame age + edge pass) | ~400 ms avg, <600 ms worst |
| INT8 vs fp32 trajectory drift (worst waypoint) | 0.17 units ≈ 1.7 cm |
| Edge Adapter on laptop (PyTorch, reference) | 95 ms/pass (10.5 Hz) |

For comparison: a conventional non-async design that sends every frame to the
server and waits for the full VLA would be bounded by VLA inference + round
trip (**>2 s** per action over the internet). The async split brings
sensor-to-action to **<450 ms** on this hardware, because the robot never
waits on the network - it fuses whatever guidance last arrived.

Validation: goal left/right/straight produce distinct, physically sensible
trajectories; a mirrored scene flips the behavior (proves the current image is
being used); an empty synthetic scene with a straight goal yields a dead
straight trajectory (<2 mm lateral drift over ~1 m).

## Repo layout

```
server/   server.py                    VLA server: loads OmniVLA, serves guidance over TCP
          setup_env.sh                 one-time environment setup on the GPU machine
          download_full_checkpoint.py  pull the full 17 GB OmniVLA checkpoint (run on GPU machine)
edge/     mp2_client.py                robot client (STM32MP2): onnxruntime + numpy only
          edge_client.py               laptop client (PyTorch): reference implementation
models/   download_checkpoint.py       pull Edge Adapter weights (153 MB) from HuggingFace
          export_onnx.py               PyTorch checkpoint -> ONNX (with parity check)
          quantize_int8.py             ONNX fp32 -> INT8 (calibrated, with drift check)
tools/    start_server.py              deploy + launch server on the GPU machine, wait for LISTENING
          watch_server.py              live-stream the server log (IPs masked)
          recover_pod.py               re-enable SSH + refresh address after a pod restart
          fetch_embs.py                grab real guidance embeddings (quantization calibration)
          test_edge_adapter.py         standalone edge-model smoke test + benchmark
          make_open_scene.py           generate the synthetic empty-scene test images
          make_mirror.py               generate mirrored test images
          plot_run.py                  render trajectory/latency figure from a run's JSON log
assets/   past.png, cur.png            office test scene (from the AsyncVLA repo)
          *_mirror.png                 mirrored office scene
          open_*.png                   synthetic empty scene
```

Model binaries and run outputs are intentionally not committed (see
`.gitignore`) - they are fully reproducible with the `models/` pipeline in
Step 3 below. Every script reads the server address from
`~/edgevla/pod_address.txt` (one line: `<ip> <ssh_port> <app_port>`), so
nothing sensitive lives in the code.

## Replicating the demo, end to end

Three machines. The dev-machine scripts assume a working directory at
`~/edgevla` (they use `~` expansion, so any username works).

### Step 1 - GPU server (tested: RunPod RTX 4090, 50 GB disk)

1. Create the pod with **TCP ports 22 and 15001 exposed** (the RunPod SSH
   proxy cannot forward ports; direct TCP is required).
2. On the pod, clone [AsyncVLA](https://github.com/NHirose/AsyncVLA) and
   [MBRA](https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA)
   into `/workspace/asyncvla/`, then run `server/setup_env.sh`. The
   `protobuf==4.25.9` / `tensorflow-metadata==1.16.1` pins in that script are
   load-bearing - newer protobuf breaks the model's imports.
3. Download the full checkpoint (~17 GB, detached because it takes a while):

   ```bash
   cd /workspace/asyncvla
   nohup env/bin/python download_full_checkpoint.py > download.log 2>&1 &
   ```

4. You never launch the server by hand - `tools/start_server.py` (Step 2)
   uploads `server/server.py` and launches it for you.

### Step 2 - Dev laptop (tested: WSL2 Ubuntu, Python 3.10)

```bash
mkdir -p ~/edgevla && cd ~/edgevla
git clone https://github.com/NHirose/AsyncVLA
git clone https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA MBRA
pip install -r requirements-laptop.txt
cp <repo>/edge/edge_client.py <repo>/models/*.py <repo>/tools/*.py <repo>/assets/*.png .
python download_checkpoint.py          # Edge Adapter weights -> checkpoints/
echo "<pod_ip> <ssh_port> <app_port>" > pod_address.txt
python start_server.py                 # idempotent: uploads, launches, waits for LISTENING
```

The AsyncVLA/MBRA clones are needed because `edge_client.py` and
`export_onnx.py` import the Edge Adapter class (`small_head.py`) and its
vision encoder from them. If the pod ever restarts (new IP/ports, sshd dead),
set the proxy ID at the top of `recover_pod.py` and run it - it repairs SSH
and rewrites `pod_address.txt` in one shot.

### Step 3 - Build the robot's model (run on the dev laptop)

```bash
python export_onnx.py     # checkpoint -> edge_adapter.onnx  (245 MB, parity-checked vs PyTorch)
python fetch_embs.py      # pulls REAL guidance embeddings from the live server -> emb_*.npy
python quantize_int8.py   # -> edge_adapter_int8qdq.onnx (64 MB, drift-checked vs fp32)
```

`quantize_int8.py` prints a per-scene trajectory-drift report; worst waypoint
should be ~0.17 normalized units (≈1.7 cm). If you retrain or change the
adapter, rerun all three and check that report before deploying.

### Step 4 - Robot (tested: MYD-LD25X STM32MP2, 2 GB RAM, Python 3.12)

```bash
# on the board - note: the rootfs may be full; /usr/local has space
mkdir -p /usr/local/edgevla && cd /usr/local/edgevla
python3 -m venv --without-pip env       # ensurepip fails on full rootfs
TMPDIR=/tmp pip install --python env/bin/python -r requirements-mp2.txt
echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
```

Copy over `edge/mp2_client.py`, `edge_adapter_int8qdq.onnx`, the `assets/`
images, and `pod_address.txt` (e.g. `scp` from the laptop). No PyTorch, no
checkpoint - the board only ever sees the 64 MB ONNX file.

### Step 5 - Run it

```bash
# dev laptop: watch the server live (IPs are masked in the stream)
python watch_server.py

# robot: three test cases with known-good expected behavior
python mp2_client.py 30 --goal 100 0   --img-past open_past.png --img-cur open_cur.png --tag open_straight
python mp2_client.py 30 --goal 10 -100 --tag office_right
python mp2_client.py 30 --goal 10 100  --img-past past_mirror.png --img-cur cur_mirror.png --tag mirror_left
```

`--goal X_FWD Y_LEFT` is a relative goal pose in normalized units (~0.1 m
each); positive Y is left. Expected results: the empty scene + straight goal
gives `vel=(0.30 m/s, ~0.00 rad/s)`; the office scene bends the trajectory
around the furniture toward the goal side; the mirrored scene flips it. Each
run prints live control lines (edge latency, guidance staleness, velocity
commands) and saves full stats to `<tag>.json` - copy that back to the laptop
and render it with `python plot_run.py <tag>.json`.

To sanity-check the model alone (no server needed):
`python test_edge_adapter.py` on the laptop, or
`python mp2_client.py --bench` on the board for a pure-inference benchmark.

## Reading the code

Suggested order:

1. **`server/server.py`** (~120 lines) - loads OmniVLA once, then a simple
   loop: receive `(jpeg, goal)` over TCP, run the VLA, send back the 8x1024
   guidance embedding. The protocol is length-prefixed pickle; deliberately
   boring.
2. **`edge/mp2_client.py`** - the heart of the system. Three threads: a
   camera thread filling a latest-frame buffer, a network thread shipping
   frames to the server and storing whatever guidance comes back (with its
   timestamp), and the main control loop running the Edge Adapter at full
   rate on the *current* frame plus the *latest available* (stale) guidance.
   This asymmetry - control never blocks on the network - is Algorithm 1 of
   the paper, and every design choice in the file serves it.
3. **`edge/edge_client.py`** - same structure in PyTorch on the laptop. Use
   it to debug logic without quantization in the picture.
4. **`models/export_onnx.py` and `models/quantize_int8.py`** - the port
   pipeline. Both scripts verify their own output (ONNX-vs-PyTorch parity,
   INT8-vs-fp32 drift); keep it that way.
5. **`tools/`** - operational scripts; each has a docstring explaining when
   you'd reach for it.

## Performance notes (hard-won, do not regress)

- **PyTorch on the edge:** `torch.set_num_threads(1)` in `edge_client.py` is
  load-bearing. Default thread-per-core spin-waiting degrades the forward pass
  ~7.5x (653 ms vs 87 ms measured) once any background thread (network,
  camera) needs CPU.
- **onnxruntime on the edge:** intra-op spinning is disabled in
  `mp2_client.py` (`session.intra_op.allow_spinning = 0`) for the same
  reason. On a 2-core board, a spinning worker starves the camera and network
  threads.
- **INT8 calibration:** `quantize_int8.py` calibrates with *real* guidance
  embeddings (via `tools/fetch_embs.py`). Their std is ~1.8; synthetic N(0,1)
  calibration data would mis-scale the transformer activations by ~2x.
- **Dynamic quantization does not work here:** it emits `ConvInteger` ops that
  onnxruntime's CPU provider cannot execute. Static QDQ quantization is
  required (that's why `quantize_int8.py` needs calibration data at all).

## Attribution

Model weights, the base VLA, and the Edge Adapter architecture are from
Hirose et al., *AsyncVLA* ([arXiv:2602.13476](https://arxiv.org/abs/2602.13476))
and its [released checkpoint](https://huggingface.co/NHirose/AsyncVLA_release).
The office test images (`assets/past.png`, `assets/cur.png`) are from the
AsyncVLA repository. This repo contains the deployment/serving code, the
ONNX/INT8 port to the STM32MP2, and the validation tooling built around them.
