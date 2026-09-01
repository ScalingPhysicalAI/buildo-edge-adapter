"""Download the full AsyncVLA release checkpoint (~17 GB) onto the GPU machine.

Run ON the GPU machine (not the laptop), inside the server env, ideally
detached since it takes a while:

    cd /workspace/asyncvla
    nohup env/bin/python download_full_checkpoint.py > download.log 2>&1 &
    tail -f download.log

The server (server.py / inference code) expects the checkpoint at
AsyncVLA/AsyncVLA_release next to the repo's inference scripts.
"""
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="NHirose/AsyncVLA_release",
    local_dir="/workspace/asyncvla/AsyncVLA/AsyncVLA_release",
    max_workers=8,
)
print("DOWNLOAD_DONE", path)
