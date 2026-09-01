"""Download the Edge Adapter checkpoint (153 MB) from the AsyncVLA release.

This is the only weight file the edge side ever needs; the full 17 GB
checkpoint (server/download_full_checkpoint.py) stays on the GPU machine.
"""
import os

from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="NHirose/AsyncVLA_release",
    filename="shead--750000_checkpoint.pt",
    local_dir=os.path.expanduser("~/edgevla/checkpoints"),
)
print("Downloaded to:", path)
