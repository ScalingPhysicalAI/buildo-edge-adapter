set -e
cd /workspace
mkdir -p asyncvla
cd asyncvla

# Repos
if [ ! -d AsyncVLA ]; then
  git clone --depth 1 https://github.com/NHirose/AsyncVLA.git
fi
if [ ! -d Learning-to-Drive-Anywhere-with-MBRA ]; then
  git clone --depth 1 https://github.com/NHirose/Learning-to-Drive-Anywhere-with-MBRA.git
fi

# Python 3.10 venv via uv
if [ ! -f /root/.local/bin/uv ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH=/root/.local/bin:$PATH
if [ ! -d env ]; then
  uv venv --python 3.10 env
fi

# Pinned deps per SETUP.md (torch 2.2.0 from PyPI ships CUDA 12.1 for x86_64)
uv pip install --python /workspace/asyncvla/env/bin/python \
  numpy==1.26.4 torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
uv pip install --python /workspace/asyncvla/env/bin/python -e /workspace/asyncvla/AsyncVLA
uv pip install --python /workspace/asyncvla/env/bin/python \
  efficientnet_pytorch utm packaging ninja

# Prebuilt flash-attn 2.5.5 wheel (avoids a long compile)
uv pip install --python /workspace/asyncvla/env/bin/python \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

/workspace/asyncvla/env/bin/python -c "import torch, flash_attn; print('torch', torch.__version__, 'cuda_ok', torch.cuda.is_available()); print('flash_attn', flash_attn.__version__)"
echo SETUP_ENV_DONE
exit
