#!/usr/bin/env bash
# Set up the DiffusionNet backbone inside WSL/Ubuntu, where robust_laplacian and
# potpourri3d import cleanly (they SEGFAULT on native Windows).
#
# Run from Windows:
#   wsl --install -d Ubuntu          # once, if WSL has no distro (no reboot if the
#                                    # WSL feature is already enabled)
#   wsl -d Ubuntu -e bash -lc 'bash /mnt/c/Users/DE00024082/Desktop/code/setup_wsl.sh'
#
# Notes baked in from a real run on this machine:
#  * fresh Ubuntu lacks pip/venv -> apt installs them first
#  * a corporate TLS-inspecting proxy breaks SSL to some hosts -> --trusted-host
#  * diffusion-net has no setup.py -> clone + add src/ to the venv via a .pth file
set -euo pipefail

PTORCH_TRUST="--trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org"

echo "==> [1/7] System packages (venv/pip/build tools)"
sudo apt-get update -qq
sudo apt-get install -y python3-venv python3-pip build-essential git

echo "==> [2/7] Creating venv ~/cpenv"
python3 -m venv ~/cpenv
source ~/cpenv/bin/activate
pip install --upgrade pip

echo "==> [3/7] Installing numpy/scipy/ijson (PyPI works through the proxy)"
pip install numpy scipy ijson scikit-learn

echo "==> [4/7] Installing torch"
# CPU build (default). For the GPU (CUDA 12.x) build instead, swap the index to
# cu128 and ALSO add: --trusted-host pypi.nvidia.com  (it pulls cuda-toolkit there)
pip install torch --index-url https://download.pytorch.org/whl/cpu $PTORCH_TRUST 2>&1 | tail -3
# GPU alternative:
#   pip install torch --index-url https://download.pytorch.org/whl/cu128 \
#       $PTORCH_TRUST --trusted-host pypi.nvidia.com 2>&1 | tail -3

echo "==> [5/7] Installing native geometry libs (these segfault on Windows)"
pip install robust_laplacian potpourri3d

echo "==> [6/7] Installing diffusion-net (no setup.py -> clone + .pth)"
# git through the proxy needs verification off for this clone
git -c http.sslVerify=false clone --depth 1 \
    https://github.com/nmwsharp/diffusion-net.git ~/diffusion-net 2>&1 | tail -1
SP=$(python3 -c "import site; print(site.getsitepackages()[0])")
echo "$HOME/diffusion-net/src" > "$SP/diffusion_net.pth"

echo "==> [7/7] IMPORT CHECK (the proof the segfault is gone)"
python3 -c "import robust_laplacian, potpourri3d, torch, diffusion_net; \
print('IMPORT OK | torch', torch.__version__, '| cuda', torch.cuda.is_available())"

echo ""
echo "==> SETUP DONE. Train the real DiffusionNet backbone on the corpus:"
echo "    source ~/cpenv/bin/activate"
echo "    cd /mnt/c/Users/DE00024082/Desktop/code"
echo "    python3 train_cp.py /mnt/c/Users/DE00024082/Desktop/JSON \\"
echo "        --backbone diffusionnet --device cuda \\"
echo "        --op-cache-dir /mnt/c/Users/DE00024082/Desktop/op_cache \\"
echo "        --epochs 200 --eval-every 10 --patience 6 \\"
echo "        --ckpt /mnt/c/Users/DE00024082/Desktop/cp_model.pt \\"
echo "        --out /mnt/c/Users/DE00024082/Desktop/results_full.json"
echo "    # (or simply: python3 run_full.py --eval-every 10 --patience 6)"
