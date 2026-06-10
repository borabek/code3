#!/usr/bin/env bash
set -euo pipefail

echo "==> [1/6] Creating venv ~/cpenv"
python3 -m venv ~/cpenv
source ~/cpenv/bin/activate
pip install --upgrade pip

echo "==> [2/6] Installing numpy/scipy/ijson"
pip install numpy scipy ijson

echo "==> [3/6] Installing CPU torch"
pip install torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -5

echo "==> [4/6] Installing native geometry libs (these segfault on Windows)"
pip install robust_laplacian potpourri3d

echo "==> [5/6] Installing diffusion-net"
pip install git+https://github.com/nmwsharp/diffusion-net.git

echo "==> [6/6] IMPORT CHECK (the proof the segfault is gone)"
python3 -c "import robust_laplacian, potpourri3d, torch, diffusion_net; print('IMPORT OK | torch', torch.__version__)"

echo ""
echo "==> SETUP DONE. To run the real DiffusionNet backbone on a sample folder:"
echo "    source ~/cpenv/bin/activate"
echo "    cd /mnt/c/Users/Administrator/repos/code/code"
echo "    python3 -c \"import sys; sys.path.insert(0,'.'); import train_cp; train_cp.train_and_eval('/mnt/c/Users/Administrator/abb_samples', backbone='diffusionnet', epochs=120, op_cache_dir='/mnt/c/Users/Administrator/op_cache')\""
