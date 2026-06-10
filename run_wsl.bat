@echo off
REM One-click: set up WSL (if needed) and train the DiffusionNet backbone.
REM Edit the paths below if your layout differs.
setlocal

set REPO_WSL=/mnt/c/Users/DE00024082/Desktop/code
set CORPUS_WSL=/mnt/c/Users/DE00024082/Desktop/JSON
set CACHE_WSL=/mnt/c/Users/DE00024082/Desktop/op_cache
set CKPT_WSL=/mnt/c/Users/DE00024082/Desktop/cp_model.pt
set OUT_WSL=/mnt/c/Users/DE00024082/Desktop/results_full.json
set EPOCHS=200

echo === Checking WSL ===
wsl -l -v
if errorlevel 1 (
  echo No WSL distro. Installing Ubuntu ^(no reboot if the WSL feature is enabled^)...
  wsl --install -d Ubuntu --no-launch
  echo If this is the first install, reboot and re-run this script.
)

echo === Setting up the venv inside WSL (idempotent) ===
wsl -d Ubuntu -e bash -lc "test -d ~/cpenv || bash %REPO_WSL%/setup_wsl.sh"
if errorlevel 1 (
  echo Setup failed. See output above.
  pause
  exit /b 1
)

echo === Training DiffusionNet on the corpus (GPU; big meshes auto-fall-back to CPU) ===
wsl -d Ubuntu -e bash -lc "source ~/cpenv/bin/activate && cd %REPO_WSL% && python3 -u train_cp.py %CORPUS_WSL% --backbone diffusionnet --device cuda --op-cache-dir %CACHE_WSL% --epochs %EPOCHS% --eval-every 10 --patience 6 --min-votes 2 --lr-decay-every 80 --ckpt %CKPT_WSL% --out %OUT_WSL%"

echo.
echo === Done. Best model: %CKPT_WSL%   Metrics: %OUT_WSL% ===
pause
