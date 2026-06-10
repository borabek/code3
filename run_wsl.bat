@echo off
setlocal

REM ---- config (edit if your paths differ) ----
set REPO_WSL=/mnt/c/Users/Administrator/repos/code/code
set SAMPLES_WSL=/mnt/c/Users/Administrator/abb_samples
set CACHE_WSL=/mnt/c/Users/Administrator/op_cache
set EPOCHS=120

echo === Checking WSL ===
wsl -l -v
if errorlevel 1 (
  echo WSL not installed. Run:  wsl --install -d Ubuntu   then reboot and re-run this.
  pause
  exit /b 1
)

echo === Running setup_wsl.sh inside WSL ===
wsl -e bash -lc "cd %REPO_WSL% && bash setup_wsl.sh"
if errorlevel 1 (
  echo Setup failed. See output above.
  pause
  exit /b 1
)

echo === Running DiffusionNet backbone on samples ===
wsl -e bash -lc "source ~/cpenv/bin/activate && cd %REPO_WSL% && python3 -c \"import sys; sys.path.insert(0,'.'); import train_cp; r=train_cp.train_and_eval('%SAMPLES_WSL%', backbone='diffusionnet', epochs=%EPOCHS%, op_cache_dir='%CACHE_WSL%'); print(r)\""

pause
