# Thin convenience wrapper: run the full DiffusionNet training on the real ABB
# corpus with the known-good flags baked in, so nobody has to retype the long
# command (or fat-finger a /mnt/c path). All it does is assemble argv and hand
# off to train_cp.main -- no logic lives here, so there is nothing to duplicate.
#
# Usage:
#   python run_full.py                      # uses the defaults below
#   python run_full.py --epochs 50          # any train_cp.py flag overrides/extends
#   python run_full.py /other/corpus ...    # positional corpus overrides the default
#
# Defaults target the WSL layout (corpus/cache/outputs under Desktop, NOT /tmp,
# which WSL wipes when the distro restarts).

import sys
import train_cp

# (flag, value) defaults; a flag already present in argv is NOT overridden.
DEFAULT_CORPUS = "/mnt/c/Users/DE00024082/Desktop/JSON"
DEFAULTS = [
    ("--backbone", "diffusionnet"),
    ("--device", "cuda"),
    ("--epochs", "200"),
    ("--op-cache-dir", "/mnt/c/Users/DE00024082/Desktop/op_cache"),
    ("--ckpt", "/mnt/c/Users/DE00024082/Desktop/cp_model.pt"),
    ("--ckpt-every", "10"),
    ("--out", "/mnt/c/Users/DE00024082/Desktop/results_full.json"),
]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # positional corpus: if the user gave none (argv empty or starts with a flag),
    # prepend the default so train_cp's required `source` arg is satisfied.
    if not argv or argv[0].startswith("-"):
        argv = [DEFAULT_CORPUS] + argv
    for flag, value in DEFAULTS:
        if flag not in argv:
            argv += [flag, value]
    train_cp.main(argv)


if __name__ == "__main__":
    main()
