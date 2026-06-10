# Single entry point to run every module's _selftest() (CI-friendly).
#
#   python run_selftests.py
#
# Each listed module exposes a `_selftest()` that builds a tiny synthetic case
# and asserts the expected result. Modules whose optional deps are missing
# (torch / diffusion_net) are reported as SKIP, not FAIL, so the pure-numpy data
# path can be verified anywhere. Exit code is non-zero iff something FAILED.

import importlib
import sys
import traceback

# pure-numpy modules first (always runnable), then ones needing torch.
MODULES = ["json_dataset", "cp_targets", "metrics", "cp_regressor"]


def run_one(name):
    try:
        mod = importlib.import_module(name)
    except ImportError as exc:
        return "SKIP", f"import failed ({exc})"
    fn = getattr(mod, "_selftest", None)
    if fn is None:
        return "SKIP", "no _selftest()"
    try:
        fn()
        return "PASS", ""
    except ImportError as exc:
        return "SKIP", f"missing optional dep ({exc})"
    except Exception as exc:                            # noqa: BLE001
        traceback.print_exc()
        return "FAIL", str(exc)


def main(argv=None):
    names = list(argv) if argv else MODULES
    results = []
    for name in names:
        status, detail = run_one(name)
        results.append((name, status, detail))
        print(f"[{status:4}] {name}" + (f"  -- {detail}" if detail else ""))
    failed = [n for n, s, _ in results if s == "FAIL"]
    print("\n%d pass, %d skip, %d fail"
          % (sum(s == "PASS" for _, s, _ in results),
             sum(s == "SKIP" for _, s, _ in results),
             len(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
