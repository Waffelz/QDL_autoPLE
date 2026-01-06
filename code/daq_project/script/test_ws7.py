#!/usr/bin/env python3
"""
WS7 smoke test (direct driver) with air/vacuum selection.

Run (from code/daq_project):
  python script/test_ws7.py --n 50 --dt 0.05 --medium vac
  python script/test_ws7.py --n 50 --dt 0.05 --medium air
"""

import time
import argparse
import statistics
import sys
from pathlib import Path
from ctypes import c_long, byref


# --- Make sure the folder containing "functions26" is on sys.path ---
HERE = Path(__file__).resolve()
root = None
for p in [HERE.parent] + list(HERE.parents):
    if (p / "functions26").exists():
        root = p
        break
if root is None:
    raise RuntimeError("Could not find a 'functions26' folder in any parent directory.")
sys.path.insert(0, str(root))

from functions26.instruments.ws7 import WS7  # noqa: E402


# HighFinesse constants (fallback)
cReturnWavelengthVac = 0
cReturnWavelengthAir = 1


def vacuum_to_air_nm(lambda_vac_nm: float) -> float:
    """
    Convert vacuum wavelength (nm) -> standard air wavelength (nm)
    using a common Ciddor/Edlén-style formula (good enough for this purpose).
    """
    lam_um = lambda_vac_nm * 1e-3
    inv_lam2 = 1.0 / (lam_um * lam_um)

    b1 = 5.792105e-2
    b2 = 1.67917e-3
    c1 = 238.0185
    c2 = 57.362

    n_minus_1 = b1 / (c1 - inv_lam2) + b2 / (c2 - inv_lam2)
    n = 1.0 + n_minus_1
    return lambda_vac_nm / n


def get_result_mode(lib):
    """Return mode int if supported, else None."""
    if not hasattr(lib, "GetResultMode"):
        return None
    m = c_long()
    try:
        lib.GetResultMode(byref(m))
        return int(m.value)
    except TypeError:
        # Some DLL variants have different signatures
        return None


def set_result_mode(lib, mode: int) -> bool:
    """Try to set mode; return True if it *seems* to work."""
    if not hasattr(lib, "SetResultMode"):
        return False
    try:
        lib.SetResultMode(c_long(int(mode)))
        return True
    except TypeError:
        try:
            lib.SetResultMode(int(mode))
            return True
        except Exception:
            return False
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--ch2", action="store_true")
    ap.add_argument("--medium", choices=["air", "vac"], default="air",
                    help="Display wavelength in air (default) or vacuum.")
    args = ap.parse_args()

    ws7 = WS7()
    print("WS7 initialized.")
    print("Has SetResultMode:", hasattr(ws7.lib, "SetResultMode"))
    print("Has GetResultMode:", hasattr(ws7.lib, "GetResultMode"))

    # Try to set the WS7's native output mode (best if supported)
    target_mode = cReturnWavelengthAir if args.medium == "air" else cReturnWavelengthVac
    did_set = set_result_mode(ws7.lib, target_mode)
    mode_after = get_result_mode(ws7.lib)

    print("SetResultMode attempted:", did_set)
    print("ResultMode after set:", mode_after, "(0=vac, 1=air, None=unknown)")

    samples = []
    t0 = time.monotonic()
    print(f"Sampling... showing '{args.medium}'")

    for i in range(args.n):
        try:
            if args.ch2 and hasattr(ws7.lib, "GetWavelength2"):
                wl_raw = float(ws7.lib.GetWavelength2(0.0))
            else:
                wl_raw = float(ws7.lib.GetWavelength(0.0))
        except Exception:
            wl_raw = -1.0

        if wl_raw > 0:
            # If WS7 mode is known and matches, trust wl_raw.
            # Otherwise, assume wl_raw is vacuum and convert when showing air.
            if args.medium == "vac":
                wl_show = wl_raw
            else:
                if mode_after == cReturnWavelengthAir:
                    wl_show = wl_raw
                else:
                    wl_show = vacuum_to_air_nm(wl_raw)

            samples.append(wl_show)
        else:
            wl_show = None

        if i < 5 or (i + 1) % 50 == 0:
            elapsed = time.monotonic() - t0
            if wl_raw > 0 and args.medium == "air" and mode_after != cReturnWavelengthAir:
                print(f"[{i+1:4d}/{args.n}] t={elapsed:7.3f}s  wl_air={wl_show:.9f}  (assumed vac={wl_raw:.9f})")
            else:
                print(f"[{i+1:4d}/{args.n}] t={elapsed:7.3f}s  wl_{args.medium}={wl_show}")

        time.sleep(args.dt)

    if not samples:
        print("\nNo valid wavelength samples (>0).")
        return

    mean = statistics.mean(samples)
    stdev = statistics.pstdev(samples) if len(samples) > 1 else 0.0

    print(f"\n--- WS7 Results ({args.medium}) ---")
    print(f"Valid samples: {len(samples)}/{args.n}")
    print(f"Mean (nm):     {mean:.9f}")
    print(f"Std (nm):      {stdev:.9f}")
    print(f"Min (nm):      {min(samples):.9f}")
    print(f"Max (nm):      {max(samples):.9f}")
    print(f"Span (pm):     {(max(samples)-min(samples))*1e3:.3f} pm")
    print("---------------------------")


if __name__ == "__main__":
    main()
