#!/usr/bin/env python3
"""
WS7 smoke test with vacuum<->air mode switching (via SetResultMode).

Run from project root:
  python script/test_ws7.py --n 50 --dt 0.05 --medium vac
  python script/test_ws7.py --n 50 --dt 0.05 --medium air
"""

import time
import argparse
import statistics
import sys
from pathlib import Path

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

# HighFinesse WLM result modes (from your snippet)
WLM_RETURN_WAVELENGTH_VAC = 0
WLM_RETURN_WAVELENGTH_AIR = 1


def read_wavelength_nm(ws7: WS7, ch2: bool) -> float:
    """Read wavelength using the driver. Returns -1.0 if invalid."""
    try:
        if ch2 and hasattr(ws7.lib, "GetWavelength2"):
            wl = float(ws7.lib.GetWavelength2(0.0))
        else:
            wl = float(ws7.lib.GetWavelength(0.0))
        return wl if wl > 0 else -1.0
    except Exception:
        return -1.0


def try_set_result_mode(ws7: WS7, mode: int) -> bool:
    """
    Try to call SetResultMode(mode). Return True if it didn't throw.
    (Some installs return an int status; we don't rely on it.)
    """
    if not hasattr(ws7.lib, "SetResultMode"):
        return False
    try:
        ws7.lib.SetResultMode(int(mode))
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--ch2", action="store_true")
    ap.add_argument("--medium", choices=["air", "vac"], default="air",
                    help="Display wavelength in air or vacuum.")
    args = ap.parse_args()

    ws7 = WS7()
    print("WS7 initialized.")
    print("Has SetResultMode:", hasattr(ws7.lib, "SetResultMode"))

    # --- One-time sanity check: VAC then AIR (if SetResultMode exists) ---
    wl_vac0 = None
    wl_air0 = None
    if hasattr(ws7.lib, "SetResultMode"):
        ok_v = try_set_result_mode(ws7, WLM_RETURN_WAVELENGTH_VAC)
        time.sleep(0.05)
        wl_vac0 = read_wavelength_nm(ws7, args.ch2)

        ok_a = try_set_result_mode(ws7, WLM_RETURN_WAVELENGTH_AIR)
        time.sleep(0.05)
        wl_air0 = read_wavelength_nm(ws7, args.ch2)

        print(f"SetResultMode(VAC) ok={ok_v}, read={wl_vac0 if wl_vac0 > 0 else None}")
        print(f"SetResultMode(AIR) ok={ok_a}, read={wl_air0 if wl_air0 > 0 else None}")
        if (wl_vac0 is not None and wl_air0 is not None and wl_vac0 > 0 and wl_air0 > 0):
            print(f"VAC-AIR delta ~ {wl_vac0 - wl_air0:.6f} nm (expected ~0.2 nm near 740 nm)")

    # --- Set requested mode for the main acquisition loop ---
    desired_mode = WLM_RETURN_WAVELENGTH_AIR if args.medium == "air" else WLM_RETURN_WAVELENGTH_VAC
    if hasattr(ws7.lib, "SetResultMode"):
        try_set_result_mode(ws7, desired_mode)
        time.sleep(0.05)

    samples = []
    t0 = time.monotonic()
    print(f"\nSampling ({args.medium})...")

    for i in range(args.n):
        wl = read_wavelength_nm(ws7, args.ch2)
        wl_show = wl if wl > 0 else None
        if wl_show is not None:
            samples.append(wl_show)

        if i < 5 or (i + 1) % 50 == 0:
            print(f"[{i+1:4d}/{args.n}] t={time.monotonic()-t0:7.3f}s  wl_{args.medium}={wl_show}")

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
