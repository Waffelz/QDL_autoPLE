#!/usr/bin/env python3
"""
WS7 smoke test (direct driver).

Run:
  python scripts/test_ws7.py --n 200 --dt 0.02
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

from functions26.instruments.ws7 import WS7  # now this should work


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--ch2", action="store_true")
    args = ap.parse_args()

    ws7 = WS7()

    samples = []
    t0 = time.monotonic()
    print("WS7 initialized. Sampling...")

    for i in range(args.n):
        try:
            if args.ch2 and hasattr(ws7.lib, "GetWavelength2"):
                wl = float(ws7.lib.GetWavelength2(0.0))
            else:
                wl = float(ws7.lib.GetWavelength(0.0))
        except Exception:
            wl = -1.0

        if wl > 0:
            samples.append(wl)

        if i < 5 or (i + 1) % 50 == 0:
            print(f"[{i+1:4d}/{args.n}] t={time.monotonic()-t0:7.3f}s  wl={wl if wl > 0 else None}")

        time.sleep(args.dt)

    if not samples:
        print("\nNo valid wavelength samples (>0).")
        return

    mean = statistics.mean(samples)
    stdev = statistics.pstdev(samples) if len(samples) > 1 else 0.0

    print("\n--- WS7 Results ---")
    print(f"Valid samples: {len(samples)}/{args.n}")
    print(f"Mean (nm):     {mean:.9f}")
    print(f"Std (nm):      {stdev:.9f}")
    print(f"Min (nm):      {min(samples):.9f}")
    print(f"Max (nm):      {max(samples):.9f}")
    print(f"Span (pm):     {(max(samples)-min(samples))*1e3:.3f} pm")
    print("-------------------")


if __name__ == "__main__":
    main()
