#!/usr/bin/env python3
"""
WS7 smoke test (direct driver, no PLE dependency).
Run on the instrument PC where wlmData.dll is available.

Run:
  python scripts/test_ws7.py --n 200 --dt 0.02
  python scripts/test_ws7.py --n 200 --dt 0.02 --ch2
"""

import time
import argparse
import statistics

from functions26.instruments.ws7 import WS7  # <-- your driver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="number of samples")
    ap.add_argument("--dt", type=float, default=0.02, help="sleep between samples (s)")
    ap.add_argument("--ch2", action="store_true", help="read GetWavelength2 instead of GetWavelength")
    args = ap.parse_args()

    ws7 = WS7()  # loads wlmData.dll and exposes ws7.lib

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

        wl_valid = wl > 0
        if wl_valid:
            samples.append(wl)

        if i < 5 or (i + 1) % 50 == 0:
            print(f"[{i+1:4d}/{args.n}] t={time.monotonic()-t0:7.3f}s  wl={wl if wl_valid else None}")

        time.sleep(args.dt)

    if not samples:
        print("\nNo valid wavelength samples (>0).")
        print("Common causes: WS7 app/service not running, no signal, wrong channel, DLL not found on PATH.")
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
