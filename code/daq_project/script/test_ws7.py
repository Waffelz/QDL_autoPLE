#!/usr/bin/env python3
"""
WS7 smoke test (run on the instrument PC).

What it does:
- Initializes WS7 via matisse_controller.shamrock_ple.ple.PLE.setup_ws7()
- Reads wavelength repeatedly
- Prints basic stats + a few sample lines

Run:
  python scripts/test_ws7.py
  python scripts/test_ws7.py --n 200 --dt 0.02
"""

import time
import argparse
import statistics

import matisse_controller.shamrock_ple.ple as ple_mod


def read_wl(ws7, use_channel2: bool = False):
    """
    Returns a float wavelength in nm or None if invalid.
    use_channel2=True tries GetWavelength2 (your driver uses both).
    """
    try:
        if use_channel2 and hasattr(ws7.lib, "GetWavelength2"):
            wl = float(ws7.lib.GetWavelength2(0.0))
        else:
            wl = float(ws7.lib.GetWavelength(0.0))
        if wl > 0:
            return wl
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="number of samples")
    ap.add_argument("--dt", type=float, default=0.02, help="sleep between samples (s)")
    ap.add_argument("--ch2", action="store_true", help="use GetWavelength2 instead of GetWavelength")
    args = ap.parse_args()

    # Initialize WS7 the same way run_scan.py does
    ple = ple_mod.PLE()
    ple.setup_ws7()
    ws7 = ple_mod.ws7

    if ws7 is None:
        raise RuntimeError("WS7 setup failed: ple_mod.ws7 is None")

    print("WS7 initialized. Sampling...")
    samples = []
    t0 = time.monotonic()

    for i in range(args.n):
        wl = read_wl(ws7, use_channel2=args.ch2)
        t = time.monotonic() - t0
        if wl is not None:
            samples.append(wl)

        # print a few live lines
        if i < 5 or (i + 1) % 50 == 0:
            print(f"[{i+1:4d}/{args.n}] t={t:7.3f}s  wl={wl}")

        time.sleep(args.dt)

    # Cleanup (safe)
    try:
        ws7.stop_acquisition()
    except Exception:
        pass
    try:
        ple.clean_up_globals()
    except Exception:
        pass

    if not samples:
        print("\nNo valid WS7 wavelength samples (>0) were received.")
        print("Common causes: WS7 software not running / wrong channel / no light / DLL not found.")
        return

    mean = statistics.mean(samples)
    stdev = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    print("\n--- WS7 Results ---")
    print(f"Valid samples: {len(samples)}/{args.n}")
    print(f"Mean (nm):     {mean:.9f}")
    print(f"Std (nm):      {stdev:.9f}")
    print(f"Min (nm):      {min(samples):.9f}")
    print(f"Max (nm):      {max(samples):.9f}")
    print(f"Span (pm):     {(max(samples)-min(samples))*1e3:.3f} pm")  # nm -> pm
    print("-------------------")


if __name__ == "__main__":
    main()
