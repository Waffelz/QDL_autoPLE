#!/usr/bin/env python3
import time
import argparse
import statistics

import matisse_controller.shamrock_ple.ple as ple_mod


def read_wl(ws7, use_channel2: bool = False):
    try:
        if use_channel2 and hasattr(ws7.lib, "GetWavelength2"):
            wl = float(ws7.lib.GetWavelength2(0.0))
        else:
            wl = float(ws7.lib.GetWavelength(0.0))
        return wl if wl > 0 else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--ch2", action="store_true")
    args = ap.parse_args()

    # IMPORTANT: do not instantiate PLE (your PLE __init__ requires args)
    ple_mod.PLE.setup_ws7()
    ws7 = ple_mod.ws7

    if ws7 is None:
        raise RuntimeError("WS7 setup failed: ple_mod.ws7 is None")

    print("WS7 initialized. Sampling...")
    samples = []
    t0 = time.monotonic()

    for i in range(args.n):
        wl = read_wl(ws7, use_channel2=args.ch2)
        if wl is not None:
            samples.append(wl)

        if i < 5 or (i + 1) % 50 == 0:
            print(f"[{i+1:4d}/{args.n}] t={time.monotonic()-t0:7.3f}s  wl={wl}")

        time.sleep(args.dt)

    # Cleanup
    try:
        ws7.stop_acquisition()
    except Exception:
        pass
    try:
        ple_mod.PLE.clean_up_globals()
    except Exception:
        pass

    if not samples:
        print("\nNo valid WS7 wavelength samples (>0) received.")
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
