#!/usr/bin/env python3
"""
WS7 smoke test (direct driver) with vacuum->air conversion.

Run:
  python script/test_ws7.py --n 200 --dt 0.02
  python script/test_ws7.py --medium vac
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

from functions26.instruments.ws7 import WS7


def vacuum_to_air_nm(lambda_vac_nm: float) -> float:
    """
    Convert vacuum wavelength (nm) -> standard air wavelength (nm)
    using Eq. (1) with Ciddor (1996) constants as tabulated in the APOGEE note. :contentReference[oaicite:1]{index=1}

    λ_vac = n(λ_vac) * λ_air  ->  λ_air = λ_vac / n
    Eq. (1): n - 1 = a + b1/(c1 - 1/λ_vac^2) + b2/(c2 - 1/λ_vac^2), λ_vac in µm
    """
    lam_um = lambda_vac_nm * 1e-3
    inv_lam2 = 1.0 / (lam_um * lam_um)

    # Ciddor 1996 constants (Table 1 in the APOGEE note) :contentReference[oaicite:2]{index=2}
    a = 0.0
    b1 = 5.792105e-2
    b2 = 1.67917e-3
    c1 = 238.0185
    c2 = 57.362

    n_minus_1 = a + b1 / (c1 - inv_lam2) + b2 / (c2 - inv_lam2)
    n = 1.0 + n_minus_1
    return lambda_vac_nm / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--ch2", action="store_true")
    ap.add_argument("--medium", choices=["air", "vac"], default="air",
                    help="Display wavelength in air (default) or vacuum.")
    args = ap.parse_args()

    ws7 = WS7()

    samples = []
    t0 = time.monotonic()
    print("WS7 initialized. Sampling...")

    for i in range(args.n):
        try:
            if args.ch2 and hasattr(ws7.lib, "GetWavelength2"):
                wl_vac = float(ws7.lib.GetWavelength2(0.0))
            else:
                wl_vac = float(ws7.lib.GetWavelength(0.0))
        except Exception:
            wl_vac = -1.0

        if wl_vac > 0:
            wl_show = vacuum_to_air_nm(wl_vac) if args.medium == "air" else wl_vac
            samples.append(wl_show)
        else:
            wl_show = None

        if i < 5 or (i + 1) % 50 == 0:
            extra = ""
            if wl_vac > 0 and args.medium == "air":
                extra = f"  (vac={wl_vac:.9f} nm)"
            print(f"[{i+1:4d}/{args.n}] t={time.monotonic()-t0:7.3f}s  wl_{args.medium}={wl_show}{extra}")

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
