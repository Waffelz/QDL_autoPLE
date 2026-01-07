#!/usr/bin/env python3
"""
WS7 smoke test using ConvertUnit(vac -> air), matching wavemaster.py behavior.

Run:
  python script/test_ws7.py --n 200 --dt 0.02 --medium air
  python script/test_ws7.py --n 50  --dt 0.05 --medium vac
"""

import time
import argparse
import statistics
import sys
from pathlib import Path
import ctypes


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


# wlmConst equivalents (from your snippet)
cReturnWavelengthVac = 0
cReturnWavelengthAir = 1


ERR_DICT = {
    0: "ErrNoValue",
    -1: "ErrNoSignal",
    -2: "ErrBadSignal",
    -3: "ErrLowSignal",
    -4: "ErrBigSignal",
    -5: "ErrWimMissing",
    -6: "ErrNotAvailable",
    -7: "InfNothingChanged",
    -8: "ErrNoPulse",
    -10: "ErrChannelNotAvailable",
    -13: "ErrDiv0",
    -14: "ErrOutOfRange",
    -15: "ErrUnitNotAvailable",
}


def bind_ctypes_prototypes(lib) -> None:
    """Ensure ctypes knows ConvertUnit/GetWavelength return doubles."""
    if hasattr(lib, "GetWavelength"):
        lib.GetWavelength.argtypes = [ctypes.c_double]
        lib.GetWavelength.restype = ctypes.c_double
    if hasattr(lib, "GetWavelength2"):
        lib.GetWavelength2.argtypes = [ctypes.c_double]
        lib.GetWavelength2.restype = ctypes.c_double
    if hasattr(lib, "ConvertUnit"):
        lib.ConvertUnit.argtypes = [ctypes.c_double, ctypes.c_long, ctypes.c_long]
        lib.ConvertUnit.restype = ctypes.c_double


def read_wavelength_nm(ws7: WS7, medium: str = "air", ch2: bool = False, max_err_print: int = 3) -> float:
    """
    Read wavelength in nm.
    - raw is vacuum wavelength from GetWavelength*()
    - air uses ConvertUnit(vac -> air)
    Returns <=0 on error (WS7 convention).
    """
    lib = ws7.lib

    # raw (vacuum)
    if ch2 and hasattr(lib, "GetWavelength2"):
        raw = float(lib.GetWavelength2(0.0))
    else:
        raw = float(lib.GetWavelength(0.0))

    if raw <= 0:
        # print a couple errors but don't spam
        if max_err_print > 0:
            msg = ERR_DICT.get(int(raw), f"UnknownError({raw})")
            print(f"WS7 error: {raw} ({msg})")
        return raw

    if medium == "vac":
        return raw

    # Convert vacuum -> air (this is the key change)
    if not hasattr(lib, "ConvertUnit"):
        raise RuntimeError("WS7 DLL has no ConvertUnit() - cannot convert vac->air.")
    air = float(lib.ConvertUnit(raw, cReturnWavelengthVac, cReturnWavelengthAir))
    return air


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--ch2", action="store_true")
    ap.add_argument("--medium", choices=["air", "vac"], default="air")
    args = ap.parse_args()

    ws7 = WS7()
    bind_ctypes_prototypes(ws7.lib)

    print("WS7 initialized.")
    print("Has ConvertUnit:", hasattr(ws7.lib, "ConvertUnit"))

    # quick sanity check: vac vs air should differ by ~0.2 nm near 740 nm
    wl_vac = read_wavelength_nm(ws7, medium="vac", ch2=args.ch2)
    wl_air = read_wavelength_nm(ws7, medium="air", ch2=args.ch2)
    if wl_vac > 0 and wl_air > 0:
        print(f"Check: vac={wl_vac:.9f} nm, air={wl_air:.9f} nm, (vac-air)={wl_vac - wl_air:+.6f} nm")
        print("NOTE: Expected (vac-air) ~ +0.20 nm near 740 nm (air smaller than vac).")
    else:
        print("Check: could not read valid wavelength for vac/air sanity check.")

    samples = []
    t0 = time.monotonic()
    print(f"\nSampling ({args.medium})...")

    for i in range(args.n):
        wl = read_wavelength_nm(ws7, medium=args.medium, ch2=args.ch2, max_err_print=3 if i < 3 else 0)

        if wl > 0:
            samples.append(wl)
            wl_show = f"{wl:.12f}"
        else:
            wl_show = "None"

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
