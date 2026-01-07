#!/usr/bin/env python3
"""
scripts/test_ple_setup.py

Matches your current matisse_controller/shamrock_ple/ple.py API:
  class PLE:
      def __init__(self, matisse): ...
      @staticmethod
      def load_andor_libs(): ... creates global ccd, shamrock

This script:
  - Imports ple module and prints file path
  - Builds a PLE instance (you can pass --no-matisse to pass None)
  - Calls PLE.load_andor_libs()
  - Performs lightweight CCD + Shamrock sanity checks

Run from project root:
  python scripts\\test_ple_setup.py

Optional:
  python scripts\\test_ple_setup.py --quick-acq --exp 0.05
"""

import sys
import time
import argparse
import inspect
from pathlib import Path


# --- Ensure project root is importable ---
ROOT = Path(__file__).resolve().parents[1]  # project root = parent of scripts/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def try_import_matisse():
    """
    Optional: import + construct a Matisse object.
    If you only want to test Andor libs, you can pass None to PLE.
    """
    from matisse_controller.matisse.matisse import Matisse
    sig = inspect.signature(Matisse)
    print("Matisse class path:", inspect.getfile(Matisse))
    print("Matisse signature :", sig)

    # Best guess constructor call based on your earlier stack:
    # Matisse(wavemeter_type, port)
    # If your Matisse requires different args, edit here.
    try:
        return Matisse("WS7", None)
    except TypeError:
        # Fallback: try no-arg
        return Matisse()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-matisse", action="store_true",
                    help="Construct PLE(None) instead of trying to construct a Matisse object.")
    ap.add_argument("--quick-acq", action="store_true",
                    help="Do a quick single acquisition (no cooldown). Requires Andor connected + Solis closed.")
    ap.add_argument("--exp", type=float, default=0.05, help="Exposure time for quick acquisition (s)")
    args = ap.parse_args()

    import matisse_controller.shamrock_ple.ple as ple_mod

    print("PLE module path:", ple_mod.__file__)
    print("PLE signature  :", inspect.signature(ple_mod.PLE))

    # Construct matisse if requested; otherwise None is fine for Andor-only tests
    matisse = None
    if not args.no_matisse:
        try:
            matisse = try_import_matisse()
            print("Constructed matisse:", type(matisse))
        except Exception as e:
            print("WARNING: Could not construct Matisse; continuing with matisse=None.")
            print("Reason:", repr(e))
            matisse = None

    # Construct PLE(matisse)
    ple = ple_mod.PLE(matisse)
    print("Constructed PLE:", type(ple))

    # Load Andor libs (creates global ccd/shamrock inside ple_mod)
    print("\nCalling PLE.load_andor_libs() ...")
    ple_mod.PLE.load_andor_libs()

    ccd = getattr(ple_mod, "ccd", None)
    shamrock = getattr(ple_mod, "shamrock", None)

    if ccd is None:
        raise RuntimeError("CCD is None after load_andor_libs(). Check Andor SDK/DLL access.")
    if shamrock is None:
        raise RuntimeError("Shamrock is None after load_andor_libs(). Check Shamrock SDK/DLL access.")

    print("\n--- Andor Globals ---")
    print("ccd     :", type(ccd))
    print("shamrock:", type(shamrock))

    # Lightweight CCD checks (should be fast)
    try:
        w, h = ccd.get_camera_size()
        print(f"CCD detector size: {w} x {h}")
    except Exception as e:
        print("CCD get_camera_size failed:", repr(e))

    try:
        temp = ccd.get_temperature()
        print(f"CCD temperature: {temp:.2f} C")
    except Exception as e:
        print("CCD get_temperature failed:", repr(e))

    # Lightweight Shamrock checks (depends on what methods exist)
    # We'll just print available public methods to guide next steps.
    shamrock_methods = [m for m in dir(shamrock) if not m.startswith("_")]
    print(f"Shamrock public methods (sample): {shamrock_methods[:25]} ...")

    # Optional quick acquisition (no cooldown; avoids long waits)
    if args.quick_acq:
        print("\n--- Quick Acquisition ---")
        print("NOTE: Ensure Andor Solis is CLOSED, otherwise the SDK often fails.")
        try:
            # Your CCD.setup() signature supports cool_down flag; keep it False for fast test.
            # acquisition_mode/readout_mode default to SINGLE/FVB in your CCD code.
            ccd.setup(exposure_time=args.exp, cool_down=False)
            data = ccd.take_acquisition()
            print("Acquisition OK.")
            print("Data shape:", getattr(data, "shape", None))
            # Print a couple values
            try:
                import numpy as np
                arr = np.asarray(data)
                print("Min/Max:", int(arr.min()), int(arr.max()))
                print("Mean:", float(arr.mean()))
            except Exception:
                pass
        except Exception as e:
            print("Quick acquisition FAILED:", repr(e))

    # Cleanup (your clean_up_globals only nulls globals; CCD.__del__ shuts down on GC)
    try:
        ple_mod.PLE.clean_up_globals()
        print("\nCleaned up ple globals.")
    except Exception as e:
        print("\nCleanup failed:", repr(e))

    print("\nDONE")


if __name__ == "__main__":
    main()
