#!/usr/bin/env python3
"""
scripts/test_ple_setup.py

For your current repo where:
  - matisse_controller.shamrock_ple.ple.PLE signature is (matisse)
  - PLE does NOT provide setup_ws7()

This script:
  1) imports ple.py and prints its path
  2) constructs Matisse (best-effort via signature introspection)
  3) constructs PLE(matisse)
  4) reads wavemeter wavelength repeatedly via Matisse (best-effort)

Run from project root:
  python scripts\\test_ple_setup.py

Options:
  python scripts\\test_ple_setup.py --n 200 --dt 0.03 --wavemeter-type WS7
"""

import sys
import time
import argparse
import inspect
from pathlib import Path


# --- Ensure project root is importable (parent of scripts/) ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _try_call(obj, method_names, *args, **kwargs):
    """
    Try calling the first existing callable attribute from method_names on obj.
    Returns the result or raises AttributeError if none exist.
    """
    for name in method_names:
        if hasattr(obj, name):
            fn = getattr(obj, name)
            if callable(fn):
                return fn(*args, **kwargs)
    raise AttributeError(f"None of these methods exist/callable: {method_names}")


def _construct_with_signature(cls, preferred_by_name):
    """
    Construct cls by matching constructor parameter names to preferred_by_name,
    and filling any remaining required positional parameters with None.

    This avoids hard-coding signatures.
    """
    sig = inspect.signature(cls)
    params = list(sig.parameters.values())

    # Build positional args in order for POSITIONAL_ONLY / POSITIONAL_OR_KEYWORD
    args = []
    kwargs = {}

    # Skip "self" if present (signature() for classes usually omits it, but just in case)
    # For classes, signature(Matisse) usually shows (arg1, arg2, ...) without self.
    for p in params:
        # If parameter can be provided by name and we have a preferred value, use it.
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            if p.name in preferred_by_name:
                args.append(preferred_by_name[p.name])
            else:
                # If required, use None as placeholder; if optional, omit to use default
                if p.default is inspect._empty:
                    args.append(None)
        elif p.kind == p.KEYWORD_ONLY:
            if p.name in preferred_by_name:
                kwargs[p.name] = preferred_by_name[p.name]
        # Ignore VAR_POSITIONAL/VAR_KEYWORD; not needed here.

    return cls(*args, **kwargs)


def get_matisse_wavelength_nm(matisse):
    """
    Best-effort wavelength read from Matisse object.
    Tries common APIs used in Matisse controller codebases.
    """
    # Most likely from your earlier code:
    #   matisse.wavemeter_wavelength()
    # or perhaps:
    #   matisse.wavemeter.wavelength()
    # or:
    #   matisse.get_wavelength()
    try:
        return float(_try_call(matisse, ["wavemeter_wavelength", "get_wavelength", "wavelength"]))
    except AttributeError:
        pass

    # Try nested wavemeter object patterns
    if hasattr(matisse, "wavemeter"):
        wvm = getattr(matisse, "wavemeter")
        try:
            return float(_try_call(wvm, ["wavelength", "get_wavelength", "GetWavelength"], 0.0))
        except Exception:
            pass

    raise RuntimeError(
        "Couldn't find a wavelength-read method on Matisse. "
        "Inspect matisse object with dir(matisse) and update get_matisse_wavelength_nm()."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wavemeter-type", default="WS7", help="Passed into Matisse constructor when possible")
    ap.add_argument("--port", default=None, help="Port for wavemeter / matisse if your constructor needs it")
    ap.add_argument("--n", type=int, default=50, help="Number of wavelength samples")
    ap.add_argument("--dt", type=float, default=0.05, help="Delay between samples (s)")
    ap.add_argument("--print-matisse-scan-speeds", action="store_true", help="Try querying scan speeds if available")
    args = ap.parse_args()

    # Import ple + Matisse
    import matisse_controller.shamrock_ple.ple as ple_mod
    from matisse_controller.matisse.matisse import Matisse

    print("PLE module path:", ple_mod.__file__)
    print("PLE signature   :", inspect.signature(ple_mod.PLE))
    print("Matisse path    :", inspect.getfile(Matisse))
    print("Matisse signature:", inspect.signature(Matisse))

    # Construct Matisse as robustly as possible
    # Common signatures in your older code were: Matisse(wavemeter_type, matisse_wavemeter_port)
    preferred = {
        # guess common names:
        "wavemeter_type": args.wavemeter_type,
        "wavemeter": args.wavemeter_type,
        "wavemeter_name": args.wavemeter_type,
        "matisse_wavemeter_port": args.port,
        "wavemeter_port": args.port,
        "port": args.port,
        "instrument_port": args.port,
    }

    try:
        matisse = _construct_with_signature(Matisse, preferred)
    except Exception as e:
        print("FAILED to construct Matisse with introspection.")
        print("Error:", repr(e))
        print("\nTry editing this script to call Matisse(...) exactly as your code expects.")
        raise

    print("\nConstructed Matisse object:", type(matisse))

    # Construct PLE(matisse)
    try:
        ple = ple_mod.PLE(matisse)
    except Exception as e:
        print("FAILED to construct PLE(matisse). Error:", repr(e))
        raise

    print("Constructed PLE object:", type(ple))
    print("PLE methods (subset):", [m for m in dir(ple) if not m.startswith("_")][:25], "...\n")

    # Optional: query Matisse scan speeds (if query() exists)
    if args.print_matisse_scan_speeds and hasattr(matisse, "query"):
        try:
            rise = matisse.query("SCAN:RISINGSPEED?", True)
            fall = matisse.query("SCAN:FALLINGSPEED?", True)
            print("Matisse scan speeds:", rise, fall)
        except Exception as e:
            print("Matisse speed query failed:", repr(e))
        print()

    # Wavelength sampling via matisse
    print("Sampling wavelength via Matisse...")
    wls = []
    t0 = time.time()
    for i in range(args.n):
        wl = get_matisse_wavelength_nm(matisse)
        wls.append(wl)
        dt = time.time() - t0
        print(f"[{i+1:4d}/{args.n}] t={dt:7.3f}s  wl={wl:.12f} nm")
        time.sleep(args.dt)

    # Simple stats without numpy
    wls_valid = [x for x in wls if x > 0]
    if not wls_valid:
        raise RuntimeError("No valid wavelength samples (>0).")

    mean = sum(wls_valid) / len(wls_valid)
    var = sum((x - mean) ** 2 for x in wls_valid) / max(1, (len(wls_valid) - 1))
    std = var ** 0.5

    print("\n--- Matisse/Wavemeter Results ---")
    print(f"Valid samples: {len(wls_valid)}/{len(wls)}")
    print(f"Mean (nm):     {mean:.12f}")
    print(f"Std (nm):      {std:.12f}")
    print(f"Min (nm):      {min(wls_valid):.12f}")
    print(f"Max (nm):      {max(wls_valid):.12f}")
    print("--------------------------------\n")

    # Cleanup if your PLE module has global cleanup (some versions do)
    if hasattr(ple_mod, "PLE") and hasattr(ple_mod.PLE, "clean_up_globals"):
        try:
            ple_mod.PLE.clean_up_globals()
        except Exception:
            pass

    print("DONE")


if __name__ == "__main__":
    main()
