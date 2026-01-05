#!/usr/bin/env python3
"""
scripts/test_ple_setup.py

Smoke test for your instrument stack through matisse_controller.shamrock_ple.ple:

- Imports PLE module and prints where it's imported from
- Creates a PLE instance even if __init__ signature differs (fallback via __new__)
- Calls:
    ple.setup_ws7()         -> verifies WS7 DLL + reading
    ple.setup_powermeter()  -> verifies Newport GPIB reading (optional)
    ple.setup_matisse()     -> verifies Matisse connection (optional; can skip)

Usage (from project root):
  python scripts/test_ple_setup.py
  python scripts/test_ple_setup.py --pm --pm-ch A
  python scripts/test_ple_setup.py --matisse
"""

import sys
import time
import argparse
import inspect
from pathlib import Path

# --- Make imports work regardless of working directory ---
ROOT = Path(__file__).resolve().parents[1]  # project root = parent of scripts/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matisse_controller.shamrock_ple.ple as ple_mod


def make_ple_instance():
    """
    Robust PLE construction:
    1) Try PLE()
    2) Try filling required positional args with None
    3) Last resort: bypass __init__ with __new__ and set minimal attributes
    """
    PLE = ple_mod.PLE
    print("PLE imported from:", ple_mod.__file__)
    print("PLE signature:", inspect.signature(PLE))

    # 1) Try no-arg
    try:
        ple = PLE()
        print("Constructed PLE() with no args.")
        return ple
    except TypeError as e:
        print("PLE() no-arg failed:", e)

    # 2) Try required positional args = None
    try:
        sig = inspect.signature(PLE)
        required = []
        for name, p in list(sig.parameters.items())[1:]:  # skip self
            if p.default is inspect._empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                required.append(None)
        ple = PLE(*required)
        print(f"Constructed PLE(*{required}).")
        return ple
    except TypeError as e:
        print("PLE(required positional args) failed:", e)

    # 3) Last resort: bypass __init__
    print("Falling back to PLE.__new__ (bypassing __init__).")
    ple = PLE.__new__(PLE)

    # Set attributes that setup_* methods often rely on (safe defaults)
    # If your PLE implementation doesn't use these, harmless.
    if not hasattr(ple, "powermeter_port"):
        ple.powermeter_port = None
    if not hasattr(ple, "spcm_port"):
        ple.spcm_port = None
    if not hasattr(ple, "wa1600_port"):
        ple.wa1600_port = None
    if not hasattr(ple, "matisse_wavemeter_port"):
        ple.matisse_wavemeter_port = None
    if not hasattr(ple, "ws7_sleep_time"):
        ple.ws7_sleep_time = 0.05

    return ple


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws7-n", type=int, default=10, help="Number of WS7 samples")
    ap.add_argument("--ws7-dt", type=float, default=0.05, help="Delay between WS7 samples (s)")

    ap.add_argument("--pm", action="store_true", help="Also test powermeter")
    ap.add_argument("--pm-ch", type=str, default="A", help="Powermeter channel: A/B/AB")

    ap.add_argument("--matisse", action="store_true", help="Also test Matisse queries")
    ap.add_argument("--wavemeter-type", type=str, default="WS7", help="Matisse wavemeter type passed to setup_matisse")
    ap.add_argument("--scan-speed", type=float, default=None, help="Optional scanning speed to set in setup_matisse")

    args = ap.parse_args()

    ple = make_ple_instance()

    # -----------------
    # WS7 test
    # -----------------
    if not hasattr(ple, "setup_ws7"):
        raise RuntimeError("PLE object has no method setup_ws7(). Your PLE implementation differs from expected.")

    print("\nSetting up WS7...")
    ple.setup_ws7()
    ws7 = getattr(ple_mod, "ws7", None)
    if ws7 is None:
        raise RuntimeError("WS7 setup failed: ple_mod.ws7 is None after ple.setup_ws7().")

    print("WS7 initialized. Sampling...")
    wl_vals = []
    t0 = time.time()
    for i in range(args.ws7_n):
        wl = float(ws7.lib.GetWavelength(0.0))
        wl_vals.append(wl)
        dt = time.time() - t0
        print(f"[{i+1:3d}/{args.ws7_n}] t={dt:7.3f}s  wl={wl:.12f}")
        time.sleep(args.ws7_dt)

    wl_arr = [w for w in wl_vals if w > 0]
    if not wl_arr:
        raise RuntimeError("No valid WS7 wavelength samples (>0).")
    wl_arr = list(map(float, wl_arr))
    print("\n--- WS7 Results ---")
    print(f"Valid samples: {len(wl_arr)}/{len(wl_vals)}")
    print(f"Mean (nm):     {sum(wl_arr)/len(wl_arr):.12f}")
    # manual std to avoid needing numpy
    mean = sum(wl_arr)/len(wl_arr)
    var = sum((x-mean)**2 for x in wl_arr) / max(1, (len(wl_arr)-1))
    std = var**0.5
    print(f"Std (nm):      {std:.12f}")
    print(f"Min (nm):      {min(wl_arr):.12f}")
    print(f"Max (nm):      {max(wl_arr):.12f}")
    print("-------------------\n")

    # -----------------
    # Powermeter test (optional)
    # -----------------
    if args.pm:
        if not hasattr(ple, "setup_powermeter"):
            raise RuntimeError("PLE object has no method setup_powermeter(channel).")

        print(f"Setting up powermeter channel={args.pm_ch}...")
        ple.setup_powermeter(args.pm_ch)
        powermeter = getattr(ple_mod, "powermeter", None)
        if powermeter is None:
            raise RuntimeError("Powermeter setup failed: ple_mod.powermeter is None after ple.setup_powermeter().")

        print("Initializing powermeter session...")
        powermeter.powermeter.initialize_instrument()
        try:
            # Some drivers have _empty_buffer; if not, ignore.
            if hasattr(powermeter, "_empty_buffer"):
                powermeter._empty_buffer()

            # One read
            rs = powermeter.powermeter.get_instrument_reading_string_all()
            vals_uW = []
            for s in rs:
                try:
                    vals_uW.append(powermeter.convert_reading_string_to_float(s))  # µW per your driver
                except Exception:
                    pass
            print("Raw strings:", rs)
            print("Parsed uW:", vals_uW)
            if vals_uW:
                mean_uW = sum(vals_uW) / len(vals_uW)
                print(f"Mean power: {mean_uW:.6f} µW  ({mean_uW*1e-6:.6e} W)")
        finally:
            print("Terminating powermeter session...")
            powermeter.powermeter.terminate_instrument()
        print()

    # -----------------
    # Matisse test (optional)
    # -----------------
    if args.matisse:
        if not hasattr(ple, "setup_matisse"):
            raise RuntimeError("PLE object has no method setup_matisse().")

        print("Setting up Matisse...")
        ple.setup_matisse(args.wavemeter_type, scanning_speed=args.scan_speed)
        matisse = getattr(ple_mod, "matisse", None)
        if matisse is None:
            raise RuntimeError("Matisse setup failed: ple_mod.matisse is None after setup_matisse().")

        # Try a couple safe queries
        try:
            rs = matisse.query("SCAN:RISINGSPEED?", True)
            fs = matisse.query("SCAN:FALLINGSPEED?", True)
            print("Matisse speeds:", rs, fs)
        except Exception as e:
            print("Matisse query failed:", e)
        print()

    # Cleanup
    if hasattr(ple, "clean_up_globals"):
        ple.clean_up_globals()
    else:
        # fallback: module-level cleanup if present
        if hasattr(ple_mod, "PLE") and hasattr(ple_mod.PLE, "clean_up_globals"):
            try:
                ple_mod.PLE.clean_up_globals()
            except Exception:
                pass

    print("DONE")


if __name__ == "__main__":
    main()
