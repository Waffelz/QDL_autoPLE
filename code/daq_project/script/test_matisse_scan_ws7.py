#!/usr/bin/env python3
"""
Matisse scan + WS7 monitor smoke test.

Run from project root (important):
  cd D:\Dropbox Folder\Dropbox\35share\Python\Xingyi\autoPLE\QDL_autoPLE-main\code\daq_project

Example (tiny safe scan):
  python script\test_matisse_scan_ws7.py ^
    --resource "USB0::0x17E7::0x0102::07-40-01::INSTR" ^
    --start 739.500 ^
    --end   739.510 ^
    --speed 0.002 ^
    --ws7-dt 0.05 ^
    --max-s 120

Notes:
- Close any Matisse/laser control GUI that might hold the VISA session if you see "already in use".
- Start with a VERY small range first.
"""

import argparse
import sys
import time
from pathlib import Path

# --- ensure repo root is on sys.path (so `matisse_controller` + `functions26` import works) ---
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]  # project root = parent of script/
sys.path.insert(0, str(ROOT))

from functions26.instruments.ws7 import WS7


def connect_matisse(resource: str):
    """
    Connect using your repo's Matisse class.
    Your earlier debug showed matisse_controller/matisse/__init__.py exports 'Matisse'.
    """
    import inspect
    from matisse_controller.matisse import Matisse

    print("Matisse class:", Matisse)
    print("Matisse signature:", inspect.signature(Matisse))

    last_err = None
    for ctor in [
        lambda: Matisse(resource),
        lambda: Matisse(device_id=resource),
        lambda: Matisse(resource_id=resource),
        lambda: Matisse(visa_resource=resource),
    ]:
        try:
            m = ctor()
            print("Constructed Matisse instance:", m)
            return m
        except Exception as e:
            last_err = e

    raise RuntimeError(
        "Could not construct Matisse() with the VISA resource.\n"
        f"Resource: {resource}\n"
        f"Last error: {last_err}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resource", required=True, help='VISA resource, e.g. USB0::...::INSTR')
    ap.add_argument("--start", type=float, required=True, help="start wavelength (nm)")
    ap.add_argument("--end", type=float, required=True, help="end wavelength (nm)")
    ap.add_argument("--speed", type=float, default=0.002, help="scan speed (nm/s) for both rising/falling")
    ap.add_argument("--ws7-dt", type=float, default=0.05, help="WS7 sampling period (s)")
    ap.add_argument("--max-s", type=float, default=180.0, help="max runtime (s)")
    ap.add_argument("--print-every", type=int, default=10, help="print every N samples")
    ap.add_argument("--tol", type=float, default=0.002, help="tolerance (nm) for lock-at-start wait")
    args = ap.parse_args()

    start_nm = float(args.start)
    end_nm = float(args.end)
    speed = float(args.speed)

    # 0 up, 1 down (matching your existing code convention)
    scan_dir = 0 if (end_nm - start_nm) > 0 else 1
    stop_wl = end_nm

    print("\nConnecting WS7...")
    ws7 = WS7()
    print("WS7 OK.")

    print("\nConnecting Matisse...")
    matisse = connect_matisse(args.resource)
    print("Matisse OK.")

    # Quick sanity query if Matisse exposes query()
    if hasattr(matisse, "query"):
        try:
            ans = matisse.query("*IDN?", True)
            print("*IDN? ->", ans)
        except Exception:
            pass

    # Optional: turn off stabilize if supported (your scan code did this)
    if hasattr(matisse, "stabilize_off"):
        try:
            matisse.stabilize_off()
        except Exception:
            pass

    # Set scan speeds (same pattern you already use)
    if hasattr(matisse, "query"):
        try:
            matisse.query(f"SCAN:RISINGSPEED {speed:.20f}")
            matisse.query(f"SCAN:FALLINGSPEED {speed:.20f}")
        except Exception as e:
            print("Warning: could not set SCAN:*SPEED via query():", e)

    # “Lock at start” (minimal version): set target wavelength if supported, then wait until WS7 near it
    if hasattr(matisse, "target_wavelength"):
        try:
            matisse.target_wavelength = round(start_nm, 6)
        except Exception:
            pass

    print(f"\nWaiting until WS7 ~ start ({start_nm:.6f} nm) within ±{args.tol} nm ...")
    t0_wait = time.monotonic()
    while True:
        wl = float(ws7.lib.GetWavelength(0.0))
        if wl > 0 and abs(wl - start_nm) <= args.tol:
            print(f"Locked-ish: WS7={wl:.6f} nm")
            break
        if (time.monotonic() - t0_wait) > 60.0:
            print(f"Proceeding anyway (didn't reach tol within 60s). WS7={wl if wl>0 else None}")
            break
        time.sleep(0.2)

    print(f"\nStarting scan dir={scan_dir} from {start_nm:.6f} -> {end_nm:.6f} at speed={speed} nm/s")
    start_wall = time.time()

    # Start scan using your existing API if present
    if not hasattr(matisse, "start_scan"):
        raise RuntimeError(
            "Your Matisse object has no start_scan().\n"
            "Tell me what methods it has (dir(matisse)) and I’ll adapt the script."
        )
    matisse.start_scan(scan_dir)

    # Monitor WS7
    samples = 0
    last_print = 0

    try:
        while True:
            if (time.time() - start_wall) > args.max_s:
                raise TimeoutError(f"Timed out after {args.max_s}s before reaching target.")

            wl = float(ws7.lib.GetWavelength(0.0))
            samples += 1

            if samples <= 5 or (samples - last_print) >= args.print_every:
                print(f"[{samples:5d}] t={time.time()-start_wall:7.2f}s  WS7={wl if wl>0 else None}")
                last_print = samples

            # stop when measured wavelength crosses target
            if wl > 0:
                if scan_dir == 0 and wl >= stop_wl:
                    print(f"Reached target: WS7={wl:.6f} nm >= {stop_wl:.6f} nm")
                    break
                if scan_dir == 1 and wl <= stop_wl:
                    print(f"Reached target: WS7={wl:.6f} nm <= {stop_wl:.6f} nm")
                    break

            time.sleep(args.ws7_dt)

    finally:
        print("\nStopping scan...")
        if hasattr(matisse, "stop_scan"):
            try:
                matisse.stop_scan()
            except Exception:
                pass

        if hasattr(matisse, "stabilize_on"):
            try:
                matisse.stabilize_on()
            except Exception:
                pass

        # If your class has a close/disconnect, call it
        for name in ("close", "disconnect", "shutdown"):
            if hasattr(matisse, name):
                try:
                    getattr(matisse, name)()
                except Exception:
                    pass

    print("\nDONE")


if __name__ == "__main__":
    main()
