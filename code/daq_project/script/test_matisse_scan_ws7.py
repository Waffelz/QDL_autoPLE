#!/usr/bin/env python3
"""
Matisse scan + WS7 monitor smoke test (using your repo's Matisse driver).

Run from project root:
  cd D:\Dropbox Folder\Dropbox\35share\Python\Xingyi\autoPLE\QDL_autoPLE-main\code\daq_project

Example:
  python script\test_matisse_scan_ws7.py --start 739.500 --end 739.510 --speed 0.002 --ws7-dt 0.05 --max-s 180
"""

import argparse
import sys
import time
from pathlib import Path

# --- ensure repo root is on sys.path ---
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]  # project root = parent of script/
sys.path.insert(0, str(ROOT))

from functions26.instruments.ws7 import WS7


def connect_matisse(wavemeter_type: str = "WS7", wavemeter_port=None):
    """
    Your matisse_controller.matisse.Matisse signature is (wavemeter_type='WaveMaster', wavemeter_port=None).
    It likely reads the VISA resource from matisse_controller.config/configuration.
    """
    import inspect
    from matisse_controller.matisse import Matisse

    print("Matisse class:", Matisse)
    print("Matisse signature:", inspect.signature(Matisse))

    # IMPORTANT: do NOT pass VISA resource string; driver likely reads from config.
    return Matisse(wavemeter_type=wavemeter_type, wavemeter_port=wavemeter_port)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, required=True, help="start wavelength (nm)")
    ap.add_argument("--end", type=float, required=True, help="end wavelength (nm)")
    ap.add_argument("--speed", type=float, default=0.002, help="scan speed (nm/s)")
    ap.add_argument("--ws7-dt", type=float, default=0.05, help="WS7 sampling period (s)")
    ap.add_argument("--max-s", type=float, default=180.0, help="max runtime (s)")
    ap.add_argument("--print-every", type=int, default=10, help="print every N samples")
    ap.add_argument("--tol", type=float, default=0.002, help="tolerance (nm) to start scan")
    ap.add_argument("--matisse-wavemeter-type", default="WS7", help="force matisse wavemeter_type (avoid WaveMaster)")
    args = ap.parse_args()

    start_nm = float(args.start)
    end_nm = float(args.end)
    speed = float(args.speed)

    scan_dir = 0 if (end_nm - start_nm) > 0 else 1
    stop_wl = end_nm

    print("\nConnecting WS7...")
    ws7 = WS7()
    print("WS7 OK.")

    print("\nConnecting Matisse...")
    matisse = connect_matisse(wavemeter_type=str(args.matisse_wavemeter_type))
    print("Matisse OK.")

    # Some drivers expose query(); if yours does, show IDN
    if hasattr(matisse, "query"):
        try:
            ans = matisse.query("*IDN?", True)
            print("*IDN? ->", ans)
        except Exception:
            pass

    # Optional stabilize off
    if hasattr(matisse, "stabilize_off"):
        try:
            matisse.stabilize_off()
        except Exception:
            pass

    # Set scan speed (your driver may forward SCPI to the laser)
    if hasattr(matisse, "query"):
        try:
            matisse.query(f"SCAN:RISINGSPEED {speed:.20f}")
            matisse.query(f"SCAN:FALLINGSPEED {speed:.20f}")
        except Exception as e:
            print("Warning: could not set SCAN:*SPEED via matisse.query():", e)

    # Set target wavelength if supported
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

    if not hasattr(matisse, "start_scan"):
        raise RuntimeError("Your Matisse object has no start_scan(). Print dir(matisse) and we’ll adapt.")

    start_wall = time.time()
    samples = 0
    last_print = 0

    try:
        matisse.start_scan(scan_dir)

        while True:
            if (time.time() - start_wall) > args.max_s:
                raise TimeoutError(f"Timed out after {args.max_s}s before reaching target.")

            wl = float(ws7.lib.GetWavelength(0.0))
            samples += 1

            if samples <= 5 or (samples - last_print) >= args.print_every:
                print(f"[{samples:5d}] t={time.time()-start_wall:7.2f}s  WS7={wl if wl>0 else None}")
                last_print = samples

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

    print("\nDONE")


if __name__ == "__main__":
    main()
