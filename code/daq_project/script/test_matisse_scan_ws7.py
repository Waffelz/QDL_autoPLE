#!/usr/bin/env python3
"""
Test: Matisse scan while monitoring WS7 (optionally displaying air wavelength).

Default behavior (recommended):
  - Measure current WS7 vacuum wavelength
  - Scan by +delta (or -delta) from the current wavelength
This guarantees the scan doesn't "finish instantly" because you're already past the target.

Run:
  python script/test_matisse_scan_ws7.py --delta 0.01 --speed 0.002
  python script/test_matisse_scan_ws7.py --delta -0.01 --speed 0.002
  python script/test_matisse_scan_ws7.py --start 739.30 --end 739.31 --speed 0.002 --force-absolute
"""

import time
import argparse
import sys
from pathlib import Path

# --- Ensure project root is on sys.path so matisse_controller/functions26 import works ---
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]  # adjust if your folder layout differs: script/ is under project root
sys.path.insert(0, str(ROOT))

from functions26.instruments.ws7 import WS7
from matisse_controller.matisse import Matisse


def vacuum_to_air_nm(lambda_vac_nm: float) -> float:
    # Ciddor 1996 constants via APOGEE note :contentReference[oaicite:3]{index=3}
    lam_um = lambda_vac_nm * 1e-3
    inv_lam2 = 1.0 / (lam_um * lam_um)
    a = 0.0
    b1 = 5.792105e-2
    b2 = 1.67917e-3
    c1 = 238.0185
    c2 = 57.362
    n = 1.0 + (a + b1 / (c1 - inv_lam2) + b2 / (c2 - inv_lam2))
    return lambda_vac_nm / n


def read_ws7_vac_nm(ws7: WS7) -> float:
    wl = float(ws7.lib.GetWavelength(0.0))
    return wl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=0.002, help="scan speed (nm/s)")
    ap.add_argument("--dt", type=float, default=0.2, help="print period (s)")
    ap.add_argument("--timeout", type=float, default=120.0, help="max run time (s)")
    ap.add_argument("--medium", choices=["air", "vac"], default="air")

    # Relative scan (default)
    ap.add_argument("--delta", type=float, default=0.01,
                    help="scan delta (nm) relative to current (vacuum nm). Use negative to scan down.")

    # Absolute scan (optional)
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--force-absolute", action="store_true",
                    help="If set, use --start/--end as absolute targets; abort if current is already past end.")

    args = ap.parse_args()

    print("Connecting WS7...")
    ws7 = WS7()
    print("WS7 OK.\n")

    print("Connecting Matisse...")
    # Matisse constructor in your repo is (wavemeter_type='WaveMaster', wavemeter_port=None)
    # so do NOT pass visa_resource here.
    matisse = Matisse(wavemeter_type="WS7", wavemeter_port=None)
    print("Matisse OK.\n")

    # Read current wavelength (vacuum)
    wl0_vac = read_ws7_vac_nm(ws7)
    if wl0_vac <= 0:
        raise RuntimeError("WS7 returned invalid wavelength (<=0).")

    # Decide scan targets (in VACUUM nm for control)
    if args.force_absolute:
        if args.start is None or args.end is None:
            raise ValueError("--force-absolute requires --start and --end.")
        start_vac = float(args.start)
        end_vac = float(args.end)
    else:
        start_vac = wl0_vac
        end_vac = wl0_vac + float(args.delta)

    scan_dir = 0 if end_vac >= start_vac else 1  # 0 up, 1 down

    def disp(wl_vac: float) -> float:
        return vacuum_to_air_nm(wl_vac) if args.medium == "air" else wl_vac

    # If absolute mode, sanity check so we don't “finish instantly”
    if args.force_absolute:
        if scan_dir == 0 and wl0_vac >= end_vac:
            raise RuntimeError(
                f"Already past end: current={wl0_vac:.6f} nm (vac) >= end={end_vac:.6f} nm. "
                "Either retune first or use relative mode (default)."
            )
        if scan_dir == 1 and wl0_vac <= end_vac:
            raise RuntimeError(
                f"Already past end: current={wl0_vac:.6f} nm (vac) <= end={end_vac:.6f} nm. "
                "Either retune first or use relative mode (default)."
            )

    print(f"Current WS7: {disp(wl0_vac):.9f} nm ({args.medium})  [vac={wl0_vac:.9f}]")
    print(f"Scanning dir={scan_dir} from start_vac={start_vac:.6f} -> end_vac={end_vac:.6f} at speed={args.speed} nm/s")

    # Configure scan speed (your wrapper already used these successfully in earlier code)
    try:
        matisse.query(f"SCAN:RISINGSPEED {args.speed:.20f}")
        matisse.query(f"SCAN:FALLINGSPEED {args.speed:.20f}")
    except Exception as e:
        print(f"WARNING: Could not set scan speed via SCAN:*SPEED ({e}). Continuing...")

    # If your wrapper supports it, set target_wavelength for safety
    try:
        matisse.target_wavelength = round(float(end_vac), 6)
    except Exception:
        pass

    t_start = time.monotonic()
    last_print = 0.0

    try:
        # Start the scan
        matisse.start_scan(scan_dir)

        # Poll WS7 until we cross end
        i = 0
        while True:
            now = time.monotonic()
            if (now - t_start) > args.timeout:
                raise TimeoutError(f"Timed out after {args.timeout}s without reaching target.")

            wl_vac = read_ws7_vac_nm(ws7)
            if wl_vac > 0 and (now - last_print) >= args.dt:
                i += 1
                print(f"[{i:4d}] t={now - t_start:7.2f}s  WS7_{args.medium}={disp(wl_vac):.9f}  (vac={wl_vac:.9f})")
                last_print = now

            if scan_dir == 0 and wl_vac >= end_vac:
                print(f"Reached target (vac): {wl_vac:.6f} >= {end_vac:.6f}")
                break
            if scan_dir == 1 and wl_vac <= end_vac:
                print(f"Reached target (vac): {wl_vac:.6f} <= {end_vac:.6f}")
                break

            time.sleep(0.02)

    finally:
        print("\nStopping scan...")
        try:
            matisse.stop_scan()
        except Exception:
            pass

        # Try to close gracefully if your class supports it (prevents thread errors on exit)
        if hasattr(matisse, "close"):
            try:
                matisse.close()
            except Exception:
                pass

    print("\nDONE")
    print(f"End WS7: {disp(read_ws7_vac_nm(ws7)):.9f} nm ({args.medium})")


if __name__ == "__main__":
    main()
