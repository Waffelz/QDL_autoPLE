#!/usr/bin/env python3
"""
Test: Matisse wavelength scan + WS7 readback

- Locks to start_nm
- Starts scan toward end_nm at scan_speed_nm_per_s
- Samples WS7 during scan (dt_s)
- Stops when WS7 crosses end_nm (or timeout)
- Prints basic slope/summary and saves a CSV

Run:
  python script/test_matisse_scan_ws7.py --start 737.0 --end 737.2 --speed 0.005
"""

from __future__ import annotations

import argparse
import csv, sys
import time
from pathlib import Path
# --- Ensure project root is importable ---
ROOT = Path(__file__).resolve().parents[1]  # project root = parent of scripts/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matisse_controller.shamrock_ple.ple as ple_mod


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=float, required=True, help="start wavelength (nm)")
    p.add_argument("--end", type=float, required=True, help="end wavelength (nm)")
    p.add_argument("--speed", type=float, default=0.005, help="scan speed (nm/s)")
    p.add_argument("--dt", type=float, default=0.05, help="WS7 sample period (s)")
    p.add_argument("--timeout", type=float, default=120.0, help="timeout (s)")
    p.add_argument("--out", type=str, default="data/test_scan_ws7.csv", help="CSV output path")
    return p.parse_args()


def main():
    args = parse_args()
    start_nm = float(args.start)
    end_nm = float(args.end)
    speed = float(args.speed)
    dt_s = float(args.dt)
    timeout_s = float(args.timeout)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: Your PLE class takes (matisse) in __init__, and load_andor_libs is separate.
    # For this scan test we only need WS7 + matisse.
    # If your environment provides a matisse object elsewhere, ple_mod likely creates it in setup_matisse().
    # So we construct PLE by bypassing __init__ (same approach you used in earlier tests).
    ple = ple_mod.PLE.__new__(ple_mod.PLE)

    # Setup instruments
    ple.setup_ws7()
    ple.setup_matisse("WS7", scanning_speed=None)
    ple._setup_wavelength_tolerance("WS7")

    matisse = ple_mod.matisse
    ws7 = ple_mod.ws7
    if matisse is None or ws7 is None:
        raise RuntimeError("Setup failed: matisse/ws7 is None.")

    # Lock at start
    ple.lock_at_wavelength(round(start_nm, 6))
    time.sleep(0.2)

    # Save original speeds so we can restore
    try:
        orig_rise = float(matisse.query("SCAN:RISINGSPEED?", True))
        orig_fall = float(matisse.query("SCAN:FALLINGSPEED?", True))
    except Exception:
        orig_rise = None
        orig_fall = None

    scan_dir = int((end_nm - start_nm) < 0)  # 0 up, 1 down

    rows = []
    wall0 = time.time()
    t0 = time.monotonic()

    def ws7_wl_nm() -> float:
        return float(ws7.lib.GetWavelength(0.0))

    try:
        # Configure scan speed
        try:
            matisse.stabilize_off()
        except Exception:
            pass

        matisse.query(f"SCAN:RISINGSPEED {speed:.20f}")
        matisse.query(f"SCAN:FALLINGSPEED {speed:.20f}")

        # Optional: set target wavelength (if driver supports it)
        try:
            matisse.target_wavelength = round(float(end_nm), 6)
        except Exception:
            pass

        # Start scan
        matisse.start_scan(scan_dir)
        print(f"Started scan: {start_nm:.6f} -> {end_nm:.6f} nm @ {speed} nm/s  (dir={scan_dir})")

        # Sample loop
        i = 0
        while True:
            if (time.time() - wall0) > timeout_s:
                raise TimeoutError(f"Timeout after {timeout_s}s before reaching target.")

            now_m = time.monotonic()
            wl = ws7_wl_nm()
            rows.append((now_m - t0, wl))
            i += 1

            if i <= 5 or i % 20 == 0:
                print(f"[{i:4d}] t={now_m - t0:7.3f}s  wl={wl:.9f} nm")

            # Stop condition
            if scan_dir == 0 and wl >= end_nm:
                break
            if scan_dir == 1 and wl <= end_nm:
                break

            time.sleep(dt_s)

    finally:
        # Stop scan and restore
        try:
            matisse.stop_scan()
        except Exception:
            pass

        try:
            if orig_rise is not None:
                matisse.query(f"SCAN:RISINGSPEED {orig_rise:.20f}")
            if orig_fall is not None:
                matisse.query(f"SCAN:FALLINGSPEED {orig_fall:.20f}")
        except Exception:
            pass

        try:
            matisse.stabilize_on()
        except Exception:
            pass

        try:
            ple.clean_up_globals()
        except Exception:
            pass

    # Save CSV
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "wl_nm"])
        w.writerows(rows)

    # Summary
    if len(rows) >= 2:
        t_first, wl_first = rows[0]
        t_last, wl_last = rows[-1]
        dt = t_last - t_first
        dwl = wl_last - wl_first
        slope = dwl / dt if dt > 0 else float("nan")
        print("\n--- Summary ---")
        print(f"Samples:      {len(rows)}")
        print(f"Start wl:     {wl_first:.9f} nm")
        print(f"End wl:       {wl_last:.9f} nm")
        print(f"Duration:     {dt:.3f} s")
        print(f"Delta wl:     {dwl:.6f} nm")
        print(f"Measured rate:{slope:.6f} nm/s")
        print(f"Saved CSV:    {out_path.resolve()}")
        print("--------------")


if __name__ == "__main__":
    main()
