#!/usr/bin/env python3
"""
Matisse scan + WS7 monitor smoke test.

Goal:
- Start a Matisse scan
- Sample WS7 during scan
- Stop when WS7 crosses end_nm (or timeout)
- Save CSV for quick inspection

Run:
  python scripts/test_matisse_scan_ws7.py --start 737.0 --end 737.05 --speed 0.005 --dt 0.05 --timeout 180
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import importlib
from pathlib import Path


# ---------- sys.path helper (same style as your WS7 test) ----------
def add_root_to_syspath():
    HERE = Path(__file__).resolve()
    root = None
    for p in [HERE.parent] + list(HERE.parents):
        if (p / "functions26").exists() and (p / "matisse_controller").exists():
            root = p
            break
    if root is None:
        raise RuntimeError("Could not find BOTH 'functions26' and 'matisse_controller' in any parent directory.")
    sys.path.insert(0, str(root))
    return root


# ---------- WS7 connection (FILLED from your working script) ----------
def connect_ws7():
    from functions26.instruments.ws7 import WS7
    return WS7()


# ---------- Matisse connection (auto-tries + helpful error) ----------
def connect_matisse():
    """
    Tries a few common import locations. If none work, edit this function.

    Your matisse object must support:
      - start_scan(scan_dir)
      - stop_scan()
      - query(cmd, ...) OR query(cmd)
      - stabilize_off()/stabilize_on() (optional)
    """
    candidates = [
        ("matisse_controller.matisse", "Matisse"),
        ("matisse_controller.matisse_controller", "Matisse"),
        ("matisse_controller.matisse", "MatisseController"),
    ]

    last_err = None
    for mod_name, cls_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
            try:
                return cls()  # most drivers
            except TypeError:
                # If it needs args, you’ll need to pass them here.
                return cls
        except Exception as e:
            last_err = e

    raise RuntimeError(
        "Could not auto-import your Matisse class.\n"
        "Edit connect_matisse() with the exact import/constructor you use for Matisse.\n"
        f"Last error: {last_err}"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--end", type=float, required=True)
    p.add_argument("--speed", type=float, default=0.005, help="scan speed (same units your driver expects)")
    p.add_argument("--dt", type=float, default=0.05, help="WS7 sample period (s)")
    p.add_argument("--timeout", type=float, default=180.0, help="seconds")
    p.add_argument("--out", type=str, default="data/test_matisse_scan_ws7.csv")
    return p.parse_args()


def main():
    args = parse_args()
    add_root_to_syspath()

    ws7 = connect_ws7()
    matisse = connect_matisse()

    start_nm = float(args.start)
    end_nm = float(args.end)
    speed = float(args.speed)
    dt_s = float(args.dt)
    timeout_s = float(args.timeout)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def read_ws7_nm() -> float:
        return float(ws7.lib.GetWavelength(0.0))

    # Your earlier convention: scan_dir = int((end-start) < 0)  # 0 up, 1 down
    scan_dir = int((end_nm - start_nm) < 0)

    # Save original speeds if available
    orig_rise = orig_fall = None
    try:
        orig_rise = float(matisse.query("SCAN:RISINGSPEED?", True))
        orig_fall = float(matisse.query("SCAN:FALLINGSPEED?", True))
    except Exception:
        pass

    rows = []
    wall0 = time.time()
    t0 = time.monotonic()

    try:
        try:
            matisse.stabilize_off()
        except Exception:
            pass

        try:
            matisse.query(f"SCAN:RISINGSPEED {speed:.20f}")
            matisse.query(f"SCAN:FALLINGSPEED {speed:.20f}")
        except Exception:
            print("Warning: could not set scan speeds via SCAN:*SPEED (driver may differ).")

        print(f"Starting scan dir={scan_dir} (0=up,1=down), target end={end_nm:.6f} nm")

        matisse.start_scan(scan_dir)

        i = 0
        while True:
            if (time.time() - wall0) > timeout_s:
                raise TimeoutError(f"Timeout after {timeout_s}s before reaching end_nm.")

            t = time.monotonic() - t0
            wl = read_ws7_nm()
            rows.append((t, wl))
            i += 1

            if i <= 5 or i % 20 == 0:
                print(f"[{i:4d}] t={t:7.3f}s  wl={wl:.9f} nm")

            if scan_dir == 0 and wl >= end_nm:
                break
            if scan_dir == 1 and wl <= end_nm:
                break

            time.sleep(dt_s)

    finally:
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

    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "wl_nm"])
        w.writerows(rows)

    if len(rows) >= 2:
        t_first, wl_first = rows[0]
        t_last, wl_last = rows[-1]
        dt = t_last - t_first
        dwl = wl_last - wl_first
        rate = (dwl / dt) if dt > 0 else float("nan")
        print("\n--- Summary ---")
        print(f"Samples:       {len(rows)}")
        print(f"Start wl:      {wl_first:.9f} nm")
        print(f"End wl:        {wl_last:.9f} nm")
        print(f"Duration:      {dt:.3f} s")
        print(f"Delta wl:      {dwl:.6f} nm")
        print(f"Measured rate: {rate:.6f} nm/s")
        print(f"Saved CSV:     {out_path.resolve()}")
        print("--------------")


if __name__ == "__main__":
    main()
