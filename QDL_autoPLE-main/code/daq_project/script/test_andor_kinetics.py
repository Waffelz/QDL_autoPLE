#!/usr/bin/env python3
"""
test_andor_kinetics.py

End-to-end test of Andor CCD kinetic series using matisse_controller.shamrock_ple.

Run:
  python script/test_andor_kinetics.py --exp 0.1 --cycle 0.12 --n 50 --out data/test_kinetics.sif

IMPORTANT:
  - Close Andor Solis / any Andor GUI before running.
"""

import os
import sys
import time
import argparse
from pathlib import Path

# --- Ensure project root is on sys.path so `import matisse_controller` works ---
ROOT = Path(__file__).resolve().parents[1]  # .../daq_project
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matisse_controller.shamrock_ple.ple import PLE
import matisse_controller.shamrock_ple.ple as ple_mod

from matisse_controller.shamrock_ple.constants import (
    READ_MODE_FVB,
    READ_MODE_SINGLE_TRACK,
    COSMIC_RAY_FILTER_ON,
    COSMIC_RAY_FILTER_OFF,
)

def parse_readout_mode(s: str) -> int:
    s = (s or "FVB").strip().upper()
    if s == "FVB":
        return READ_MODE_FVB
    if s in ("SINGLE_TRACK", "SINGLETRACK"):
        return READ_MODE_SINGLE_TRACK
    raise ValueError(f"Unknown readout_mode: {s}")

def parse_cosmic(val: str) -> int:
    s = (val or "ON").strip().upper()
    if s in ("ON", "TRUE", "1", "YES"):
        return COSMIC_RAY_FILTER_ON
    if s in ("OFF", "FALSE", "0", "NO"):
        return COSMIC_RAY_FILTER_OFF
    raise ValueError(f"Unknown cosmic setting: {val}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp", type=float, default=-65.0, help="CCD temperature setpoint (C)")
    ap.add_argument("--tol", type=float, default=1.0, help="Temp tolerance (C)")
    ap.add_argument("--wait", action="store_true", help="Wait until temperature reaches setpoint (slow)")
    ap.add_argument("--wait-timeout", type=float, default=1800, help="Max seconds to wait for cooldown")
    ap.add_argument("--exp", type=float, default=0.10, help="Exposure time per frame (s)")
    ap.add_argument("--cycle", type=float, default=0.12, help="Kinetic cycle time (s)")
    ap.add_argument("--n", type=int, default=50, help="Number of frames")
    ap.add_argument("--readout", type=str, default="FVB", help="FVB or SINGLE_TRACK")
    ap.add_argument("--cosmic", type=str, default="ON", help="ON or OFF")
    ap.add_argument("--out", type=str, default="data/test_kinetics.sif", help="Output .sif path")
    args = ap.parse_args()

    print("NOTE: Close Andor Solis / any Andor GUI before running this test.")
    print("Loading Andor libs ...")
    PLE.load_andor_libs()

    ccd = ple_mod.ccd
    shamrock = ple_mod.shamrock  # may be None / unused in this test
    if ccd is None:
        raise RuntimeError("CCD global is None after PLE.load_andor_libs().")

    # Keep cooling on (non-blocking)
    ccd.ensure_cooling(args.temp, persist_on_shutdown=True)

    if args.wait:
        print(f"Waiting until CCD <= {args.temp}+{args.tol} C (timeout={args.wait_timeout}s)")
        ccd.wait_to_cooldown(target_C=args.temp, tol_C=args.tol, poll_s=5.0, timeout_s=args.wait_timeout)
    else:
        t_now = ccd.wait_until_cold(args.temp, tol_C=args.tol, timeout_s=0.0)
        print(f"Temp check (non-blocking): current CCD temp = {t_now:.1f} C")

    readout_mode = parse_readout_mode(args.readout)
    cosmic_mode = parse_cosmic(args.cosmic)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure kinetics (do NOT block on cooldown again)
    print("\nConfiguring kinetics...")
    exp_actual, cycle_actual = ccd.setup_kinetics(
        exposure_time=args.exp,
        cycle_time=args.cycle,
        n_frames=args.n,
        readout_mode=readout_mode,
        temperature=args.temp,
        cool_down=False,  # important: don't wait here
        cosmic_ray_filter=cosmic_mode,
    )

    print(f"Kinetics configured: exp_actual={exp_actual:.6f}s, cycle_actual={cycle_actual:.6f}s")
    print(f"Estimated duration ~ {args.n * cycle_actual:.3f}s for {args.n} frames")

    # Acquire
    print("\nStarting acquisition...")
    t0 = time.time()
    ccd.start_acquisition()
    ccd.wait_for_acquisition()
    t1 = time.time()
    print(f"Acquisition finished in {t1 - t0:.3f}s")

    # Save SIF robustly: write to CWD then move
    tmp_name = out_path.name
    print(f"\nSaving SIF (tmp) to {Path.cwd() / tmp_name}")
    ccd.save_as_sif(tmp_name)

    # Move into desired folder
    if out_path.exists():
        out_path.unlink()
    os.replace(str(Path.cwd() / tmp_name), str(out_path))

    temp_after = ccd.get_temperature()
    print(f"CCD temperature after acquisition: {temp_after:.1f} C")
    print(f"\nSaved SIF: {out_path}")

    print("\nDONE.")

    # Clean up globals (your updated CCD can keep cooler on depending on settings)
    try:
        PLE.clean_up_globals()
    except Exception:
        pass

if __name__ == "__main__":
    main()
