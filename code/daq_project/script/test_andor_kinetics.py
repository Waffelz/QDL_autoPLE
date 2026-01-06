#!/usr/bin/env python3
"""
test_andor_kinetics.py

End-to-end test of Andor CCD kinetic series using matisse_controller.shamrock_ple.

What it does:
  - Loads Andor libs (CCD + Shamrock globals)
  - Ensures cooler is ON and setpoint is applied (non-blocking)
  - Optionally waits until temperature is cold enough
  - Configures kinetics (n_frames, exposure, cycle)
  - Runs acquisition (blocking until done)
  - Saves a .sif file
  - Prints timing + temperature before/after

IMPORTANT:
  - Close Andor Solis / any Andor GUI before running.
"""

import os
import time
import argparse

from matisse_controller.shamrock_ple.ple import PLE, ccd, shamrock
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

    global ccd, shamrock
    if ccd is None:
        raise RuntimeError("CCD global is None after load_andor_libs().")
    if shamrock is None:
        print("Warning: Shamrock global is None (spectrograph not required for pure CCD kinetics test).")

    # Keep cooling on across shutdowns
    ccd.ensure_cooling(args.temp, persist_on_shutdown=True)

    # Optional wait for cold (slow)
    if args.wait:
        print(f"Waiting until CCD <= {args.temp}+{args.tol} C (timeout={args.wait_timeout}s)")
        ccd.wait_to_cooldown(target_C=args.temp, tol_C=args.tol, poll_s=5.0, timeout_s=args.wait_timeout)
    else:
        t_now = ccd.wait_until_cold(args.temp, tol_C=args.tol, timeout_s=0.0)
        print(f"Temp check (non-blocking): current CCD temp = {t_now:.1f} C")

    readout_mode = parse_readout_mode(args.readout)
    cosmic_mode = parse_cosmic(args.cosmic)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Configure kinetics (do NOT block on cooldown again)
    print("\nConfiguring kinetics...")
    exp_actual, cycle_actual = ccd.setup_kinetics(
        exposure_time=args.exp,
        cycle_time=args.cycle,
        n_frames=args.n,
        readout_mode=readout_mode,
        temperature=args.temp,
        cool_down=False,                # <-- important: do not wait here
        cosmic_ray_filter=cosmic_mode,
    )

    print(f"Kinetics configured: exp_actual={exp_actual:.6f}s, cycle_actual={cycle_actual:.6f}s")
    est = args.n * cycle_actual
    print(f"Estimated duration ~ {est:.3f}s for {args.n} frames")

    # Start acquisition
    print("\nStarting acquisition...")
    t0 = time.time()
    ccd.start_acquisition()
    ccd.wait_for_acquisition()
    t1 = time.time()
    print(f"Acquisition finished in {t1 - t0:.3f}s")

    # Save SIF (Andor SDK writes into current working dir if given a relative path)
    out_path = os.path.abspath(args.out)
    out_name = os.path.basename(out_path)

    print(f"\nSaving SIF to {out_path}")
    # SaveAsSif writes where filename points; some SDK builds require CWD write perms.
    # Use a temp in CWD then move, for robustness:
    tmp_name = out_name
    ccd.save_as_sif(tmp_name)

    # Move to desired output folder if needed
    if os.path.abspath(tmp_name) != out_path:
        if os.path.exists(out_path):
            os.remove(out_path)
        os.replace(tmp_name, out_path)

    temp_after = ccd.get_temperature()
    print(f"CCD temperature after acquisition: {temp_after:.1f} C")

    print("\nDONE. If you see a valid .sif and no SDK errors, kinetics path is good.")

    # Important: do NOT force cooler off here
    # Clean up globals (will call shutdown; our CCD.shutdown honors keep_cooler_on if set)
    try:
        PLE.clean_up_globals()
    except Exception:
        pass


if __name__ == "__main__":
    main()
