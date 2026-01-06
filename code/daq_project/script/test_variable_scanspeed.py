#!/usr/bin/env python3
"""
Test Matisse scanning + WS7 readout (AIR) with:
  1) "fixed" start wavelength: lock/stabilize at start_nm first
  2) variable scan speed: piecewise scan segments, each with its own speed

Examples:

# Fast pre-lock (assumes you're already close and laser can stabilize quickly):
python script/test_matisse_scan_ws7.py --start 739.300 --tol 0.002 --prelock stabilize \
  --segments "739.310,0.002;739.330,0.010;739.340,0.001"

# Full pre-lock (calls matisse.set_wavelength(start), can take longer):
python script/test_matisse_scan_ws7.py --start 739.300 --tol 0.002 --prelock full \
  --segments "739.310,0.002;739.330,0.010"

# Relative segments (delta_nm,speed), e.g. +0.010 then +0.020:
python script/test_matisse_scan_ws7.py --start 739.300 --relative --prelock stabilize \
  --segments "0.010,0.002;0.030,0.010"
"""

import sys
import time
import argparse
from pathlib import Path
import ctypes
from typing import List, Tuple, Optional


# ---------------------------
# Robust sys.path injection
# ---------------------------
HERE = Path(__file__).resolve()
root_funcs = None
root_matisse = None

for p in [HERE.parent] + list(HERE.parents):
    if root_funcs is None and (p / "functions26").exists():
        root_funcs = p
    if root_matisse is None and (p / "matisse_controller").exists():
        root_matisse = p
    if root_funcs and root_matisse:
        break

if root_funcs is None:
    raise RuntimeError("Could not find 'functions26' in any parent directory.")
if root_matisse is None:
    raise RuntimeError("Could not find 'matisse_controller' in any parent directory.")

sys.path.insert(0, str(root_funcs))
sys.path.insert(0, str(root_matisse))


# ---------------------------
# Imports (after sys.path)
# ---------------------------
from functions26.instruments.ws7 import WS7  # noqa: E402
from matisse_controller.matisse.matisse import Matisse  # noqa: E402


# wlmConst snippet equivalents
cReturnWavelengthVac = 0
cReturnWavelengthAir = 1


ERR_DICT = {
    0: "ErrNoValue",
    -1: "ErrNoSignal",
    -2: "ErrBadSignal",
    -3: "ErrLowSignal",
    -4: "ErrBigSignal",
    -5: "ErrWimMissing",
    -6: "ErrNotAvailable",
    -7: "InfNothingChanged",
    -8: "ErrNoPulse",
    -10: "ErrChannelNotAvailable",
    -13: "ErrDiv0",
    -14: "ErrOutOfRange",
    -15: "ErrUnitNotAvailable",
}


def bind_ws7_prototypes(lib) -> None:
    """Make ctypes return correct floats so ConvertUnit doesn't look like a no-op."""
    if hasattr(lib, "GetWavelength"):
        lib.GetWavelength.argtypes = [ctypes.c_double]
        lib.GetWavelength.restype = ctypes.c_double
    if hasattr(lib, "GetWavelength2"):
        lib.GetWavelength2.argtypes = [ctypes.c_double]
        lib.GetWavelength2.restype = ctypes.c_double
    if hasattr(lib, "ConvertUnit"):
        lib.ConvertUnit.argtypes = [ctypes.c_double, ctypes.c_long, ctypes.c_long]
        lib.ConvertUnit.restype = ctypes.c_double


def ws7_read_air_nm(ws7: WS7, ch2: bool = False) -> float:
    """Read WS7 wavelength in AIR nm using ConvertUnit(vac->air). Returns <=0 on error."""
    lib = ws7.lib
    if ch2 and hasattr(lib, "GetWavelength2"):
        raw = float(lib.GetWavelength2(0.0))
    else:
        raw = float(lib.GetWavelength(0.0))

    if raw <= 0:
        return raw

    if not hasattr(lib, "ConvertUnit"):
        raise RuntimeError("WS7 DLL missing ConvertUnit(); cannot convert vac->air.")
    air = float(lib.ConvertUnit(raw, cReturnWavelengthVac, cReturnWavelengthAir))
    return air


def parse_segments(s: str) -> List[Tuple[float, float]]:
    """
    Parse: "end_nm,speed;end_nm,speed;..."
    Returns list[(end_value, speed_nm_per_s)]
    """
    out: List[Tuple[float, float]] = []
    if not s.strip():
        return out
    parts = [x.strip() for x in s.split(";") if x.strip()]
    for p in parts:
        a, b = [x.strip() for x in p.split(",")]
        out.append((float(a), float(b)))
    return out


def wait_until_close(ws7: WS7, target_nm: float, tol_nm: float, timeout_s: float, dt_s: float, ch2: bool) -> float:
    """Wait until |wl-target| <= tol; returns last wl."""
    t0 = time.monotonic()
    last = float("nan")
    while True:
        wl = ws7_read_air_nm(ws7, ch2=ch2)
        last = wl
        if wl > 0 and abs(wl - target_nm) <= tol_nm:
            return wl
        if (time.monotonic() - t0) >= timeout_s:
            return last
        time.sleep(dt_s)


def set_scan_speed(matisse: Matisse, speed_nm_s: float) -> None:
    """Set both rising/falling speeds to same magnitude."""
    sp = abs(float(speed_nm_s))
    # Many controllers accept float formatting; keep plenty of precision
    matisse.query(f"SCAN:RISINGSPEED {sp:.12f}")
    matisse.query(f"SCAN:FALLINGSPEED {sp:.12f}")


def start_scan_to_target(
    matisse: Matisse,
    ws7: WS7,
    start_nm: float,
    end_nm: float,
    speed_nm_s: float,
    tol_nm: float,
    poll_dt_s: float,
    ch2: bool,
    verbose_every: float = 0.25,
) -> None:
    """
    Perform ONE scan segment:
      - set scan speed
      - set target wavelength (for later stabilization + for user logs)
      - start scan in correct direction
      - monitor WS7 until reached end (by inequality with tol)
      - stop scan
    """
    if speed_nm_s <= 0:
        raise ValueError("speed_nm_s must be > 0")

    direction = 0 if (end_nm - start_nm) >= 0 else 1  # 0=up, 1=down

    set_scan_speed(matisse, speed_nm_s)
    matisse.target_wavelength = float(end_nm)

    print(f"\nSegment: {start_nm:.6f} -> {end_nm:.6f}  speed={abs(speed_nm_s):.6f} nm/s  dir={direction}")

    # make sure stabilization isn't running during scan
    try:
        matisse.stabilize_off()
    except Exception:
        pass

    # start scan
    matisse.start_scan(direction)

    t0 = time.monotonic()
    t_last_print = -1e9
    while True:
        wl = ws7_read_air_nm(ws7, ch2=ch2)
        t = time.monotonic() - t0

        if (time.monotonic() - t_last_print) >= verbose_every:
            print(f"  t={t:7.2f}s  WS7_air={wl if wl > 0 else None}")
            t_last_print = time.monotonic()

        # reached condition (with tolerance)
        if wl > 0:
            if direction == 0 and wl >= (end_nm - tol_nm):
                break
            if direction == 1 and wl <= (end_nm + tol_nm):
                break

        time.sleep(poll_dt_s)

    # stop scan
    matisse.stop_scan()
    wl_end = ws7_read_air_nm(ws7, ch2=ch2)
    print(f"Reached end: WS7_air={wl_end:.9f} nm (target {end_nm:.9f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, required=True, help="Start wavelength (nm, AIR) you want to align to.")
    ap.add_argument("--tol", type=float, default=0.002, help="Tolerance (nm) for 'start reached' and segment end.")
    ap.add_argument("--timeout", type=float, default=120.0, help="Timeout (s) for initial alignment.")
    ap.add_argument("--dt", type=float, default=0.05, help="Polling dt for WS7 (s).")
    ap.add_argument("--ch2", action="store_true", help="Use WS7 channel 2 if supported.")
    ap.add_argument("--prelock", choices=["none", "stabilize", "full"], default="stabilize",
                    help="How to fix beginning wavelength: "
                         "none=do nothing; stabilize=lock+stabilize only; full=call set_wavelength(start).")
    ap.add_argument("--segments", type=str, required=True,
                    help="Scan plan: 'end_nm,speed;end_nm,speed;...' OR if --relative, 'delta_nm,speed;...'")
    ap.add_argument("--relative", action="store_true",
                    help="Interpret segment endpoints as delta from START (cumulative).")
    args = ap.parse_args()

    start_nm = float(args.start)
    tol_nm = float(args.tol)

    # Connect WS7 (air conversion via ConvertUnit)
    print("Connecting WS7...")
    ws7 = WS7()
    bind_ws7_prototypes(ws7.lib)
    wl0 = ws7_read_air_nm(ws7, ch2=args.ch2)
    print(f"WS7 OK. Current WS7_air={wl0 if wl0 > 0 else None}")

    # Connect Matisse controller
    print("\nConnecting Matisse...")
    matisse = Matisse(wavemeter_type="WS7")  # uses your existing config device_id internally
    print("Matisse OK.")
    print("Laser locked?:", getattr(matisse, "laser_locked", lambda: None)())

    # Safety: stop any running scan
    try:
        if matisse.is_scanning():
            matisse.stop_scan()
    except Exception:
        pass

    # -----------------------
    # Pre-lock / align start
    # -----------------------
    print(f"\nPre-lock mode: {args.prelock}")
    if args.prelock == "full":
        # Full set_wavelength routine (can take time)
        print(f"Calling matisse.set_wavelength({start_nm:.6f}) ...")
        matisse.set_wavelength(start_nm)

    elif args.prelock == "stabilize":
        # Fast path: try to ensure locked + stabilization on at start
        matisse.target_wavelength = start_nm

        # Start lock correction if available and not running
        try:
            if hasattr(matisse, "is_lock_correction_on") and hasattr(matisse:="dummy","x"):  # placeholder
                pass
        except Exception:
            pass

        try:
            # If lock correction isn't on, start it (helps ensure laser_locked)
            if hasattr(matisse, "is_lock_correction_on") and hasattr(matisse, "start_laser_lock_correction"):
                if not matisse.is_lock_correction_on():
                    matisse.start_laser_lock_correction()
        except Exception:
            pass

        # Start stabilization toward start
        try:
            matisse.stabilize_on()
        except Exception:
            pass

    elif args.prelock == "none":
        matisse.target_wavelength = start_nm

    print(f"\nWaiting until WS7_air ~ start ({start_nm:.6f} nm) within ±{tol_nm:.6f} nm ...")
    wl_start = wait_until_close(ws7, start_nm, tol_nm, timeout_s=args.timeout, dt_s=args.dt, ch2=args.ch2)
    if wl_start > 0 and abs(wl_start - start_nm) <= tol_nm:
        print(f"Start aligned: WS7_air={wl_start:.9f} nm")
    else:
        print(f"Proceeding anyway (didn't reach tol within {args.timeout:.1f}s). Last WS7_air={wl_start}")

    # Turn OFF stabilization during scanning to avoid fighting SCAN:STATUS RUN
    try:
        matisse.stabilize_off()
    except Exception:
        pass

    # -----------------------
    # Build scan plan
    # -----------------------
    segs = parse_segments(args.segments)
    if not segs:
        raise ValueError("No segments parsed. Example: --segments '739.310,0.002;739.330,0.010'")

    # Convert relative to absolute if requested (cumulative from start)
    plan: List[Tuple[float, float]] = []
    if args.relative:
        # endpoints are cumulative deltas from start, e.g. 0.01 then 0.03 means end at start+0.03 on 2nd segment
        for delta, sp in segs:
            plan.append((start_nm + float(delta), float(sp)))
    else:
        plan = [(float(end), float(sp)) for end, sp in segs]

    # -----------------------
    # Execute segments
    # -----------------------
    cur = ws7_read_air_nm(ws7, ch2=args.ch2)
    if not (cur > 0):
        cur = start_nm

    print("\n=== Running scan plan ===")
    print(f"Start (measured): {cur:.9f} nm (target start {start_nm:.9f})")
    for i, (end_nm, sp) in enumerate(plan, start=1):
        # Use current WS7 as "start" for the segment direction logic
        seg_start = ws7_read_air_nm(ws7, ch2=args.ch2)
        if not (seg_start > 0):
            seg_start = cur

        print(f"\n--- Segment {i}/{len(plan)} ---")
        start_scan_to_target(
            matisse=matisse,
            ws7=ws7,
            start_nm=seg_start,
            end_nm=end_nm,
            speed_nm_s=sp,
            tol_nm=tol_nm,
            poll_dt_s=args.dt,
            ch2=args.ch2,
            verbose_every=0.25,
        )
        cur = ws7_read_air_nm(ws7, ch2=args.ch2)

    # -----------------------
    # Optional: re-stabilize at start
    # -----------------------
    print("\nRe-stabilizing at start wavelength (optional)...")
    try:
        matisse.target_wavelength = start_nm
        matisse.stabilize_on()
        wl_back = wait_until_close(ws7, start_nm, tol_nm, timeout_s=30.0, dt_s=args.dt, ch2=args.ch2)
        print(f"After re-stabilize: WS7_air={wl_back}")
    except Exception:
        pass

    print("\nDONE")


if __name__ == "__main__":
    main()
