#!/usr/bin/env python3
"""
Test Matisse scanning + WS7 readout (AIR) with:
  1) fixed start wavelength: SEEK to start using SCAN, then stabilize
  2) variable scan speed: piecewise scan segments, each with its own speed

Examples:

# Absolute endpoints:
python script/test_matisse_scan_ws7.py --start 739.300 --tol 0.002 --prelock stabilize \
  --segments "739.310,0.002;739.330,0.010;739.340,0.001"

# Relative endpoints (cumulative delta from start):
python script/test_matisse_scan_ws7.py --start 739.300 --tol 0.002 --relative --prelock stabilize \
  --segments "0.010,0.002;0.030,0.010"
"""

import sys
import time
import argparse
from pathlib import Path
import ctypes
from typing import List, Tuple


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


from functions26.instruments.ws7 import WS7  # noqa: E402
from matisse_controller.matisse.matisse import Matisse  # noqa: E402


# wlmConst values
cReturnWavelengthVac = 0
cReturnWavelengthAir = 1


def bind_ws7_prototypes(lib) -> None:
    """Make ctypes return correct float types so ConvertUnit isn't silently broken."""
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
    """Read WS7 wavelength in AIR (nm). Returns <=0 on error codes."""
    lib = ws7.lib
    raw = float(lib.GetWavelength2(0.0)) if (ch2 and hasattr(lib, "GetWavelength2")) else float(lib.GetWavelength(0.0))
    if raw <= 0:
        return raw
    if not hasattr(lib, "ConvertUnit"):
        raise RuntimeError("WS7 DLL missing ConvertUnit(); cannot convert vac->air.")
    return float(lib.ConvertUnit(raw, cReturnWavelengthVac, cReturnWavelengthAir))


def parse_segments(s: str) -> List[Tuple[float, float]]:
    """
    Parse: "end_nm,speed;end_nm,speed;..."
    Returns list[(end_value, speed_nm_per_s)]
    """
    out: List[Tuple[float, float]] = []
    parts = [x.strip() for x in s.split(";") if x.strip()]
    for p in parts:
        a, b = [x.strip() for x in p.split(",")]
        out.append((float(a), float(b)))
    return out


def set_scan_speed(matisse: Matisse, speed_nm_s: float) -> None:
    sp = abs(float(speed_nm_s))
    if sp <= 0:
        raise ValueError("scan speed must be > 0")
    matisse.query(f"SCAN:RISINGSPEED {sp:.12f}")
    matisse.query(f"SCAN:FALLINGSPEED {sp:.12f}")


def seek_to_wavelength(
    matisse: Matisse,
    ws7: WS7,
    target_nm: float,
    tol_nm: float,
    seek_speed_nm_s: float,
    timeout_s: float,
    poll_dt_s: float,
    ch2: bool,
) -> float:
    """
    Actively move wavelength toward target using SCAN (up/down) until within tol or timeout.
    Returns last WS7_air nm.
    """
    # Don't let stabilization fight us
    try:
        matisse.stabilize_off()
    except Exception:
        pass

    wl0 = ws7_read_air_nm(ws7, ch2=ch2)
    if wl0 > 0 and abs(wl0 - target_nm) <= tol_nm:
        return wl0

    set_scan_speed(matisse, seek_speed_nm_s)
    matisse.target_wavelength = float(target_nm)

    # Decide scan direction based on current reading
    wl_now = wl0
    if not (wl_now > 0):
        # if WS7 invalid, default to "up"
        direction = 0
    else:
        direction = 0 if (target_nm - wl_now) > 0 else 1  # 0 up, 1 down

    print(f"Seeking start via SCAN dir={direction} at {abs(seek_speed_nm_s):.6f} nm/s "
          f"(now={wl_now if wl_now>0 else None}, target={target_nm:.6f}±{tol_nm:.6f})")

    matisse.start_scan(direction)

    t0 = time.monotonic()
    last = wl_now
    try:
        while True:
            last = ws7_read_air_nm(ws7, ch2=ch2)
            if last > 0 and abs(last - target_nm) <= tol_nm:
                break
            if (time.monotonic() - t0) >= timeout_s:
                break
            time.sleep(poll_dt_s)
    finally:
        try:
            matisse.stop_scan()
        except Exception:
            pass

    return last


def start_scan_to_target(
    matisse: Matisse,
    ws7: WS7,
    end_nm: float,
    speed_nm_s: float,
    tol_nm: float,
    poll_dt_s: float,
    ch2: bool,
    verbose_every_s: float = 0.25,
) -> None:
    """
    One scan segment: set speed, start scan, watch WS7 until we hit end, stop scan.
    """
    wl_start = ws7_read_air_nm(ws7, ch2=ch2)
    direction = 0 if (end_nm - wl_start) >= 0 else 1

    # Make sure stabilization isn't running during scan
    try:
        matisse.stabilize_off()
    except Exception:
        pass

    set_scan_speed(matisse, speed_nm_s)
    matisse.target_wavelength = float(end_nm)

    print(f"\nSegment: start~{wl_start:.6f} -> {end_nm:.6f}  speed={abs(speed_nm_s):.6f} nm/s  dir={direction}")

    matisse.start_scan(direction)

    t0 = time.monotonic()
    t_last = -1e9
    try:
        while True:
            wl = ws7_read_air_nm(ws7, ch2=ch2)
            now = time.monotonic()
            if (now - t_last) >= verbose_every_s:
                print(f"  t={now-t0:7.2f}s  WS7_air={wl if wl>0 else None}")
                t_last = now

            if wl > 0:
                if direction == 0 and wl >= (end_nm - tol_nm):
                    break
                if direction == 1 and wl <= (end_nm + tol_nm):
                    break

            time.sleep(poll_dt_s)
    finally:
        matisse.stop_scan()

    wl_end = ws7_read_air_nm(ws7, ch2=ch2)
    print(f"Reached end: WS7_air={wl_end:.9f} nm (target {end_nm:.9f})")

def safe_stop_scan(matisse):
    try:
        if matisse is not None:
            # If your Matisse wrapper has stop_scan():
            if hasattr(matisse, "stop_scan"):
                matisse.stop_scan()
            else:
                matisse.query("SCAN:STATUS STOP")
    except Exception as e:
        print(f"(warn) failed to stop scan: {e}")


def main():
    try:
        # run all segments
        ap = argparse.ArgumentParser()
        ap.add_argument("--start", type=float, required=True, help="Start wavelength to align to (nm, AIR).")
        ap.add_argument("--tol", type=float, default=0.002, help="Tolerance (nm) for start and segment ends.")
        ap.add_argument("--dt", type=float, default=0.05, help="WS7 poll dt (s).")
        ap.add_argument("--ch2", action="store_true", help="Use WS7 channel 2 if supported.")
        ap.add_argument("--prelock", choices=["none", "stabilize", "full"], default="stabilize",
                        help="none=do nothing; stabilize=seek then stabilize; full=call set_wavelength(start)")
        ap.add_argument("--seek_speed", type=float, default=0.01, help="nm/s used to SEEK to start wavelength.")
        ap.add_argument("--seek_timeout", type=float, default=60.0, help="seconds allowed to SEEK to start.")
        ap.add_argument("--segments", type=str, required=True,
                        help="Scan plan: 'end_nm,speed;end_nm,speed;...' (or delta,speed with --relative)")
        ap.add_argument("--relative", action="store_true",
                        help="Interpret segment endpoints as cumulative delta from start.")
        args = ap.parse_args()

        start_nm = float(args.start)
        tol_nm = float(args.tol)

        # WS7
        print("Connecting WS7...")
        ws7 = WS7()
        bind_ws7_prototypes(ws7.lib)
        wl0 = ws7_read_air_nm(ws7, ch2=args.ch2)
        print(f"WS7 OK. Current WS7_air={wl0 if wl0 > 0 else None}")

        # Matisse
        print("\nConnecting Matisse...")
        matisse = Matisse(wavemeter_type="WS7")
        print("Matisse OK.")
        try:
            print("Laser locked?:", matisse.laser_locked())
        except Exception:
            print("Laser locked?: (unknown)")

        # Stop any running scan
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
            print(f"Calling matisse.set_wavelength({start_nm:.6f}) ...")
            matisse.set_wavelength(start_nm)

        elif args.prelock == "stabilize":
            # Ensure lock correction if your class supports it (optional)
            try:
                if hasattr(matisse, "is_lock_correction_on") and hasattr(matisse, "start_laser_lock_correction"):
                    if not matisse.is_lock_correction_on():
                        matisse.start_laser_lock_correction()
            except Exception:
                pass

            # SEEK if not within tolerance
            wl_now = ws7_read_air_nm(ws7, ch2=args.ch2)
            if wl_now > 0 and abs(wl_now - start_nm) > tol_nm:
                wl_seek = seek_to_wavelength(
                    matisse=matisse,
                    ws7=ws7,
                    target_nm=start_nm,
                    tol_nm=tol_nm,
                    seek_speed_nm_s=args.seek_speed,
                    timeout_s=args.seek_timeout,
                    poll_dt_s=args.dt,
                    ch2=args.ch2,
                )
                print(f"Seek result: WS7_air={wl_seek if wl_seek > 0 else None}")
            else:
                print("Already within tolerance; skipping SEEK.")

            # Now stabilize at start
            try:
                matisse.target_wavelength = start_nm
                matisse.stabilize_on()
            except Exception:
                pass

        else:
            # none
            matisse.target_wavelength = start_nm

        wl_start = ws7_read_air_nm(ws7, ch2=args.ch2)
        print(f"\nStart check: WS7_air={wl_start if wl_start > 0 else None} (target {start_nm:.6f}±{tol_nm:.6f})")

        # Turn OFF stabilization during scanning
        try:
            matisse.stabilize_off()
        except Exception:
            pass

        # -----------------------
        # Build scan plan
        # -----------------------
        segs = parse_segments(args.segments)
        if not segs:
            raise ValueError("No segments parsed.")

        if args.relative:
            plan = [(start_nm + d, sp) for (d, sp) in segs]
        else:
            plan = [(end, sp) for (end, sp) in segs]

        # -----------------------
        # Execute segments
        # -----------------------
        print("\n=== Running scan plan ===")
        for i, (end_nm, sp) in enumerate(plan, start=1):
            print(f"\n--- Segment {i}/{len(plan)} ---")
            start_scan_to_target(
                matisse=matisse,
                ws7=ws7,
                end_nm=end_nm,
                speed_nm_s=sp,
                tol_nm=tol_nm,
                poll_dt_s=args.dt,
                ch2=args.ch2,
            )

        # Optional: re-stabilize at start
        print("\nRe-stabilizing at start (optional)...")
        try:
            matisse.target_wavelength = start_nm
            matisse.stabilize_on()
            time.sleep(1.0)
        except Exception:
            pass

        print("\nDONE")

finally:
        safe_stop_scan(matisse)


if __name__ == "__main__":
    main()
