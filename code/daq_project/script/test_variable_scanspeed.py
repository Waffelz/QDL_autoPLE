#!/usr/bin/env python3
"""
test_variable_scanspeed.py

Test Matisse scanning + WS7 readout (AIR) with:
  1) fixed start wavelength: SEEK to start using SCAN (with auto-reverse + slow approach), then stabilize
  2) variable scan speed: piecewise scan segments, each with its own speed

Examples:

# Absolute endpoints:
python script/test_variable_scanspeed.py --start 739.300 --tol 0.002 --prelock stabilize \
  --segments "739.310,0.002;739.312,0.010;739.315,0.001"

# Relative endpoints (cumulative delta from start):
python script/test_variable_scanspeed.py --start 739.300 --tol 0.002 --relative --prelock stabilize \
  --segments "0.010,0.002;0.030,0.010"

# Tight start tol + slow seek near target:
python script/test_variable_scanspeed.py --start 739.308 --tol 0.00002 --prelock stabilize \
  --seek_speed 0.010 --seek_ramp 0.00020 --seek_min_speed 0.00050 \
  --segments "739.310,0.002;739.312,0.0005;739.315,0.001"
"""

import sys
import time
import math
import argparse
from pathlib import Path
import ctypes
from typing import List, Tuple, Callable, Optional


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

# Common WS7 error codes (optional)
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
    """Bind ctypes prototypes so ConvertUnit/GetWavelength return correct float types."""
    if hasattr(lib, "GetWavelength"):
        lib.GetWavelength.argtypes = [ctypes.c_double]
        lib.GetWavelength.restype = ctypes.c_double
    if hasattr(lib, "GetWavelength2"):
        lib.GetWavelength2.argtypes = [ctypes.c_double]
        lib.GetWavelength2.restype = ctypes.c_double
    if hasattr(lib, "ConvertUnit"):
        lib.ConvertUnit.argtypes = [ctypes.c_double, ctypes.c_long, ctypes.c_long]
        lib.ConvertUnit.restype = ctypes.c_double


def read_wavelength_nm(
    ws7: WS7,
    medium: str = "air",
    ch2: bool = False,
    max_err_print: int = 0,
) -> float:
    """
    Read wavelength in nm.
    - raw is vacuum wavelength from GetWavelength*()
    - air uses ConvertUnit(vac -> air)
    Returns <=0 on error (WS7 convention).
    """
    lib = ws7.lib

    if ch2 and hasattr(lib, "GetWavelength2"):
        raw = float(lib.GetWavelength2(0.0))
    else:
        raw = float(lib.GetWavelength(0.0))

    if raw <= 0:
        if max_err_print > 0:
            msg = ERR_DICT.get(int(raw), f"UnknownError({raw})")
            print(f"WS7 error: {raw} ({msg})")
        return raw

    if medium == "vac":
        return raw

    if not hasattr(lib, "ConvertUnit"):
        raise RuntimeError("WS7 DLL has no ConvertUnit() - cannot convert vac->air.")
    return float(lib.ConvertUnit(raw, cReturnWavelengthVac, cReturnWavelengthAir))


def make_ws7_air_reader(ws7: WS7, ch2: bool, err_print_budget: int = 3) -> Callable[[], float]:
    """
    Returns a zero-arg callable that reads WS7 AIR wavelength.
    Prints up to `err_print_budget` error messages total.
    """
    budget = {"n": int(err_print_budget)}

    def _read() -> float:
        n = budget["n"]
        wl = read_wavelength_nm(ws7, medium="air", ch2=ch2, max_err_print=n)
        if wl <= 0 and budget["n"] > 0:
            budget["n"] -= 1
        return wl

    return _read


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


def safe_stop_scan(matisse: Optional[Matisse]) -> None:
    try:
        if matisse is None:
            return
        if hasattr(matisse, "stop_scan"):
            matisse.stop_scan()
        else:
            matisse.query("SCAN:STATUS STOP")
    except Exception as e:
        print(f"(warn) failed to stop scan: {e}")


def set_scan_speed(matisse: Matisse, speed_nm_s: float) -> float:
    sp = abs(float(speed_nm_s))
    if sp <= 0:
        raise ValueError("scan speed must be > 0")
    matisse.query(f"SCAN:RISINGSPEED {sp:.12f}")
    matisse.query(f"SCAN:FALLINGSPEED {sp:.12f}")
    return sp


def choose_speed(abs_err_nm: float, base_speed_nm_s: float, ramp_window_nm: float, min_speed_nm_s: float) -> float:
    """
    Speed ramp-down near target:
      - abs_err >= ramp_window -> base_speed
      - abs_err <  ramp_window -> base_speed * (abs_err/ramp_window), floored at min_speed
    """
    base = abs(float(base_speed_nm_s))
    mn = abs(float(min_speed_nm_s))
    w = abs(float(ramp_window_nm))

    if base <= 0:
        base = 0.001
    if mn <= 0:
        mn = min(base, 1e-4)
    if w <= 0:
        return max(mn, base)

    if abs_err_nm >= w:
        return max(mn, base)

    sp = base * (abs_err_nm / w)
    return max(mn, min(base, sp))


def start_scan_dir(matisse: Matisse, direction: int) -> None:
    """
    Start scan in given direction (0 up, 1 down).
    We stop first to avoid weird behavior when changing direction.
    """
    safe_stop_scan(matisse)
    matisse.start_scan(int(direction))


def seek_to_wavelength(
    matisse: Matisse,
    read_wl_nm: Callable[[], float],
    target_nm: float,
    tol_nm: float,
    base_speed_nm_s: float = 0.010,
    ramp_window_nm: float = 0.00020,
    min_speed_nm_s: float = 0.00050,
    poll_dt_s: float = 0.05,
    max_time_s: float = 60.0,
    verbose: bool = True,
) -> float:
    """
    Actively move wavelength toward target using SCAN (up/down) until within tol or timeout.
    - Skips invalid WS7 readings (<=0 / nan)
    - Slows down near target (ramp_window_nm) and floors at min_speed_nm_s
    - If it passes target (sign flip), auto-reverses direction.

    Returns last valid WS7_air wavelength (or last read, possibly invalid if nothing valid).
    """
    # Ensure stabilization isn't fighting us
    try:
        matisse.stabilize_off()
    except Exception:
        pass

    t0 = time.monotonic()
    last_valid = float("nan")

    # Get an initial valid reading
    wl = read_wl_nm()
    for _ in range(50):
        if wl > 0 and (not math.isnan(wl)):
            last_valid = wl
            break
        if (time.monotonic() - t0) > min(2.0, max_time_s):
            break
        time.sleep(poll_dt_s)
        wl = read_wl_nm()

    if not math.isnan(last_valid):
        if abs(last_valid - target_nm) <= tol_nm:
            if verbose:
                print(f"Seek: already within tol. WS7_air={last_valid:.12f}")
            return last_valid

    # Decide initial direction from last_valid if available, else default up
    if math.isnan(last_valid):
        direction = 0
        err = float("inf")
    else:
        err = target_nm - last_valid
        direction = 0 if err > 0 else 1

    flips = 0
    last_speed = None

    if verbose:
        now_str = f"{last_valid:.12f}" if not math.isnan(last_valid) else "None"
        print(
            f"Seeking start via SCAN dir={direction} "
            f"(now={now_str}, target={target_nm:.6f}±{tol_nm:.6f})"
        )

    # Start scanning
    start_scan_dir(matisse, direction)

    try:
        while True:
            if (time.monotonic() - t0) >= max_time_s:
                if verbose:
                    print("Seek timeout reached.")
                break

            wl = read_wl_nm()
            if not (wl > 0) or math.isnan(wl):
                time.sleep(poll_dt_s)
                continue

            last_valid = wl
            err = target_nm - wl
            abs_err = abs(err)

            if abs_err <= tol_nm:
                if verbose:
                    print(f"Seek reached tol: WS7_air={wl:.12f}")
                break

            desired_dir = 0 if err > 0 else 1

            # If we've crossed the target, reverse direction
            if desired_dir != direction:
                flips += 1
                direction = desired_dir
                if verbose:
                    print(f"Seek overshoot -> reversing dir={direction} (WS7_air={wl:.12f}, err={err:+.6e} nm)")
                start_scan_dir(matisse, direction)

            # Ramp speed near target
            sp = choose_speed(abs_err, base_speed_nm_s, ramp_window_nm, min_speed_nm_s)

            # Avoid spamming VISA: update only if speed changes meaningfully
            if (last_speed is None) or (abs(sp - last_speed) / max(last_speed, 1e-12) > 0.25):
                set_scan_speed(matisse, sp)
                last_speed = sp
                if verbose:
                    print(f"  seek speed -> {sp:.6f} nm/s  (abs_err={abs_err:.6e} nm)")

            # Safety: if we are flip-flopping too much, exit
            if flips > 20:
                print("(warn) too many direction flips during seek; stopping.")
                break

            time.sleep(poll_dt_s)

    finally:
        safe_stop_scan(matisse)

    return float(last_valid) if not math.isnan(last_valid) else float(wl)


def start_scan_to_target(
    matisse: Matisse,
    read_wl_nm: Callable[[], float],
    end_nm: float,
    speed_nm_s: float,
    tol_nm: float,
    poll_dt_s: float,
    verbose_every_s: float = 0.25,
    max_time_s: Optional[float] = None,
) -> float:
    """
    One scan segment: set speed, start scan, watch WS7 until we hit end, stop scan.
    Skips invalid WS7 readings.
    Returns last valid wavelength at segment end (AIR nm).
    """
    # Ensure stabilization isn't running during scan
    try:
        matisse.stabilize_off()
    except Exception:
        pass

    # Determine initial direction from current valid wl
    t_start = time.monotonic()
    wl0 = read_wl_nm()
    while (not (wl0 > 0) or math.isnan(wl0)) and (time.monotonic() - t_start) < 2.0:
        time.sleep(poll_dt_s)
        wl0 = read_wl_nm()

    if not (wl0 > 0) or math.isnan(wl0):
        # if no valid reading, guess direction from target
        direction = 0
        wl0_disp = None
    else:
        direction = 0 if (end_nm - wl0) >= 0 else 1
        wl0_disp = wl0

    sp = set_scan_speed(matisse, speed_nm_s)
    matisse.target_wavelength = float(end_nm)

    if max_time_s is None:
        # generous time limit: 2x ideal travel time + 10s
        if wl0_disp is not None:
            ideal = abs(end_nm - wl0_disp) / max(sp, 1e-12)
        else:
            ideal = 30.0
        max_time_s = 2.0 * ideal + 10.0

    print(
        f"\nSegment: start~{wl0_disp if wl0_disp is not None else 'None'} -> {end_nm:.6f}  "
        f"speed={sp:.6f} nm/s  dir={direction}  (max_time~{max_time_s:.1f}s)"
    )

    start_scan_dir(matisse, direction)

    last_valid = wl0_disp if wl0_disp is not None else float("nan")
    t0 = time.monotonic()
    t_last_print = -1e9

    try:
        while True:
            if (time.monotonic() - t0) >= float(max_time_s):
                print("(warn) segment timed out; stopping scan.")
                break

            wl = read_wl_nm()
            now = time.monotonic()

            if wl > 0 and (not math.isnan(wl)):
                last_valid = wl

            if (now - t_last_print) >= verbose_every_s:
                show = last_valid if not math.isnan(last_valid) else (wl if (wl > 0 and not math.isnan(wl)) else None)
                print(f"  t={now-t0:7.2f}s  WS7_air={show}")
                t_last_print = now

            if not math.isnan(last_valid):
                if direction == 0 and last_valid >= (end_nm - tol_nm):
                    break
                if direction == 1 and last_valid <= (end_nm + tol_nm):
                    break

            time.sleep(poll_dt_s)

    finally:
        safe_stop_scan(matisse)

    wl_end = last_valid if not math.isnan(last_valid) else read_wl_nm()
    print(f"Reached end: WS7_air={wl_end if (wl_end > 0 and not math.isnan(wl_end)) else None}  (target {end_nm:.9f})")
    return float(wl_end)


def main():
    matisse: Optional[Matisse] = None
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument("--start", type=float, required=True, help="Start wavelength to align to (nm, AIR).")
        ap.add_argument("--tol", type=float, default=0.002, help="Tolerance (nm) for start and segment ends.")
        ap.add_argument("--dt", type=float, default=0.05, help="WS7 poll dt (s).")
        ap.add_argument("--ch2", action="store_true", help="Use WS7 channel 2 if supported.")

        ap.add_argument(
            "--prelock",
            choices=["none", "stabilize", "full"],
            default="stabilize",
            help="none=do nothing; stabilize=seek then stabilize; full=call set_wavelength(start)",
        )

        # SEEK controls (new)
        ap.add_argument("--seek_speed", type=float, default=0.010, help="Base nm/s used to SEEK toward start.")
        ap.add_argument("--seek_min_speed", type=float, default=0.00050, help="Minimum nm/s near target during SEEK.")
        ap.add_argument("--seek_ramp", type=float, default=0.00020, help="Ramp window (nm) for SEEK speed slowdown.")
        ap.add_argument("--seek_timeout", type=float, default=60.0, help="Seconds allowed to SEEK to start.")
        ap.add_argument("--seek_poll", type=float, default=None, help="Override SEEK poll dt (s); defaults to --dt.")
        ap.add_argument("--settle_s", type=float, default=1.0, help="Seconds to let stabilization settle after seek.")

        ap.add_argument(
            "--segments",
            type=str,
            required=True,
            help="Scan plan: 'end_nm,speed;end_nm,speed;...' (or delta,speed with --relative)",
        )
        ap.add_argument("--relative", action="store_true", help="Interpret segment endpoints as cumulative delta from start.")
        args = ap.parse_args()

        start_nm = float(args.start)
        tol_nm = float(args.tol)
        poll_dt = float(args.dt)
        seek_poll = float(args.seek_poll) if args.seek_poll is not None else poll_dt

        # WS7
        print("Connecting WS7...")
        ws7 = WS7()
        bind_ws7_prototypes(ws7.lib)
        read_ws7_air_nm = make_ws7_air_reader(ws7, ch2=args.ch2, err_print_budget=3)

        wl0 = read_ws7_air_nm()
        print(f"WS7 OK. Current WS7_air={wl0 if wl0 > 0 else None}")

        # Matisse
        print("\nConnecting Matisse...")
        matisse = Matisse(wavemeter_type="WS7")
        print("Matisse OK.")
        try:
            print("Laser locked?:", matisse.laser_locked())
        except Exception:
            print("Laser locked?: (unknown)")

        # Ensure any scan is stopped
        safe_stop_scan(matisse)

        # -----------------------
        # Pre-lock / align start
        # -----------------------
        print(f"\nPre-lock mode: {args.prelock}")
        if args.prelock == "full":
            print(f"Calling matisse.set_wavelength({start_nm:.6f}) ...")
            matisse.set_wavelength(start_nm)

        elif args.prelock == "stabilize":
            # optional: ensure lock correction thread is running
            try:
                if hasattr(matisse, "is_lock_correction_on") and hasattr(matisse, "start_laser_lock_correction"):
                    if not matisse.is_lock_correction_on():
                        matisse.start_laser_lock_correction()
            except Exception:
                pass

            wl_now = read_ws7_air_nm()
            if wl_now > 0 and abs(wl_now - start_nm) > tol_nm:
                wl_seek = seek_to_wavelength(
                    matisse=matisse,
                    read_wl_nm=read_ws7_air_nm,
                    target_nm=start_nm,
                    tol_nm=tol_nm,
                    base_speed_nm_s=float(args.seek_speed),
                    ramp_window_nm=float(args.seek_ramp),
                    min_speed_nm_s=float(args.seek_min_speed),
                    poll_dt_s=seek_poll,
                    max_time_s=float(args.seek_timeout),
                    verbose=True,
                )
                print(f"Seek result: WS7_air={wl_seek if wl_seek > 0 else None}")
            else:
                print("Already within tolerance; skipping SEEK.")

            # stabilize briefly at start
            try:
                matisse.target_wavelength = start_nm
                matisse.stabilize_on()
                time.sleep(max(0.0, float(args.settle_s)))
            except Exception:
                pass

        else:
            # none
            try:
                matisse.target_wavelength = start_nm
            except Exception:
                pass

        wl_start = read_ws7_air_nm()
        print(f"\nStart check: WS7_air={wl_start if wl_start > 0 else None} (target {start_nm:.6f}±{tol_nm:.6f})")

        # Turn OFF stabilization during scanning plan
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
                read_wl_nm=read_ws7_air_nm,
                end_nm=float(end_nm),
                speed_nm_s=float(sp),
                tol_nm=tol_nm,
                poll_dt_s=poll_dt,
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

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt: stopping scan safely...")
        raise
    finally:
        safe_stop_scan(matisse)


if __name__ == "__main__":
    main()
