#!/usr/bin/env python3
"""
test_timing_alignment.py

Goal:
  Quantify timing alignment between:
    - Matisse scan commands (start/stop/resume)
    - Andor kinetics acquisition start (software timestamp)
    - WS7 timestamped samples

Method:
  - Start a scan at constant speed
  - Start Andor kinetics
  - After hold_after_s from scan start: STOP scan (hold)
  - After hold_s: RESUME scan
  - Compute per-frame mean WS7 wavelength using assumed frame windows
  - Detect plateau region (delta ~ 0) and estimate time offset:
        offset = (observed plateau center time) - (command plateau center time)

Run example:
  python script/test_timing_alignment.py --start 739.300 --end 739.320 --speed 0.010 \
    --exp 0.050 --cycle 0.060 --n 200 \
    --hold_after 2.0 --hold 1.0 --tol 0.002 --dt 0.02 --out data/timing_align.npz

Notes:
  - This reads WS7 in AIR using ConvertUnit (vac->air).
  - Uses ONE WS7 instance (the one inside Matisse), to avoid DLL state weirdness.
"""

import sys
import time
import math
import argparse
import threading
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import ctypes

import numpy as np


# ---------------------------
# Robust sys.path injection
# ---------------------------
HERE = Path(__file__).resolve()
root = None
for p in [HERE.parent] + list(HERE.parents):
    if (p / "matisse_controller").exists():
        root = p
        break
if root is None:
    raise RuntimeError("Could not find 'matisse_controller' in any parent directory.")
sys.path.insert(0, str(root))


from matisse_controller.matisse.matisse import Matisse  # noqa: E402
from matisse_controller.shamrock_ple.ccd import CCD  # noqa: E402

# Try to import Andor status enum if present
try:
    from matisse_controller.shamrock_ple.constants import CCDErrorCode  # type: ignore
except Exception:
    CCDErrorCode = None  # fallback


# wlmConst values
cReturnWavelengthVac = 0
cReturnWavelengthAir = 1


def estimate_hold_center_from_ws7(
    t_ws7: np.ndarray,
    wl_ws7: np.ndarray,
    scan_speed_nm_s: float,
    window_s: float = 0.25,
    frac_of_scan_speed: float = 0.10,
):
    """
    Estimate hold (plateau) center time directly from WS7 data by finding the longest
    region where the measured slope is near zero.

    Returns:
      center_t, (t_start, t_end), debug_dict
    """
    t_ws7 = np.asarray(t_ws7, dtype=float)
    wl_ws7 = np.asarray(wl_ws7, dtype=float)

    # guard
    if len(t_ws7) < 10:
        raise ValueError("Not enough WS7 samples to estimate hold center.")

    dt = np.median(np.diff(t_ws7))
    win = max(5, int(round(window_s / max(dt, 1e-6))))

    # finite-difference slope over a longer window to reduce WS7 quantization effects
    denom = (t_ws7[win:] - t_ws7[:-win])
    slope = (wl_ws7[win:] - wl_ws7[:-win]) / denom
    t_slope = t_ws7[:-win] + 0.5 * denom

    thr = frac_of_scan_speed * abs(scan_speed_nm_s)
    mask = np.abs(slope) < thr

    # longest contiguous run of True in mask
    best = None
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            if best is None or (j - i) > (best[1] - best[0]):
                best = (i, j)
            i = j
        else:
            i += 1

    if best is None:
        raise RuntimeError("Could not find a hold/plateau in WS7 slope data. Try longer hold_s.")

    i0, i1 = best  # [i0, i1)
    t0 = float(t_slope[i0])
    t1 = float(t_slope[i1 - 1])
    center = 0.5 * (t0 + t1)

    dbg = {
        "dt_med": float(dt),
        "win_samples": int(win),
        "slope_thr_nm_s": float(thr),
        "run_len": int(i1 - i0),
        "t0": t0,
        "t1": t1,
    }
    return center, (t0, t1), dbg

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
    if hasattr(lib, "GetStatus"):
        lib.GetStatus.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.GetStatus.restype = ctypes.c_int


def ws7_read_air_nm(ws7_obj, ch2: bool = False) -> float:
    """
    Read WS7 AIR wavelength in nm.
    ws7_obj is matisse._wavemeter (WS7 wrapper from matisse_controller.wavemaster)
    Returns <=0 on error (WS7 convention).
    """
    lib = ws7_obj.lib
    raw = float(lib.GetWavelength2(0.0)) if (ch2 and hasattr(lib, "GetWavelength2")) else float(lib.GetWavelength(0.0))
    if raw <= 0 or math.isnan(raw):
        return raw
    if not hasattr(lib, "ConvertUnit"):
        raise RuntimeError("WS7 DLL missing ConvertUnit(); cannot convert vac->air.")
    return float(lib.ConvertUnit(raw, cReturnWavelengthVac, cReturnWavelengthAir))


class WS7Sampler:
    """Background WS7 sampler with monotonic timestamps."""
    def __init__(self, ws7_obj, ch2: bool, dt_s: float):
        self.ws7 = ws7_obj
        self.ch2 = bool(ch2)
        self.dt_s = float(dt_s)
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None
        self.t: List[float] = []
        self.wl: List[float] = []

    def start(self):
        if self._th and self._th.is_alive():
            return
        self._stop.clear()

        def run():
            next_t = time.monotonic()
            while not self._stop.is_set():
                now = time.monotonic()
                if now < next_t:
                    time.sleep(min(0.005, next_t - now))
                    continue

                wl = ws7_read_air_nm(self.ws7, ch2=self.ch2)
                if wl > 0 and not math.isnan(wl):
                    self.t.append(now)
                    self.wl.append(wl)

                next_t += self.dt_s

        self._th = threading.Thread(target=run, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()
        if self._th:
            self._th.join(timeout=1.0)


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


def ccd_wait_until_acquiring(ccd: CCD, timeout_s: float = 2.0, poll_s: float = 0.01) -> float:
    """
    Best-effort: return a better t0 than "command issue time" by waiting until CCD status != IDLE.
    If we can't detect IDLE, return current time right away.
    """
    if not hasattr(ccd.lib, "GetStatus"):
        return time.monotonic()

    idle_val = None
    if CCDErrorCode is not None and hasattr(CCDErrorCode, "DRV_IDLE"):
        idle_val = int(CCDErrorCode.DRV_IDLE.value)

    st = ctypes.c_int()
    t0 = time.monotonic()

    # If we don't know IDLE, just return quickly (still useful)
    if idle_val is None:
        return t0

    while (time.monotonic() - t0) < float(timeout_s):
        try:
            ccd.lib.GetStatus(ctypes.byref(st))
            if int(st.value) != idle_val:
                return time.monotonic()
        except Exception:
            break
        time.sleep(poll_s)

    return t0


def frame_means_from_samples(
    t_samp: np.ndarray,
    wl_samp: np.ndarray,
    t0: float,
    exposure_s: float,
    cycle_s: float,
    n_frames: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-frame mean wavelength using sample points that fall inside each exposure window.
    Returns (frame_t_mid, frame_mean, frame_count).
    """
    frame_mid = np.empty(n_frames, dtype=float)
    frame_mean = np.full(n_frames, np.nan, dtype=float)
    frame_count = np.zeros(n_frames, dtype=int)

    for i in range(n_frames):
        a = t0 + i * cycle_s
        b = a + exposure_s
        frame_mid[i] = a + 0.5 * exposure_s

        m = (t_samp >= a) & (t_samp <= b)
        if np.any(m):
            xs = wl_samp[m]
            frame_mean[i] = float(xs.mean())
            frame_count[i] = int(xs.size)

    return frame_mid, frame_mean, frame_count


def find_longest_plateau(delta_nm: np.ndarray, thresh_nm: float, min_len: int = 3) -> Optional[Tuple[int, int]]:
    """
    Find longest run of consecutive indices where abs(delta) <= thresh.
    delta has length n_frames-1, index i corresponds to (frame i -> i+1).
    Returns (i0, i1) inclusive indices into delta for the plateau run, or None.
    """
    ok = np.isfinite(delta_nm) & (np.abs(delta_nm) <= thresh_nm)
    best = None
    i = 0
    while i < ok.size:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < ok.size and ok[j]:
            j += 1
        # run is [i, j-1]
        if (j - i) >= min_len:
            if best is None or (j - i) > (best[1] - best[0] + 1):
                best = (i, j - 1)
        i = j
    return best


def main():
    matisse: Optional[Matisse] = None
    ccd: Optional[CCD] = None

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, required=True, help="Start wavelength (nm, AIR).")
    ap.add_argument("--end", type=float, required=True, help="End wavelength (nm, AIR).")
    ap.add_argument("--speed", type=float, required=True, help="Scan speed (nm/s).")
    ap.add_argument("--tol", type=float, default=0.002, help="Tolerance for end condition (nm).")
    ap.add_argument("--dt", type=float, default=0.02, help="WS7 sample period (s).")
    ap.add_argument("--ch2", action="store_true", help="Use WS7 channel 2 if supported.")
    ap.add_argument("--cool_down", action="store_true", help="Cool down CCD before kinetics (slower).")
    ap.add_argument("--temp", type=float, default=-70.0, help="CCD temperature setpoint (C).")

    ap.add_argument("--exp", type=float, required=True, help="CCD exposure time (s).")
    ap.add_argument("--cycle", type=float, required=True, help="CCD kinetic cycle time (s).")
    ap.add_argument("--n", type=int, required=True, help="Number of kinetic frames.")

    ap.add_argument("--hold_after", type=float, default=2.0, help="Seconds after scan start to STOP scan.")
    ap.add_argument("--hold", type=float, default=1.0, help="Seconds to HOLD (scan stopped) before resuming.")

    ap.add_argument("--out", type=str, default="data/timing_align.npz", help="Output NPZ path.")
    args = ap.parse_args()

    start_nm = float(args.start)
    end_nm = float(args.end)
    scan_speed = abs(float(args.speed))
    scan_dir = 0 if (end_nm - start_nm) >= 0 else 1

    # Connect Matisse (includes WS7 wavemeter instance)
    print("Connecting Matisse (and WS7)...")
    matisse = Matisse(wavemeter_type="WS7")
    print("Matisse OK.")
    try:
        print("Laser locked?:", matisse.laser_locked())
    except Exception:
        pass

    # Grab the WS7 object inside Matisse
    ws7_obj = getattr(matisse, "_wavemeter", None)
    if ws7_obj is None or not hasattr(ws7_obj, "lib"):
        raise RuntimeError("Could not access matisse._wavemeter.lib (WS7).")
    bind_ws7_prototypes(ws7_obj.lib)

    wl_now = ws7_read_air_nm(ws7_obj, ch2=args.ch2)
    print(f"WS7_air now: {wl_now if wl_now > 0 else None}")

    # Connect CCD
    print("\nConnecting Andor CCD...")
    ccd = CCD(initialize_on_definition=True)
    print("CCD OK.")

    # Ensure we start from a sane state
    safe_stop_scan(matisse)
    try:
        matisse.stabilize_off()
    except Exception:
        pass

    # Lock/stabilize at start (simple: set target + stabilize; you can replace with your SEEK if desired)
    print(f"\nStabilizing at start {start_nm:.6f} nm (AIR)...")
    try:
        matisse.target_wavelength = start_nm
        matisse.stabilize_on()
        time.sleep(1.0)
    except Exception:
        pass

    # Stop stabilization before scanning
    try:
        matisse.stabilize_off()
    except Exception:
        pass

    # Configure kinetics
    print("\nConfiguring CCD kinetics...")
    exp_actual, cycle_actual = ccd.setup_kinetics(
        exposure_time=float(args.exp),
        cycle_time=float(args.cycle),
        n_frames=int(args.n),
        temperature=float(args.temp),
        cool_down=bool(args.cool_down),
    )
    print(f"Kinetics actual: exp={exp_actual:.6f}s cycle={cycle_actual:.6f}s frames={args.n}")

    # Start WS7 sampler
    sampler = WS7Sampler(ws7_obj, ch2=args.ch2, dt_s=float(args.dt))
    sampler.start()

    # Schedule scan pause/resume
    times: Dict[str, float] = {}

    def pause_worker():
        # Wait until scan has been started and we have t_scan_start
        while "t_scan_start" not in times:
            time.sleep(0.001)

        t_scan_start = times["t_scan_start"]
        t_stop = t_scan_start + float(args.hold_after)
        t_resume = t_stop + float(args.hold)

        # wait to stop
        while time.monotonic() < t_stop:
            time.sleep(0.001)
        times["t_stop_cmd"] = time.monotonic()
        print(f"\n>>> STOP scan (hold) at t={times['t_stop_cmd'] - t_scan_start:.3f}s")
        safe_stop_scan(matisse)

        # wait to resume
        while time.monotonic() < t_resume:
            time.sleep(0.001)
        times["t_resume_cmd"] = time.monotonic()
        print(f"\n>>> RESUME scan at t={times['t_resume_cmd'] - t_scan_start:.3f}s")
        try:
            matisse.start_scan(scan_dir)
        except Exception:
            matisse.query(f"SCAN:MODE {scan_dir}")
            matisse.query("SCAN:STATUS RUN")

    pause_th = threading.Thread(target=pause_worker, daemon=True)

    try:
        # Configure scan speed + target end
        set_scan_speed(matisse, scan_speed)
        matisse.target_wavelength = float(end_nm)

        # Start scan
        print(f"\nStarting scan dir={scan_dir} {start_nm:.6f}->{end_nm:.6f} at {scan_speed:.6f} nm/s")
        times["t_scan_start"] = time.monotonic()
        matisse.start_scan(scan_dir)

        pause_th.start()

        # Start acquisition
        print("Starting CCD kinetics...")
        t_cmd = time.monotonic()
        ccd.start_acquisition()
        t_effective = ccd_wait_until_acquiring(ccd, timeout_s=2.0)
        times["t_ccd_cmd"] = t_cmd
        times["t_ccd_t0_used"] = t_effective
        print(f"CCD start: cmd_t0={t_cmd:.6f}, used_t0={t_effective:.6f} (monotonic)")

        # Wait for acquisition to finish
        ccd.wait_for_acquisition()
        times["t_ccd_done"] = time.monotonic()

        print("\nCCD kinetics finished.")

    finally:
        safe_stop_scan(matisse)
        sampler.stop()

    # Convert samples to arrays
    t_samp = np.asarray(sampler.t, dtype=float)
    wl_samp = np.asarray(sampler.wl, dtype=float)

    # Frame stats using our chosen t0
    t0_used = float(times["t_ccd_t0_used"])
    frame_mid, frame_mean, frame_n = frame_means_from_samples(
        t_samp=t_samp,
        wl_samp=wl_samp,
        t0=t0_used,
        exposure_s=float(exp_actual),
        cycle_s=float(cycle_actual),
        n_frames=int(args.n),
    )

    # Detect plateau (scan hold) using delta between consecutive frames
    delta = np.diff(frame_mean)
    expected_step = scan_speed * float(cycle_actual)
    thresh = max(0.00002, 0.2 * expected_step)  # nm
    plateau = find_longest_plateau(delta, thresh_nm=thresh, min_len=3)

    print("\n=== Timing alignment report ===")
    print(f"WS7 samples: {t_samp.size}")
    print(f"Expected scan step per frame ~ {expected_step:.6e} nm")
    print(f"Plateau threshold: {thresh:.6e} nm (|delta| <= thresh)")

    if plateau is None or ("t_stop_cmd" not in times) or ("t_resume_cmd" not in times):
        print("Could not find a clear plateau region (or missing stop/resume times).")
        print("Try increasing hold time (e.g., --hold 2.0) and/or increasing frames.")
        offset_est = float("nan")
    else:
        i0, i1 = plateau
        # plateau in delta indices corresponds to frames i0..i1+1
        frame_i_center = int(round(0.5 * (i0 + (i1 + 1))))
        t_obs_center = float(frame_mid[frame_i_center])  # observed by our frame windowing
        t_cmd_center = 0.5 * (float(times["t_stop_cmd"]) + float(times["t_resume_cmd"]))  # commanded plateau center
        offset_est = t_obs_center - t_cmd_center

        print(f"Plateau run (delta idx): {i0}..{i1}  (~frames {i0}..{i1+1})")
        print(f"Plateau center frame: {frame_i_center}")
        print(f"Observed plateau center time (frame_mid): {t_obs_center:.6f}")
        print(f"Command plateau center time (stop/resume avg): {t_cmd_center:.6f}")
        print(f"Estimated timing offset (frame_mid - cmd_center): {offset_est:+.6f} s")

    # Save NPZ for plotting/debug
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = dict(
        start_nm=start_nm,
        end_nm=end_nm,
        scan_speed_nm_s=scan_speed,
        scan_dir=scan_dir,
        exp_s=float(exp_actual),
        cycle_s=float(cycle_actual),
        n_frames=int(args.n),
        hold_after_s=float(args.hold_after),
        hold_s=float(args.hold),
        tol_nm=float(args.tol),
        ws7_dt_s=float(args.dt),
        times=times,
        plateau_thresh_nm=float(thresh),
        offset_est_s=float(offset_est) if np.isfinite(offset_est) else None,
    )

    np.savez(
        str(out_path),
        t_ws7=t_samp,
        wl_ws7=wl_samp,
        frame_mid=frame_mid,
        frame_mean=frame_mean,
        frame_n=frame_n,
        meta=np.array([meta], dtype=object),
    )

    print(f"\nSaved: {out_path}")
    print("DONE")


if __name__ == "__main__":
    main()
