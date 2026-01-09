#!/usr/bin/env python3
"""
run_scan.py (place at project root)

Modes (selected in YAML):
  1) timetagger_counts (recommended to test first):
     - Align to start_nm using SEEK (SCAN) + optional stabilize dwell
     - Execute scan plan as piecewise segments, each with its own speed
     - During each exposure window:
         * mean/std wavelength from WS7 AIR samples within the window
         * TimeTagger integrated counts over exposure_s
     - Save points to HDF5 (preferred) or NPZ fallback (+ meta.json)

  2) andor_kinetic:
     - Configure Andor CCD kinetics
     - Optionally auto set scan speed so scan duration matches kinetic duration
     - Start scan + start kinetic acquisition
     - Sample WS7 AIR + optional power until kinetics ends (or timeout)
     - Save .sif + arrays/traces to HDF5/NPZ (+ meta.json)

Usage:
  python run_scan.py                 # uses run_scan.yml in same directory
  python run_scan.py path/to.yml     # use specific config file
"""

from __future__ import annotations

import os
import sys
import time
import json
import shutil
import threading
import ctypes
from dataclasses import dataclass
from pathlib import Path
from collections import deque
from typing import Deque, Optional, Dict, Any, List, Tuple

import numpy as np

from matisse_controller.shamrock_ple import ple

try:
    from Swabian import TimeTagger
except ImportError:
    import TimeTagger

# ---------------------------
# Robust sys.path injection  (MUST be before importing project modules)
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

# NOW import your repo modules
import matisse_controller.shamrock_ple.ple as ple_mod
import yaml
from functions26.instruments.ws7 import WS7
from functions26.instruments.powermeter import PowerMeter
from matisse_controller.matisse.matisse import Matisse

DEFAULT_CONFIG_PATH = "run_scan.yml"


# WS7 wlmConst values (we avoid importing wlmConst as a module due to package layout issues)
cReturnWavelengthVac = 0
cReturnWavelengthAir = 1


# -------------------------
# Data structures
# -------------------------
@dataclass
class ScanPoint:
    t0: float
    t1: float
    wl_nm: float
    wl_std_nm: float
    n_wl_samples: int
    power_W: float
    counts: int


@dataclass(frozen=True)
class WlSample:
    t: float          # time.monotonic()
    wl_nm: float


@dataclass(frozen=True)
class PowerSample:
    t: float          # time.monotonic()
    power_W: float    # Watts


# -------------------------
# Helpers: config
# -------------------------
import threading
import time
from typing import Any, Dict, List, Tuple

def start_midpoint_power_read(powermeter_instance, exposure_s: float) -> Tuple[threading.Thread, Dict[str, Any]]:
    """
    One-shot Newport power reading around the midpoint of an exposure window.

    Assumes your powermeter object supports:
      - powermeter_instance.powermeter.get_instrument_reading_string_all()
      - powermeter_instance.convert_reading_string_to_float(str) -> value in µW
    Returns (thread, out_dict) where out_dict["power_W"] is filled when the thread completes.
    """
    out: Dict[str, Any] = {"power_W": None, "error": None}

    def worker():
        try:
            # try to sample near the midpoint of the exposure
            time.sleep(max(0.0, 0.5 * float(exposure_s)))

            reading_strings = powermeter_instance.powermeter.get_instrument_reading_string_all()

            vals_uW: List[float] = []
            for s in reading_strings:
                try:
                    vals_uW.append(powermeter_instance.convert_reading_string_to_float(s))  # µW
                except Exception:
                    pass

            if vals_uW:
                out["power_W"] = float(sum(vals_uW) / len(vals_uW)) * 1e-6  # µW -> W

        except Exception as e:
            out["error"] = e

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    return th, out

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        top = yaml.safe_load(f) or {}

    cfg = top.get("RunScan", {})
    if not cfg:
        raise ValueError("YAML missing top-level key: RunScan")

    # Keep entire YAML so we can read legacy PLEScan fields too
    cfg["_full_yaml"] = top
    return cfg



def cfg_get(d: Dict[str, Any], path: str, default=None):
    """Safe nested get: cfg_get(cfg, 'scan.start_nm')"""
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def enforce_horizontal_shift_speed(ccd, hss=(0, 0, 0.05)) -> None:
    """
    Force Andor horizontal shift speed using the underlying Andor SDK functions.

    hss = (ad_channel, output_amp, speed_MHz)
      - ad_channel: int (usually 0)
      - output_amp: int (usually 0)
      - speed_MHz: float (e.g. 0.05)

    This maps speed_MHz -> closest available HSSpeed index, then calls SetHSSpeed().
    """
    import ctypes

    ad_ch, out_amp, target_mhz = int(hss[0]), int(hss[1]), float(hss[2])

    # Try common attribute names where the ctypes DLL handle might live
    sdk = (
        getattr(ccd, "atmcd", None)
        or getattr(ccd, "_atmcd", None)
        or getattr(ccd, "lib", None)
        or getattr(ccd, "_lib", None)
        or getattr(ccd, "dll", None)
        or getattr(ccd, "_dll", None)
        or getattr(ccd, "andor", None)
        or getattr(ccd, "_andor", None)
    )

    if sdk is None:
        raise RuntimeError(
            "Can't locate the Andor SDK DLL handle inside `ccd`. "
            "Print attributes like `dir(ccd)` and look for atmcd/lib/dll."
        )

    def has(name: str) -> bool:
        return hasattr(sdk, name)

    def call(name: str, *args):
        fn = getattr(sdk, name, None)
        if fn is None:
            return None
        try:
            return fn(*args)
        except Exception:
            return None

    # Sanity: these are the key SDK functions we rely on
    needed = ("GetNumberHSSpeeds", "GetHSSpeed", "SetHSSpeed")
    if not all(has(n) for n in needed):
        raise RuntimeError(
            f"Andor SDK handle found, but missing required functions: "
            f"{[n for n in needed if not has(n)]}. "
            "Your wrapper may be using a different API surface."
        )

    # Set AD channel + output amplifier if available
    if has("SetADChannel"):
        call("SetADChannel", ctypes.c_int(ad_ch))
    if has("SetOutputAmplifier"):
        call("SetOutputAmplifier", ctypes.c_int(out_amp))

    # Enumerate available HS speeds and pick closest to target_mhz
    n = ctypes.c_int()
    # Andor signature: GetNumberHSSpeeds(int channel, int typ, int* speeds)
    call("GetNumberHSSpeeds", ctypes.c_int(ad_ch), ctypes.c_int(out_amp), ctypes.byref(n))

    if n.value <= 0:
        raise RuntimeError(f"GetNumberHSSpeeds returned n={n.value}. Cannot set HS speed.")

    best_i = 0
    best_speed = None
    best_diff = float("inf")

    for i in range(n.value):
        sp = ctypes.c_float()
        # Andor signature: GetHSSpeed(int channel, int typ, int index, float* speed)
        call("GetHSSpeed", ctypes.c_int(ad_ch), ctypes.c_int(out_amp), ctypes.c_int(i), ctypes.byref(sp))
        diff = abs(float(sp.value) - target_mhz)
        if diff < best_diff:
            best_diff = diff
            best_i = i
            best_speed = float(sp.value)

    # Andor signature: SetHSSpeed(int typ, int index)   (typ == output amplifier)
    call("SetHSSpeed", ctypes.c_int(out_amp), ctypes.c_int(best_i))

    print(
        f"Enforced horizontal_shift_speed: (AD={ad_ch}, AMP={out_amp}, target={target_mhz} MHz) "
        f"-> index={best_i}, actual={best_speed} MHz"
    )

def stop_stabilization_everywhere(matisse):
    # 1) turn off stabilize mode if it exists
    try:
        matisse.stabilize_off()
    except Exception:
        pass

    # 2) common method names
    for name in (
        "stop_stabilization_thread",
        "stop_stabilisation_thread",
        "stop_stabilization",
        "stop_stabilisation",
    ):
        if hasattr(matisse, name):
            try:
                getattr(matisse, name)()
                return
            except Exception:
                pass

    # 3) common attribute names (thread object)
    th = None
    for attr in ("stabilization_thread", "_stabilization_thread", "stab_thread", "_stab_thread"):
        if hasattr(matisse, attr):
            th = getattr(matisse, attr)
            break

    if th is not None:
        try:
            if hasattr(th, "stop"):
                th.stop()
        except Exception:
            pass
        try:
            if hasattr(th, "join"):
                th.join(timeout=2.0)
        except Exception:
            pass

# -------------------------
# WS7 helpers (AIR)
# -------------------------
def bind_ws7_prototypes(lib) -> None:
    """Bind ctypes prototypes so ConvertUnit/GetWavelength return correct doubles."""
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
    """Read WS7 wavelength in AIR nm. Returns <=0 on WS7 error codes."""
    lib = ws7.lib
    raw = float(lib.GetWavelength2(0.0)) if (ch2 and hasattr(lib, "GetWavelength2")) else float(lib.GetWavelength(0.0))
    if raw <= 0:
        return raw
    if not hasattr(lib, "ConvertUnit"):
        raise RuntimeError("WS7 DLL missing ConvertUnit(); cannot convert vac->air.")
    return float(lib.ConvertUnit(raw, cReturnWavelengthVac, cReturnWavelengthAir))


# -------------------------
# WS7 sampler (background)
# -------------------------
class WavelengthSampler:
    """
    Background WS7 sampler in AIR nm, timestamped with time.monotonic().
    """
    def __init__(self, ws7_instance, ch2: bool = False, maxlen: int = 200_000):
        self.ws7 = ws7_instance
        self.ch2 = bool(ch2)
        self.buf: Deque[WlSample] = deque(maxlen=maxlen)
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None
        self._io_lock = threading.Lock()

    def _read_wl_nm_once(self) -> Optional[float]:
        try:
            with self._io_lock:
                wl = float(ws7_read_air_nm(self.ws7, ch2=self.ch2))
            return wl if wl > 0 else None
        except Exception:
            return None

    def start(self, sample_period_s: float) -> None:
        if self._th and self._th.is_alive():
            return
        self._stop.clear()

        def run():
            while not self._stop.is_set():
                t = time.monotonic()
                wl = self._read_wl_nm_once()
                if wl is not None:
                    self.buf.append(WlSample(t=t, wl_nm=wl))
                time.sleep(sample_period_s)

        self._th = threading.Thread(target=run, daemon=True)
        self._th.start()

    def stop(self) -> None:
        self._stop.set()
        if self._th:
            self._th.join(timeout=1.0)

    def stats_between(self, t0: float, t1: float) -> Tuple[float, float, int]:
        xs = [s.wl_nm for s in self.buf if t0 <= s.t <= t1]
        if xs:
            arr = np.asarray(xs, dtype=float)
            return float(arr.mean()), float(arr.std()), int(arr.size)

        if not self.buf:
            raise RuntimeError("No WS7 samples available.")
        mid = 0.5 * (t0 + t1)
        nearest = min(self.buf, key=lambda s: abs(s.t - mid))
        return float(nearest.wl_nm), 0.0, 0

@dataclass(frozen=True)
class PowerSample:
    t: float          # time.monotonic()
    power_W: float    # Watts


class PowerSampler:
    """
    Continuous timestamped sampling of Newport power meter.

    Expects `powermeter_instance` to have:
      - powermeter_instance.powermeter.get_instrument_reading_string_all()
      - powermeter_instance.convert_reading_string_to_float(s) -> µW
    """
    def __init__(self, powermeter_instance, maxlen: int = 200_000):
        self.pm = powermeter_instance
        self.buf: Deque[PowerSample] = deque(maxlen=maxlen)
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None
        self._io_lock = threading.Lock()

    def _read_power_W_once(self) -> Optional[float]:
        try:
            with self._io_lock:
                reading_strings = self.pm.powermeter.get_instrument_reading_string_all()

            vals_uW: List[float] = []
            for s in reading_strings:
                try:
                    vals_uW.append(self.pm.convert_reading_string_to_float(s))  # µW
                except Exception:
                    pass

            if not vals_uW:
                return None

            mean_uW = sum(vals_uW) / len(vals_uW)
            return float(mean_uW) * 1e-6  # µW -> W
        except Exception:
            return None

    def start(self, sample_period_s: float) -> None:
        if self._th and self._th.is_alive():
            return
        self._stop.clear()

        def run():
            next_t = time.monotonic()
            while not self._stop.is_set():
                now = time.monotonic()
                if now < next_t:
                    time.sleep(min(0.01, next_t - now))
                    continue

                p = self._read_power_W_once()
                if p is not None:
                    self.buf.append(PowerSample(t=now, power_W=p))

                next_t += float(sample_period_s)

        self._th = threading.Thread(target=run, daemon=True)
        self._th.start()

    def stop(self) -> None:
        self._stop.set()
        if self._th:
            self._th.join(timeout=1.0)

    def mean_between(self, t0: float, t1: float) -> Optional[float]:
        if not self.buf:
            return None

        xs = [s.power_W for s in self.buf if t0 <= s.t <= t1]
        if xs:
            return float(sum(xs) / len(xs))

        # fallback: nearest sample
        mid = 0.5 * (t0 + t1)
        nearest = min(self.buf, key=lambda s: abs(s.t - mid))
        return float(nearest.power_W)


# -------------------------
# Scan plan parsing (variable speed)
# -------------------------
def parse_segments(s: str) -> List[Tuple[float, float]]:
    """
    Parse: "end_nm,speed;end_nm,speed;..."
    Returns list[(end_value, speed_nm_per_s)]
    """
    out: List[Tuple[float, float]] = []
    parts = [x.strip() for x in str(s).split(";") if x.strip()]
    for p in parts:
        a, b = [x.strip() for x in p.split(",")]
        out.append((float(a), float(b)))
    return out

def _import_timetagger_module():
    """
    Return the Swabian TimeTagger python module object.
    Raises RuntimeError with a clear message if not importable.
    """
    try:
        import TimeTagger as TT  # type: ignore
        return TT
    except Exception as e1:
        try:
            from Swabian import TimeTagger as TT  # type: ignore
            return TT
        except Exception as e2:
            raise RuntimeError(
                "Could not import Swabian TimeTagger bindings.\n"
                f"import TimeTagger failed with: {repr(e1)}\n"
                f"from Swabian import TimeTagger failed with: {repr(e2)}\n"
                "Check your TimeTagger installation / PATH / DLL load."
            )

def build_scan_plan(
    start_nm: float,
    end_nm: float,
    scan_speed_nm_s: float,
    segments_str: Optional[str],
    relative: bool,
) -> List[Tuple[float, float]]:
    """
    If segments_str is provided, use piecewise plan.
    If relative=True: endpoints are cumulative deltas from start.
    Else: endpoints are absolute nm.

    Fallback: single segment (end_nm, scan_speed_nm_s).
    """
    if segments_str:
        segs = parse_segments(segments_str)
        if not segs:
            raise ValueError("scan.segments provided but parsed as empty.")
        if relative:
            return [(start_nm + d, sp) for (d, sp) in segs]
        return [(end, sp) for (end, sp) in segs]

    return [(end_nm, scan_speed_nm_s)]


def set_scan_speed(matisse: Matisse, speed_nm_s: float) -> None:
    sp = abs(float(speed_nm_s))
    if sp <= 0:
        raise ValueError("scan speed must be > 0")
    matisse.query(f"SCAN:RISINGSPEED {sp:.12f}")
    matisse.query(f"SCAN:FALLINGSPEED {sp:.12f}")


def safe_stop_scan(matisse: Optional[Matisse]) -> None:
    try:
        if matisse is None:
            return
        if hasattr(matisse, "stop_scan"):
            matisse.stop_scan()
        else:
            matisse.query("SCAN:STATUS STOP")
    except Exception:
        pass


# -------------------------
# SEEK to start wavelength (robust: slow down + reverse on overshoot)
# -------------------------
def seek_to_wavelength(
    matisse: Matisse,
    ws7: WS7,
    target_nm: float,
    tol_nm: float,
    poll_dt_s: float,
    timeout_s: float,
    coarse_speed_nm_s: float,
    fine_speed_nm_s: float,
    fine_window_nm: float,
    ch2: bool,
    verbose_every_s: float = 0.5,
    min_move_nm: float = 2e-6,     # 0.002 pm
    no_progress_s: float = 4.0,
    verify_dir_s: float = 0.6,     # short "tap" time used for dir detection
) -> float:
    """
    Seek to target_nm using SCAN with:
      - coarse speed far away
      - fine speed inside fine_window_nm
      - auto-detect scan direction mapping (which dir increases wl)
      - always command motion that reduces |error|
      - flip if moving away (guard against inverted dir mapping / loop fighting)
      - stop on stall/timeout
    Returns last valid WS7_air wavelength (nm).
    """

    def read_wl() -> float:
        return ws7_read_air_nm(ws7, ch2=ch2)

    def set_speed(sp: float):
        set_scan_speed(matisse, float(abs(sp)))

    def start_scan(dirn: int):
        safe_stop_scan(matisse)
        matisse.start_scan(int(dirn))

    def stop_scan():
        safe_stop_scan(matisse)

    # -------------------------
    # Make sure scan can respond
    # -------------------------
    # (match what worked in test_variable_scanspeed: lock correction + loops)
    try:
        if hasattr(matisse, "is_lock_correction_on") and hasattr(matisse, "start_laser_lock_correction"):
            if not matisse.is_lock_correction_on():
                matisse.start_laser_lock_correction()
    except Exception:
        pass

    try:
        if hasattr(matisse, "start_control_loops"):
            matisse.start_control_loops()
    except Exception:
        pass

    # stop stabilization fighting us
    try:
        matisse.stabilize_off()
    except Exception:
        pass
    stop_scan()

    t0 = time.monotonic()
    t_last_print = t0
    t_last_move = t0

    wl0 = read_wl()
    if wl0 > 0 and abs(wl0 - target_nm) <= tol_nm:
        return wl0

    # Some controllers behave better if you also set target_wavelength
    try:
        matisse.target_wavelength = float(target_nm)
    except Exception:
        pass

    # -------------------------
    # Auto-detect direction map
    # -------------------------
    # Determine which `dir` makes wavelength increase.
    dir_up = 0  # default guess
    dir_confident = False

    # Use a moderate tap speed so motion is measurable but not huge
    tap_speed = max(0.001, min(abs(coarse_speed_nm_s), 0.01))  # clamp a bit
    tap_dt = max(0.2, float(verify_dir_s))

    try:
        set_speed(tap_speed)

        deltas = {}
        for d in (0, 1):
            # Re-baseline before each tap (important!)
            wl_a = read_wl()
            if wl_a <= 0:
                continue

            start_scan(d)
            time.sleep(tap_dt)
            stop_scan()

            wl_b = read_wl()
            if wl_b > 0:
                deltas[d] = wl_b - wl_a

            time.sleep(0.15)

        if 0 in deltas and 1 in deltas:
            d0, d1 = deltas[0], deltas[1]

            # If one is clearly positive and the other negative, it's unambiguous.
            if (d0 > min_move_nm and d1 < -min_move_nm) or (d1 > min_move_nm and d0 < -min_move_nm):
                dir_up = 0 if d0 > d1 else 1
                dir_confident = True
            else:
                # Otherwise, pick the one with larger (more positive) delta if either moved enough
                if abs(d0) > min_move_nm or abs(d1) > min_move_nm:
                    dir_up = 0 if d0 > d1 else 1
                    dir_confident = True

    except Exception:
        pass
    finally:
        stop_scan()

    dir_down = 1 - dir_up

    if not dir_confident:
        print(
            "(seek) WARNING: Could not confidently detect SCAN direction (wl barely moved). "
            "If seek runs away, SCAN may be ignored or control loops are overpowering it."
        )

    def desired_dir_from_wl(wl_now: float) -> int:
        # If below target -> go UP (increase wl), else go DOWN
        if wl_now <= 0:
            return dir_up
        return dir_up if wl_now < target_nm else dir_down

    def desired_speed_from_err(err_nm: float) -> float:
        return abs(coarse_speed_nm_s) if abs(err_nm) > fine_window_nm else abs(fine_speed_nm_s)

    # initial settings
    wl = read_wl()
    err = (wl - target_nm) if wl > 0 else float("inf")
    direction = desired_dir_from_wl(wl)
    speed = desired_speed_from_err(err)

    set_speed(speed)
    print(
        f"Seeking start via SCAN dir={direction} "
        f"(dir_up={dir_up}, dir_down={dir_down}, coarse={coarse_speed_nm_s:.6f}, "
        f"fine={fine_speed_nm_s:.6f}, window={fine_window_nm:.6f}) "
        f"target={target_nm:.6f}±{tol_nm:.6f} now={wl if wl > 0 else None}"
    )
    start_scan(direction)

    last_wl = wl
    last_err_abs = abs(err) if wl > 0 else float("inf")

    try:
        while True:
            now = time.monotonic()
            if (now - t0) >= timeout_s:
                raise TimeoutError(f"SEEK timeout after {timeout_s:.1f}s (last wl={last_wl}).")

            wl = read_wl()
            if wl <= 0:
                time.sleep(poll_dt_s)
                continue

            err = wl - target_nm
            err_abs = abs(err)

            # success
            if err_abs <= tol_nm:
                break

            # progress tracking
            if last_wl > 0 and abs(wl - last_wl) >= min_move_nm:
                t_last_move = now
                last_wl = wl

            # stall detection
            if (now - t_last_move) >= no_progress_s:
                raise RuntimeError(
                    f"SEEK stalled for {no_progress_s:.1f}s. "
                    f"wl={wl:.9f}, target={target_nm:.9f}, dir={direction}, sp={speed}"
                )

            # update speed based on distance to target
            desired_speed = desired_speed_from_err(err)
            if desired_speed != speed:
                speed = desired_speed
                set_speed(speed)

            # direction from sign(wl-target) (auto-reverses when crossing target)
            desired_dir = desired_dir_from_wl(wl)

            # safeguard: if |err| is trending worse by more than noise floor, flip immediately
            moving_away = (err_abs > last_err_abs + min_move_nm)
            last_err_abs = err_abs
            if moving_away:
                desired_dir = dir_down if direction == dir_up else dir_up

            if desired_dir != direction:
                direction = desired_dir
                start_scan(direction)

            if (now - t_last_print) >= verbose_every_s:
                print(
                    f"  seek t={now-t0:6.1f}s  wl={wl:.9f}  err={err*1e3:+.3f} pm  "
                    f"dir={direction}  sp={speed:.6f}"
                )
                t_last_print = now

            time.sleep(poll_dt_s)

    finally:
        stop_scan()

    return wl

# -------------------------
# Time Tagger counts detector
# -------------------------
def _edge_channel(phys_ch: int, edge: str) -> int:
    edge = edge.strip().lower()
    if edge == "rising":
        return int(abs(phys_ch))
    if edge == "falling":
        return -int(abs(phys_ch))
    raise ValueError("edge must be 'rising' or 'falling'")


class TimeTaggerCounts:
    def __init__(self, click_phys_ch: int, click_trigger_v: float, click_edge: str, serial: Optional[str] = None):
        self.click_phys_ch = int(click_phys_ch)
        self.click_trigger_v = float(click_trigger_v)
        self.click_sw_ch = _edge_channel(self.click_phys_ch, click_edge)
        self.serial = serial
        self._tagger = None
        self._counter = None
        self._binwidth_ps = None

    def connect(self) -> None:
        if self._tagger is not None:
            return
        self._tagger = TimeTagger.createTimeTagger(self.serial) if self.serial else TimeTagger.createTimeTagger()
        self._tagger.setTriggerLevel(self.click_phys_ch, self.click_trigger_v)

    def close(self) -> None:
        if self._tagger is not None:
            try:
                self._counter = None
                self._binwidth_ps = None
            finally:
                TimeTagger.freeTimeTagger(self._tagger)
                self._tagger = None

    @property
    def tagger(self):
        if self._tagger is None:
            raise RuntimeError("TimeTagger not connected.")
        return self._tagger

    def _ensure_counter(self, exposure_s: float):
        bw = int(round(exposure_s * 1e12))
        if self._counter is None or self._binwidth_ps != bw:
            self._counter = TimeTagger.Counter(self.tagger, [self.click_sw_ch], binwidth=bw, n_values=1)
            self._binwidth_ps = bw

    def acquire_counts_timed(self, exposure_s: float) -> int:
        if exposure_s <= 0:
            raise ValueError("exposure_s must be > 0")
        self.connect()

        binwidth_ps = int(round(exposure_s * 1e12))
        meas = TimeTagger.Counter(self.tagger, [self.click_sw_ch], binwidth=binwidth_ps, n_values=1)
        meas.startFor(binwidth_ps, clear=True)
        meas.waitUntilFinished()

        # API compatibility across TimeTagger versions
        try:
            data = meas.getData()
        except TypeError:
            data = meas.getData(rolling=True)

        # Some versions return (data, aux) or similar
        if isinstance(data, tuple):
            data = data[0]

        arr = np.asarray(data, dtype=np.int64).reshape(-1)
        return int(arr[-1]) if arr.size else 0


# -------------------------
# Save helpers
# -------------------------
def save_meta_json_if_enabled(meta_path: str, meta: Dict[str, Any], enabled: bool) -> None:
    if not enabled:
        return
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def save_points_h5(path: str, points: List[ScanPoint], meta: Dict[str, Any]) -> None:
    import h5py

    wl = np.array([p.wl_nm for p in points], dtype=float)
    wl_std = np.array([p.wl_std_nm for p in points], dtype=float)
    n_wl = np.array([p.n_wl_samples for p in points], dtype=int)
    t0 = np.array([p.t0 for p in points], dtype=float)
    t1 = np.array([p.t1 for p in points], dtype=float)
    power_W = np.array([p.power_W for p in points], dtype=float)
    counts = np.array([p.counts for p in points], dtype=np.int64)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["meta_json"] = json.dumps(meta)
        f.create_dataset("wl_nm_air", data=wl)
        f.create_dataset("wl_std_nm_air", data=wl_std)
        f.create_dataset("n_wl_samples", data=n_wl)
        f.create_dataset("t0_monotonic_s", data=t0)
        f.create_dataset("t1_monotonic_s", data=t1)
        f.create_dataset("power_W", data=power_W)
        f.create_dataset("counts", data=counts)


def save_points_npz(path: str, points: List[ScanPoint], meta: Dict[str, Any]) -> str:
    base = os.path.splitext(path)[0] + ".npz"

    wl = np.array([p.wl_nm for p in points], dtype=float)
    wl_std = np.array([p.wl_std_nm for p in points], dtype=float)
    n_wl = np.array([p.n_wl_samples for p in points], dtype=int)
    t0 = np.array([p.t0 for p in points], dtype=float)
    t1 = np.array([p.t1 for p in points], dtype=float)
    power_W = np.array([p.power_W for p in points], dtype=float)
    counts = np.array([p.counts for p in points], dtype=np.int64)

    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    np.savez(
        base,
        wl_nm_air=wl,
        wl_std_nm_air=wl_std,
        n_wl_samples=n_wl,
        t0_monotonic_s=t0,
        t1_monotonic_s=t1,
        power_W=power_W,
        counts=counts,
        meta=np.array([meta], dtype=object),
    )
    return base


# -------------------------
# Mode 1: TimeTagger counts (variable scan speed + start alignment)
# -------------------------
def run_scan_timetagger_counts(cfg: Dict[str, Any]) -> None:
    scan = cfg["scan"]
    out = cfg["output"]
    inst = cfg["instruments"]
    det_cfg = inst["detector"]["timetagger_counts"]
    pow_cfg = inst.get("powermeter", {}) or {}
    pow_enabled = bool(pow_cfg.get("enabled", True))
    pow_channel = str(pow_cfg.get("channel", "B")).strip().upper()

    # -----------------------
    # start/end + plan
    # -----------------------
    start_nm = float(scan["start_nm"])
    end_nm_cfg = float(scan.get("end_nm", start_nm))

    segments_str = scan.get("segments", None)
    relative = bool(scan.get("relative", False))
    default_scan_speed = float(scan.get("scan_speed_nm_per_s", scan.get("scan_speed", 0.005)))
    plan = build_scan_plan(start_nm, end_nm_cfg, default_scan_speed, segments_str, relative)
    if not plan:
        raise ValueError("Empty scan plan. Check scan.segments / scan.end_nm.")
    end_nm = float(plan[-1][0])

    # exposure / timing
    exposure_s = float(det_cfg.get("exposure_s", scan.get("exposure_s", 0.1)))
    ws7_sample_period_s = float(scan.get("ws7_sample_period_s", 0.02))
    poll_dt_s = float(scan.get("poll_dt_s", 0.05))
    max_run_time_s = float(scan.get("max_run_time_s", 600.0))

    # WS7
    ch2 = bool(scan.get("ws7_ch2", False))

    # start alignment
    tol_nm = float(scan.get("start_tolerance_nm", scan.get("tol_nm", 0.0002)))
    prelock = str(scan.get("prelock", "stabilize")).strip().lower()  # none | stabilize | full
    stabilize_settle_s = float(scan.get("stabilize_settle_s", 0.5))

    # seek settings
    seek_timeout_s = float(scan.get("seek_timeout_s", 90.0))
    seek_coarse_speed = float(scan.get("seek_coarse_speed_nm_per_s", 0.001))
    seek_fine_speed = float(scan.get("seek_fine_speed_nm_per_s", 0.0005))
    seek_fine_window = float(scan.get("seek_fine_window_nm", max(5 * tol_nm, 5e-4)))

    # -----------------------
    # output
    # -----------------------
    out_dir = out.get("directory", "data")
    base = out.get("basename", "scan")
    fmt = out.get("format", "h5").lower()
    overwrite = bool(out.get("overwrite", True))
    save_meta = bool(out.get("save_meta_json", True))

    os.makedirs(out_dir, exist_ok=True)
    data_path = os.path.join(out_dir, f"{base}.{ 'h5' if fmt == 'h5' else 'npz' }")
    meta_path = os.path.join(out_dir, f"{base}.meta.json")
    if (not overwrite) and os.path.exists(data_path):
        raise FileExistsError(f"Output exists and overwrite=false: {data_path}")

    # -----------------------
    # detector
    # -----------------------
    # ---- warm up TimeTagger BEFORE scanning ----

    click = det_cfg["click"]
    det = TimeTaggerCounts(
        click_phys_ch=int(click["physical_channel"]),
        click_trigger_v=float(click["trigger_v"]),
        click_edge=str(click.get("edge", "falling")),
        serial=det_cfg.get("serial", None),
    )
    #warm up time tagger to avoid long exposure time at the start of a scan
    det.connect()
    _ = det.acquire_counts_timed(min(0.02, exposure_s))  # throw away
    _ = det.acquire_counts_timed(min(0.02, exposure_s))  # throw away (optional)


    # -----------------------
    # connect instruments (DIRECT, like test_variable_scanspeed)
    # -----------------------
    print("Connecting WS7...")
    ws7 = WS7()
    bind_ws7_prototypes(ws7.lib)
    wl_now = ws7_read_air_nm(ws7, ch2=ch2)
    print(f"WS7 OK. Current WS7_air={wl_now if wl_now > 0 else None}")

    print("\nConnecting Matisse...")
    matisse = Matisse(wavemeter_type="WS7")
    print("Matisse OK.")
    try:
        print("Laser locked?:", matisse.laser_locked())
    except Exception:
        print("Laser locked?: (unknown)")

    powermeter = None
    if pow_enabled:
        print(f"\nConnecting powermeter (channel {pow_channel})...")
        powermeter = PowerMeter(pow_channel)
        if powermeter is None:
            raise RuntimeError("powermeter enabled but ple_mod.powermeter is None after setup_powermeter().")
        powermeter.powermeter.initialize_instrument()
        try:
            powermeter._empty_buffer()
        except Exception:
            pass
        print('Power meter ok')
    else:
        print("\nPowermeter disabled.")

    # -----------------------
    # metadata
    # -----------------------
    meta: Dict[str, Any] = {
        "mode": "timetagger_counts",
        "start_nm_air": start_nm,
        "end_nm_air": end_nm,
        "start_tolerance_nm": tol_nm,
        "prelock": prelock,
        "segments": [(float(a), float(b)) for (a, b) in plan],
        "exposure_s": exposure_s,
        "ws7_sample_period_s": ws7_sample_period_s,
        "ws7_ch2": ch2,
        "seek": {
            "timeout_s": seek_timeout_s,
            "coarse_speed_nm_per_s": seek_coarse_speed,
            "fine_speed_nm_per_s": seek_fine_speed,
            "fine_window_nm": seek_fine_window,
            "poll_dt_s": poll_dt_s,
        },
        "timetagger": {
            "click_physical_channel": int(click["physical_channel"]),
            "click_edge": str(click.get("edge", "falling")),
            "click_trigger_v": float(click["trigger_v"]),
            "serial": det_cfg.get("serial", None),
        },
        "created_unix_s": time.time(),
        "powermeter": {
            "enabled": pow_enabled,
            "channel": pow_channel if pow_enabled else None,
            "mode": "midpoint_one_shot_per_exposure",
        },
    }

    points: List[ScanPoint] = []
    start_wall = time.time()

    # IMPORTANT: your WavelengthSampler must return AIR nm (via ws7_read_air_nm)
    wl_sampler = WavelengthSampler(ws7, ch2=ch2)
    wl_sampler.start(sample_period_s=ws7_sample_period_s)


    pm_sampler = None
    pm_dt = float(scan.get("power_sample_period_s", 0.05))  # add this config

    if pow_enabled and powermeter is not None:
        # powermeter.powermeter.initialize_instrument()
        powermeter._empty_buffer()
        pm_sampler = PowerSampler(powermeter)
        pm_sampler.start(sample_period_s=pm_dt)

    def ensure_lock_and_loops():
        # mirrors your working test script behavior
        try:
            if hasattr(matisse, "start_laser_lock_correction"):
                matisse.start_laser_lock_correction()
        except Exception:
            pass
        try:
            if hasattr(matisse, "start_control_loops"):
                matisse.start_control_loops()
        except Exception:
            pass

    try:
        safe_stop_scan(matisse)
        ensure_lock_and_loops()

        # -----------------------
        # Pre-lock / align start
        # -----------------------
        print(f"\nPre-lock mode: {prelock}")

        if prelock == "full":
            # heavier path; if this exists and is slow but reliable, keep it
            print(f"Calling matisse.set_wavelength({start_nm:.6f}) ...")
            matisse.set_wavelength(start_nm)

        elif prelock == "stabilize":
            # make sure stabilize is OFF while SEEKing
            try:
                matisse.stabilize_off()
            except Exception:
                pass
            safe_stop_scan(matisse)

            wl_now = ws7_read_air_nm(ws7, ch2=ch2)
            if wl_now > 0 and abs(wl_now - start_nm) > tol_nm:
                wl_seek = seek_to_wavelength(
                    matisse=matisse,
                    ws7=ws7,
                    target_nm=start_nm,
                    tol_nm=tol_nm,
                    poll_dt_s=poll_dt_s,
                    timeout_s=seek_timeout_s,
                    coarse_speed_nm_s=seek_coarse_speed,
                    fine_speed_nm_s=seek_fine_speed,
                    fine_window_nm=seek_fine_window,
                    ch2=ch2,
                )
                print(f"Seek result: WS7_air={wl_seek if wl_seek > 0 else None}")
            else:
                print("Already within tolerance; skipping SEEK.")

            # now “lock” by holding target + stabilize briefly (then OFF before scanning)
            try:
                matisse.target_wavelength = float(start_nm)
                matisse.stabilize_on()
                if stabilize_settle_s > 0:
                    time.sleep(stabilize_settle_s)
            finally:
                try:
                    matisse.stabilize_off()
                except Exception:
                    pass

        elif prelock == "none":
            pass
        else:
            raise ValueError(f"Unknown prelock={prelock!r}. Use none|stabilize|full.")

        wl_start = ws7_read_air_nm(ws7, ch2=ch2)
        print(f"\nStart check: WS7_air={wl_start if wl_start > 0 else None} "
              f"(target {start_nm:.6f}±{tol_nm:.6f})")

        # Ensure stabilization is OFF before scanning
        try:
            matisse.stabilize_off()
        except Exception:
            pass
        safe_stop_scan(matisse)

        # -----------------------
        # Execute scan plan
        # -----------------------
        wl_sampler.buf.clear()
        time.sleep(ws7_sample_period_s * 2)  # let it refill with fresh samples
        last_power_W = None  # cache last valid power
        if pm_sampler is not None:
            time.sleep(0.2)  # or pm_dt * 3

        print("\n=== Running scan plan ===")
        for seg_i, (seg_end_nm, seg_speed) in enumerate(plan, start=1):
            seg_end_nm = float(seg_end_nm)
            seg_speed = float(seg_speed)

            wl_here = ws7_read_air_nm(ws7, ch2=ch2)
            direction = 0 if (seg_end_nm - (wl_here if wl_here > 0 else start_nm)) >= 0 else 1

            print(f"\n--- Segment {seg_i}/{len(plan)} ---")
            print(f"Segment: target_end={seg_end_nm:.6f} nm  speed={abs(seg_speed):.6f} nm/s  dir={direction}")

            safe_stop_scan(matisse)
            try:
                matisse.target_wavelength = float(seg_end_nm)
            except Exception:
                pass
            set_scan_speed(matisse, seg_speed)
            matisse.start_scan(direction)

            try:
                while True:
                    if (time.time() - start_wall) > max_run_time_s:
                        raise TimeoutError(f"Run exceeded max_run_time_s={max_run_time_s}s.")

                    t0 = time.monotonic()

                    counts = det.acquire_counts_timed(exposure_s)
                    if isinstance(counts, tuple):
                        counts = counts[0]

                    t1 = time.monotonic()

                    wl_mean, wl_std, n = wl_sampler.stats_between(t0, t1)

                    # --- Power reading (window mean) + forward fill ---
                    power_W = None
                    if pm_sampler is not None:
                        power_W = pm_sampler.mean_between(t0, t1)

                    # If missing/invalid, reuse last valid value
                    if power_W is None or (isinstance(power_W, float) and not np.isfinite(power_W)):
                        power_W = last_power_W
                    else:
                        last_power_W = power_W

                    # If we STILL don't have power (common at very beginning), do a single blocking read once
                    # so the dataset starts with a real number instead of NaN.
                    if power_W is None and powermeter is not None:
                        try:
                            reading_strings = powermeter.powermeter.get_instrument_reading_string_all()
                            readings_uW = []
                            for s in reading_strings:
                                try:
                                    readings_uW.append(powermeter.convert_reading_string_to_float(s))  # µW
                                except Exception:
                                    pass
                            if readings_uW:
                                power_W = float(sum(readings_uW) / len(readings_uW)) * 1e-6
                                last_power_W = power_W
                        except Exception:
                            pass

                    points.append(
                        ScanPoint(
                            t0=t0,
                            t1=t1,
                            wl_nm=wl_mean,
                            wl_std_nm=wl_std,
                            n_wl_samples=n,
                            power_W=power_W,  # now forward-filled
                            counts=int(counts),
                        )
                    )

                    # stop condition for this segment
                    if wl_mean > 0:
                        if direction == 0 and wl_mean >= (seg_end_nm - tol_nm):
                            break
                        if direction == 1 and wl_mean <= (seg_end_nm + tol_nm):
                            break

                    time.sleep(poll_dt_s)
            finally:
                safe_stop_scan(matisse)

            wl_after = ws7_read_air_nm(ws7, ch2=ch2)
            print(f"Segment end reached: WS7_air={wl_after if wl_after > 0 else None}")

        print("\nDONE (acquisition complete)")

    finally:
        safe_stop_scan(matisse)
        try:
            wl_sampler.stop()
        except Exception:
            pass
        try:
            det.close()
        except Exception:
            pass
        try:
            if pm_sampler is not None:
                pm_sampler.stop()
        except Exception:
            pass
        try:
            if pow_enabled and powermeter is not None:
                powermeter.powermeter.terminate_instrument()
        except Exception:
            pass

    # save
    save_meta_json_if_enabled(meta_path, meta, save_meta)
    if fmt == "h5":
        try:
            save_points_h5(data_path, points, meta)
            print(f"Saved HDF5: {data_path}")
        except Exception as e:
            fallback = save_points_npz(data_path, points, meta)
            print(f"HDF5 save failed ({e}). Saved NPZ: {fallback}")
    else:
        fallback = save_points_npz(data_path, points, meta)
        print(f"Saved NPZ: {fallback}")

    print(f"Acquired {len(points)} points.")
    if points:
        print(f"First wl_air={points[0].wl_nm:.6f} nm, last wl_air={points[-1].wl_nm:.6f} nm")

# -------------------------
# Mode 2: Andor kinetic series (kept similar, WS7 AIR sampling)
# -------------------------
def run_scan_andor_kinetic(cfg: Dict[str, Any], top_cfg: Optional[Dict[str, Any]] = None) -> None:

    import os, time, json, shutil, threading
    import numpy as np
    import matisse_controller.shamrock_ple.ple as ple_mod

    # ---------- helpers ----------
    def cfg_get(d, path, default=None):
        cur = d
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def wait_for_ccd_with_timeout(ccd, timeout_s: float) -> bool:
        done = {"ok": False}
        def worker():
            try:
                ccd.wait_for_acquisition()
                done["ok"] = True
            except Exception:
                done["ok"] = False
        th = threading.Thread(target=worker, daemon=True)
        th.start()
        th.join(timeout=max(0.0, float(timeout_s)))
        if th.is_alive():
            try:
                ccd.exit_flag = True
            except Exception:
                pass
            th.join(timeout=1.0)
            return False
        return bool(done["ok"])

    def frame_windows(t0: float, exposure_s: float, cycle_s: float, n_frames: int):
        return [(t0 + i * cycle_s, t0 + i * cycle_s + exposure_s) for i in range(n_frames)]


    def get_horizontal_shift_speed_from_yaml(top_cfg: Dict[str, Any]) -> Optional[Tuple[int, int, float]]:
        """
        Supports legacy location:
          PLEScan: DAQController: configure: horizontal_shift_speed: [AD, AMP, MHz]
        """
        try:
            hss = top_cfg.get("PLEScan", {}).get("DAQController", {}).get("configure", {}).get("horizontal_shift_speed",
                                                                                               None)
            if hss is None:
                return None
            ad, amp, mhz = hss
            return int(ad), int(amp), float(mhz)
        except Exception:
            return None

    def ensure_lock_and_loops():
        # Match the behavior that made SCAN responsive in timetagger mode
        try:
            # some versions expose "start_laser_lock()" instead of correction
            if hasattr(matisse, "start_laser_lock"):
                matisse.start_laser_lock()
        except Exception:
            pass

        try:
            if hasattr(matisse, "start_laser_lock_correction"):
                matisse.start_laser_lock_correction()
        except Exception:
            pass

        try:
            if hasattr(matisse, "start_control_loops"):
                matisse.start_control_loops()
        except Exception:
            pass

    # ---------- config ----------
    scan = cfg["scan"]
    out = cfg["output"]
    inst = cfg["instruments"]
    det = inst["detector"]["andor_kinetic"]

    start_nm = float(scan["start_nm"])
    end_nm   = float(scan["end_nm"])

    ws7_dt = float(scan.get("ws7_sample_period_s", 0.02))
    pm_dt  = float(scan.get("power_sample_period_s", 0.05))
    max_run_time_s = float(scan.get("max_run_time_s", 600))

    # kinetics
    exposure_s_req = float(det["exposure_s"])
    cycle_s_req    = float(det["cycle_s"])
    n_frames       = int(det["n_frames"])

    # temp/cooling keys (support both new + “old” names)
    temperature_C = float(det.get("temperature_C", det.get("target_sensor_temperature", -65.0)))
    temp_tol_C    = float(det.get("temp_tol_C", det.get("tol_C", det.get("temperature_tolerance_C", 1.0))))
    wait_for_cooldown = bool(det.get("wait_for_cooldown", det.get("reach_temperature_before_acquisition", False)))
    wait_timeout_s = float(det.get("wait_timeout_s", det.get("wait_timeout", 1800)))
    persist_cooling = bool(det.get("persist_cooling_on_shutdown", det.get("cooler_persistence", True)))

    # scan speed
    auto_speed = bool(scan.get("auto_scan_speed_from_kinetic", True))
    manual_speed = float(scan.get("scan_speed_nm_per_s", scan.get("scan_speed", 0.005)))

    # output
    out_dir = out.get("directory", "data")
    base = out.get("basename", "andor_kinetic")
    fmt = out.get("format", "h5").lower()
    overwrite = bool(out.get("overwrite", True))
    save_meta = bool(out.get("save_meta_json", True))

    os.makedirs(out_dir, exist_ok=True)
    sif_path  = os.path.join(out_dir, f"{base}.sif")
    meta_path = os.path.join(out_dir, f"{base}.meta.json")
    data_path = os.path.join(out_dir, f"{base}.{'h5' if fmt=='h5' else 'npz'}")

    if (not overwrite) and (os.path.exists(sif_path) or os.path.exists(data_path)):
        raise FileExistsError(f"Output exists and overwrite=false: {sif_path} / {data_path}")

    from matisse_controller.shamrock_ple.constants import (
        READ_MODE_FVB,
        READ_MODE_SINGLE_TRACK,
        COSMIC_RAY_FILTER_ON,
        COSMIC_RAY_FILTER_OFF,
    )

    def parse_andor_readout_mode(mode) -> int:
        # Accept strings, bool/int, None
        if mode is None:
            return READ_MODE_FVB
        if isinstance(mode, (int, np.integer)):
            return int(mode)
        s = str(mode).strip().upper()
        if s == "FVB":
            return READ_MODE_FVB
        if s in ("SINGLE_TRACK", "SINGLETRACK"):
            return READ_MODE_SINGLE_TRACK
        raise ValueError(f"Unknown Andor readout_mode: {mode!r}")

    def parse_cosmic_ray_filter(val) -> int:
        if val is None:
            return COSMIC_RAY_FILTER_ON
        if isinstance(val, bool):
            return COSMIC_RAY_FILTER_ON if val else COSMIC_RAY_FILTER_OFF
        if isinstance(val, (int, np.integer)):
            return int(val)
        s = str(val).strip().upper()
        if s in ("ON", "TRUE", "1", "YES"):
            return COSMIC_RAY_FILTER_ON
        if s in ("OFF", "FALSE", "0", "NO"):
            return COSMIC_RAY_FILTER_OFF
        raise ValueError(f"Unknown cosmic_ray_filter: {val!r}")

    # ---------- connect WS7 ----------
    print("Connecting WS7...")
    ws7 = WS7()
    bind_ws7_prototypes(ws7.lib)
    wl_now = ws7_read_air_nm(ws7, ch2=bool(scan.get("ws7_ch2", False)))
    print(f"WS7 OK. Current WS7_air={wl_now if wl_now>0 else None}")

    # ---------- connect Matisse ----------
    print("\nConnecting Matisse...")
    matisse = Matisse(wavemeter_type="WS7")
    print("Matisse OK.")
    try:
        print("Laser locked?:", matisse.laser_locked())
    except Exception:
        print("Laser locked?: (unknown)")

    # ---------- load Andor libs  ----------
    print("\nLoading Andor libs ...")
    # Some versions allow device indexes; try, then fallback:
    ccd_device_index = cfg_get(cfg, "PLEScan.DAQController.configure.ccd_device_index", None)
    spg_device_index = cfg_get(cfg, "PLEScan.DAQController.configure.spg_device_index", None)
    try:
        if ccd_device_index is not None or spg_device_index is not None:
            ple_mod.PLE.load_andor_libs(ccd_device_index=ccd_device_index, spg_device_index=spg_device_index)
        else:
            ple_mod.PLE.load_andor_libs()
    except TypeError:
        ple_mod.PLE.load_andor_libs()

    ccd = ple_mod.ccd
    shamrock = getattr(ple_mod, "shamrock", None)  # may be None
    if ccd is None:
        raise RuntimeError("CCD global is None after PLE.load_andor_libs().")

    # ---------- cooling (match your working test) ----------
    try:
        ccd.ensure_cooling(temperature_C, persist_on_shutdown=persist_cooling)
    except Exception as e:
        print(f"WARNING: ensure_cooling failed: {e}")

    if wait_for_cooldown:
        print(f"Waiting for CCD to reach {temperature_C}±{temp_tol_C} C (timeout={wait_timeout_s}s)")
        try:
            ccd.wait_to_cooldown(target_C=temperature_C, tol_C=temp_tol_C, poll_s=5.0, timeout_s=wait_timeout_s)
        except Exception as e:
            raise RuntimeError(f"Cooldown wait failed: {e}")
    else:
        try:
            t_now = ccd.wait_until_cold(temperature_C, tol_C=temp_tol_C, timeout_s=0.0)
            print(f"Temp check (non-blocking): current CCD temp = {t_now:.1f} C")
        except Exception:
            pass

    #enforce horizontal shift speed
    # -----------------------------
    # Enforce horizontal shift speed
    # -----------------------------
    full = cfg.get("_full_yaml", {})

    legacy_hss = (
        (((full.get("PLEScan") or {}).get("DAQController") or {}).get("configure") or {})
        .get("horizontal_shift_speed", None)
    )

    yaml_hss = det.get("horizontal_shift_speed", None)  # if you also allow it in RunScan

    # Always enforce to the known-good setting
    HSS_ENFORCED = (0, 0, 0.05)

    if yaml_hss is not None and tuple(yaml_hss) != HSS_ENFORCED:
        print(f"WARNING: overriding RunScan horizontal_shift_speed={yaml_hss} -> {HSS_ENFORCED}")
    if legacy_hss is not None and tuple(legacy_hss) != HSS_ENFORCED:
        print(f"WARNING: overriding PLEScan horizontal_shift_speed={legacy_hss} -> {HSS_ENFORCED}")

    def _set_horizontal_shift_speed(ccd_obj, hss_tuple):
        # Try common method names (depends on your CCD wrapper)
        for name in (
                "set_horizontal_shift_speed",
                "set_horiz_shift_speed",
                "set_hs_speed",
                "setHorizontalShiftSpeed",
        ):
            if hasattr(ccd_obj, name):
                getattr(ccd_obj, name)(hss_tuple)
                return True
        return False

    ok = _set_horizontal_shift_speed(ccd, HSS_ENFORCED)
    if ok:
        print(f"CCD horizontal_shift_speed enforced: {HSS_ENFORCED}")
    # else:
        # raise RuntimeError(
        #     "Your CCD wrapper doesn't expose a setter like set_horizontal_shift_speed(...). "
        #     "Add a method for it (recommended), or expose it through setup_kinetics in the wrapper."
        # )

    # ---------- optional shamrock config from old YAML ----------
    # Only apply if device exposes methods; otherwise just store in meta.
    old_cfg = cfg_get(cfg, "PLEScan.DAQController.configure", {}) or {}
    try:
        if shamrock is not None:
            center_wl = old_cfg.get("center_wavelength", None)
            if center_wl is not None and hasattr(shamrock, "set_wavelength"):
                shamrock.set_wavelength(float(center_wl))
            grating = old_cfg.get("grating", None)
            if grating is not None and hasattr(shamrock, "set_grating"):
                shamrock.set_grating(grating)
    except Exception as e:
        print(f"WARNING: shamrock config skipped: {e}")

    # ---------- power meter ----------
    pow_cfg = inst.get("powermeter", {}) or {}
    pow_enabled = bool(pow_cfg.get("enabled", False))
    pow_channel = str(pow_cfg.get("channel", "B"))

    powermeter = None
    pm_sampler = None
    if pow_enabled:
        print(f"\nConnecting powermeter (channel {pow_channel})...")
        powermeter = PowerMeter(pow_channel)
        powermeter.powermeter.initialize_instrument()
        try:
            powermeter._empty_buffer()
        except Exception:
            pass
        pm_sampler = PowerSampler(powermeter)
        pm_sampler.start(sample_period_s=pm_dt)
        print("Power meter OK.")
    else:
        print("\nPowermeter disabled.")

    # ---------- ws7 sampler ----------
    wl_sampler = WavelengthSampler(ws7, ch2=False)
    wl_sampler.start(sample_period_s=ws7_dt)

    # ---------- configure kinetics ----------
    print("\nConfiguring kinetics...")
    exp_actual_s, cycle_actual_s = ccd.setup_kinetics(
        exposure_time=exposure_s_req,
        cycle_time=cycle_s_req,
        n_frames=n_frames,
        readout_mode=parse_andor_readout_mode(det.get("readout_mode", "FVB")),
        temperature=temperature_C,
        cool_down=False,  # do not block here (matches your test script)
        cosmic_ray_filter=parse_cosmic_ray_filter(det.get("cosmic_ray_filter", True)),
    )
    print(f"Kinetics configured: exp_actual={exp_actual_s:.6f}s, cycle_actual={cycle_actual_s:.6f}s")


    print(f"\nStarting Andor kinetics and segment scan ...")

    # -----------------------
    # Pre-lock / align start (same logic as timetagger mode)
    # -----------------------
    tol_nm = float(scan.get("start_tolerance_nm", scan.get("tol_nm", 5e-5)))
    prelock = str(scan.get("prelock", "stabilize")).strip().lower()  # none|stabilize|full
    stabilize_settle_s = float(scan.get("stabilize_settle_s", 0.5))

    seek_timeout_s = float(scan.get("seek_timeout_s", 90.0))
    seek_coarse_speed = float(scan.get("seek_coarse_speed_nm_per_s", 0.01))
    seek_fine_speed = float(scan.get("seek_fine_speed_nm_per_s", 0.0005))
    seek_fine_window = float(scan.get("seek_fine_window_nm", max(5 * tol_nm, 5e-4)))
    poll_dt_s = float(scan.get("poll_dt_s", 0.05))

    def ensure_lock_and_loops_local():
        # mirrors your timetagger behavior
        try:
            if hasattr(matisse, "start_laser_lock_correction"):
                matisse.start_laser_lock_correction()
        except Exception:
            pass
        try:
            if hasattr(matisse, "start_control_loops"):
                matisse.start_control_loops()
        except Exception:
            pass

    print(f"\nPre-lock mode: {prelock}")
    safe_stop_scan(matisse)
    ensure_lock_and_loops_local()

    # never let stabilize fight seeking
    try:
        matisse.stabilize_off()
    except Exception:
        pass
    safe_stop_scan(matisse)

    if prelock == "full":
        # heavy/slow absolute path if your driver supports it
        print(f"Calling matisse.set_wavelength({start_nm:.6f}) ...")
        matisse.set_wavelength(start_nm)

    elif prelock == "stabilize":
        wl_now = ws7_read_air_nm(ws7, ch2=False)
        if wl_now > 0 and abs(wl_now - start_nm) > tol_nm:
            wl_seek = seek_to_wavelength(
                matisse=matisse,
                ws7=ws7,
                target_nm=start_nm,
                tol_nm=tol_nm,
                poll_dt_s=poll_dt_s,
                timeout_s=seek_timeout_s,
                coarse_speed_nm_s=seek_coarse_speed,
                fine_speed_nm_s=seek_fine_speed,
                fine_window_nm=seek_fine_window,
                ch2=False,
            )
            print(f"Seek result: WS7_air={wl_seek:.9f}")
        else:
            print("Already within tolerance; skipping SEEK.")

        # “lock at start”: prefer your proven lock path if available, else stabilize briefly
        locked = False
        try:
            if "ple" in locals() and hasattr(ple, "lock_at_wavelength"):
                print(f"Locking at start {start_nm:.6f} nm via ple.lock_at_wavelength() ...")
                ple.lock_at_wavelength(round(start_nm, 6))
                locked = True
        except Exception:
            locked = False

        if not locked:
            # fallback: brief stabilize settle at start
            try:
                matisse.target_wavelength = float(start_nm)
            except Exception:
                pass
            try:
                matisse.stabilize_on()
                if stabilize_settle_s > 0:
                    time.sleep(stabilize_settle_s)
            finally:
                try:
                    matisse.stabilize_off()
                except Exception:
                    pass

    elif prelock == "none":
        pass
    else:
        raise ValueError(f"Unknown prelock={prelock!r}. Use none|stabilize|full.")

    wl_start = ws7_read_air_nm(ws7, ch2=False)
    print(f"Start check: WS7_air={wl_start if wl_start > 0 else None} (target {start_nm:.6f}±{tol_nm:.6f})")

    # critical: stabilize OFF before scanning
    try:
        matisse.stabilize_off()
    except Exception:
        pass
    safe_stop_scan(matisse)

    # ---- Build segment plan from YAML (same behavior as timetagger mode) ----
    segments_str = scan.get("segments", None)
    relative = bool(scan.get("relative", False))
    default_scan_speed = float(scan.get("scan_speed_nm_per_s", scan.get("scan_speed", 0.005)))
    plan = build_scan_plan(start_nm, end_nm, default_scan_speed, segments_str, relative)
    if not plan:
        raise ValueError("Empty scan plan. Check scan.segments / scan.end_nm.")
    end_nm_plan = float(plan[-1][0])

    # tolerance / polling for segment boundaries
    seg_tol_nm = float(scan.get("segment_tolerance_nm", scan.get("start_tolerance_nm", scan.get("tol_nm", 5e-5))))
    poll_dt_s = float(scan.get("poll_dt_s", 0.05))

    # ---- detect which dir increases wavelength (best-effort) ----
    def detect_dir_map(
            tap_s: float = 0.4,
            test_speed: float = 0.01,
            min_move_nm: float = 2e-6,
    ) -> tuple[int, int]:
        """
        Returns (dir_up, dir_down) where dir_up SHOULD make wavelength increase.

        IMPORTANT: drift/lock correction can cause both taps to move the same sign.
        If ambiguous, we fall back to (0,1) and rely on per-segment sanity checks.
        """
        try:
            set_scan_speed(matisse, float(test_speed))
            deltas: dict[int, float] = {}

            for d in (0, 1):
                safe_stop_scan(matisse)

                wl_before = ws7_read_air_nm(ws7, ch2=False)
                if wl_before <= 0:
                    continue

                matisse.start_scan(int(d))
                time.sleep(max(0.2, float(tap_s)))
                safe_stop_scan(matisse)

                wl_after = ws7_read_air_nm(ws7, ch2=False)
                if wl_after > 0:
                    deltas[d] = wl_after - wl_before

                time.sleep(0.1)

            if 0 in deltas and 1 in deltas:
                d0, d1 = deltas[0], deltas[1]
                print(f"(dir_map) tap deltas: dir0={d0:+.6e} nm, dir1={d1:+.6e} nm", flush=True)

                # Ambiguous: both moved same sign (drift dominates)
                if (d0 > +min_move_nm and d1 > +min_move_nm) or (d0 < -min_move_nm and d1 < -min_move_nm):
                    print("(dir_map) ambiguous (both deltas same sign) -> default dir_up=0, dir_down=1", flush=True)
                    return (0, 1)

                # Clear case: opposite signs
                if d0 > +min_move_nm and d1 < -min_move_nm:
                    return (0, 1)
                if d1 > +min_move_nm and d0 < -min_move_nm:
                    return (1, 0)

                # Weak/noisy: choose "more increasing", but segment sanity check may swap later
                dir_up = 0 if d0 > d1 else 1
                return (dir_up, 1 - dir_up)

        except Exception as e:
            print(f"(dir_map) WARNING: direction detect failed: {e}", flush=True)

        finally:
            safe_stop_scan(matisse)

        return (0, 1)

    dir_up0, dir_down0 = detect_dir_map()
    dir_map = {"up": int(dir_up0), "down": int(dir_down0)}

    print(f"\nStarting Andor kinetics and segment scan (dir_up={dir_map['up']}, dir_down={dir_map['down']}) ...",
          flush=True)

    stop_evt = threading.Event()
    scan_err: dict[str, Exception | None] = {"exc": None}


    # --- start scan thread + acquisition (you should already have this around it) ---
    # ---- start acquisition + segment scanning with PRIMING GATE ----
    scan_ready_evt = threading.Event()  # scan is ready to start "for real"
    go_evt = threading.Event()  # permission to begin the real scan

    prime_timeout_s = float(scan.get("prime_timeout_s", 20.0))
    prime_seek_back = bool(scan.get("prime_seek_back_to_start", True))
    prime_tap_s = float(scan.get("prime_tap_s", 0.5))
    prime_max_tries = int(scan.get("prime_max_tries", 6))

    def prime_first_segment_direction() -> None:
        """
        Decide the correct mapping using ONLY the first segment, BEFORE Andor acquisition starts.
        Updates dir_map in-place if needed.
        """
        if not plan:
            raise RuntimeError("prime: empty plan")

        seg_end_nm, seg_speed = plan[0]
        seg_end_nm = float(seg_end_nm)
        seg_speed = float(seg_speed)

        # ensure stabilize isn't fighting
        try:
            matisse.stabilize_off()
        except Exception:
            pass
        safe_stop_scan(matisse)

        for k in range(prime_max_tries):
            wl_a = ws7_read_air_nm(ws7, ch2=False)
            if wl_a <= 0:
                time.sleep(0.05)
                continue

            err_a = abs(seg_end_nm - wl_a)
            need_up = (seg_end_nm >= wl_a)
            direction = dir_map["up"] if need_up else dir_map["down"]

            # short tap at the REAL segment speed
            safe_stop_scan(matisse)
            try:
                matisse.target_wavelength = round(seg_end_nm, 6)
            except Exception:
                pass
            set_scan_speed(matisse, seg_speed)
            matisse.start_scan(int(direction))
            time.sleep(max(0.25, prime_tap_s))
            safe_stop_scan(matisse)

            wl_b = ws7_read_air_nm(ws7, ch2=False)
            if wl_b <= 0:
                continue
            err_b = abs(seg_end_nm - wl_b)

            # if error got worse, mapping is wrong -> swap
            if err_b > err_a + float(scan.get("dir_check_err_eps_nm", 2e-5)):
                print(
                    f"(prime) WRONG DIR for seg1 (wl {wl_a:.9f}->{wl_b:.9f}, |err| {err_a:.6e}->{err_b:.6e}). "
                    f"Swapping dir_map.",
                    flush=True
                )
                dir_map["up"], dir_map["down"] = dir_map["down"], dir_map["up"]
                continue

            # good enough
            print(f"(prime) Direction looks good for seg1 after {k + 1} tries.", flush=True)
            return

        raise RuntimeError("(prime) Could not find a direction that reduces |error| for seg1.")

    def segment_scan_worker():
        try:
            # --------- PRIMING (NO Andor acquisition yet) ---------
            prime_first_segment_direction()

            # optional: return to start + lock/settle again so kinetics starts clean
            if prime_seek_back:
                safe_stop_scan(matisse)
                try:
                    matisse.stabilize_off()
                except Exception:
                    pass

                wl_seek = seek_to_wavelength(
                    matisse=matisse,
                    ws7=ws7,
                    target_nm=float(start_nm),
                    tol_nm=float(scan.get("start_tolerance_nm", scan.get("tol_nm", 5e-5))),
                    poll_dt_s=float(scan.get("poll_dt_s", 0.05)),
                    timeout_s=float(scan.get("seek_timeout_s", 90.0)),
                    coarse_speed_nm_s=float(scan.get("seek_coarse_speed_nm_per_s", 0.01)),
                    fine_speed_nm_s=float(scan.get("seek_fine_speed_nm_per_s", 0.0005)),
                    fine_window_nm=float(scan.get("seek_fine_window_nm", 5e-4)),
                    ch2=False,
                )
                print(f"(prime) Back at start: WS7_air={wl_seek:.9f}", flush=True)

                # optional brief settle (then OFF before scan)
                settle = float(scan.get("stabilize_settle_s", 0.3))
                try:
                    matisse.target_wavelength = float(start_nm)
                    matisse.stabilize_on()
                    if settle > 0:
                        time.sleep(settle)
                finally:
                    try:
                        matisse.stabilize_off()
                    except Exception:
                        pass
                safe_stop_scan(matisse)

            # signal main thread that scan is ready, then wait for go
            scan_ready_evt.set()
            if not go_evt.wait(timeout=prime_timeout_s):
                raise TimeoutError("Scan worker never received go_evt to start scan.")

            # --------- REAL segment scan (during Andor kinetics) ---------
            try:
                matisse.stabilize_off()
            except Exception:
                pass

            # from here onward: do NOT do any more tap-based direction discovery;
            # just run segments using the dir_map we already primed.
            for seg_end_nm, seg_speed in plan:
                if stop_evt.is_set():
                    break

                seg_end_nm = float(seg_end_nm)
                seg_speed = float(seg_speed)

                wl_here = ws7_read_air_nm(ws7, ch2=False)
                if wl_here <= 0:
                    wl_here = start_nm

                need_up = (seg_end_nm >= wl_here)
                direction = dir_map["up"] if need_up else dir_map["down"]

                print(
                    f"  segment -> end={seg_end_nm:.6f} nm  speed={abs(seg_speed):.6f} nm/s  "
                    f"need_up={need_up}  dir={direction} (map up={dir_map['up']} down={dir_map['down']})",
                    flush=True
                )

                safe_stop_scan(matisse)
                try:
                    matisse.target_wavelength = round(seg_end_nm, 6)
                except Exception:
                    pass
                set_scan_speed(matisse, seg_speed)
                matisse.start_scan(int(direction))

                while not stop_evt.is_set():
                    wl = ws7_read_air_nm(ws7, ch2=False)
                    if wl > 0:
                        if direction == dir_map["up"] and wl >= (seg_end_nm - seg_tol_nm):
                            break
                        if direction == dir_map["down"] and wl <= (seg_end_nm + seg_tol_nm):
                            break
                    time.sleep(poll_dt_s)

                safe_stop_scan(matisse)

            safe_stop_scan(matisse)

        except Exception as e:
            scan_err["exc"] = e
            stop_evt.set()
            safe_stop_scan(matisse)

    # start scan thread (it will PRIME, then wait)
    scan_th = threading.Thread(target=segment_scan_worker, daemon=True)
    scan_th.start()

    # wait until scan thread says it's ready (direction known + optionally back to start)
    if not scan_ready_evt.wait(timeout=prime_timeout_s):
        stop_evt.set()
        raise TimeoutError("Scan priming did not finish before prime_timeout_s.")

    # start kinetics ONLY after scan is ready, then release scan to begin real motion
    t_acq_start = time.monotonic()
    go_evt.set()
    ccd.start_acquisition()

    ok = wait_for_ccd_with_timeout(ccd, timeout_s=max_run_time_s)
    t_acq_end = time.monotonic()

    if scan_err["exc"] is not None:
        raise scan_err["exc"]
    if not ok:
        raise TimeoutError(f"Andor kinetics did not finish within max_run_time_s={max_run_time_s}s.")

    # stop samplers AFTER acquisition is done (buffer remains for stats)
    try:
        wl_sampler.stop()
    except Exception:
        pass
    if pm_sampler is not None:
        try:
            pm_sampler.stop()
        except Exception:
            pass

    # ---- Frame-aligned extraction (wl + power) ----
    if t_acq_start is None or t_acq_end is None:
        # defensive fallback (shouldn't happen now)
        t_acq_start = time.monotonic()
        t_acq_end = t_acq_start

    wins = frame_windows(t_acq_start, exp_actual_s, cycle_actual_s, n_frames)

    frame_wl = np.full(n_frames, np.nan, dtype=float)
    frame_wl_std = np.full(n_frames, np.nan, dtype=float)
    frame_power = np.full(n_frames, np.nan, dtype=float)

    last_power = np.nan
    for i, (t0, t1) in enumerate(wins):
        wl_mean, wl_std, _n = wl_sampler.stats_between(t0, t1)
        frame_wl[i] = wl_mean
        frame_wl_std[i] = wl_std

        if pm_sampler is not None:
            p = pm_sampler.mean_between(t0, t1)
            if p is None or (isinstance(p, float) and not np.isfinite(p)):
                p = last_power
            else:
                last_power = p
            frame_power[i] = p

    # ---------- save SIF (matches your working test script) ----------
    tmp_name = f"{base}.sif"
    ccd.save_as_sif(tmp_name)  # no spectrograph coeff dependency
    try:
        if os.path.exists(sif_path):
            os.remove(sif_path)
        shutil.move(tmp_name, sif_path)
    except Exception:
        pass

    # ---------- align WS7/power to frames ----------
    if t_acq_start is None:
        t_acq_start = time.monotonic()
    if t_acq_end is None:
        t_acq_end = t_acq_start

    wins = frame_windows(t_acq_start, exp_actual_s, cycle_actual_s, n_frames)

    frame_wl = np.full(n_frames, np.nan, dtype=float)
    frame_wl_std = np.full(n_frames, np.nan, dtype=float)
    frame_power = np.full(n_frames, np.nan, dtype=float)

    for i, (t0, t1) in enumerate(wins):
        try:
            wl_mean, wl_std, _n = wl_sampler.stats_between(t0, t1)
            frame_wl[i] = wl_mean
            frame_wl_std[i] = wl_std
        except Exception:
            pass

        if pm_sampler is not None:
            p = pm_sampler.mean_between(t0, t1)
            if p is None or (isinstance(p, float) and not np.isfinite(p)):
                p = last_power_W
            else:
                last_power_W = p
            if p is not None:
                frame_power[i] = float(p)

    raw_wl_t = np.array([s.t for s in wl_sampler.buf], dtype=float)
    raw_wl   = np.array([s.wl_nm for s in wl_sampler.buf], dtype=float)

    if pm_sampler is not None:
        raw_p_t = np.array([s.t for s in pm_sampler.buf], dtype=float)
        raw_p   = np.array([s.power_W for s in pm_sampler.buf], dtype=float)
    else:
        raw_p_t = np.array([], dtype=float)
        raw_p   = np.array([], dtype=float)

    # --- define scan speed fields for metadata ---
    duration_measured_s = float(t_acq_end - t_acq_start) if (
                t_acq_start is not None and t_acq_end is not None) else float("nan")
    scan_speed_effective_nm_per_s = (
        abs(end_nm - start_nm) / duration_measured_s
        if np.isfinite(duration_measured_s) and duration_measured_s > 0
        else float("nan")
    )

    meta = {
        "mode": "andor_kinetic",
        "start_nm_air": start_nm,
        "end_nm_air": end_nm,
        "scan_speed_nm_per_s_effective": float(scan_speed_effective_nm_per_s),
        "segments": [(float(a), float(b)) for (a, b) in plan],  # if you're using plan in kinetic mode
        "auto_scan_speed_from_kinetic": auto_speed,
        "timestamps": {
            "t_acq_start_monotonic": float(t_acq_start),
            "t_acq_end_monotonic": float(t_acq_end),
        },
        "wavemeter": {"ws7_sample_period_s": ws7_dt},
        "powermeter": {
            "enabled": pow_enabled,
            "channel": pow_channel if pow_enabled else None,
            "power_sample_period_s": pm_dt if pow_enabled else None,
        },
        "andor": {
            "n_frames": n_frames,
            "exposure_s_requested": exposure_s_req,
            "cycle_s_requested": cycle_s_req,
            "exposure_s_actual": exp_actual_s,
            "cycle_s_actual": cycle_actual_s,
            "temperature_C": temperature_C,
            "persist_cooling_on_shutdown": persist_cooling,
            "sif_path": os.path.abspath(sif_path),
        },
        "old_device_config": old_cfg,  # store your legacy block
        "created_unix_s": time.time(),
    }

    if save_meta:
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    # save aligned arrays + raw traces
    if fmt == "h5":
        import h5py
        with h5py.File(data_path, "w") as h:
            h.attrs["meta_json"] = json.dumps(meta)
            h.create_dataset("frame_wl_nm_air", data=frame_wl)
            h.create_dataset("frame_wl_std_nm_air", data=frame_wl_std)
            h.create_dataset("frame_power_W", data=frame_power)
            h.create_dataset("raw_wl_t_monotonic", data=raw_wl_t)
            h.create_dataset("raw_wl_nm_air", data=raw_wl)
            h.create_dataset("raw_power_t_monotonic", data=raw_p_t)
            h.create_dataset("raw_power_W", data=raw_p)
    else:
        base_npz = os.path.splitext(data_path)[0] + ".npz"
        np.savez(
            base_npz,
            frame_wl_nm_air=frame_wl,
            frame_wl_std_nm_air=frame_wl_std,
            frame_power_W=frame_power,
            raw_wl_t_monotonic=raw_wl_t,
            raw_wl_nm_air=raw_wl,
            raw_power_t_monotonic=raw_p_t,
            raw_power_W=raw_p,
            meta=np.array([meta], dtype=object),
        )

    print(f"\nSaved SIF : {sif_path}")
    if save_meta:
        print(f"Saved meta: {meta_path}")
    print(f"Saved data: {data_path}")
    print(f"Kinetics frames: {n_frames}, measured duration: {(t_acq_end - t_acq_start):.3f}s")

# -------------------------
# Main dispatch
# -------------------------
def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    cfg = load_config(cfg_path)

    mode = cfg_get(cfg, "instruments.detector.mode", None)
    if mode is None:
        raise ValueError("Config missing: RunScan.instruments.detector.mode")

    mode = str(mode).strip().lower()
    if mode == "timetagger_counts":
        run_scan_timetagger_counts(cfg)
    elif mode == "andor_kinetic":
        run_scan_andor_kinetic(cfg)
    else:
        raise ValueError(f"Unknown detector mode: {mode!r} (expected 'timetagger_counts' or 'andor_kinetic')")


if __name__ == "__main__":
    main()
