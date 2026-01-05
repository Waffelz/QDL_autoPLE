#!/usr/bin/env python3
"""
run_scan.py (place at project root)

Supports two detector modes (selected in YAML):
  1) timetagger_counts:
     - Start scan from start_nm to end_nm
     - For each exposure window:
         * mean/std wavelength from WS7 samples within the window
         * one power reading near midpoint (Newport)
         * TimeTagger integrated counts over exposure_s
     - Save points to HDF5 (preferred) or NPZ fallback

  2) andor_kinetic:
     - Configure Andor CCD kinetics (exposure_s, cycle_s, n_frames, readout_mode, cosmic_ray_filter)
     - Optionally auto set scan speed so scan duration matches kinetic duration
     - Start scan + start kinetic acquisition
     - Stop WS7/power sampling exactly when kinetic ends (or timeout)
     - Save raw .sif kinetic series
     - Save per-frame wavelength/power arrays + raw traces to HDF5/NPZ (+ meta.json)

Usage:
  python run_scan.py                 # uses configs/run_scan.yml
  python run_scan.py path/to.yml     # use specific config file
"""

from __future__ import annotations

import os
import sys
import time
import json
import shutil
import threading
from dataclasses import dataclass
from collections import deque
from typing import Deque, Optional, Dict, Any, List, Tuple

import numpy as np

# YAML config
try:
    import yaml
except ImportError as e:
    raise RuntimeError("Missing dependency: pyyaml. Install with: pip install pyyaml") from e

# Import PLE module so we can access module-level globals after setup_* assigns them
import matisse_controller.shamrock_ple.ple as ple_mod

# Swabian Time Tagger Python bindings:
# Most installs use `import TimeTagger`. Some environments provide `from Swabian import TimeTagger`.
try:
    import TimeTagger  # type: ignore
except ImportError:
    from Swabian import TimeTagger  # type: ignore

from matisse_controller.shamrock_ple.constants import (
    READ_MODE_FVB,
    READ_MODE_SINGLE_TRACK,
    READ_MODE_MULTI_TRACK,
    READ_MODE_RANDOM_TRACK,
    READ_MODE_IMAGE,
    COSMIC_RAY_FILTER_ON,
    COSMIC_RAY_FILTER_OFF,
)

DEFAULT_CONFIG_PATH = "configs/run_scan.yml"


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
    power_W: Optional[float]
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
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        top = yaml.safe_load(f) or {}
    cfg = top.get("RunScan", {})
    if not cfg:
        raise ValueError("YAML missing top-level key: RunScan")
    return cfg


def cfg_get(d: Dict[str, Any], path: str, default=None):
    """Safe nested get: cfg_get(cfg, 'scan.start_nm')"""
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


# -------------------------
# Andor YAML -> constants parsing
# -------------------------
def parse_andor_readout_mode(mode) -> int:
    """
    Map YAML string/int -> Andor SDK constant.

    Accepts: FVB, SINGLE_TRACK, MULTI_TRACK, RANDOM_TRACK, IMAGE (case-insensitive),
             or numeric values.
    """
    if mode is None:
        return READ_MODE_FVB

    # allow numeric
    if isinstance(mode, (int, float)):
        return int(mode)

    s = str(mode).strip().upper()
    mapping = {
        "FVB": READ_MODE_FVB,
        "SINGLE_TRACK": READ_MODE_SINGLE_TRACK,
        "SINGLETRACK": READ_MODE_SINGLE_TRACK,
        "MULTI_TRACK": READ_MODE_MULTI_TRACK,
        "MULTITRACK": READ_MODE_MULTI_TRACK,
        "RANDOM_TRACK": READ_MODE_RANDOM_TRACK,
        "RANDOMTRACK": READ_MODE_RANDOM_TRACK,
        "IMAGE": READ_MODE_IMAGE,
    }
    if s in mapping:
        return mapping[s]

    raise ValueError(
        f"Unknown Andor readout_mode: {mode!r}. "
        "Use FVB / SINGLE_TRACK / MULTI_TRACK / RANDOM_TRACK / IMAGE (or an int)."
    )


def parse_cosmic_ray_filter(val) -> int:
    """
    Accept bool/string/int -> Andor SDK constant.

    true/ON -> COSMIC_RAY_FILTER_ON (2)
    false/OFF -> COSMIC_RAY_FILTER_OFF (0)
    """
    if val is None:
        return COSMIC_RAY_FILTER_ON

    if isinstance(val, bool):
        return COSMIC_RAY_FILTER_ON if val else COSMIC_RAY_FILTER_OFF

    if isinstance(val, (int, float)):
        return int(val)

    s = str(val).strip().upper()
    if s in ("ON", "TRUE", "1", "YES"):
        return COSMIC_RAY_FILTER_ON
    if s in ("OFF", "FALSE", "0", "NO"):
        return COSMIC_RAY_FILTER_OFF

    raise ValueError(f"Unknown cosmic_ray_filter: {val!r}. Use bool, ON/OFF, or 0/2.")


# -------------------------
# WS7 sampler (background)
# -------------------------
class WavelengthSampler:
    """
    Background WS7 sampler using ws7.lib.GetWavelength(0.0).
    Uses time.monotonic() so exposure windows align correctly.
    """
    def __init__(self, ws7_instance, maxlen: int = 200_000):
        self.ws7 = ws7_instance
        self.buf: Deque[WlSample] = deque(maxlen=maxlen)
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None
        self._io_lock = threading.Lock()

    def _read_wl_nm_once(self) -> Optional[float]:
        try:
            with self._io_lock:
                wl = float(self.ws7.lib.GetWavelength(0.0))
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

        # fallback: nearest sample
        if not self.buf:
            raise RuntimeError("No WS7 samples available.")
        mid = 0.5 * (t0 + t1)
        nearest = min(self.buf, key=lambda s: abs(s.t - mid))
        return float(nearest.wl_nm), 0.0, 0


# -------------------------
# Power meter sampler (background) - for Andor kinetics alignment
# -------------------------
class PowerSampler:
    """
    Continuous timestamped sampling of Newport power meter.

    Expects `powermeter_instance` to be your functions26.instruments.powermeter.PowerMeter object:
      - powermeter_instance.powermeter.get_instrument_reading_string_all()
      - powermeter_instance.convert_reading_string_to_float() -> µW (per your driver)
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

                next_t += sample_period_s

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

        mid = 0.5 * (t0 + t1)
        nearest = min(self.buf, key=lambda s: abs(s.t - mid))
        return float(nearest.power_W)


# -------------------------
# Newport: one read near midpoint of exposure (for TimeTagger per-point)
# -------------------------
def start_midpoint_power_read(powermeter_instance, exposure_s: float) -> Tuple[threading.Thread, Dict[str, Any]]:
    """
    One-shot power reading around exposure midpoint.
    Returns (thread, out_dict) where out_dict["power_W"] is filled when thread completes.
    """
    out: Dict[str, Any] = {"power_W": None, "error": None}

    def worker():
        try:
            time.sleep(max(0.0, 0.5 * exposure_s))
            reading_strings = powermeter_instance.powermeter.get_instrument_reading_string_all()
            readings_uW: List[float] = []
            for s in reading_strings:
                try:
                    readings_uW.append(powermeter_instance.convert_reading_string_to_float(s))  # µW
                except Exception:
                    pass
            if readings_uW:
                out["power_W"] = float(sum(readings_uW) / len(readings_uW)) * 1e-6  # W
        except Exception as e:
            out["error"] = e

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    return th, out


# -------------------------
# Time Tagger counts
# -------------------------
def _edge_channel(phys_ch: int, edge: str) -> int:
    edge = edge.strip().lower()
    if edge == "rising":
        return int(abs(phys_ch))
    if edge == "falling":
        return -int(abs(phys_ch))
    raise ValueError("edge must be 'rising' or 'falling'")


class TimeTaggerCounts:
    """
    Minimal counts detector using TimeTagger.Counter:
      acquire_counts(exposure_s) -> int
    """
    def __init__(self, click_phys_ch: int, click_trigger_v: float, click_edge: str, serial: Optional[str] = None):
        self.click_phys_ch = int(click_phys_ch)
        self.click_trigger_v = float(click_trigger_v)
        self.click_sw_ch = _edge_channel(self.click_phys_ch, click_edge)
        self.serial = serial
        self._tagger = None

    def connect(self) -> None:
        if self._tagger is not None:
            return
        self._tagger = TimeTagger.createTimeTagger(self.serial) if self.serial else TimeTagger.createTimeTagger()
        # Trigger level set on physical channel (positive integer)
        self._tagger.setTriggerLevel(self.click_phys_ch, self.click_trigger_v)

    def close(self) -> None:
        if self._tagger is not None:
            TimeTagger.freeTimeTagger(self._tagger)
            self._tagger = None

    @property
    def tagger(self):
        if self._tagger is None:
            raise RuntimeError("TimeTagger not connected.")
        return self._tagger

    def acquire_counts(self, exposure_s: float) -> int:
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
    power_W = np.array([np.nan if p.power_W is None else p.power_W for p in points], dtype=float)
    counts = np.array([p.counts for p in points], dtype=np.int64)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["meta_json"] = json.dumps(meta)
        f.create_dataset("wl_nm", data=wl)
        f.create_dataset("wl_std_nm", data=wl_std)
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
    power_W = np.array([np.nan if p.power_W is None else p.power_W for p in points], dtype=float)
    counts = np.array([p.counts for p in points], dtype=np.int64)

    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    np.savez(
        base,
        wl_nm=wl,
        wl_std_nm=wl_std,
        n_wl_samples=n_wl,
        t0_monotonic_s=t0,
        t1_monotonic_s=t1,
        power_W=power_W,
        counts=counts,
        meta=np.array([meta], dtype=object),
    )
    return base


def save_andor_arrays(path: str,
                      frame_wl: np.ndarray,
                      frame_wl_std: np.ndarray,
                      frame_power: np.ndarray,
                      raw_wl_t: np.ndarray,
                      raw_wl: np.ndarray,
                      raw_p_t: np.ndarray,
                      raw_p: np.ndarray,
                      meta: Dict[str, Any],
                      fmt: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if fmt == "h5":
        import h5py
        with h5py.File(path, "w") as h:
            h.attrs["meta_json"] = json.dumps(meta)
            n_frames = int(frame_wl.size)
            h.create_dataset("frame_index", data=np.arange(n_frames, dtype=int))
            h.create_dataset("frame_wl_nm", data=frame_wl)
            h.create_dataset("frame_wl_std_nm", data=frame_wl_std)
            h.create_dataset("frame_power_W", data=frame_power)
            h.create_dataset("raw_wl_t_monotonic", data=raw_wl_t)
            h.create_dataset("raw_wl_nm", data=raw_wl)
            h.create_dataset("raw_power_t_monotonic", data=raw_p_t)
            h.create_dataset("raw_power_W", data=raw_p)
        return path

    base = os.path.splitext(path)[0] + ".npz"
    np.savez(
        base,
        frame_index=np.arange(frame_wl.size, dtype=int),
        frame_wl_nm=frame_wl,
        frame_wl_std_nm=frame_wl_std,
        frame_power_W=frame_power,
        raw_wl_t_monotonic=raw_wl_t,
        raw_wl_nm=raw_wl,
        raw_power_t_monotonic=raw_p_t,
        raw_power_W=raw_p,
        meta=np.array([meta], dtype=object),
    )
    return base


# -------------------------
# Utility: frame windows for kinetics
# -------------------------
def frame_windows(t0: float, exposure_s: float, cycle_s: float, n_frames: int) -> List[Tuple[float, float]]:
    return [(t0 + i * cycle_s, t0 + i * cycle_s + exposure_s) for i in range(n_frames)]


def wait_for_ccd_with_timeout(ccd, timeout_s: float) -> bool:
    """
    Wait for CCD acquisition to complete, but don’t hang forever.
    Returns True if acquisition completed, False if timed out.
    """
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
        # Ask CCD wait loop to exit if it honors exit_flag (your CCD class does).
        try:
            ccd.exit_flag = True
        except Exception:
            pass
        th.join(timeout=1.0)
        return False

    return bool(done["ok"])


# -------------------------
# Mode 1: TimeTagger counts
# -------------------------
def run_scan_timetagger_counts(cfg: Dict[str, Any]) -> None:
    scan = cfg["scan"]
    out = cfg["output"]
    inst = cfg["instruments"]
    det_cfg = inst["detector"]["timetagger_counts"]

    start_nm = float(scan["start_nm"])
    end_nm = float(scan["end_nm"])
    exposure_s = float(det_cfg.get("exposure_s", scan.get("exposure_s", 0.1)))
    scan_speed = float(scan.get("scan_speed_nm_per_s", scan.get("scan_speed", 0.005)))
    ws7_sample_period_s = float(scan.get("ws7_sample_period_s", 0.02))
    max_run_time_s = float(scan.get("max_run_time_s", 600))
    stop_margin_nm = float(scan.get("stop_margin_nm", 0.0))

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

    click = det_cfg["click"]
    det = TimeTaggerCounts(
        click_phys_ch=int(click["physical_channel"]),
        click_trigger_v=float(click["trigger_v"]),
        click_edge=str(click.get("edge", "falling")),
        serial=det_cfg.get("serial", None),
    )

    ple = ple_mod.PLE(
        powermeter_port=None,
        spcm_port=None,
        wa1600_port=None,
        matisse_wavemeter_port=None,
        keithley2450_port=None,
        powermeter_sampling_time=0.05,
        spcm_sampling_time=0.05,
        wa1600_sampling_time=0.05,
        ws7_sleep_time=0.05,
        keithley2450_sleep_time=0.2,
    )

    ple.setup_ws7()

    pow_cfg = inst.get("powermeter", {})
    pow_enabled = bool(pow_cfg.get("enabled", True))
    pow_channel = str(pow_cfg.get("channel", "A"))

    if pow_enabled:
        ple.setup_powermeter(pow_channel)

    ple.setup_matisse("WS7", scanning_speed=scan_speed)
    ple._setup_wavelength_tolerance("WS7")

    matisse = ple_mod.matisse
    ws7 = ple_mod.ws7
    powermeter = ple_mod.powermeter

    if matisse is None or ws7 is None:
        raise RuntimeError("Setup failed: matisse/ws7 is None after setup_*().")

    ple.lock_at_wavelength(round(start_nm, 6))

    wl_sampler = WavelengthSampler(ws7)
    wl_sampler.start(sample_period_s=ws7_sample_period_s)

    if pow_enabled and powermeter is not None:
        powermeter.powermeter.initialize_instrument()
        powermeter._empty_buffer()

    try:
        orig_rise = float(matisse.query("SCAN:RISINGSPEED?", True))
        orig_fall = float(matisse.query("SCAN:FALLINGSPEED?", True))
    except Exception:
        orig_rise = None
        orig_fall = None

    scan_dir = int((end_nm - start_nm) < 0)  # 0 up, 1 down
    stop_wl = (end_nm - stop_margin_nm) if scan_dir == 0 else (end_nm + stop_margin_nm)

    points: List[ScanPoint] = []
    start_wall = time.time()

    meta: Dict[str, Any] = {
        "mode": "timetagger_counts",
        "start_nm": start_nm,
        "end_nm": end_nm,
        "stop_wl_nm": stop_wl,
        "scan_speed_nm_per_s": scan_speed,
        "exposure_s": exposure_s,
        "ws7_sample_period_s": ws7_sample_period_s,
        "powermeter_enabled": pow_enabled,
        "powermeter_channel": pow_channel if pow_enabled else None,
        "timetagger": {
            "click_physical_channel": int(click["physical_channel"]),
            "click_edge": str(click.get("edge", "falling")),
            "click_trigger_v": float(click["trigger_v"]),
            "serial": det_cfg.get("serial", None),
        },
        "created_unix_s": time.time(),
    }

    try:
        try:
            matisse.stabilize_off()
        except Exception:
            pass

        try:
            matisse.query(f"SCAN:RISINGSPEED {scan_speed:.20f}")
            matisse.query(f"SCAN:FALLINGSPEED {scan_speed:.20f}")
        except Exception:
            pass

        try:
            matisse.target_wavelength = round(float(end_nm), 6)
        except Exception:
            pass

        matisse.start_scan(scan_dir)

        while True:
            if (time.time() - start_wall) > max_run_time_s:
                raise TimeoutError(f"Scan exceeded max_run_time_s={max_run_time_s}s.")

            t0 = time.monotonic()

            if pow_enabled and powermeter is not None:
                p_th, p_out = start_midpoint_power_read(powermeter, exposure_s)
            else:
                p_th, p_out = None, {"power_W": None}

            counts = det.acquire_counts(exposure_s)

            t1 = time.monotonic()

            wl_mean, wl_std, n = wl_sampler.stats_between(t0, t1)

            if p_th is not None:
                p_th.join(timeout=1.0)
            power_W = p_out.get("power_W", None)

            points.append(
                ScanPoint(
                    t0=t0,
                    t1=t1,
                    wl_nm=wl_mean,
                    wl_std_nm=wl_std,
                    n_wl_samples=n,
                    power_W=power_W,
                    counts=int(counts),
                )
            )

            if scan_dir == 0 and wl_mean >= stop_wl:
                break
            if scan_dir == 1 and wl_mean <= stop_wl:
                break

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

        try:
            wl_sampler.stop()
        except Exception:
            pass

        try:
            if pow_enabled and powermeter is not None:
                powermeter.powermeter.terminate_instrument()
        except Exception:
            pass

        try:
            det.close()
        except Exception:
            pass

        try:
            ple.clean_up_globals()
        except Exception:
            pass

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
        print(f"First wl={points[0].wl_nm:.6f} nm, last wl={points[-1].wl_nm:.6f} nm")


# -------------------------
# Mode 2: Andor kinetic series
# -------------------------
def run_scan_andor_kinetic(cfg: Dict[str, Any]) -> None:
    scan = cfg["scan"]
    out = cfg["output"]
    inst = cfg["instruments"]
    det = inst["detector"]["andor_kinetic"]

    start_nm = float(scan["start_nm"])
    end_nm = float(scan["end_nm"])
    ws7_dt = float(scan.get("ws7_sample_period_s", 0.02))
    pm_dt = float(scan.get("power_sample_period_s", 0.05))
    max_run_time_s = float(scan.get("max_run_time_s", 600))
    stop_margin_nm = float(scan.get("stop_margin_nm", 0.0))

    exposure_s_req = float(det["exposure_s"])
    cycle_s_req = float(det["cycle_s"])
    n_frames = int(det["n_frames"])
    cool_down = bool(det.get("cool_down", True))
    temperature_C = float(det.get("temperature_C", -70))

    # Support either top-level readout_mode/cosmic_ray_filter OR inside ccd_kwargs
    ccd_kwargs = det.get("ccd_kwargs", {}) or {}
    readout_mode = parse_andor_readout_mode(det.get("readout_mode", ccd_kwargs.get("readout_mode", "FVB")))
    cosmic_ray_filter = parse_cosmic_ray_filter(det.get("cosmic_ray_filter", ccd_kwargs.get("cosmic_ray_filter", True)))

    auto_speed = bool(scan.get("auto_scan_speed_from_kinetic", True))
    manual_speed = float(scan.get("scan_speed_nm_per_s", scan.get("scan_speed", 0.005)))

    out_dir = out.get("directory", "data")
    base = out.get("basename", "scan")
    fmt = out.get("format", "h5").lower()
    overwrite = bool(out.get("overwrite", True))
    save_meta = bool(out.get("save_meta_json", True))

    os.makedirs(out_dir, exist_ok=True)
    sif_path = os.path.join(out_dir, f"{base}.sif")
    meta_path = os.path.join(out_dir, f"{base}.meta.json")
    data_path = os.path.join(out_dir, f"{base}.{ 'h5' if fmt == 'h5' else 'npz' }")

    if (not overwrite) and (os.path.exists(sif_path) or os.path.exists(data_path)):
        raise FileExistsError(f"Output exists and overwrite=false: {sif_path} / {data_path}")

    pow_cfg = inst.get("powermeter", {})
    pow_enabled = bool(pow_cfg.get("enabled", True))
    pow_channel = str(pow_cfg.get("channel", "A"))

    ple = ple_mod.PLE(powermeter_port=None, spcm_port=None, wa1600_port=None, matisse_wavemeter_port=None)

    ple.setup_ws7()
    if pow_enabled:
        ple.setup_powermeter(pow_channel)

    ple.setup_andor()
    ple.setup_matisse("WS7", scanning_speed=None)
    ple._setup_wavelength_tolerance("WS7")

    ws7 = ple_mod.ws7
    powermeter = ple_mod.powermeter
    ccd = ple_mod.ccd
    spectrograph = ple_mod.spectrograph
    matisse = ple_mod.matisse

    if ws7 is None or ccd is None or spectrograph is None or matisse is None:
        raise RuntimeError("Setup failed: ws7/ccd/spectrograph/matisse is None after setup.")

    ple.lock_at_wavelength(round(start_nm, 6))

    exp_actual_s, cycle_actual_s = ccd.setup_kinetics(
        exposure_time=exposure_s_req,
        cycle_time=cycle_s_req,
        n_frames=n_frames,
        readout_mode=readout_mode,
        temperature=temperature_C,
        cool_down=cool_down,
        cosmic_ray_filter=cosmic_ray_filter,
    )

    kinetic_duration_s = n_frames * cycle_actual_s
    if auto_speed:
        scan_speed = abs(end_nm - start_nm) / kinetic_duration_s if kinetic_duration_s > 0 else manual_speed
    else:
        scan_speed = manual_speed

    scan_dir = int((end_nm - start_nm) < 0)
    stop_wl = (end_nm - stop_margin_nm) if scan_dir == 0 else (end_nm + stop_margin_nm)

    wl_sampler = WavelengthSampler(ws7)
    wl_sampler.start(sample_period_s=ws7_dt)

    pm_sampler = None
    if pow_enabled and powermeter is not None:
        powermeter.powermeter.initialize_instrument()
        powermeter._empty_buffer()
        pm_sampler = PowerSampler(powermeter)
        pm_sampler.start(sample_period_s=pm_dt)

    try:
        orig_rise = float(matisse.query("SCAN:RISINGSPEED?", True))
        orig_fall = float(matisse.query("SCAN:FALLINGSPEED?", True))
    except Exception:
        orig_rise = None
        orig_fall = None

    t_acq_start = None
    t_acq_end = None

    try:
        try:
            matisse.stabilize_off()
        except Exception:
            pass

        try:
            matisse.query(f"SCAN:RISINGSPEED {scan_speed:.20f}")
            matisse.query(f"SCAN:FALLINGSPEED {scan_speed:.20f}")
        except Exception:
            pass

        try:
            matisse.target_wavelength = round(float(end_nm), 6)
        except Exception:
            pass

        matisse.start_scan(scan_dir)

        # Start kinetics right after scan begins
        t_acq_start = time.monotonic()
        ccd.start_acquisition()

        # Don’t hang forever
        ok = wait_for_ccd_with_timeout(ccd, timeout_s=max_run_time_s)
        t_acq_end = time.monotonic()

        if not ok:
            raise TimeoutError(f"Andor kinetics did not finish within max_run_time_s={max_run_time_s}s.")

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

        try:
            wl_sampler.stop()
        except Exception:
            pass

        try:
            if pm_sampler is not None:
                pm_sampler.stop()
        except Exception:
            pass

    # Save SIF exactly as produced by Andor SDK
    tmp_name = f"{base}.sif"
    ccd.save_as_sif(tmp_name, spectrograph.calibration_coefficients)

    try:
        if os.path.exists(sif_path):
            os.remove(sif_path)
        shutil.move(tmp_name, sif_path)
    except Exception:
        # If move fails, keep tmp_name and record it in meta
        pass

    if t_acq_start is None or t_acq_end is None:
        t_acq_start = time.monotonic()
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
            if p is not None:
                frame_power[i] = p

    raw_wl_t = np.array([s.t for s in wl_sampler.buf], dtype=float)
    raw_wl = np.array([s.wl_nm for s in wl_sampler.buf], dtype=float)

    if pm_sampler is not None:
        raw_p_t = np.array([s.t for s in pm_sampler.buf], dtype=float)
        raw_p = np.array([s.power_W for s in pm_sampler.buf], dtype=float)
    else:
        raw_p_t = np.array([], dtype=float)
        raw_p = np.array([], dtype=float)

    try:
        if pow_enabled and powermeter is not None:
            powermeter.powermeter.terminate_instrument()
    except Exception:
        pass

    try:
        ple.clean_up_globals()
    except Exception:
        pass

    meta: Dict[str, Any] = {
        "mode": "andor_kinetic",
        "start_nm": start_nm,
        "end_nm": end_nm,
        "stop_wl_nm": stop_wl,
        "auto_scan_speed_from_kinetic": auto_speed,
        "scan_speed_nm_per_s": scan_speed,
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
            "readout_mode": det.get("readout_mode", ccd_kwargs.get("readout_mode", "FVB")),
            "cosmic_ray_filter": det.get("cosmic_ray_filter", ccd_kwargs.get("cosmic_ray_filter", True)),
            "duration_s_requested": float(kinetic_duration_s),
            "duration_s_measured": float(t_acq_end - t_acq_start),
            "sif_path": os.path.abspath(sif_path) if os.path.exists(sif_path) else os.path.abspath(tmp_name),
        },
        "timestamps": {
            "t_acq_start_monotonic": float(t_acq_start),
            "t_acq_end_monotonic": float(t_acq_end),
        },
        "created_unix_s": time.time(),
    }

    save_meta_json_if_enabled(meta_path, meta, save_meta)

    saved = save_andor_arrays(
        data_path,
        frame_wl=frame_wl,
        frame_wl_std=frame_wl_std,
        frame_power=frame_power,
        raw_wl_t=raw_wl_t,
        raw_wl=raw_wl,
        raw_p_t=raw_p_t,
        raw_p=raw_p,
        meta=meta,
        fmt=fmt,
    )

    print(f"Saved SIF : {sif_path}")
    if save_meta:
        print(f"Saved meta: {meta_path}")
    print(f"Saved data: {saved}")
    print(f"Kinetics frames: {n_frames}, measured duration: {float(t_acq_end - t_acq_start):.3f}s")


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
