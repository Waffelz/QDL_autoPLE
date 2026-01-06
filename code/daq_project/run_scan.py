#!/usr/bin/env python3
"""
run_scan.py (project root)

Two detector modes (selected in YAML):
  1) timetagger_counts:
     - Optional Matisse scan (enable_matisse)
     - For each exposure window:
         * mean/std wavelength from WS7 samples within window
         * one power reading near midpoint (Newport)
         * TimeTagger integrated counts over exposure_s
     - Save points to HDF5 (preferred) or NPZ fallback

  2) andor_kinetic:
     - Configure Andor CCD kinetics (exposure_s, cycle_s, n_frames, readout_mode, cosmic_ray_filter)
     - Cooling behavior:
         * always call ccd.ensure_cooling(setpoint)
         * optionally REQUIRE temperature is cold enough before acquisition (quick check or timeout)
         * do NOT block-cool during scans unless cool_down=true
     - Optional Matisse scan (enable_matisse)
     - Sample WS7/power continuously during kinetic acquisition
     - Stop sampling exactly when kinetic ends (or timeout)
     - Save raw .sif kinetic series + matched wl/power arrays + raw traces

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
import importlib
from dataclasses import dataclass
from collections import deque
from typing import Deque, Optional, Dict, Any, List, Tuple, Callable

import numpy as np

# ---- Make imports work no matter where you run from ----
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# YAML
try:
    import yaml
except ImportError as e:
    raise RuntimeError("Missing dependency: pyyaml. Install with: pip install pyyaml") from e

# Andor globals live in this module
import matisse_controller.shamrock_ple.ple as ple_mod

# TimeTagger bindings
try:
    import TimeTagger  # type: ignore
except ImportError:
    from Swabian import TimeTagger  # type: ignore

from pathlib import Path
DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().with_name("run_scan.yml"))

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
    t: float
    wl_nm: float


@dataclass(frozen=True)
class PowerSample:
    t: float
    power_W: float


# -------------------------
# Config helpers
# -------------------------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        top = yaml.safe_load(f) or {}
    cfg = top.get("RunScan", {})
    if not cfg:
        raise ValueError("YAML missing top-level key: RunScan")
    return cfg


def cfg_get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def import_class(import_path: str, class_name: str):
    mod = importlib.import_module(import_path)
    if not hasattr(mod, class_name):
        raise ImportError(f"Module '{import_path}' has no attribute '{class_name}'")
    return getattr(mod, class_name)


# -------------------------
# Andor YAML -> constants parsing
# -------------------------
def parse_andor_readout_mode(mode) -> int:
    if mode is None:
        return READ_MODE_FVB
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
    raise ValueError(f"Unknown Andor readout_mode: {mode!r}")


def parse_cosmic_ray_filter(val) -> int:
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
    raise ValueError(f"Unknown cosmic_ray_filter: {val!r}")


# -------------------------
# Samplers (background)
# -------------------------
class WavelengthSampler:
    def __init__(self, read_wl_nm: Callable[[], Optional[float]], maxlen: int = 200_000):
        self.read_wl_nm = read_wl_nm
        self.buf: Deque[WlSample] = deque(maxlen=maxlen)
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None

    def start(self, sample_period_s: float) -> None:
        if self._th and self._th.is_alive():
            return
        self._stop.clear()

        def run():
            while not self._stop.is_set():
                t = time.monotonic()
                wl = self.read_wl_nm()
                if wl is not None and wl > 0:
                    self.buf.append(WlSample(t=t, wl_nm=float(wl)))
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


class PowerSampler:
    def __init__(self, read_power_W: Callable[[], Optional[float]], maxlen: int = 200_000):
        self.read_power_W = read_power_W
        self.buf: Deque[PowerSample] = deque(maxlen=maxlen)
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None

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
                p = self.read_power_W()
                if p is not None:
                    self.buf.append(PowerSample(t=now, power_W=float(p)))
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
# Power: one midpoint read (for per-point TimeTagger mode)
# -------------------------
def start_midpoint_power_read(read_power_W: Callable[[], Optional[float]], exposure_s: float):
    out: Dict[str, Any] = {"power_W": None, "error": None}

    def worker():
        try:
            time.sleep(max(0.0, 0.5 * exposure_s))
            out["power_W"] = read_power_W()
        except Exception as e:
            out["error"] = e

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    return th, out


# -------------------------
# TimeTagger counts
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

    def connect(self) -> None:
        if self._tagger is not None:
            return
        self._tagger = TimeTagger.createTimeTagger(self.serial) if self.serial else TimeTagger.createTimeTagger()
        self._tagger.setTriggerLevel(self.click_phys_ch, self.click_trigger_v)

    def close(self) -> None:
        if self._tagger is not None:
            TimeTagger.freeTimeTagger(self._tagger)
            self._tagger = None

    def acquire_counts(self, exposure_s: float) -> int:
        if exposure_s <= 0:
            raise ValueError("exposure_s must be > 0")
        self.connect()
        binwidth_ps = int(round(exposure_s * 1e12))
        meas = TimeTagger.Counter(self._tagger, [self.click_sw_ch], binwidth=binwidth_ps, n_values=1)
        meas.startFor(binwidth_ps, clear=True)
        meas.waitUntilFinished()
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


def frame_windows(t0: float, exposure_s: float, cycle_s: float, n_frames: int) -> List[Tuple[float, float]]:
    return [(t0 + i * cycle_s, t0 + i * cycle_s + exposure_s) for i in range(n_frames)]


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


# -------------------------
# Instrument factories (WS7 / Powermeter / Matisse optional)
# -------------------------
def make_ws7_reader(ws7_obj) -> Callable[[], Optional[float]]:
    if hasattr(ws7_obj, "get_wavelength"):
        return lambda: float(ws7_obj.get_wavelength())
    if hasattr(ws7_obj, "lib") and hasattr(ws7_obj.lib, "GetWavelength"):
        return lambda: float(ws7_obj.lib.GetWavelength(0.0))
    raise RuntimeError("WS7 object has no get_wavelength() and no .lib.GetWavelength().")


def init_ws7(cfg: Dict[str, Any]):
    ws7_cfg = cfg_get(cfg, "instruments.ws7", {}) or {}
    if not bool(ws7_cfg.get("enabled", True)):
        return None, None

    import_path = ws7_cfg.get("import_path", "functions26.instruments.ws7")
    class_name = ws7_cfg.get("class_name", "WS7")
    init_kwargs = ws7_cfg.get("init_kwargs", {}) or {}

    WS7Class = import_class(import_path, class_name)
    ws7 = WS7Class(**init_kwargs) if init_kwargs else WS7Class()

    # best-effort init
    for m in ("initialize_instrument", "initialize", "connect"):
        if hasattr(ws7, m):
            getattr(ws7, m)()
            break

    return ws7, make_ws7_reader(ws7)


def init_powermeter(cfg: Dict[str, Any]):
    pm_cfg = cfg_get(cfg, "instruments.powermeter", {}) or {}
    if not bool(pm_cfg.get("enabled", True)):
        return None, None

    import_path = pm_cfg.get("import_path", "functions26.instruments.powermeter")
    class_name = pm_cfg.get("class_name", "PowerMeter")
    init_kwargs = pm_cfg.get("init_kwargs", {}) or {}

    PMClass = import_class(import_path, class_name)
    pm = PMClass(**init_kwargs) if init_kwargs else PMClass()

    # If your driver uses an internal GPIBInstrument, open once + empty buffer
    if hasattr(pm, "powermeter") and hasattr(pm.powermeter, "initialize_instrument"):
        pm.powermeter.initialize_instrument()
    if hasattr(pm, "_empty_buffer"):
        pm._empty_buffer()

    def read_power_W() -> Optional[float]:
        try:
            if hasattr(pm, "powermeter") and hasattr(pm.powermeter, "get_instrument_reading_string_all") and hasattr(pm, "convert_reading_string_to_float"):
                reading_strings = pm.powermeter.get_instrument_reading_string_all()
                vals_uW: List[float] = []
                for s in reading_strings:
                    try:
                        vals_uW.append(pm.convert_reading_string_to_float(s))  # µW
                    except Exception:
                        pass
                if not vals_uW:
                    return None
                return float(sum(vals_uW) / len(vals_uW)) * 1e-6  # -> W
            # fallback: user can implement their own method
            if hasattr(pm, "read_power_W"):
                return float(pm.read_power_W())
        except Exception:
            return None
        return None

    return pm, read_power_W


def maybe_close_powermeter(pm):
    try:
        if pm is None:
            return
        if hasattr(pm, "powermeter") and hasattr(pm.powermeter, "terminate_instrument"):
            pm.powermeter.terminate_instrument()
    except Exception:
        pass


def init_matisse(cfg: Dict[str, Any]):
    m_cfg = cfg_get(cfg, "instruments.matisse", {}) or {}
    enabled = bool(cfg_get(cfg, "scan.enable_matisse", False)) and bool(m_cfg.get("enabled", True))
    if not enabled:
        return None

    import_path = m_cfg.get("import_path", None)
    class_name = m_cfg.get("class_name", None)
    init_kwargs = m_cfg.get("init_kwargs", {}) or {}

    if not import_path or not class_name:
        raise ValueError("enable_matisse=true but instruments.matisse.import_path/class_name not set in YAML.")

    MClass = import_class(import_path, class_name)
    return MClass(**init_kwargs) if init_kwargs else MClass()


# -------------------------
# Mode 2 (CURRENT TEST GOAL): Andor kinetics + ws7/power alignment
# -------------------------
def run_scan_andor_kinetic(cfg: Dict[str, Any]) -> None:
    scan = cfg["scan"]
    out = cfg["output"]
    det = cfg["instruments"]["detector"]["andor_kinetic"]

    # Output
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

    # Scan params (Matisse optional)
    start_nm = float(scan.get("start_nm", 0.0))
    end_nm = float(scan.get("end_nm", 0.0))
    enable_matisse = bool(scan.get("enable_matisse", False))
    auto_speed = bool(scan.get("auto_scan_speed_from_kinetic", True))
    manual_speed = float(scan.get("scan_speed_nm_per_s", 0.005))
    max_run_time_s = float(scan.get("max_run_time_s", 600))

    # Sampling
    ws7_dt = float(scan.get("ws7_sample_period_s", 0.02))
    pm_dt = float(scan.get("power_sample_period_s", 0.05))

    # Detector params
    exposure_s_req = float(det["exposure_s"])
    cycle_s_req = float(det["cycle_s"])
    n_frames = int(det["n_frames"])

    temperature_C = float(det.get("temperature_C", -65))
    cool_down = bool(det.get("cool_down", False))  # default False for your “don’t wait every scan” goal

    require_temp = bool(det.get("require_temperature", True))
    temp_tol = float(det.get("temperature_tolerance_C", 1.0))
    temp_timeout = float(det.get("temperature_timeout_s", 0.0))  # 0 = instant check

    readout_mode = parse_andor_readout_mode(det.get("readout_mode", "FVB"))
    cosmic_ray_filter = parse_cosmic_ray_filter(det.get("cosmic_ray_filter", True))

    # Init WS7 + powermeter
    ws7, read_wl_nm = init_ws7(cfg)
    pm, read_power_W = init_powermeter(cfg)

    if read_wl_nm is None:
        raise RuntimeError("WS7 is disabled or failed to initialize; andor_kinetic mode requires WS7 for alignment.")

    # Init Andor globals
    # NOTE: Close Andor Solis GUI before running (as you already learned).
    ple_mod.PLE.load_andor_libs()
    ccd = ple_mod.ccd
    shamrock = ple_mod.shamrock  # optional, info-only

    if ccd is None:
        raise RuntimeError("Andor CCD not initialized (ple_mod.ccd is None).")

    # Cooling behavior: always ensure cooling ON, only wait if cool_down=True
    try:
        ccd.ensure_cooling(target_C=temperature_C, persist_on_shutdown=True)
        temp_now = ccd.wait_until_cold(target_C=temperature_C, tol_C=temp_tol, timeout_s=temp_timeout, poll_s=5.0)
        if require_temp and (temp_now is not None) and (not np.isnan(temp_now)) and (temp_now > temperature_C + temp_tol):
            raise RuntimeError(
                f"CCD too warm for acquisition: {temp_now:.1f} °C (need <= {temperature_C + temp_tol:.1f} °C). "
                f"Either wait longer or set require_temperature=false."
            )
    except AttributeError:
        # If your CCD class doesn't have these helpers, fall back to setup_kinetics(cool_down=True)
        if require_temp:
            print("WARNING: CCD.ensure_cooling/wait_until_cold not found. Consider updating ccd.py as discussed.")

    # Configure kinetics (your CCD.setup_kinetics already calls ensure_cooling and only waits if cool_down=True)
    exp_actual_s, cycle_actual_s = ccd.setup_kinetics(
        exposure_time=exposure_s_req,
        cycle_time=cycle_s_req,
        n_frames=n_frames,
        readout_mode=readout_mode,
        temperature=temperature_C,
        cool_down=cool_down,
        cosmic_ray_filter=cosmic_ray_filter,
    )

    kinetic_duration_s = n_frames * float(cycle_actual_s)
    if auto_speed and kinetic_duration_s > 0:
        scan_speed = abs(end_nm - start_nm) / kinetic_duration_s if enable_matisse else 0.0
    else:
        scan_speed = manual_speed if enable_matisse else 0.0

    # Start samplers
    wl_sampler = WavelengthSampler(read_wl_nm)
    wl_sampler.start(sample_period_s=ws7_dt)

    pm_sampler = None
    if read_power_W is not None:
        pm_sampler = PowerSampler(read_power_W)
        pm_sampler.start(sample_period_s=pm_dt)

    # Optional Matisse
    matisse = init_matisse(cfg) if enable_matisse else None

    # --- Run acquisition ---
    t_acq_start = None
    t_acq_end = None
    tmp_name = f"{base}.sif"

    try:
        # Start scan if enabled
        if enable_matisse and matisse is not None:
            try:
                matisse.stabilize_off()
            except Exception:
                pass

            try:
                matisse.query(f"SCAN:RISINGSPEED {scan_speed:.20f}")
                matisse.query(f"SCAN:FALLINGSPEED {scan_speed:.20f}")
            except Exception:
                pass

            scan_dir = int((end_nm - start_nm) < 0)
            try:
                matisse.target_wavelength = round(float(end_nm), 6)
            except Exception:
                pass

            matisse.start_scan(scan_dir)

        # Start kinetics
        t_acq_start = time.monotonic()
        ccd.start_acquisition()
        ok = wait_for_ccd_with_timeout(ccd, timeout_s=max_run_time_s)
        t_acq_end = time.monotonic()

        if not ok:
            raise TimeoutError(f"Andor kinetics did not finish within max_run_time_s={max_run_time_s}s.")

    finally:
        # Stop scan
        if enable_matisse and matisse is not None:
            try:
                matisse.stop_scan()
            except Exception:
                pass
            try:
                matisse.stabilize_on()
            except Exception:
                pass

        # Stop samplers
        try:
            wl_sampler.stop()
        except Exception:
            pass
        try:
            if pm_sampler is not None:
                pm_sampler.stop()
        except Exception:
            pass

    # Save SIF (raw)
    ccd.save_as_sif(tmp_name)
    try:
        if os.path.exists(sif_path):
            os.remove(sif_path)
        shutil.move(tmp_name, sif_path)
    except Exception:
        # If move fails, keep tmp_name where it is and record in meta
        pass

    if t_acq_start is None or t_acq_end is None:
        t_acq_start = time.monotonic()
        t_acq_end = t_acq_start

    # Per-frame windows aligned to kinetic timing
    wins = frame_windows(t_acq_start, float(exp_actual_s), float(cycle_actual_s), n_frames)

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

    # Raw traces
    raw_wl_t = np.array([s.t for s in wl_sampler.buf], dtype=float)
    raw_wl = np.array([s.wl_nm for s in wl_sampler.buf], dtype=float)
    if pm_sampler is not None:
        raw_p_t = np.array([s.t for s in pm_sampler.buf], dtype=float)
        raw_p = np.array([s.power_W for s in pm_sampler.buf], dtype=float)
    else:
        raw_p_t = np.array([], dtype=float)
        raw_p = np.array([], dtype=float)

    # Close powermeter (WS7 driver usually doesn't need explicit close)
    maybe_close_powermeter(pm)

    # Meta
    meta: Dict[str, Any] = {
        "mode": "andor_kinetic",
        "enable_matisse": enable_matisse,
        "start_nm": start_nm,
        "end_nm": end_nm,
        "auto_scan_speed_from_kinetic": auto_speed,
        "scan_speed_nm_per_s": scan_speed,
        "wavemeter": {"ws7_sample_period_s": ws7_dt},
        "powermeter": {"enabled": pm is not None, "power_sample_period_s": pm_dt if pm is not None else None},
        "andor": {
            "n_frames": n_frames,
            "exposure_s_requested": exposure_s_req,
            "cycle_s_requested": cycle_s_req,
            "exposure_s_actual": float(exp_actual_s),
            "cycle_s_actual": float(cycle_actual_s),
            "temperature_C_setpoint": temperature_C,
            "require_temperature": require_temp,
            "temperature_tolerance_C": temp_tol,
            "readout_mode": det.get("readout_mode", "FVB"),
            "cosmic_ray_filter": det.get("cosmic_ray_filter", True),
            "duration_s_requested": float(kinetic_duration_s),
            "duration_s_measured": float(t_acq_end - t_acq_start),
            "sif_path": os.path.abspath(sif_path) if os.path.exists(sif_path) else os.path.abspath(tmp_name),
        },
        "shamrock": {
            "present": shamrock is not None,
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
    if mode == "andor_kinetic":
        run_scan_andor_kinetic(cfg)
    else:
        raise ValueError(
            f"This updated script is aligned to the current test goal (andor_kinetic). "
            f"Set mode='andor_kinetic'. Got: {mode!r}"
        )


if __name__ == "__main__":
    main()
