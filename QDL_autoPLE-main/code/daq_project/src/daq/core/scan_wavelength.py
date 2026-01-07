from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Union
import numpy as np

from daq.protocols import Laser, Wavemeter, DetectorCounts, DetectorSpectrum, PowerMeter
from daq.core.sampler import WavelengthSampler


import threading, time
from typing import Optional, Dict, Any

def start_midpoint_power_read(powermeter, exposure_s: float):
    out: Dict[str, Any] = {"power": None, "error": None}

    def worker():
        try:
            time.sleep(max(0.0, exposure_s * 0.5))
            out["power"] = float(powermeter.read_power_W())
        except Exception as e:
            out["error"] = e

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    return th, out


@dataclass
class CountsPoint:
    t0: float
    t1: float
    wl_nm: float
    wl_std_nm: float
    n_wl_samples: int
    counts: int
    power_W: Optional[float] = None


@dataclass
class SpectrumPoint:
    t0: float
    t1: float
    wl_nm: float
    wl_std_nm: float
    n_wl_samples: int
    intensity: np.ndarray  # (M,)
    power_W: Optional[float] = None


@dataclass
class ScanCountsResult:
    points: List[CountsPoint]
    meta: Dict


@dataclass
class ScanSpectrumResult:
    axis: np.ndarray       # (M,)
    points: List[SpectrumPoint]
    meta: Dict


def _direction_from_range(start_nm: float, end_nm: float) -> str:
    return "up" if end_nm >= start_nm else "down"


def run_scan_counts(
    *,
    laser: Laser,
    wavemeter: Wavemeter,
    detector: DetectorCounts,
    start_nm: float,
    end_nm: float,
    scan_speed: float,
    exposure_s: float,
    powermeter: Optional[PowerMeter] = None,
    wl_sample_period_s: float = 0.02,
    stop_margin_nm: float = 0.0,
) -> ScanCountsResult:
    """
    Locks at start_nm, begins scan, then repeats:
      - acquire counts for exposure_s
      - compute mean wavemeter wavelength during exposure
      - append CountsPoint
    Stops when mean wavelength crosses end_nm (with margin).
    """
    direction = _direction_from_range(start_nm, end_nm)
    meta = dict(
        detector="counts",
        start_nm=start_nm,
        end_nm=end_nm,
        direction=direction,
        scan_speed=scan_speed,
        exposure_s=exposure_s,
        wl_sample_period_s=wl_sample_period_s,
        created_unix_s=time.time(),
        stop_margin_nm=stop_margin_nm,
    )

    laser.lock_to_wavelength_nm(start_nm)

    sampler = WavelengthSampler(wavemeter)
    sampler.start(sample_period_s=wl_sample_period_s)

    laser.set_scan_speed(scan_speed)
    laser.start_scan(direction=direction)

    points: List[CountsPoint] = []
    try:
        while True:
            t0 = time.monotonic()

            # Detector acquisition (blocks for ~exposure_s)
            counts = detector.acquire_counts(exposure_s)

            t1 = time.monotonic()
            wl_mean, wl_std, n = sampler.stats_between(t0, t1)

            pwr = powermeter.read_power_W() if powermeter else None
            points.append(CountsPoint(
                t0=t0, t1=t1,
                wl_nm=wl_mean, wl_std_nm=wl_std, n_wl_samples=n,
                counts=int(counts),
                power_W=pwr,
            ))

            # Stop condition based on measured wavelength
            if direction == "up" and wl_mean >= (end_nm - stop_margin_nm):
                break
            if direction == "down" and wl_mean <= (end_nm + stop_margin_nm):
                break

    finally:
        laser.stop_scan()
        sampler.stop()

    return ScanCountsResult(points=points, meta=meta)


def run_scan_spectrum(
    *,
    laser: Laser,
    wavemeter: Wavemeter,
    detector: DetectorSpectrum,
    start_nm: float,
    end_nm: float,
    scan_speed: float,
    exposure_s: float,
    axis: np.ndarray,
    powermeter: Optional[PowerMeter] = None,
    wl_sample_period_s: float = 0.02,
    stop_margin_nm: float = 0.0,
) -> ScanSpectrumResult:
    direction = _direction_from_range(start_nm, end_nm)
    meta = dict(
        detector="spectrum",
        start_nm=start_nm,
        end_nm=end_nm,
        direction=direction,
        scan_speed=scan_speed,
        exposure_s=exposure_s,
        wl_sample_period_s=wl_sample_period_s,
        created_unix_s=time.time(),
        stop_margin_nm=stop_margin_nm,
    )

    axis = np.asarray(axis, dtype=float)

    laser.lock_to_wavelength_nm(start_nm)

    detector.set_exposure(exposure_s)

    sampler = WavelengthSampler(wavemeter)
    sampler.start(sample_period_s=wl_sample_period_s)

    laser.set_scan_speed(scan_speed)
    laser.start_scan(direction=direction)

    points: List[SpectrumPoint] = []
    try:
        while True:
            t0 = time.monotonic()
            det_axis, intensity = detector.acquire_spectrum()
            t1 = time.monotonic()

            # enforce consistent axis (common with fixed ROI)
            det_axis = np.asarray(det_axis, dtype=float)
            if det_axis.shape != axis.shape or np.max(np.abs(det_axis - axis)) > 1e-9:
                raise ValueError("Spectrometer axis changed. Store per-frame axis or lock ROI/calibration.")

            wl_mean, wl_std, n = sampler.stats_between(t0, t1)
            pwr = powermeter.read_power_W() if powermeter else None

            points.append(SpectrumPoint(
                t0=t0, t1=t1,
                wl_nm=wl_mean, wl_std_nm=wl_std, n_wl_samples=n,
                intensity=np.asarray(intensity, dtype=float),
                power_W=pwr,
            ))

            if direction == "up" and wl_mean >= (end_nm - stop_margin_nm):
                break
            if direction == "down" and wl_mean <= (end_nm + stop_margin_nm):
                break

    finally:
        laser.stop_scan()
        sampler.stop()

    return ScanSpectrumResult(axis=axis, points=points, meta=meta)
