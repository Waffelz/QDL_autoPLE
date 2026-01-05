#!/usr/bin/env python3
"""
test_point_acquisition.py

Integration test (no scanning):
- WS7 wavemeter: continuous sampling during each exposure window -> mean/std
- Newport power meter: one read near midpoint of exposure window
- Swabian Time Tagger Ultra: integrated counts over exposure time

Run this BEFORE trying full run_scan.py to validate timing + connectivity.

Examples:
  python scripts/test_point_acquisition.py --n 20 --exp 0.1
  python scripts/test_point_acquisition.py --n 50 --exp 0.1 --dt 0.02 --pm-ch A
  python scripts/test_point_acquisition.py --lock-nm 739.50 --n 10 --exp 0.2
"""

import time
import threading
import argparse
from dataclasses import dataclass
from collections import deque
from typing import Deque, Optional, List, Tuple, Dict, Any

import numpy as np

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root = parent of scripts/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import PLE module so module-level globals (ws7, powermeter, matisse) are accessible
import matisse_controller.shamrock_ple.ple as ple_mod

# TimeTagger bindings (new + legacy)
try:
    from Swabian import TimeTagger  # newer namespace on some installs
except ImportError:
    import TimeTagger  # legacy namespace


@dataclass(frozen=True)
class WlSample:
    t: float          # time.monotonic()
    wl_nm: float


class WavelengthSampler:
    """Background WS7 sampler using ws7.lib.GetWavelength(0.0) with monotonic timestamps."""
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


def start_midpoint_power_read(powermeter_instance, exposure_s: float):
    """
    One-shot power read around the exposure midpoint.
    Uses your Newport driver: convert_reading_string_to_float() returns µW.
    """
    out: Dict[str, Any] = {"power_W": None, "error": None}

    def worker():
        try:
            time.sleep(max(0.0, 0.5 * exposure_s))
            reading_strings = powermeter_instance.powermeter.get_instrument_reading_string_all()
            vals_uW: List[float] = []
            for s in reading_strings:
                try:
                    vals_uW.append(powermeter_instance.convert_reading_string_to_float(s))  # µW
                except Exception:
                    pass
            if vals_uW:
                out["power_W"] = float(sum(vals_uW) / len(vals_uW)) * 1e-6  # W
        except Exception as e:
            out["error"] = e

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    return th, out


def _edge_channel(phys_ch: int, edge: str) -> int:
    edge = edge.strip().lower()
    if edge == "rising":
        return int(abs(phys_ch))
    if edge == "falling":
        return -int(abs(phys_ch))
    raise ValueError("edge must be 'rising' or 'falling'")


class TimeTaggerCounts:
    """Counts detector using TimeTagger.Counter."""
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

    @property
    def tagger(self):
        if self._tagger is None:
            raise RuntimeError("TimeTagger not connected.")
        return self._tagger

    def acquire_counts(self, exposure_s: float) -> int:
        self.connect()
        binwidth_ps = int(round(exposure_s * 1e12))
        meas = TimeTagger.Counter(self.tagger, [self.click_sw_ch], binwidth=binwidth_ps, n_values=1)
        meas.startFor(binwidth_ps, clear=True)
        meas.waitUntilFinished()
        # API compatibility
        try:
            data = meas.getData()
        except TypeError:
            data = meas.getData(rolling=True)
        arr = np.asarray(data, dtype=np.int64).reshape(-1)
        return int(arr[-1]) if arr.size else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="Number of points")
    ap.add_argument("--exp", type=float, default=0.1, help="Exposure/integration time (s)")
    ap.add_argument("--dt", type=float, default=0.02, help="WS7 sample period (s)")
    ap.add_argument("--pm-ch", type=str, default="A", help="Powermeter channel: A, B, or AB")
    ap.add_argument("--no-pm", action="store_true", help="Disable powermeter read")
    ap.add_argument("--serial", type=str, default=None, help="TimeTagger serial (optional)")
    ap.add_argument("--ch", type=int, default=1, help="TimeTagger click physical channel")
    ap.add_argument("--trig", type=float, default=-0.08, help="TimeTagger trigger level (V)")
    ap.add_argument("--edge", type=str, default="falling", choices=["rising", "falling"], help="Pulse edge")
    ap.add_argument("--lock-nm", type=float, default=None, help="Optional: lock Matisse at this wavelength first")
    args = ap.parse_args()

    ple = ple_mod.PLE(powermeter_port=None, matisse_wavemeter_port=None)
    det = TimeTaggerCounts(args.ch, args.trig, args.edge, serial=args.serial)

    try:
        # Setup WS7
        ple.setup_ws7()
        ws7 = ple_mod.ws7
        if ws7 is None:
            raise RuntimeError("WS7 setup failed (ple_mod.ws7 is None).")

        # Optional: setup powermeter
        powermeter = None
        if not args.no_pm:
            ple.setup_powermeter(args.pm_ch)
            powermeter = ple_mod.powermeter
            if powermeter is None:
                raise RuntimeError("Powermeter setup failed (ple_mod.powermeter is None).")
            powermeter.powermeter.initialize_instrument()
            powermeter._empty_buffer()

        # Optional: lock matisse to a wavelength (nice for stability tests)
        if args.lock_nm is not None:
            ple.setup_matisse("WS7", scanning_speed=None)
            ple._setup_wavelength_tolerance("WS7")
            ple.lock_at_wavelength(round(float(args.lock_nm), 6))

        # Start WS7 background sampling
        wl_sampler = WavelengthSampler(ws7)
        wl_sampler.start(sample_period_s=args.dt)

        print("Running point acquisition...")
        print(f"WS7 dt={args.dt}s, exposure={args.exp}s, N={args.n}")
        print(f"TimeTagger: phys_ch={args.ch}, edge={args.edge}, trig={args.trig}V (sw_ch={_edge_channel(args.ch,args.edge)})")
        if powermeter is None:
            print("Powermeter: disabled")
        else:
            print(f"Powermeter: enabled channel={args.pm_ch}")

        wl_list = []
        p_list = []
        c_list = []

        t_start = time.time()
        for i in range(args.n):
            t0 = time.monotonic()

            # power read near midpoint
            if powermeter is not None:
                p_th, p_out = start_midpoint_power_read(powermeter, args.exp)
            else:
                p_th, p_out = None, {"power_W": None}

            # integrated counts
            counts = det.acquire_counts(args.exp)

            t1 = time.monotonic()

            wl_mean, wl_std, n_wl = wl_sampler.stats_between(t0, t1)

            if p_th is not None:
                p_th.join(timeout=1.0)
            pW = p_out.get("power_W", None)

            wl_list.append(wl_mean)
            p_list.append(np.nan if pW is None else pW)
            c_list.append(counts)

            if (i < 5) or ((i + 1) % 10 == 0) or (i + 1 == args.n):
                dt_wall = time.time() - t_start
                print(f"[{i+1:3d}/{args.n}] t={dt_wall:6.2f}s  wl={wl_mean:12.9f} nm  std={wl_std*1e3:7.3f} pm  n={n_wl:4d}  "
                      f"power={pW if pW is not None else None}  counts={counts}")

        wl_arr = np.asarray(wl_list, dtype=float)
        p_arr = np.asarray(p_list, dtype=float)
        c_arr = np.asarray(c_list, dtype=float)

        print("\n--- Summary ---")
        print(f"WL mean ± std: {wl_arr.mean():.9f} ± {wl_arr.std():.9f} nm")
        if np.isfinite(p_arr).any():
            print(f"Power mean ± std: {np.nanmean(p_arr):.3e} ± {np.nanstd(p_arr):.3e} W")
        print(f"Counts mean ± std: {c_arr.mean():.1f} ± {c_arr.std():.1f} per {args.exp}s")
        print("---------------")

    finally:
        try:
            wl_sampler.stop()
        except Exception:
            pass
        try:
            if not args.no_pm and ple_mod.powermeter is not None:
                ple_mod.powermeter.powermeter.terminate_instrument()
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


if __name__ == "__main__":
    main()
