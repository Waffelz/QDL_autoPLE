#!/usr/bin/env python3
"""
Stationary "one-point" acquisition test:
WS7 mean/std over exposure window + Newport midpoint power + TimeTagger counts.

Run:
  python scripts/test_point_acquisition.py --n 30 --exp 0.1
"""

import time
import argparse
import sys
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import Optional, Deque, List, Tuple
import numpy as np
import threading

# ---- sys.path fix to find functions26 ----
HERE = Path(__file__).resolve()
root = None
for p in [HERE.parent] + list(HERE.parents):
    if (p / "functions26").exists():
        root = p
        break
if root is None:
    raise RuntimeError("Could not find a 'functions26' folder in any parent directory.")
sys.path.insert(0, str(root))

from functions26.instruments.ws7 import WS7
from functions26.instruments.powermeter import PowerMeter

# Swabian Time Tagger
try:
    from Swabian import TimeTagger
except ImportError:
    import TimeTagger


@dataclass(frozen=True)
class WlSample:
    t: float
    wl_nm: float


class WavelengthSampler:
    def __init__(self, ws7: WS7, maxlen: int = 200_000):
        self.ws7 = ws7
        self.buf: Deque[WlSample] = deque(maxlen=maxlen)
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self, dt: float):
        if self._th and self._th.is_alive():
            return
        self._stop.clear()

        def run():
            while not self._stop.is_set():
                t = time.monotonic()
                try:
                    with self._lock:
                        wl = float(self.ws7.lib.GetWavelength(0.0))
                    if wl > 0:
                        self.buf.append(WlSample(t=t, wl_nm=wl))
                except Exception:
                    pass
                time.sleep(dt)

        self._th = threading.Thread(target=run, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()
        if self._th:
            self._th.join(timeout=1.0)

    def stats_between(self, t0: float, t1: float) -> Tuple[float, float, int]:
        xs = [s.wl_nm for s in self.buf if t0 <= s.t <= t1]
        if xs:
            a = np.asarray(xs, float)
            return float(a.mean()), float(a.std()), int(a.size)
        # fallback to nearest
        if not self.buf:
            raise RuntimeError("No WS7 samples")
        mid = 0.5 * (t0 + t1)
        nearest = min(self.buf, key=lambda s: abs(s.t - mid))
        return float(nearest.wl_nm), 0.0, 0


def _edge_channel(phys_ch: int, edge: str) -> int:
    edge = edge.strip().lower()
    if edge == "rising":
        return int(abs(phys_ch))
    if edge == "falling":
        return -int(abs(phys_ch))
    raise ValueError("edge must be rising or falling")


class TTCounts:
    def __init__(self, phys_ch: int, trig_v: float, edge: str, serial: Optional[str] = None):
        self.phys_ch = phys_ch
        self.trig_v = trig_v
        self.sw_ch = _edge_channel(phys_ch, edge)
        self.serial = serial
        self.tagger = None

    def open(self):
        if self.tagger is not None:
            return
        self.tagger = TimeTagger.createTimeTagger(self.serial) if self.serial else TimeTagger.createTimeTagger()
        self.tagger.setTriggerLevel(self.phys_ch, self.trig_v)

    def close(self):
        if self.tagger is not None:
            TimeTagger.freeTimeTagger(self.tagger)
            self.tagger = None

    def counts(self, exp_s: float) -> int:
        self.open()
        bin_ps = int(round(exp_s * 1e12))
        meas = TimeTagger.Counter(self.tagger, [self.sw_ch], binwidth=bin_ps, n_values=1)
        meas.startFor(bin_ps, clear=True)
        meas.waitUntilFinished()
        try:
            data = meas.getData()
        except TypeError:
            data = meas.getData(rolling=True)
        arr = np.asarray(data, dtype=np.int64).reshape(-1)
        return int(arr[-1]) if arr.size else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--exp", type=float, default=0.1)
    ap.add_argument("--ws7_dt", type=float, default=0.02)
    ap.add_argument("--pm_channel", default="A", choices=["A", "B", "AB"])
    ap.add_argument("--tt_ch", type=int, default=1)
    ap.add_argument("--tt_trig", type=float, default=-0.08)
    ap.add_argument("--tt_edge", default="falling", choices=["rising", "falling"])
    args = ap.parse_args()

    ws7 = WS7()
    pm = PowerMeter(args.pm_channel)

    # init powermeter session once
    pm.powermeter.initialize_instrument()
    try:
        if hasattr(pm, "_empty_buffer"):
            pm._empty_buffer()
    except Exception:
        pass

    tt = TTCounts(args.tt_ch, args.tt_trig, args.tt_edge)

    wl_s = WavelengthSampler(ws7)
    wl_s.start(args.ws7_dt)

    try:
        print("Stationary acquisition loop...")
        for i in range(args.n):
            t0 = time.monotonic()

            # midpoint power read (simple sync read here)
            time.sleep(0.5 * args.exp)
            try:
                rs = pm.powermeter.get_instrument_reading_string_all()
                vals_uW = [pm.convert_reading_string_to_float(s) for s in rs]
                p_W = (sum(vals_uW) / len(vals_uW)) * 1e-6 if vals_uW else float("nan")
            except Exception:
                p_W = float("nan")

            # remaining half exposure (approx)
            time.sleep(max(0.0, 0.5 * args.exp))

            # counts over the same exposure length (separate but same duration)
            c = tt.counts(args.exp)

            t1 = time.monotonic()
            wl_mean, wl_std, nwl = wl_s.stats_between(t0, t1)

            print(f"[{i+1:3d}/{args.n}] wl={wl_mean:.9f}±{wl_std:.9f} nm (n={nwl:3d})  "
                  f"P={p_W:.3e} W  counts={c}")

    finally:
        wl_s.stop()
        tt.close()
        try:
            pm.powermeter.terminate_instrument()
        except Exception:
            pass


if __name__ == "__main__":
    main()
