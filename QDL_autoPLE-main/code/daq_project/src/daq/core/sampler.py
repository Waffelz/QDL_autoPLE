from __future__ import annotations
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple, Optional, List

import numpy as np
from daq.protocols import Wavemeter


@dataclass(frozen=True)
class WlSample:
    t: float          # time.monotonic()
    wl_nm: float


class WavelengthSampler:
    """
    Background sampler:
    - continuously reads wavemeter
    - stores (t, wl) in a ring buffer
    """
    def __init__(self, wavemeter: Wavemeter, maxlen: int = 200_000):
        self.wavemeter = wavemeter
        self.buf: Deque[WlSample] = deque(maxlen=maxlen)
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None

    def start(self, sample_period_s: float = 0.02) -> None:
        if self._th and self._th.is_alive():
            return
        self._stop.clear()

        def run():
            while not self._stop.is_set():
                t = time.monotonic()
                wl = self.wavemeter.get_wavelength_nm()
                self.buf.append(WlSample(t=t, wl_nm=wl))
                time.sleep(sample_period_s)

        self._th = threading.Thread(target=run, daemon=True)
        self._th.start()

    def stop(self) -> None:
        self._stop.set()
        if self._th:
            self._th.join(timeout=1.0)

    def stats_between(self, t0: float, t1: float) -> Tuple[float, float, int]:
        """
        Returns (mean_wl_nm, std_wl_nm, n_samples) for samples in [t0, t1].
        """
        xs: List[float] = [s.wl_nm for s in self.buf if t0 <= s.t <= t1]
        if not xs:
            # fallback: nearest sample
            if not self.buf:
                raise RuntimeError("No wavemeter samples available.")
            mid = 0.5 * (t0 + t1)
            nearest = min(self.buf, key=lambda s: abs(s.t - mid))
            return nearest.wl_nm, 0.0, 0
        arr = np.asarray(xs, dtype=float)
        return float(arr.mean()), float(arr.std()), int(arr.size)

    def latest(self) -> float:
        if not self.buf:
            return self.wavemeter.get_wavelength_nm()
        return self.buf[-1].wl_nm
