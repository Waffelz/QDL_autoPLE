"""
Swabian Time Tagger Ultra "driver" for photon counting.

Config used here (as requested):
- Start: physical channel 2, trigger +1.0 V, rising edge -> +2
- Click: physical channel 1, trigger -0.08 V, falling edge -> -1

Requires Swabian Time Tagger software / Python bindings installed.
API docs: createTimeTagger(), setTriggerLevel(), Countrate, Counter, etc.
"""

from __future__ import annotations

import time
import threading
import queue
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

# Swabian API import: depending on install, either "TimeTagger" works,
# or (less commonly) "Swabian.TimeTagger".
try:
    import TimeTagger  # type: ignore
except ImportError as exc:
    raise ImportError(
        "Could not import Swabian TimeTagger Python module. "
        "Install/repair the Swabian Time Tagger software so the Python bindings are available."
    ) from exc

# Your existing data container (same pattern as WS7 / PowerMeter)
try:
    from functions26.filing.QDLFiling import QDLFDataManager
except Exception:
    QDLFDataManager = None  # allows standalone testing without your repo


@dataclass(frozen=True)
class TTConfig:
    # Physical inputs
    start_phys_ch: int = 2
    click_phys_ch: int = 1

    # Trigger levels in Volts (applied to *physical* channel numbers)
    start_trigger_v: float = 1.0
    click_trigger_v: float = -0.08

    # Which edge to timestamp for measurements:
    # Swabian convention: rising edge = +channel, falling edge = -channel  (see docs)
    start_edge: str = "rising"   # "rising" or "falling"
    click_edge: str = "falling"  # "rising" or "falling"

    # Optional: if you have multiple taggers connected
    serial: Optional[str] = None


def _edge_channel(phys_ch: int, edge: str) -> int:
    e = edge.strip().lower()
    if e in ("rising", "rise", "pos", "positive"):
        return int(abs(phys_ch))
    if e in ("falling", "fall", "neg", "negative"):
        return -int(abs(phys_ch))
    raise ValueError(f"Unknown edge '{edge}'. Use 'rising' or 'falling'.")


class SwabianTimeTaggerUltra:
    """
    Low-level wrapper: connect, configure trigger levels, and provide a couple counting primitives.
    """

    def __init__(self, config: TTConfig = TTConfig()):
        self.config = config
        self._tagger = None
        self._lock = threading.Lock()

        # Precomputed "edge channels" used in measurements
        self.start_ch = _edge_channel(config.start_phys_ch, config.start_edge)
        self.click_ch = _edge_channel(config.click_phys_ch, config.click_edge)

    @staticmethod
    def list_connected_serials() -> List[str]:
        """
        List serial numbers of connected but not-instantiated Time Taggers.
        """
        return list(TimeTagger.scanTimeTagger())

    def connect(self) -> None:
        """
        Create the TimeTagger instance and apply channel trigger levels.
        """
        with self._lock:
            if self._tagger is not None:
                return

            if self.config.serial:
                self._tagger = TimeTagger.createTimeTagger(self.config.serial)
            else:
                self._tagger = TimeTagger.createTimeTagger()

            # Trigger levels are set per *physical channel number* (positive)
            # (API: setTriggerLevel(channel, voltage))
            self._tagger.setTriggerLevel(self.config.start_phys_ch, float(self.config.start_trigger_v))
            self._tagger.setTriggerLevel(self.config.click_phys_ch, float(self.config.click_trigger_v))

    def close(self) -> None:
        with self._lock:
            if self._tagger is not None:
                TimeTagger.freeTimeTagger(self._tagger)
                self._tagger = None

    @property
    def tagger(self):
        if self._tagger is None:
            raise RuntimeError("TimeTagger not connected. Call connect() first.")
        return self._tagger

    def get_device_info(self) -> Dict[str, Any]:
        """
        Handy for logging.
        """
        return {
            "serial": self.tagger.getSerial(),
            "model": self.tagger.getModel(),
            "start_phys_ch": self.config.start_phys_ch,
            "click_phys_ch": self.config.click_phys_ch,
            "start_edge_channel": self.start_ch,
            "click_edge_channel": self.click_ch,
            "start_trigger_v": self.tagger.getTriggerLevel(self.config.start_phys_ch),
            "click_trigger_v": self.tagger.getTriggerLevel(self.config.click_phys_ch),
        }

    def count_for(self, exposure_s: float, clear: bool = True) -> Dict[str, int]:
        """
        Blocking integration for a fixed exposure time using Counter:
          - binwidth is in picoseconds
          - n_values=1 gives you a single integrated bin per channel

        Returns dict with integer counts for click and start channels.
        """
        if exposure_s <= 0:
            raise ValueError("exposure_s must be > 0")

        binwidth_ps = int(round(exposure_s * 1e12))  # Time Tagger API uses ps. :contentReference[oaicite:2]{index=2}
        channels = [self.click_ch, self.start_ch]

        meas = TimeTagger.Counter(self.tagger, channels, binwidth=binwidth_ps, n_values=1)
        meas.startFor(binwidth_ps, clear=bool(clear))         # duration in ps :contentReference[oaicite:3]{index=3}
        meas.waitUntilFinished()

        data = np.array(meas.getData(rolling=True))  # shape: (n_channels, 1) :contentReference[oaicite:4]{index=4}
        # data[0,0] corresponds to channels[0], etc.
        return {
            "click": int(data[0, 0]),
            "start": int(data[1, 0]),
        }

    def countrate_for(self, exposure_s: float, clear: bool = True) -> Dict[str, float]:
        """
        Blocking average countrate (Hz) over exposure_s using Countrate. :contentReference[oaicite:5]{index=5}
        """
        if exposure_s <= 0:
            raise ValueError("exposure_s must be > 0")

        duration_ps = int(round(exposure_s * 1e12))
        channels = [self.click_ch, self.start_ch]
        cr = TimeTagger.Countrate(self.tagger, channels)
        cr.startFor(duration_ps, clear=bool(clear))
        cr.waitUntilFinished()
        rates = np.array(cr.getData(), dtype=float)  # counts/s per channel :contentReference[oaicite:6]{index=6}
        return {
            "click_hz": float(rates[0]),
            "start_hz": float(rates[1]),
        }


class TimeTaggerSPCM:
    """
    "SPCM-like" threaded acquisition, matching your WS7 / PowerMeter style:

    - start_acquisition(start_time, start_event, stop_event, sleep_time)
    - stop_acquisition() returns QDLFDataManager with (t, click_counts_per_bin, start_counts_per_bin)

    Internally uses Countrate.getCountsTotal() and differences it to get counts per sample interval,
    which is robust for long/continuous runs. :contentReference[oaicite:7]{index=7}
    """

    def __init__(self, config: TTConfig = TTConfig()):
        self.tt = SwabianTimeTaggerUltra(config)
        self.pipeline: "queue.Queue[Any]" = queue.Queue()
        self.thread: Optional[threading.Thread] = None

        self._countrate = None
        self._last_totals: Optional[np.ndarray] = None

    def start_acquisition(
        self,
        start_time: float,
        start_event: threading.Event,
        stop_event: threading.Event,
        sleep_time: float = 0.05,
    ) -> bool:
        self.tt.connect()

        # Create Countrate measurement on (edge) channels
        channels = [self.tt.click_ch, self.tt.start_ch]
        self._countrate = TimeTagger.Countrate(self.tt.tagger, channels)

        self.thread = threading.Thread(
            target=self._acquisition_loop,
            args=(start_time, sleep_time, start_event, stop_event),
            daemon=True,
        )
        self.thread.start()
        return True

    def _acquisition_loop(
        self,
        start_time: float,
        sleep_time: float,
        start_event: threading.Event,
        stop_event: threading.Event,
    ) -> None:
        # Wait until the scan says "go"
        while not start_event.is_set() and not stop_event.is_set():
            time.sleep(sleep_time)

        if stop_event.is_set():
            self.pipeline.put(self._empty_dm())
            return

        # Reset counters at t0
        assert self._countrate is not None
        self._countrate.clear()
        self._last_totals = np.array(self._countrate.getCountsTotal(), dtype=np.int64)

        t_list: List[float] = []
        click_list: List[int] = []
        start_list: List[int] = []

        while not stop_event.is_set():
            time.sleep(sleep_time)
            now = time.time()

            totals = np.array(self._countrate.getCountsTotal(), dtype=np.int64)  # totals since clear() :contentReference[oaicite:8]{index=8}
            delta = totals - self._last_totals
            self._last_totals = totals

            t_list.append(now - start_time)
            click_list.append(int(delta[0]))
            start_list.append(int(delta[1]))

        data = {
            "x1": np.array(t_list, dtype=float),
            "y1": np.array(click_list, dtype=np.int64),
            "y2": np.array(start_list, dtype=np.int64),
        }
        params = {
            "sleep_time": sleep_time,
            "click_edge_channel": self.tt.click_ch,
            "start_edge_channel": self.tt.start_ch,
            "click_trigger_v": self.tt.tagger.getTriggerLevel(self.tt.config.click_phys_ch),
            "start_trigger_v": self.tt.tagger.getTriggerLevel(self.tt.config.start_phys_ch),
        }

        if QDLFDataManager is None:
            self.pipeline.put({"data": data, "parameters": params, "datatype": "timetagger_counts"})
        else:
            self.pipeline.put(QDLFDataManager(data, parameters=params, datatype="timetagger_counts"))

    def _empty_dm(self):
        if QDLFDataManager is None:
            return {"data": {"x1": [], "y1": [], "y2": []}, "parameters": {}, "datatype": "timetagger_counts"}
        return QDLFDataManager({"x1": [], "y1": [], "y2": []}, parameters={}, datatype="timetagger_counts")

    def stop_acquisition(self):
        if self.thread is None:
            return self._empty_dm()

        self.thread.join()
        self.thread = None

        dm = self.pipeline.get()

        # Countrate measurement object will be GC'd; you can explicitly stop() if desired.
        self._countrate = None
        self._last_totals = None

        return dm

    def stop_and_save_acquisition(self, filename: str):
        dm = self.stop_acquisition()
        if hasattr(dm, "save"):
            dm.save(filename)
        else:
            # standalone fallback
            import json
            with open(filename, "w") as f:
                json.dump(dm, f, indent=2)

    def close(self):
        self.tt.close()


if __name__ == "__main__":
    # Quick standalone sanity test:
    # 1) run this script
    # 2) it will integrate counts for 1.0 s and print them
    cfg = TTConfig(
        start_phys_ch=2, start_trigger_v=1.0, start_edge="rising",
        click_phys_ch=1, click_trigger_v=-0.08, click_edge="falling",
        serial=None,  # or put your serial string here
    )

    tt = SwabianTimeTaggerUltra(cfg)
    print("Connected taggers:", SwabianTimeTaggerUltra.list_connected_serials())
    tt.connect()
    print("Device:", tt.get_device_info())

    counts = tt.count_for(1.0)
    rates = tt.countrate_for(1.0)
    print("1.0 s counts:", counts)
    print("1.0 s rates (Hz):", rates)

    tt.close()
