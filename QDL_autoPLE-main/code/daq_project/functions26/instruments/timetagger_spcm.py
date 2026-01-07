import numpy as np
import threading
import time
import queue
from dataclasses import dataclass

from functions26.filing.QDLFiling import QDLFDataManager

# Swabian bindings are commonly "import TimeTagger"
# Some installations/examples use: from Swabian import TimeTagger
try:
    import TimeTagger  # type: ignore
except ImportError:
    from Swabian import TimeTagger  # type: ignore


def _edge_channel(physical_channel: int, edge: str) -> int:
    """
    Swabian convention:
      rising edge  -> +ch
      falling edge -> -ch  :contentReference[oaicite:2]{index=2}
    """
    edge = edge.strip().lower()
    if edge in ("rising", "rise", "positive", "pos"):
        return int(abs(physical_channel))
    if edge in ("falling", "fall", "negative", "neg"):
        return -int(abs(physical_channel))
    raise ValueError("edge must be 'rising' or 'falling'")


@dataclass(frozen=True)
class TimeTaggerCounterConfig:
    click_phys_ch: int = 1
    click_trigger_v: float = -0.08
    click_edge: str = "falling"     # default: falling edge => software channel -1
    serial: str | None = None       # optional if multiple taggers


class TimeTaggerSPCM:
    """
    Drop-in “SPCM-like” interface, but backed by Swabian Time Tagger Ultra.

    Produces a time trace:
      x1: time (s) since start_time
      y1: counts per time_step bin

    Internally uses Countrate.getCountsTotal() (robust total count) and diffs it. :contentReference[oaicite:3]{index=3}
    """

    def __init__(self, instrument_port=None, cfg: TimeTaggerCounterConfig | None = None):
        self.cfg = cfg or TimeTaggerCounterConfig()
        self.pipeline = queue.Queue()
        self.thread = None

        self._tagger = None
        self._countrate = None
        self._last_total = None

        self._click_sw_ch = _edge_channel(self.cfg.click_phys_ch, self.cfg.click_edge)

    def _connect(self):
        if self._tagger is not None:
            return

        # createTimeTagger() is the standard entrypoint. :contentReference[oaicite:4]{index=4}
        if self.cfg.serial:
            self._tagger = TimeTagger.createTimeTagger(self.cfg.serial)
        else:
            self._tagger = TimeTagger.createTimeTagger()

        # Trigger level set per PHYSICAL channel (in volts). :contentReference[oaicite:5]{index=5}
        self._tagger.setTriggerLevel(self.cfg.click_phys_ch, float(self.cfg.click_trigger_v))

        # Countrate: average rate + total counts since clear(). :contentReference[oaicite:6]{index=6}
        self._countrate = TimeTagger.Countrate(self._tagger, [self._click_sw_ch])

    def _disconnect(self):
        try:
            if self._countrate is not None:
                self._countrate.stop()  # common measurement method :contentReference[oaicite:7]{index=7}
        except Exception:
            pass

        self._countrate = None
        self._last_total = None

        if self._tagger is not None:
            TimeTagger.freeTimeTagger(self._tagger)
            self._tagger = None

    def start_acquisition(self, start_time: float, start_event: threading.Event, stop_event: threading.Event,
                          sleep_time: float = 0.05):
        self._connect()
        self.thread = threading.Thread(
            target=self.acquisition,
            args=(start_time, sleep_time, self.pipeline, start_event, stop_event),
            daemon=True
        )
        self.thread.start()
        return True

    def stop_and_save_acquisition(self, filename):
        dm = self.stop_acquisition()
        dm.save(filename)

    def stop_acquisition(self):
        data_manager = QDLFDataManager()
        if self.thread is not None:
            self.thread.join()
            data_manager = self.pipeline.get()
            self.thread = None

        # Match your NI-DAQ SPCM behavior: release device resources after each acquisition
        self._disconnect()
        return data_manager

    def acquisition(self, start_time: float, time_step: float, pipeline: queue.Queue,
                    start_event: threading.Event, stop_event: threading.Event):

        # Wait for "go"
        while not start_event.is_set():
            time.sleep(time_step)

        # Reset at t0: Countrate averages from the first tag after instantiation/clear(). :contentReference[oaicite:8]{index=8}
        try:
            self._countrate.clear()
        except Exception:
            pass

        # Initialize last_total
        try:
            self._last_total = int(self._countrate.getCountsTotal()[0])  # :contentReference[oaicite:9]{index=9}
        except Exception:
            self._last_total = 0

        t_list = []
        counts_list = []

        while not stop_event.is_set():
            time.sleep(time_step)
            now = time.time()

            try:
                total = int(self._countrate.getCountsTotal()[0])  # absolute total since clear() :contentReference[oaicite:10]{index=10}
                delta = total - self._last_total
                self._last_total = total

                t_list.append(now - start_time)
                counts_list.append(delta)
            except Exception:
                # Keep going like your other drivers
                pass

        t_arr = np.asarray(t_list, dtype=float)
        y_counts = np.asarray(counts_list, dtype=np.int64)

        data = {"x1": t_arr, "y1": y_counts}
        params = {
            "start_time": start_time,
            "time_step": time_step,
            "click_phys_ch": self.cfg.click_phys_ch,
            "click_trigger_v": self.cfg.click_trigger_v,
            "click_edge": self.cfg.click_edge,
            "click_sw_ch": self._click_sw_ch,
        }

        dm = QDLFDataManager(data, parameters=params, datatype="timetagger_counts")
        pipeline.put(dm)

    def is_available(self):
        try:
            self._connect()
            self._disconnect()
            return True
        except Exception:
            return False

    def shutdown(self):
        try:
            self._disconnect()
        except Exception:
            pass
