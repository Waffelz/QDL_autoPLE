# inside PowerMeter class in src/daq/drivers/powermeter_raw.py


import threading
import time

class PowerMeter:
    # ... your existing class ...

    def open(self):
        """Open VISA session once (fast scanning requires keeping it open)."""
        if getattr(self, "_is_open", False):
            return
        self.powermeter.initialize_instrument()
        self._empty_buffer()
        self._is_open = True
        if not hasattr(self, "_io_lock"):
            self._io_lock = threading.Lock()

    def close(self):
        """Close VISA session."""
        if getattr(self, "_is_open", False):
            self.powermeter.terminate_instrument()
            self._is_open = False

    def read_power_uW(self) -> float:
        """
        Single-shot read. Returns ONE scalar in µW (consistent with your existing conversion).
        If channel='AB', returns mean(A,B) as one number.
        """
        self.open()

        with self._io_lock:
            reading_strings = self.powermeter.get_instrument_reading_string_all()

        readings_uW = []
        for s in reading_strings:
            try:
                readings_uW.append(self.convert_reading_string_to_float(s))
            except Exception:
                pass

        if not readings_uW:
            raise RuntimeError("Newport power meter returned no parseable readings")

        return float(sum(readings_uW) / len(readings_uW))
