# inside WS7 class (in ws7_raw.py)

import threading
import time

class WS7:
    lib_path = 'wlmData.dll'

    def __init__(self):
        wlmData.LoadDLL(self.lib_path)
        self.lib = wlmData.dll
        self.pipeline = queue.Queue()
        self.thread = None
        self._io_lock = threading.Lock()

    def get_wavelength_nm(self, retries: int = 3, retry_sleep_s: float = 0.01) -> float:
        """
        Single-shot read of channel 1 (GetWavelength).
        Returns >0 wavelength in nm, or raises RuntimeError if it can't get a valid read.
        """
        for _ in range(retries):
            with self._io_lock:
                wl = float(self.lib.GetWavelength(0.0))
            if wl > 0:
                return wl
            time.sleep(retry_sleep_s)
        raise RuntimeError("WS7 returned invalid wavelength (<=0) after retries")

    def get_wavelength2_nm(self, retries: int = 3, retry_sleep_s: float = 0.01) -> float:
        """
        Single-shot read of channel 2 (GetWavelength2).
        """
        for _ in range(retries):
            with self._io_lock:
                wl2 = float(self.lib.GetWavelength2(0.0))
            if wl2 > 0:
                return wl2
            time.sleep(retry_sleep_s)
        raise RuntimeError("WS7 returned invalid wavelength2 (<=0) after retries")
