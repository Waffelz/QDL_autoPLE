from __future__ import annotations
import numpy as np
from daq.protocols import DetectorCounts

class TimeTaggerCountsAdapter(DetectorCounts):
    def __init__(self, tt_driver):
        """
        tt_driver: your existing wrapper around Swabian TimeTagger.
        It should already know channels etc.
        """
        self.d = tt_driver

    def acquire_counts(self, exposure_s: float) -> int:
        # Map to your driver.
        # Example possibilities:
        #   return self.d.counts_for(exposure_s)
        #   return self.d.acquire_counts(exposure_s)
        return int(self.d.acquire_counts(exposure_s))

    def close(self) -> None:
        if hasattr(self.d, "close"):
            self.d.close()
