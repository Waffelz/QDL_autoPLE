from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from daq.protocols import Wavemeter

# import your raw driver
from daq.drivers.ws7_raw import WS7


@dataclass
class WS7WavemeterAdapter(Wavemeter):
    """
    Adapts your existing WS7 class to the simple Wavemeter protocol used by scan logic.
    """
    ws7: WS7
    channel: int = 1  # 1 or 2

    def get_wavelength_nm(self) -> float:
        if self.channel == 1:
            return self.ws7.get_wavelength_nm()
        elif self.channel == 2:
            return self.ws7.get_wavelength2_nm()
        else:
            raise ValueError("WS7 channel must be 1 or 2")

    def close(self) -> None:
        # your raw driver doesn't need explicit close; keep for symmetry
        pass
