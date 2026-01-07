from __future__ import annotations
import numpy as np
from daq.protocols import DetectorSpectrum

class AndorSpectrumAdapter(DetectorSpectrum):
    def __init__(self, andor_driver, axis: np.ndarray):
        """
        axis: fixed wavelength/energy axis for the spectrometer
        """
        self.d = andor_driver
        self._axis = np.asarray(axis, dtype=float)

    def set_exposure(self, exposure_s: float) -> None:
        self.d.set_exposure(exposure_s)

    def acquire_spectrum(self):
        # You decide what your underlying driver returns.
        # Common: intensity array only -> return (axis, intensity)
        intensity = np.asarray(self.d.snap(), dtype=float)
        return self._axis, intensity

    def close(self) -> None:
        if hasattr(self.d, "close"):
            self.d.close()
