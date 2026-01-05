from __future__ import annotations
from daq.protocols import Laser

class MatisseLaserAdapter(Laser):
    def __init__(self, matisse_driver):
        """
        matisse_driver: your existing working driver instance
        """
        self.d = matisse_driver

    def lock_to_wavelength_nm(self, wl_nm: float) -> None:
        # Map to your existing method
        # Example possibilities:
        #   self.d.set_wavelength_nm(wl_nm)
        #   self.d.lock(wl_nm)
        #   self.d.goto_wavelength(wl_nm); self.d.enable_lock()
        self.d.lock_to_wavelength_nm(wl_nm)

    def set_scan_speed(self, speed: float) -> None:
        # speed units: whatever your driver expects
        self.d.set_scan_speed(speed)

    def start_scan(self, direction: str) -> None:
        # direction: "up" or "down"
        self.d.start_scan(direction=direction)

    def stop_scan(self) -> None:
        self.d.stop_scan()

    def close(self) -> None:
        # good practice even if your driver is a no-op on close
        if hasattr(self.d, "close"):
            self.d.close()
