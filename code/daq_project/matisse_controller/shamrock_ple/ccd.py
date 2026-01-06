import time
from ctypes import (
    c_float,
    c_int,
    c_char_p,
    c_int32,
    c_long,
    POINTER,
    pointer,
    byref,
)

import numpy as np

import matisse_controller.config as cfg
from matisse_controller.shamrock_ple.constants import *
from matisse_controller.shamrock_ple.utils import load_lib


# TODO: Add action to give live feed from CCD
class CCD:
    LIBRARY_NAME = "atmcd64d.dll"
    WIDTH = 1024
    HEIGHT = 256
    MIN_TEMP = -120
    MAX_TEMP = -10
    PIXEL_WIDTH = 26.0  # um
    PIXEL_HEIGHT = 26.0  # um

    def _bind_prototypes(self):
        """
        Bind argtypes/restype for the SDK calls we use.
        This prevents ctypes from guessing wrong.
        """
        lib = self.lib

        # Temperature calls
        if hasattr(lib, "GetTemperatureF"):
            lib.GetTemperatureF.argtypes = [POINTER(c_float)]
            lib.GetTemperatureF.restype = c_int

        if hasattr(lib, "GetTemperature"):
            lib.GetTemperature.argtypes = [POINTER(c_int)]
            lib.GetTemperature.restype = c_int

        if hasattr(lib, "SetTemperature"):
            lib.SetTemperature.argtypes = [c_int]
            lib.SetTemperature.restype = c_int

        if hasattr(lib, "CoolerON"):
            lib.CoolerON.argtypes = []
            lib.CoolerON.restype = c_int

        if hasattr(lib, "CoolerOFF"):
            lib.CoolerOFF.argtypes = []
            lib.CoolerOFF.restype = c_int

        # Cooler mode (persist cooling after ShutDown)
        if hasattr(lib, "SetCoolerMode"):
            lib.SetCoolerMode.argtypes = [c_int]
            lib.SetCoolerMode.restype = c_int

        # Acquisition timing helper
        if hasattr(lib, "GetAcquisitionTimings"):
            lib.GetAcquisitionTimings.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float)]
            lib.GetAcquisitionTimings.restype = c_int

    def __init__(self, initialize_on_definition: bool = True):
        self.lib = load_lib(CCD.LIBRARY_NAME)
        self._bind_prototypes()

        # Controls whether shutdown/destructor keeps cooler running.
        # Set to True in scanning scripts if you want cooling persisted between runs.
        self.keep_cooler_on: bool = False

        if initialize_on_definition:
            self.initialize()
            self.WIDTH, self.HEIGHT = self.get_camera_size()
            self.PIXEL_WIDTH, self.PIXEL_HEIGHT = self.get_pixel_size()

        self.temperature_ok = False
        self.exit_flag = False

    def __del__(self):
        # Never let destructor exceptions kill interpreter shutdown
        try:
            self.shutdown()  # honors self.keep_cooler_on
        except Exception:
            pass

    # -----------------------
    # Acquisition timing
    # -----------------------
    def get_acquisition_timings(self):
        """
        Returns (exposure_s, accumulation_s, kinetic_s) as floats.
        """
        exposure = c_float()
        accumulate = c_float()
        kinetic = c_float()
        self.lib.GetAcquisitionTimings(pointer(exposure), pointer(accumulate), pointer(kinetic))
        return exposure.value, accumulate.value, kinetic.value

    # -----------------------
    # Kinetics setup (scan)
    # -----------------------
    def setup_kinetics(
        self,
        exposure_time: float,
        cycle_time: float,
        n_frames: int,
        readout_mode=READ_MODE_FVB,
        temperature: float = -70,
        cool_down: bool = True,
        cosmic_ray_filter=COSMIC_RAY_FILTER_ON,
    ):
        """
        Configure CCD for kinetic series acquisition (N frames).
        cycle_time is the requested time between frames (includes readout).

        Behavior:
          - Always sets setpoint + ensures cooler ON (non-blocking).
          - Only waits for cooldown if cool_down=True.
        """
        self.exit_flag = False

        print(f"Setting temperature setpoint to {temperature} °C and ensuring cooler is ON")
        self.ensure_cooling(float(temperature), persist_on_shutdown=True)

        if cool_down:
            self.wait_to_cooldown(target_C=float(temperature), tol_C=1.0, poll_s=5.0, timeout_s=1200)

        print("Configuring kinetic acquisition parameters.")
        self.lib.SetAcquisitionMode(c_int(ACQ_MODE_KINETICS))
        self.lib.SetReadMode(c_int(readout_mode))
        self.lib.SetVSSpeed(c_int(1))
        self.lib.SetTriggerMode(c_int(TRIGGER_MODE_INTERNAL))
        self.lib.SetExposureTime(c_float(float(exposure_time)))

        # Kinetics-specific
        self.lib.SetNumberKinetics(c_int(int(n_frames)))
        self.lib.SetKineticCycleTime(c_float(float(cycle_time)))

        # Typically 1 unless you intentionally accumulate within each frame
        self.lib.SetNumberAccumulations(c_int(1))

        self.lib.SetFilterMode(c_int(cosmic_ray_filter))

        exp_s, acc_s, kin_s = self.get_acquisition_timings()
        print(f"Requested: exposure={exposure_time}s, cycle={cycle_time}s, frames={n_frames}")
        print(f"Actual   : exposure={exp_s}s, kinetic_cycle={kin_s}s")
        return exp_s, kin_s

    # -----------------------
    # Stationary setup
    # -----------------------
    def setup(
        self,
        exposure_time: float,
        acquisition_mode=ACQ_MODE_SINGLE,
        readout_mode=READ_MODE_FVB,
        temperature: float = -70,
        cool_down: bool = True,
        number_accumulations: int = 2,
        cosmic_ray_filter=COSMIC_RAY_FILTER_ON,
    ):
        """
        Perform setup procedures on CCD, like cooling down and setting acquisition parameters.

        Behavior:
          - Always sets setpoint + ensures cooler ON (non-blocking).
          - Only waits for cooldown if cool_down=True.
        """
        self.exit_flag = False

        print(f"Setting temperature setpoint to {temperature} °C and ensuring cooler is ON")
        self.ensure_cooling(float(temperature), persist_on_shutdown=True)

        if cool_down:
            self.wait_to_cooldown(target_C=float(temperature), tol_C=1.0, poll_s=5.0, timeout_s=1200)

        print("Configuring acquisition parameters.")
        self.set_acquisition_parameters(
            exposure_time,
            acquisition_mode,
            readout_mode,
            number_accumulations,
            cosmic_ray_filter,
        )
        print("CCD ready for acquisition.")

    def set_acquisition_parameters(
        self,
        exposure_time: float,
        acquisition_mode=ACQ_MODE_ACCUMULATE,
        readout_mode=READ_MODE_FVB,
        number_accumulations: int = 2,
        cosmic_ray_filter=COSMIC_RAY_FILTER_ON,
    ):
        self.lib.SetAcquisitionMode(c_int(acquisition_mode))
        self.lib.SetNumberAccumulations(c_int(number_accumulations))
        self.lib.SetReadMode(c_int(readout_mode))
        self.lib.SetVSSpeed(c_int(1))
        self.lib.SetTriggerMode(c_int(TRIGGER_MODE_INTERNAL))
        self.lib.SetExposureTime(c_float(float(exposure_time)))
        self.lib.SetFilterMode(c_int(cosmic_ray_filter))

    # -----------------------
    # Temperature helpers
    # -----------------------
    def get_temperature(self) -> float:
        """
        Temperature in °C as a float (even if SDK gives an int).
        """
        _code, tempC = self.get_temperature_status()
        return tempC

    def get_temperature_status(self) -> (int, float):
        """
        Returns (status_code, temperature_C).

        SDK2:
          - GetTemperature(int*) returns a status code and fills an int temperature.
          - Some builds also provide GetTemperatureF(float*) for higher precision.
        """
        if hasattr(self.lib, "GetTemperatureF"):
            t = c_float()
            code = int(self.lib.GetTemperatureF(byref(t)))
            return code, float(t.value)

        t = c_int()
        code = int(self.lib.GetTemperature(byref(t)))
        return code, float(t.value)

    def set_temperature(self, temperature: float):
        self.lib.SetTemperature(c_int(int(temperature)))

    def get_temperature_range(self) -> (float, float):
        min_temp, max_temp = c_int(), c_int()
        self.lib.GetTemperatureRange(pointer(min_temp), pointer(max_temp))
        return float(min_temp.value), float(max_temp.value)

    def ensure_cooling(self, target_C: float, persist_on_shutdown: bool = True):
        """
        Start/keep the cooler running and set the temperature setpoint.

        If persist_on_shutdown is True, we set CoolerMode=1 (when available) so that cooling can
        persist after ShutDown (useful when you don't want to wait for cooldown during scans).
        """
        # record intent for shutdown/destructor behavior
        self.keep_cooler_on = bool(persist_on_shutdown)

        try:
            if persist_on_shutdown and hasattr(self.lib, "SetCoolerMode"):
                # 1 = maintain temperature on shutdown (Andor SDK)
                self.lib.SetCoolerMode(c_int(1))
        except Exception:
            pass

        try:
            self.lib.SetTemperature(c_int(int(target_C)))
        except Exception:
            pass

        try:
            self.lib.CoolerON()
        except Exception:
            pass

    def wait_to_cooldown(
        self,
        target_C: float = -65.0,
        tol_C: float = 1.0,
        poll_s: float = 5.0,
        timeout_s: float = 1200,
    ):
        """
        Wait until CCD temperature <= target_C + tol_C (or SDK reports stabilized), or timeout.
        """
        t_start = time.time()
        self.temperature_ok = False

        while not self.temperature_ok:
            if self.exit_flag:
                return

            code, tempC = self.get_temperature_status()
            print(f"Cooling CCD: T={tempC:.1f} °C  (status={code})")

            # If SDK reports stabilized, accept it.
            try:
                if code == CCDErrorCode.DRV_TEMPERATURE_STABILIZED.value:
                    self.temperature_ok = True
                    break
            except Exception:
                pass

            if tempC <= (target_C + tol_C):
                self.temperature_ok = True
                break

            if (time.time() - t_start) > timeout_s:
                raise TimeoutError(
                    f"CCD did not reach {target_C}±{tol_C} °C within {timeout_s}s (last T={tempC:.1f} °C)"
                )

            time.sleep(poll_s)

    def wait_until_cold(
        self,
        target_C: float,
        tol_C: float = 1.0,
        timeout_s: float = 0.0,
        poll_s: float = 5.0,
    ) -> float:
        """
        Wait until CCD temperature <= target_C + tol_C.

        If timeout_s <= 0: do a single check and return immediately.
        Returns the latest measured temperature.
        """
        t0 = time.time()
        while True:
            temp = self.get_temperature()
            if (temp is not None) and (not np.isnan(temp)) and (temp <= target_C + tol_C):
                return float(temp)

            if timeout_s <= 0:
                return float(temp)

            if (time.time() - t0) >= timeout_s:
                return float(temp)

            time.sleep(poll_s)

    # -----------------------
    # Acquisition
    # -----------------------
    def take_acquisition(self, num_points=WIDTH) -> np.ndarray:
        self.exit_flag = False
        self.start_acquisition()
        acquisition_array_type = c_int32 * int(num_points)
        data = acquisition_array_type()
        self.wait_for_acquisition()
        self.lib.GetAcquiredData(data, c_int(int(num_points)))
        return np.flip(np.array(data, dtype=np.int32))

    def start_acquisition(self):
        self.lib.StartAcquisition()

    def wait_for_acquisition(self):
        # self.lib.WaitForAcquisition() often unreliable; poll status instead.
        while True:
            if self.exit_flag:
                break
            status = c_int()
            self.lib.GetStatus(pointer(status))
            if status.value == CCDErrorCode.DRV_IDLE.value:
                break
            time.sleep(1)

    # -----------------------
    # Init / info
    # -----------------------
    def initialize(self):
        """Run CCD-related initialization procedures."""
        try:
            self.lib.Initialize()
            # Turn cooler on by default; scripts can later choose persistence via ensure_cooling()
            self.lib.SetTemperature(c_int(int(cfg.get(cfg.PLE_TARGET_TEMPERATURE))))
            self.lib.CoolerON()

            num_cameras = c_long()
            self.lib.GetAvailableCameras(pointer(num_cameras))
            assert num_cameras.value > 0, "No CCD camera found."
        except OSError as err:
            raise RuntimeError("Unable to initialize Andor CCD API.") from err

    def get_camera_size(self) -> (int, int):
        width, height = c_int(), c_int()
        self.lib.GetDetector(pointer(width), pointer(height))
        return int(width.value), int(height.value)

    def get_pixel_size(self):
        width, height = c_float(), c_float()
        self.lib.GetPixelSize(pointer(width), pointer(height))
        return float(width.value), float(height.value)

    # -----------------------
    # Save
    # -----------------------
    def save_as_sif(self, filename: str, calibration_values=None):
        filename_b = filename.encode("utf-8")
        print("save_as_sif called:", filename_b)
        self.lib.SaveAsSif(c_char_p(filename_b))

    # -----------------------
    # Shutdown / cleanup
    # -----------------------
    def shutdown(self, keep_cooler_on: bool = None):
        """
        If keep_cooler_on=True: do not turn cooler off; try to persist cooling.
        If keep_cooler_on=False: turn cooler off.
        If keep_cooler_on is None: use self.keep_cooler_on.
        """
        if keep_cooler_on is None:
            keep_cooler_on = bool(getattr(self, "keep_cooler_on", False))

        try:
            if keep_cooler_on:
                # Request cooling persistence, then shut down API without turning off cooler
                try:
                    if hasattr(self.lib, "SetCoolerMode"):
                        self.lib.SetCoolerMode(c_int(1))
                except Exception:
                    pass
            else:
                # Hard off
                try:
                    self.lib.CoolerOFF()
                except Exception:
                    pass
        finally:
            try:
                self.lib.ShutDown()
            except Exception:
                pass
