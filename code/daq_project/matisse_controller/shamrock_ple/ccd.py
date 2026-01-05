import time
from ctypes import c_float, c_int, c_char_p, c_int32, c_long, pointer, c_int, c_float
from ctypes import POINTER, byref

import numpy as np

import matisse_controller.config as cfg
from matisse_controller.shamrock_ple.constants import *
from matisse_controller.shamrock_ple.utils import load_lib


# TODO: Add action to give live feed from CCD
class CCD:
    LIBRARY_NAME = 'atmcd64d.dll'
    WIDTH = 1024
    HEIGHT = 256
    MIN_TEMP = -120
    MAX_TEMP = -10
    PIXEL_WIDTH = 26.  # in um
    PIXEL_HEIGHT = 26.  # in um

    def _bind_prototypes(self):
        """
        Bind argtypes/restype for the SDK calls we use.
        This prevents ctypes from guessing wrong.
        """
        lib = self.lib

        # Temperature calls (SDK2)
        if hasattr(lib, "GetTemperatureF"):
            lib.GetTemperatureF.argtypes = [POINTER(c_float)]
            lib.GetTemperatureF.restype = c_int
        lib.GetTemperature.argtypes = [POINTER(c_int)]
        lib.GetTemperature.restype = c_int

        lib.SetTemperature.argtypes = [c_int]
        lib.SetTemperature.restype = c_int

        lib.CoolerON.argtypes = []
        lib.CoolerON.restype = c_int
        lib.CoolerOFF.argtypes = []
        lib.CoolerOFF.restype = c_int

    def __init__(self, initialize_on_definition=True):

        self.lib = load_lib(CCD.LIBRARY_NAME)
        self._bind_prototypes()  # <<< add this line

        if initialize_on_definition:
            self.initialize()
            self.WIDTH, self.HEIGHT = self.get_camera_size()
            self.PIXEL_WIDTH, self.PIXEL_HEIGHT = self.get_pixel_size()

        self.temperature_ok = False
        self.exit_flag = False

    def __del__(self):
        self.shutdown()

#setup for kinetic scan
    def get_acquisition_timings(self):
        """
        Returns (exposure_s, accumulation_s, kinetic_s) as floats.
        """
        exposure = c_float()
        accumulate = c_float()
        kinetic = c_float()
        # This is a standard Andor SDK call; your DLL should export it.
        self.lib.GetAcquisitionTimings(pointer(exposure), pointer(accumulate), pointer(kinetic))
        return exposure.value, accumulate.value, kinetic.value

    def setup_kinetics(
            self,
            exposure_time: float,
            cycle_time: float,
            n_frames: int,
            readout_mode=READ_MODE_FVB,
            temperature=-70,
            cool_down=True,
            cosmic_ray_filter=COSMIC_RAY_FILTER_ON,
    ):
        """
        Configure CCD for kinetic series acquisition (N frames).
        cycle_time is the requested time between frames (includes readout).
        """
        self.exit_flag = False

        if cool_down:
            print(f"Setting temperature to {temperature} C")
            self.lib.SetTemperature(c_int(int(temperature)))
            time.sleep(1)
            self.lib.CoolerON()
            time.sleep(1)
            self.wait_to_cooldown()

        print("Configuring kinetic acquisition parameters.")
        self.lib.SetAcquisitionMode(c_int(ACQ_MODE_KINETICS))
        self.lib.SetReadMode(c_int(readout_mode))
        self.lib.SetVSSpeed(c_int(1))
        self.lib.SetTriggerMode(c_int(TRIGGER_MODE_INTERNAL))
        self.lib.SetExposureTime(c_float(float(exposure_time)))

        # Kinetics-specific
        self.lib.SetNumberKinetics(c_int(int(n_frames)))
        self.lib.SetKineticCycleTime(c_float(float(cycle_time)))

        # In kinetics, "accumulations" is usually 1 unless you're intentionally accumulating within each frame
        self.lib.SetNumberAccumulations(c_int(1))

        self.lib.SetFilterMode(c_int(cosmic_ray_filter))

        exp_s, acc_s, kin_s = self.get_acquisition_timings()
        print(f"Requested: exposure={exposure_time}s, cycle={cycle_time}s, frames={n_frames}")
        print(f"Actual   : exposure={exp_s}s, kinetic_cycle={kin_s}s")
        return exp_s, kin_s

#setup for stationary scan
    def setup(self, exposure_time: float, acquisition_mode=ACQ_MODE_SINGLE, readout_mode=READ_MODE_FVB,
              temperature=-70, cool_down=True, number_accumulations=2, cosmic_ray_filter=COSMIC_RAY_FILTER_ON):
        """
        Perform setup procedures on CCD, like cooling down to a given temperature and setting acquisition parameters.

        Parameters
        ----------
        exposure_time
            the desired exposure time at which to configure the CCD
        acquisition_mode
            the desired acquisition mode at which to configure the CCD (default is accumulate)
        readout_mode
            the desired readout mode at which to configure the CCD (default is FVB)
        temperature
            the desired temperature in degrees centigrade at which to configure the CCD (default is -70)
        cool_down
            whether to cool down the CCD at all (sometimes we don't care, like when taking a single acquisition)
        number_accumulations
            the number of accumulations if the acquisition mode is set to Accumulation mode or kinetic mode
        cosmic_ray_filter
            determines if the cosmic ray filter is OFF (0) or ON (2).
        """
        self.exit_flag = False
        if cool_down:
            # min_temp, max_temp = self.get_temperature_range()
            # assert min_temp < temperature < max_temp, f"Temperature must be set between {min_temp} and {max_temp}"
            print(f"Setting temperature to {temperature} C")
            self.lib.SetTemperature(c_int(temperature))
            time.sleep(1)
            self.lib.CoolerON()
            time.sleep(1)
            self.wait_to_cooldown()

        print('Configuring acquisition parameters.')
        self.set_acquisition_parameters(exposure_time, acquisition_mode, readout_mode, number_accumulations,
                                        cosmic_ray_filter)
        print('CCD ready for acquisition.')

    def set_acquisition_parameters(self, exposure_time: float, acquisition_mode=ACQ_MODE_ACCUMULATE,
                                   readout_mode=READ_MODE_FVB, number_accumulations=2,
                                   cosmic_ray_filter=COSMIC_RAY_FILTER_ON):
        """
        Perform setup procedures on CCD, like cooling down to a given temperature and setting acquisition parameters.

        Parameters
        ----------
        exposure_time
            the desired exposure time at which to configure the CCD
        acquisition_mode
            the desired acquisition mode at which to configure the CCD (default is accumulate)
        readout_mode
            the desired readout mode at which to configure the CCD (default is FVB)
        number_accumulations
            the number of accumulations if the acquisition mode is set to Accumulation mode or kinetic mode
        cosmic_ray_filter
            determines if the cosmic ray filter is OFF (0) or ON (2).
        """
        self.lib.SetAcquisitionMode(c_int(acquisition_mode))
        self.lib.SetNumberAccumulations(c_int(number_accumulations))
        self.lib.SetReadMode(c_int(readout_mode))
        self.lib.SetVSSpeed(c_int(1))
        self.lib.SetTriggerMode(c_int(TRIGGER_MODE_INTERNAL))
        self.lib.SetExposureTime(c_float(exposure_time))
        self.lib.SetFilterMode(c_int(cosmic_ray_filter))

    # def get_temperature(self) -> float:
    #     """
    #     Returns
    #     -------
    #     float
    #         the current temperature of the CCD camera
    #     """
    #     temperature = c_float()
    #     self.lib.GetTemperature(pointer(temperature))
    #     return temperature.value
#updated methods:
    def get_temperature(self) -> float:
        """
        Temperature in °C as a float (even if SDK gives an int).
        """
        _code, tempC = self.get_temperature_status()
        return tempC

    def get_temperature_status(self) -> (int, float):
        """
        Returns (status_code, temperature_C).
        SDK2 GetTemperature returns a status code and fills an int temperature.
        Some builds also provide GetTemperatureF.
        """
        if hasattr(self.lib, "GetTemperatureF"):
            t = c_float()
            code = int(self.lib.GetTemperatureF(byref(t)))
            return code, float(t.value)

        t = c_int()
        code = int(self.lib.GetTemperature(byref(t)))
        return code, float(t.value)

    def set_temperature(self, temperature: float):
        """
        Parameters
        -------
        temperature: float
            the target temperature of the CCD camera
        """
        self.lib.SetTemperature(c_int(int(temperature)))

    def get_temperature_range(self) -> (float, float):
        """
        Returns
        -------
        float, float
            the temperature range minimum and maximum of the CCD camera
        """
        min_temp, max_temp = c_int(), c_int()
        self.lib.GetTemperatureRange(pointer(min_temp), pointer(max_temp))
        return min_temp.value, max_temp.value

    # def wait_to_cooldown(self):
    #     """Goes on a loop to wait until the temperature is close to the set temperature of the CCD camera."""
    #     # Cooler stops when temp is within 3 degrees of target, so wait until it's close
    #     # CCD normally takes a few minutes to fully cool down
    #     temperature = self.get_temperature_range()
    #     while not self.temperature_ok:
    #         if self.exit_flag:
    #             return
    #         current_temp = self.get_temperature()
    #         print(f"Cooling CCD. Current temperature is {round(current_temp, 2)} °C")
    #         # if current_temp < cfg.get(cfg.PLE_TARGET_TEMPERATURE) + cfg.get(cfg.PLE_TEMPERATURE_TOLERANCE):
    #         if current_temp <= -62:
    #             print("reached -62C")
    #         # if current_temp <= -55:
    #         #     print("reached -55C")
    #             self.temperature_ok = True
    #         time.sleep(10)

    def wait_to_cooldown(self, target_C: float = -65.0, tol_C: float = 1.0, poll_s: float = 5.0,
                         timeout_s: float = 1200):
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

            # If SDK reports stabilized, we can accept it.
            if hasattr(CCDErrorCode,
                       "DRV_TEMPERATURE_STABILIZED") and code == CCDErrorCode.DRV_TEMPERATURE_STABILIZED.value:
                self.temperature_ok = True
                break

            if tempC <= (target_C + tol_C):
                self.temperature_ok = True
                break

            if (time.time() - t_start) > timeout_s:
                raise TimeoutError(
                    f"CCD did not reach {target_C}±{tol_C} °C within {timeout_s}s (last T={tempC:.1f}°C)")

            time.sleep(poll_s)

    def take_acquisition(self, num_points=WIDTH) -> np.ndarray:
        """
        Parameters
        ----------
        num_points
            the number of pixels to read from the CCD camera - in FVB mode, this is the width of the screen.

        Returns
        -------
        ndarray
            an array of counts for each pixel on the CCD screen
        """
        self.exit_flag = False
        self.start_acquisition()
        acquisition_array_type = c_int32 * num_points
        data = acquisition_array_type()
        self.wait_for_acquisition()
        self.lib.GetAcquiredData(data, c_int(num_points))
        data = np.flip(np.array(data, dtype=np.int32))  # Data comes out backwards! Vasilis: not really.
        # Depends on calibration parameters
        return data

    def start_acquisition(self):
        """Starts acquisition process."""
        self.lib.StartAcquisition()

    def wait_for_acquisition(self):
        """ Waits unitl acquisition is done"""
        # self.lib.WaitForAcquisition() does not work, so use a loop instead and check the status.
        while True:
            if self.exit_flag:
                break
            status = c_int()
            self.lib.GetStatus(pointer(status))
            if status.value == CCDErrorCode.DRV_IDLE.value:
                break
            else:
                time.sleep(1)

    def initialize(self):
        """Run CCD-related initialization procedures."""
        try:
            self.lib.Initialize()
            self.lib.SetTemperature(c_int(cfg.get(cfg.PLE_TARGET_TEMPERATURE)))
            self.lib.CoolerON()

            num_cameras = c_long()
            self.lib.GetAvailableCameras(pointer(num_cameras))
            assert num_cameras.value > 0, 'No CCD camera found.'
        except OSError as err:
            raise RuntimeError('Unable to initialize Andor CCD API.') from err

    def get_camera_size(self) -> (int, int):
        """
        Get CCD number of pixels.

        Returns
        -------
        width, height: float, float
            a tuple of the width and the height of the CCD camera in pixels
        """
        width, height = c_int(), c_int()
        self.lib.GetDetector(pointer(width), pointer(height))
        return width.value, height.value

    def get_pixel_size(self):
        """
        Get CCD pixel dimensions.

        Returns
        -------
        width, height: float, float
            a tuple of the dimensions of the pixels
        """
        width, height = c_float(), c_float()
        self.lib.GetPixelSize(pointer(width), pointer(height))
        return width.value, height.value

    def save_as_sif(self, filename: str, calibration_values=None):
        """
        Saves the file as sif. If a calibration value list is provided, then it is saved with a calibration.

        Parameters
        ----------
        filename: str
            the filename to use for saving.
        calibration_values
            a sized object (e.g. list or numpy.ndarray) of 4 elements for wavelength x**3 polynomial calibration.
        """
        filename = filename.encode('utf-8')
        print('save as sif callled,', filename)
        self.lib.SaveAsSif(c_char_p(filename))
#Xingyi commented out this chunk 1/30/24
        # if calibration_values is not None:
        #     if len(calibration_values) != 4:
        #         raise ValueError('calibration_values should be a list of 4 elements')
        #         print('calibration value not none', calibration_values)
        #
        #
        #     coefficients = (c_float * len(calibration_values))(*calibration_values)
        #     self.lib.SaveAsCalibratedSif(c_char_p(filename), c_int(DATA_TYPE_WAVELENGTH), c_int(UNITS_WAVELENGTH_NM),
        #                                  pointer(coefficients))
        #     print('save sif called')
        # else:
        #     self.lib.SaveAsSif(c_char_p(filename))
        #     print('else save sif called')

    def shutdown(self):
        """Run CCD-related cleanup and shutdown procedures."""
        self.lib.CoolerOFF()
        self.lib.ShutDown()
        # TODO: Before shutting it down, we should wait for temp to hit -20 °C, otherwise it rises too fast
        # In practice, of course, we don't do this :)
