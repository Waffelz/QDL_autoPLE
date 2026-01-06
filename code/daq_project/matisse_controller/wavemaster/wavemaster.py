import threading
import warnings

import pyvisa
from serial import Serial, SerialException
from functions26.InstrumentHandler import GPIBInstrument, SerialInstrument
import numpy as np
import wlmData, wlmConst



class WaveMeter:
    """An interface to serial port communication with the Coherent WaveMaster wavemeter."""

    wavemeter_lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        self.serial = None
        self.initialize(*args, **kwargs)

    def initialize(self, *args, **kwargs):
        warnings.warn('Set your own "initialize" class method.')

    def __del__(self):
        try:
            if getattr(self, "serial", None) is not None:
                self.serial.close()
        except Exception:
            pass

    def query(self, command: str) -> str:
        """
        Wait to acquire an exclusive lock on the serial port, then send a command to the wavemeter.

        Parameters
        ----------
        command : str
            the command to send to the wavemeter

        Returns
        -------
        str
            the response from the wavemeter to the given command
        """
        warnings.warn('Set your own "query" class method.')
        return ''

    def get_raw_value(self) -> str:
        """
        Returns
        -------
        str
            the raw output from the wavemeter display
        """
        warnings.warn('Set your own "get_raw_value" class method.')
        return ''

    def get_wavelength(self) -> float:
        """
        Returns
        -------
        float
            a measurement from the wavemeter

        Notes
        -----
        Blocks the calling thread until a number is received.
        """
        warnings.warn('Set your own "get_wavelength" class method.')
        return -1.


class WaveMaster(WaveMeter):
    """An interface to serial port communication with the Coherent WaveMaster wavemeter."""

    wavemeter_lock = threading.Lock()

    def initialize(self, port: str):
        try:
            self.serial = Serial(port)
            self.serial.timeout = 10.0
            self.serial.write_timeout = 10.0
        except SerialException as err:
            raise IOError("Couldn't open connection to wavemeter.") from err

    def __del__(self):
        self.serial.close()

    def query(self, command: str) -> str:
        """
        Wait to acquire an exclusive lock on the serial port, then send a command to the wavemeter.

        Parameters
        ----------
        command : str
            the command to send to the wavemeter

        Returns
        -------
        str
            the response from the wavemeter to the given command
        """
        with WaveMaster.wavemeter_lock:
            try:
                if not self.serial.is_open:
                    self.serial.open()
                # Ensure a newline is at the end
                command = command.strip() + '\n\n'
                self.serial.write(command.encode())
                self.serial.flush()
                return self.serial.readline().strip().decode()
            except SerialException as err:
                raise IOError("Error communicating with wavemeter serial port.") from err

    def get_raw_value(self) -> str:
        """
        Returns
        -------
        str
            the raw output from the wavemeter display
        """
        return self.query('VAL?').split(',')[1].strip()

    def get_wavelength(self) -> float:
        """
        Returns
        -------
        float
            a measurement from the wavemeter

        Notes
        -----
        Blocks the calling thread until a number is received.
        """
        raw_value = self.get_raw_value()
        # Keep trying until we get a number
        while raw_value == 'NO SIGNAL' or raw_value == 'MULTI-LINE':
            raw_value = self.get_raw_value()
        return float(raw_value)


class WA1600(WaveMeter):

    def initialize(self, port, *args, **kwargs):
        # if port is None:
        #     port = 'GPIB0::18::INSTR'
        # self.device = GPIBInstrument(port, '\n', ':READ:WAV?')
        port = 'COM4'
        read_command = ':READ:WAV?\r\n'.encode('utf-8')
        self.device = SerialInstrument(port, read_command, baud_rate=9600)
        print("wa1600 initialized in wavemaster")

    def __del__(self):
        self.device.terminate_instrument()

    def query(self, command: str) -> str:
        """
        Wait to acquire an exclusive lock on the serial port, then send a command to the wavemeter.

        Parameters
        ----------
        command : str
            the command to send to the wavemeter

        Returns
        -------
        str
            the response from the wavemeter to the given command
        """
        with WA1600.wavemeter_lock:

            try:
                self.device.instrument.last_status
            except (pyvisa.errors.InvalidSession, AttributeError):
                self.device.initialize_instrument()

            return self.device.get_instrument_reading_string(command)

    def get_raw_value(self) -> str:
        raw_value = self.device.get_instrument_reading_string()
        if isinstance(raw_value, bytes):
            raw_value = raw_value.decode('utf-8')
        return raw_value

    def get_wavelength(self) -> float:
        with WaveMaster.wavemeter_lock:
            raw_value = self.get_raw_value()
            # Keep trying until we get a number
            while raw_value.startswith('0'):
                raw_value = self.get_raw_value()
            return float(raw_value)


class WS7(WaveMeter):
    lib_path = "wlmData.dll"

    ERR_DICT = {
        0: "ErrNoValue",
        -1: "ErrNoSignal",
        -2: "ErrBadSignal",
        -3: "ErrLowSignal",
        -4: "ErrBigSignal",
        -5: "ErrWimMissing",
        -6: "ErrNotAvailable",
        -7: "InfNothingChanged",
        -8: "ErrNoPulse",
        -10: "ErrChannelNotAvailable",
        -13: "ErrDiv0",
        -14: "ErrOutOfRange",
        -15: "ErrUnitNotAvailable"
    }

    def initialize(self, *args, **kwargs):
        wlmData.LoadDLL(self.lib_path)
        self.lib = wlmData.dll

    def get_raw_value(self) -> float:
        return self.lib.GetWavelength(0.0)

    def get_wavelength(self, err_nums: int = np.inf) -> float:
        err_count = 0
        raw_value = self.get_raw_value()
        while raw_value <= 0 and err_count < err_nums:
            err_count += 1
            print("Got Error:", self.ERR_DICT[raw_value])
            raw_value = self.get_raw_value()

        # if err_count == err_nums:
        #     return -1.0

        air_value = self.lib.ConvertUnit(raw_value, wlmConst.cReturnWavelengthVac, wlmConst.cReturnWavelengthAir)

        # return raw_value
        return air_value
