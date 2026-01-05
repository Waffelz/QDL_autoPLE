import numpy as np
import threading
import time
import queue

import wlmData
import wlmConst
from functions26.filing.QDLFiling import QDLFDataManager
from functions26.InstrumentHandler import GPIBInstrument


class WS7:
    lib_path = 'wlmData.dll'  # Create a class attribute called lib_path that has the path to a dll file

    def __init__(self):
        wlmData.LoadDLL(self.lib_path) # Use the LoadDLL method in the imported module to load the dll file
        self.lib = wlmData.dll
        self.pipeline = queue.Queue()
        self.thread = None

    def start_acquisition(self, start_time: float, start_event: threading.Event, stop_event: threading.Event,
                          sleep_time: float = 0.05):

        self.thread = threading.Thread(target=self.acquisition, args=(start_time, sleep_time, self.pipeline,
                                                                      start_event, stop_event), daemon=True)
        self.thread.start()

        return True

    def stop_and_save_acquisition(self, filename):
        data_manager = self.stop_acquisition()
        self.save_acquisition(filename, data_manager)

    def stop_acquisition(self):
        data_manager = QDLFDataManager()  # Empty manager
        if self.thread is not None:
            self.thread.join()
            data_manager: QDLFDataManager = self.pipeline.get()
            del self.thread
            self.thread = None

        return data_manager

    @staticmethod
    def save_acquisition(filename, data_manager: QDLFDataManager):
        data_manager.save(filename)

    def acquisition(self, start_time: float, sleep_time: float, pipeline: queue.Queue, start_event: threading.Event,
                    stop_event: threading.Event):
        time_array = np.array([], dtype=float)
        wavelength_list = []
        wavelength2_list=[]
        while not start_event.is_set():
            time.sleep(sleep_time)

        while not stop_event.is_set():
            new_time = time.time()
            try:
            #reading from switch channel 1 and reading 2 from switch channel 2
                reading = self.lib.GetWavelength(0.0)
                reading2 = self.lib.GetWavelength2(0.0)
                # reading = self.lib.ConvertUnit(reading, wlmConst.cReturnWavelengthVac, wlmConst.cReturnWavelengthAir)
                if reading > 0:
                    wavelength_list.append(reading)
                    wavelength2_list.append(reading2)
                    time_array = np.append(time_array, new_time)
            except Exception as e:
                pass
            time.sleep(sleep_time)

        data = {'x1': time_array - start_time, 'y1': np.array(wavelength_list), 'secondlaser_wl': np.array(wavelength2_list)}

        data_manager = QDLFDataManager(data, parameters={'start_time': start_time, 'sleep_time': sleep_time},
                                       datatype='wavelength')
        pipeline.put(data_manager)