import time
from daq.drivers.ws7_raw import WS7

def main():
    ws7 = WS7()
    for i in range(20):
        wl1 = ws7.get_wavelength_nm()
        wl2 = ws7.get_wavelength2_nm()
        print(f"{i:02d}: wl1={wl1:.6f} nm, wl2={wl2:.6f} nm")
        time.sleep(0.1)

if __name__ == "__main__":
    main()
