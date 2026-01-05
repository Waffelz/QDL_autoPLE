import time
from daq.drivers.powermeter_raw import PowerMeter

def main():
    pm = PowerMeter(channel="A")
    pm.open()
    try:
        for i in range(10):
            p_uW = pm.read_power_uW()
            print(f"{i}: {p_uW:.3f} uW")
            time.sleep(0.2)
    finally:
        pm.close()

if __name__ == "__main__":
    main()
