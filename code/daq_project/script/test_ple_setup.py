# scripts/test_ple_setup.py
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root = parent of scripts/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import matisse_controller.shamrock_ple.ple as ple_mod

def main():
    ple = ple_mod.PLE(powermeter_port=None, matisse_wavemeter_port=None)

    print("Setting up WS7...")
    ple.setup_ws7()
    print("Setting up powermeter A...")
    ple.setup_powermeter("A")
    print("Setting up Matisse (WS7)...")
    ple.setup_matisse("WS7", scanning_speed=None)
    ple._setup_wavelength_tolerance("WS7")

    ws7 = ple_mod.ws7
    pm = ple_mod.powermeter
    matisse = ple_mod.matisse

    print("WS7 wl:", ws7.lib.GetWavelength(0.0))
    pm.powermeter.initialize_instrument()
    pm._empty_buffer()
    print("Power strings:", pm.powermeter.get_instrument_reading_string_all())
    pm.powermeter.terminate_instrument()

    # Quick Matisse query sanity check
    try:
        print("Matisse rising speed:", matisse.query("SCAN:RISINGSPEED?", True))
    except Exception as e:
        print("Matisse query failed:", e)

    ple.clean_up_globals()
    print("DONE")

if __name__ == "__main__":
    main()
