#!/usr/bin/env python3
import time
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root (one level above scripts/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from matisse_controller.shamrock_ple.ccd import CCD

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp", type=float, default=-65.0, help="Target temperature (°C)")
    ap.add_argument("--tol", type=float, default=1.0, help="Tolerance (°C)")
    ap.add_argument("--poll", type=float, default=5.0, help="Polling period (s)")
    ap.add_argument("--timeout", type=float, default=1200.0, help="Timeout (s)")
    args = ap.parse_args()

    print("NOTE: Close Andor Solis / any Andor GUI before running this test.")
    ccd = CCD(initialize_on_definition=True)

    print(f"Setting CCD temperature setpoint to {args.temp} °C and enabling cooler...")
    ccd.set_temperature(args.temp)
    ccd.lib.CoolerON()

    t0 = time.time()
    temps = []

    while True:
        code, tempC = ccd.get_temperature_status()
        temps.append((time.time() - t0, tempC, code))
        print(f"t={temps[-1][0]:7.1f}s  T={tempC:6.1f} °C  status={code}")

        if tempC <= args.temp + args.tol:
            print(f"✅ Reached {args.temp}±{args.tol} °C (current {tempC:.1f} °C)")
            break

        if (time.time() - t0) > args.timeout:
            raise TimeoutError(f"Did not reach {args.temp}±{args.tol} °C within {args.timeout}s")

        time.sleep(args.poll)

    # Optional: keep it cold for a moment to see stability
    print("Holding for 30s to check stability...")
    for _ in range(int(30 / max(args.poll, 1))):
        code, tempC = ccd.get_temperature_status()
        print(f"hold: T={tempC:.1f} °C  status={code}")
        time.sleep(args.poll)

    print("Done. (Not shutting down cooler here; close if you want to warm up.)")

if __name__ == "__main__":
    main()
