#!/usr/bin/env python3
"""
Newport powermeter smoke test (via functions26 PowerMeter driver).

Run (from project root):
  python scripts/test_powermeter.py
  python scripts/test_powermeter.py --channel A --n 100 --dt 0.10
  python scripts/test_powermeter.py --channel AB --n 50 --dt 0.20
  python scripts/test_powermeter.py --addr "GPIB0::5::INSTR"

Notes:
- This uses your PowerMeter.convert_reading_string_to_float(), which (in your snippet)
  returns microW (µW). We also print W.
"""

import time
import argparse
import statistics
import sys
from pathlib import Path
import inspect
from typing import Optional

# --- Make sure the folder containing "functions26" is on sys.path ---
HERE = Path(__file__).resolve()
root = None
for p in [HERE.parent] + list(HERE.parents):
    if (p / "functions26").exists():
        root = p
        break
if root is None:
    raise RuntimeError("Could not find a 'functions26' folder in any parent directory.")
sys.path.insert(0, str(root))

from functions26.instruments.powermeter import PowerMeter


def make_powermeter(channel: str, addr: Optional[str]):
    """
    Be robust to different PowerMeter __init__ signatures in your repo.
    """
    sig = inspect.signature(PowerMeter.__init__)
    params = sig.parameters

    # If user gave an address but constructor doesn't accept it,
    # set the class attribute before instantiation.
    if addr is not None:
        if ("instrument_port" not in params) and ("powermeter_port" not in params) and (len(params) <= 2):
            PowerMeter.powermeter_name = addr

    # Try common constructor variants
    if "instrument_port" in params:
        return PowerMeter(channel=channel, instrument_port=addr)
    if "powermeter_port" in params:
        return PowerMeter(channel=channel, powermeter_port=addr)
    if len(params) > 2:
        # positional (channel, port) style
        return PowerMeter(channel, addr)
    return PowerMeter(channel)


def read_once(pm: PowerMeter):
    """
    Returns (vals_uW, vals_W) as lists.
    """
    reading_strings = pm.powermeter.get_instrument_reading_string_all()

    vals_uW = []
    for s in reading_strings:
        try:
            vals_uW.append(pm.convert_reading_string_to_float(s))  # µW (per your driver)
        except Exception:
            pass

    vals_W = [v * 1e-6 for v in vals_uW]
    return vals_uW, vals_W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="A", choices=["A", "B", "AB"], help="powermeter channel")
    ap.add_argument("--addr", default=None, help='VISA resource, e.g. "GPIB0::5::INSTR"')
    ap.add_argument("--n", type=int, default=100, help="number of reads")
    ap.add_argument("--dt", type=float, default=0.10, help="delay between reads (s)")
    ap.add_argument("--no-empty-buffer", action="store_true", help="skip buffer flushing")
    args = ap.parse_args()

    pm = make_powermeter(args.channel, args.addr)

    # Open VISA session
    pm.powermeter.initialize_instrument()

    # Clear any buffered responses (recommended)
    if (not args.no_empty_buffer) and hasattr(pm, "_empty_buffer"):
        try:
            pm._empty_buffer()
        except Exception:
            pass

    print("Power meter initialized. Sampling...")
    print(f"Channel={args.channel}  Addr={args.addr or getattr(pm, 'powermeter_name', 'UNKNOWN')}")

    all_means_uW = []
    t0 = time.monotonic()

    try:
        for i in range(args.n):
            t_read0 = time.monotonic()
            vals_uW, _vals_W = read_once(pm)
            t_read1 = time.monotonic()

            mean_uW = (sum(vals_uW) / len(vals_uW)) if vals_uW else None
            if mean_uW is not None:
                all_means_uW.append(mean_uW)

            if i < 5 or (i + 1) % 20 == 0:
                dt_read_ms = (t_read1 - t_read0) * 1e3
                print(
                    f"[{i+1:4d}/{args.n}] "
                    f"t={time.monotonic()-t0:7.3f}s  "
                    f"read={dt_read_ms:6.1f}ms  "
                    f"vals_uW={vals_uW if vals_uW else None}  "
                    f"mean_uW={mean_uW}"
                )

            time.sleep(args.dt)

    finally:
        try:
            pm.powermeter.terminate_instrument()
        except Exception:
            pass

    if not all_means_uW:
        print("\nNo valid readings were parsed.")
        print("If you saw VISA timeouts: check GPIB address, termination, or instrument mode.")
        return

    mean_uW = statistics.mean(all_means_uW)
    std_uW = statistics.pstdev(all_means_uW) if len(all_means_uW) > 1 else 0.0

    print("\n--- Power Meter Results ---")
    print(f"Valid reads: {len(all_means_uW)}/{args.n}")
    print(f"Mean:       {mean_uW:.6f} µW   ({mean_uW*1e-6:.6e} W)")
    print(f"Std:        {std_uW:.6f} µW   ({std_uW*1e-6:.6e} W)")
    print(f"Min:        {min(all_means_uW):.6f} µW")
    print(f"Max:        {max(all_means_uW):.6f} µW")
    print("--------------------------")


if __name__ == "__main__":
    main()
