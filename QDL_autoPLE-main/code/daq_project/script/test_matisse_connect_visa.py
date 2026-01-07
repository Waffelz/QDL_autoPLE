#!/usr/bin/env python3
"""
Minimal VISA connection test for Matisse-like USB instruments.

Run:
  python script/test_matisse_connect_visa.py --resource "USB0::0x17E7::0x0102::07-40-01::INSTR"
"""

import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resource", required=True, help='VISA resource, e.g. USB0::...::INSTR')
    ap.add_argument("--backend", default=None, help='Optional: "@ni" or "@py"')
    ap.add_argument("--timeout_ms", type=int, default=2000)
    args = ap.parse_args()

    import pyvisa

    rm = pyvisa.ResourceManager(args.backend) if args.backend else pyvisa.ResourceManager()
    print("VISA resources visible:")
    for r in rm.list_resources():
        print("  ", r)

    res = rm.open_resource(args.resource)
    res.timeout = args.timeout_ms
    print("\nOpened:", args.resource)

    # Try a couple of common ID/diagnostic queries.
    for cmd in ["*IDN?", "PRN?", "SYST:ERR?"]:
        try:
            ans = res.query(cmd)
            print(f"{cmd} -> {ans!r}")
            break
        except Exception as e:
            print(f"{cmd} failed: {e}")

    res.close()
    rm.close()
    print("\nDONE")

if __name__ == "__main__":
    main()
