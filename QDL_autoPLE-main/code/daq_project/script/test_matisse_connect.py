#!/usr/bin/env python3
"""
Minimal Matisse connection test.

Run:
  python script/test_matisse_connect.py --port COM6
"""

import sys
import argparse
from pathlib import Path

def add_project_roots():
    HERE = Path(__file__).resolve()
    root = None
    for p in [HERE.parent] + list(HERE.parents):
        if (p / "matisse_controller").exists():
            root = p
            break
    if root is None:
        raise RuntimeError("Could not find 'matisse_controller' folder in parents. Run from repo root.")
    sys.path.insert(0, str(root))
    return root

def list_com_ports():
    try:
        import serial.tools.list_ports
        ports = [p.device for p in serial.tools.list_ports.comports()]
        return ports
    except Exception:
        return []

def main():
    add_project_roots()

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="Windows COM port, e.g. COM6")
    args = ap.parse_args()

    print("Available COM ports:", list_com_ports())

    # ✅ Correct import based on your discovery
    from matisse_controller.matisse import Matisse

    import inspect
    print("Matisse class:", Matisse)
    try:
        print("Matisse signature:", inspect.signature(Matisse))
    except Exception:
        pass

    # Try common constructor patterns (positional vs keyword)
    last_err = None
    matisse = None
    for ctor in [
        lambda: Matisse(args.port),
        lambda: Matisse(port=args.port),
        lambda: Matisse(com_port=args.port),
    ]:
        try:
            matisse = ctor()
            break
        except Exception as e:
            last_err = e

    if matisse is None:
        raise RuntimeError(f"Failed to construct Matisse on {args.port}. Last error: {last_err}")

    print("Connected Matisse object:", matisse)
    print("Has methods:", [m for m in ["query","write","start_scan","stop_scan","stabilize_off","stabilize_on"] if hasattr(matisse, m)])

    # Optional: try a harmless query if supported
    if hasattr(matisse, "query"):
        for cmd in ["PRN", "PRN?", "*IDN?", "SYST:ERR?"]:
            try:
                r = matisse.query(cmd, True) if matisse.query.__code__.co_argcount >= 3 else matisse.query(cmd)
                print(f"query({cmd!r}) -> {r!r}")
                break
            except Exception as e:
                print(f"query({cmd!r}) failed: {e}")

    print("DONE")

if __name__ == "__main__":
    main()
