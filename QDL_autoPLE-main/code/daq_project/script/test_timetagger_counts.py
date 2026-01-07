#!/usr/bin/env python3
"""
Swabian Time Tagger Ultra: simple integrated counts smoke test.

Run:
  python scripts/test_timetagger_counts.py --ch 1 --trig -0.08 --edge falling --exp 0.1 --n 50

Tip:
- Block/unblock the light and verify counts drop/rise.
"""

import time
import argparse
import statistics
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Try Swabian namespace then legacy
try:
    from Swabian import TimeTagger
except ImportError:
    import TimeTagger


def _edge_channel(phys_ch: int, edge: str) -> int:
    edge = edge.strip().lower()
    if edge == "rising":
        return int(abs(phys_ch))
    if edge == "falling":
        return -int(abs(phys_ch))
    raise ValueError("edge must be 'rising' or 'falling'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default=None, help="optional serial string if multiple taggers connected")
    ap.add_argument("--ch", type=int, default=1, help="physical input channel number (e.g. 1)")
    ap.add_argument("--trig", type=float, default=-0.08, help="trigger level in volts")
    ap.add_argument("--edge", default="falling", choices=["rising", "falling"], help="edge polarity")
    ap.add_argument("--exp", type=float, default=0.1, help="integration time per point (seconds)")
    ap.add_argument("--n", type=int, default=50, help="number of points")
    ap.add_argument("--dt", type=float, default=0.0, help="extra delay between points (seconds)")
    args = ap.parse_args()

    phys_ch = int(args.ch)
    sw_ch = _edge_channel(phys_ch, args.edge)
    exp_s = float(args.exp)
    if exp_s <= 0:
        raise ValueError("--exp must be > 0")

    # Connect
    tagger = TimeTagger.createTimeTagger(args.serial) if args.serial else TimeTagger.createTimeTagger()
    try:
        tagger.setTriggerLevel(phys_ch, float(args.trig))
        print("TimeTagger connected.")
        print(f"Channel phys={phys_ch} (sw={sw_ch}), trig={args.trig} V, exp={exp_s} s")

        counts_list = []
        t0 = time.time()

        for i in range(args.n):
            binwidth_ps = int(round(exp_s * 1e12))
            meas = TimeTagger.Counter(tagger, [sw_ch], binwidth=binwidth_ps, n_values=1)
            meas.startFor(binwidth_ps, clear=True)
            meas.waitUntilFinished()

            # API compat
            try:
                data = meas.getData()
            except TypeError:
                data = meas.getData(rolling=True)

            arr = np.asarray(data, dtype=np.int64).reshape(-1)
            c = int(arr[-1]) if arr.size else 0
            counts_list.append(c)

            if i < 5 or (i + 1) % 10 == 0:
                print(f"[{i+1:3d}/{args.n}] t={time.time()-t0:6.2f}s  counts={c}")

            if args.dt > 0:
                time.sleep(args.dt)

        mean_c = statistics.mean(counts_list)
        std_c = statistics.pstdev(counts_list) if len(counts_list) > 1 else 0.0

        print("\n--- TimeTagger Counts ---")
        print(f"Valid points: {len(counts_list)}/{args.n}")
        print(f"Mean counts:  {mean_c:.3f} per {exp_s}s")
        print(f"Std counts:   {std_c:.3f}")
        print(f"Min/Max:      {min(counts_list)} / {max(counts_list)}")
        print("------------------------")

    finally:
        TimeTagger.freeTimeTagger(tagger)


if __name__ == "__main__":
    main()
