from __future__ import annotations
import json
import numpy as np
import h5py

from daq.core.scan_wavelength import ScanCountsResult, ScanSpectrumResult


def save_counts_h5(path: str, r: ScanCountsResult) -> None:
    wl = np.array([p.wl_nm for p in r.points], dtype=float)
    wl_std = np.array([p.wl_std_nm for p in r.points], dtype=float)
    n_wl = np.array([p.n_wl_samples for p in r.points], dtype=int)
    t0 = np.array([p.t0 for p in r.points], dtype=float)
    t1 = np.array([p.t1 for p in r.points], dtype=float)
    counts = np.array([p.counts for p in r.points], dtype=int)

    with h5py.File(path, "w") as f:
        f.attrs["meta_json"] = json.dumps(r.meta)
        f.create_dataset("wl_nm", data=wl)
        f.create_dataset("wl_std_nm", data=wl_std)
        f.create_dataset("n_wl_samples", data=n_wl)
        f.create_dataset("t0_monotonic_s", data=t0)
        f.create_dataset("t1_monotonic_s", data=t1)
        f.create_dataset("counts", data=counts)

        if any(p.power_W is not None for p in r.points):
            pwr = np.array([np.nan if p.power_W is None else p.power_W for p in r.points], dtype=float)
            f.create_dataset("power_W", data=pwr)


def save_spectrum_h5(path: str, r: ScanSpectrumResult) -> None:
    wl = np.array([p.wl_nm for p in r.points], dtype=float)
    wl_std = np.array([p.wl_std_nm for p in r.points], dtype=float)
    n_wl = np.array([p.n_wl_samples for p in r.points], dtype=int)
    t0 = np.array([p.t0 for p in r.points], dtype=float)
    t1 = np.array([p.t1 for p in r.points], dtype=float)
    intensity = np.stack([p.intensity for p in r.points], axis=0) if r.points else np.zeros((0, r.axis.size))

    with h5py.File(path, "w") as f:
        f.attrs["meta_json"] = json.dumps(r.meta)
        f.create_dataset("wl_nm", data=wl)
        f.create_dataset("wl_std_nm", data=wl_std)
        f.create_dataset("n_wl_samples", data=n_wl)
        f.create_dataset("t0_monotonic_s", data=t0)
        f.create_dataset("t1_monotonic_s", data=t1)
        f.create_dataset("axis", data=r.axis)
        f.create_dataset("intensity", data=intensity)

        if any(p.power_W is not None for p in r.points):
            pwr = np.array([np.nan if p.power_W is None else p.power_W for p in r.points], dtype=float)
            f.create_dataset("power_W", data=pwr)
