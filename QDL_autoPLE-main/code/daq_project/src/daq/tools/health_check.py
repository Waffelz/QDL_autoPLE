def health_check(laser, wavemeter, counts_det, spec_det):
    print("Wavemeter:", wavemeter.get_wavelength_nm())

    # optional: try a short lock
    # laser.lock_to_wavelength_nm(wavemeter.get_wavelength_nm())

    c = counts_det.acquire_counts(0.05)
    print("Counts(50ms):", c)

    try:
        spec_det.set_exposure(0.05)
        axis, y = spec_det.acquire_spectrum()
        print("Spectrum:", y.shape, "axis:", axis.shape)
    except Exception as e:
        print("Spectrum check failed:", e)
