# Experimental Control Framework for ZnO Spectroscopy

This repository contains the experimental control software developed for high-resolution photoluminescence excitation (PLE) and coherent population trapping (CPT) measurements in donor-bound exciton systems in ZnO.

The framework provides a modular and extensible platform for coordinating multiple instruments, enabling precise wavelength control, synchronized data acquisition, and efficient scan strategies for resolving narrow spectral features.

---

## Overview

Modern quantum optical experiments require tight integration between lasers, wavelength monitoring, photon detection, and data acquisition systems. This repository implements a flexible control architecture designed to:

- Coordinate multi-instrument experiments
- Perform wavelength scans with variable speed and adaptive sampling
- Support multiple acquisition modes (photon counting and spectroscopy)
- Record synchronized data streams with metadata for reproducibility

The system has been used for:
- High-resolution PLE spectroscopy
- Coherent population trapping (CPT) measurements at high magnetic fields
- Time-resolved photon counting experiments

---

## Architecture

The control system follows a layered design:

- **Scan Engine**  
  Central orchestration of experiments (`run_scan.py`), handling scan logic, timing, and synchronization.

- **Instrument Drivers**  
  Modular interfaces for individual hardware components:
  - Tunable laser (Ti:Sapphire / Matisse)
  - Wavemeter (e.g., HighFinesse WS7)
  - Data acquisition (NI DAQ / power monitoring)
  - Photon counting (Swabian TimeTagger)
  - Spectrometer (Andor + Shamrock)

- **Configuration Layer**  
  YAML-based configuration files defining scan parameters, enabling rapid reconfiguration without modifying code.

- **Data Pipeline**  
  Structured output (HDF5 / NPZ + metadata) for reproducibility and downstream analysis.

---

## Acquisition Modes

The framework supports two primary measurement modes:

### 1. TimeTagger Mode (Photon Counting)
- Records photon arrival events
- Integrates counts within defined time windows
- Used for PLE and CPT measurements collected through side excitation 

### 2. Andor Mode (Spectroscopy)
- Acquires full spectra at each wavelength step
- Enables simultaneous monitoring of multiple emission features

Both modes share the same scan interface, allowing interchangeable use within the same experimental workflow.

---

## Variable-Speed Scanning

A key feature of the framework is **variable-speed wavelength scanning**, designed to efficiently resolve narrow spectral features.

- **Coarse scans** rapidly identify resonance regions
- **Fine scans** provide high-resolution sampling near features of interest
- **Piecewise scans** allow different speeds and dwell times within a single scan

This approach is particularly important for CPT experiments, where:
- Resonances can be narrow (MHz–GHz scale)
- Measurement time is limited (multi-hour datasets)
- Laser drift and system fluctuations must be tracked over repeated scans

---

## Installation

### Requirements
- Python 3.8+
- NumPy, SciPy, Pandas, Matplotlib
- h5py
- PyVISA (for instrument communication)
- Vendor-specific SDKs:
  - TimeTagger API (Swabian Instruments)
  - Andor SDK
  - Wavemeter API

### Setup
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt

---
### Usage
Basic Scan Execution
python run_scan.py --config config/run_scan.yml

Configuration Example
scan:
  start_wavelength: 368.500
  stop_wavelength: 368.505
  scan_speed: 0.000025  # nm/s

acquisition:
  mode: timetagger
  integration_time: 0.02  # seconds

output:
  save_path: ./data/
