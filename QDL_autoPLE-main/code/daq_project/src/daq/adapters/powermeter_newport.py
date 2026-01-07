
from __future__ import annotations
from dataclasses import dataclass
from daq.protocols import PowerMeter as PowerMeterProtocol
from daq.drivers.powermeter_raw import PowerMeter as NewportPowerMeterRaw

@dataclass
class NewportPowerMeterAdapter(PowerMeterProtocol):
    pm: NewportPowerMeterRaw

    def read_power_W(self) -> float:
        # raw method returns µW -> convert to W for storage consistency
        return self.pm.read_power_uW() * 1e-6

    def close(self) -> None:
        self.pm.close()
