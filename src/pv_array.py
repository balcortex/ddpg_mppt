import os
from collections import namedtuple
from functools import partial
from typing import Dict, List, Union

import numpy as np
import matlab.engine
from scipy.optimize import minimize
from tqdm import tqdm
from collections import defaultdict

from src import utils
from src.logger import logger
from src.matlab_api import set_parameters
from src.utils import read_weather_csv

PVSimResult = namedtuple("PVSimResult", ["power", "voltage", "current"])


class PVArray:
    def __init__(
        self,
        params: Dict,
        engine,
        ckp_path: str,
        f_precision: int = 1,
    ):
        """PV Array Model, interface between MATLAB and Python

        Params:
            model_params: dictionary with the parameters
            float_precision: decimal places used by the model (for cache)
        """
        logger.info("Starting MATLAB engine . . .")
        self._params = params
        self.float_precision = f_precision
        self._model_path = os.path.join("src", "matlab_model")
        self.ckp_path = ckp_path
        self._eng = engine

        self._init()
        self._init_history()

    def __repr__(self) -> str:
        return (
            f"PVArray {float(self.params['Im']) * float(self.params['Vm']):.0f} Watts"
        )

    def simulate(
        self, voltage: float, irradiance: float, cell_temp: float
    ) -> PVSimResult:
        """
        Simulate the simulink model

        Params:
            voltage: load voltage [V]
            irradiance: solar irradiance [W/m^2]
            temperature: cell temperature [celsius]
        """
        if isinstance(voltage, np.ndarray):
            voltage = voltage[0]

        v = round(voltage, self.float_precision)
        g = round(irradiance, self.float_precision)
        t = round(cell_temp, self.float_precision)

        key = f"{v},{g},{t}"
        if self.hist[key]:
            result = PVSimResult(*self.hist[key])
        else:
            result = self._simulate(v, g, t)
            self.hist[key] = result
            self._save_history(verbose=False)

        return result

    def get_true_mpp(
        self,
        irradiance: Union[float, List[float]],
        cell_temp: Union[float, List[float]],
        ftol: float = 1e-09,
        verbose: bool = False,
    ) -> PVSimResult:
        """Get the real MPP for the specified inputs

        Params:
            irradiance: solar irradiance [w/m^2]
            temperature: cell temperature [celsius]
            ftol: tolerance of the solver (optimizer)
            verbose: show a progress bar
        """
        if isinstance(irradiance, (int, float)):
            irradiance = [irradiance]
            cell_temp = [cell_temp]
        assert len(cell_temp) == len(
            irradiance
        ), "irradiance and cell_temp lists must be the same length"

        # logger.debug("Calculating true MPP . . .")
        pv_voltages, pv_powers, pv_currents = [], [], []
        float_precision = self.float_precision
        self.float_precision = 12

        for g, t in tqdm(
            list(zip(irradiance, cell_temp)),
            desc="Calculating true MPP",
            ascii=True,
            disable=not verbose,
        ):
            result = self._get_true_mpp(g, t, ftol)
            pv_voltages.append(round(result.voltage, float_precision))
            pv_powers.append(round(result.power, float_precision))
            pv_currents.append(round(result.current, float_precision))

        self.float_precision = float_precision

        if len(pv_powers) == 1:
            return PVSimResult(pv_powers[0], pv_voltages[0], pv_currents[0])
        return PVSimResult(pv_powers, pv_voltages, pv_currents)

    def get_po_mpp(
        self,
        irradiance: List[float],
        cell_temp: List[float],
        v0: float = 0.0,
        v_step: float = 0.1,
        verbose: bool = False,
    ) -> PVSimResult:
        """
        Perform the P&O MPPT technique

        Params:
            irradiance: solar irradiance [W/m^2]
            temperature: pv array temperature [celsius]
            v0: initial voltage of the load
            v_step: delta voltage for incrementing/decrementing the load voltage
            verbose: show a progress bar
        """
        assert len(cell_temp) == len(
            irradiance
        ), "irradiance and cell_temp lists must be the same length"

        logger.debug(f"Running P&O, step={v_step} volts . . .")
        pv_voltages, pv_powers, pv_currents = [v0, v0], [0], []

        for g, t in tqdm(
            list(zip(irradiance, cell_temp)),
            desc="Calculating PO",
            ascii=True,
            disable=not verbose,
        ):
            sim_result = self.simulate(pv_voltages[-1], g, t)
            delta_v = pv_voltages[-1] - pv_voltages[-2]
            delta_p = sim_result.power - pv_powers[-1]
            pv_powers.append(sim_result.power)
            pv_currents.append(sim_result.current)

            if delta_p == 0:
                pv_voltages.append(pv_voltages[-1])
            else:
                if delta_p > 0:
                    if delta_v >= 0:
                        pv_voltages.append(pv_voltages[-1] + v_step)
                    else:
                        pv_voltages.append(pv_voltages[-1] - v_step)
                else:
                    if delta_v >= 0:
                        pv_voltages.append(pv_voltages[-1] - v_step)
                    else:
                        pv_voltages.append(pv_voltages[-1] + v_step)

        return PVSimResult(pv_powers[1:], pv_voltages[1:-1], pv_currents)

    def _init(self) -> None:
        "Load the model and initialize it"
        self._eng.eval("beep off", nargout=0)
        self._eng.eval(f"cd '{os.getcwd()}'", nargout=0)
        self._eng.eval('model = "{}";'.format(self._model_path), nargout=0)
        self._eng.eval("load_system(model)", nargout=0)
        set_parameters(self._eng, self.model_name, {"StopTime": "1e-3"})
        set_parameters(self._eng, [self.model_name, "PV Array"], self.params)
        logger.info("Model loaded succesfully.")

    def _init_history(self) -> None:
        if os.path.exists(self.ckp_path):
            self.hist = defaultdict(lambda: None, utils.load_dict(self.ckp_path))
        else:
            self.hist = defaultdict(lambda: None)

    def _save_history(self, verbose: bool = True) -> None:
        utils.save_dict(self.hist, self.ckp_path, verbose=verbose)

    def _get_true_mpp(
        self,
        irradiance: float,
        cell_temp: float,
        ftol: float = 1e-12,
    ) -> PVSimResult:
        neg_power_fn = lambda v, g, t: self.simulate(v[0], g, t)[0] * -1
        min_fn = partial(neg_power_fn, g=irradiance, t=cell_temp)
        optim_result = minimize(
            min_fn, 0.8 * self.voc, method="SLSQP", options={"ftol": ftol}
        )
        assert optim_result.success == True
        v = optim_result.x[0]
        p = optim_result.fun * -1
        i = p / v

        return PVSimResult(p, v, i)

    def _set_cell_temp(self, cell_temp: float) -> None:
        "Auxiliar function for setting the cell temperature on the Simulink model"
        set_parameters(
            self._eng, [self.model_name, "Cell Temperature"], {"Value": str(cell_temp)}
        )

    def _set_irradiance(self, irradiance: float) -> None:
        "Auxiliar function for setting the irradiance on the Simulink model"
        set_parameters(
            self._eng, [self.model_name, "Irradiance"], {"Value": str(irradiance)}
        )

    def _set_voltage(self, voltage: float) -> None:
        "Auxiliar function for setting the load voltage source on the Simulink model"
        set_parameters(
            self._eng,
            [self.model_name, "Variable DC Source", "Load Voltage"],
            {"Value": str(voltage)},
        )

    def _simulate(
        self, voltage: float, irradiance: float, cell_temp: float
    ) -> PVSimResult:
        "Cached simulate function"
        self._set_voltage(voltage)
        self._set_irradiance(irradiance)
        self._set_cell_temp(cell_temp)
        self._start_simulation()

        pv_power = self._eng.eval("P(end);", nargout=1)
        pv_voltage = self._eng.eval("V(end);", nargout=1)
        pv_current = self._eng.eval("I(end);", nargout=1)

        return PVSimResult(
            round(pv_power, self.float_precision),
            round(pv_voltage, self.float_precision),
            round(pv_current, self.float_precision),
        )

    def _start_simulation(self) -> None:
        "Start the simulation command"
        set_parameters(self._eng, self.model_name, {"SimulationCommand": "start"})

    @staticmethod
    def mppt_eff(p_real: List[float], p: List[float]) -> float:
        return sum([p1 / p2 for p1, p2 in zip(p, p_real)]) * 100 / len(p_real)

    @property
    def voc(self) -> float:
        "Nominal open-circuit voltage of the pv array"
        return float(self.params["Voc"])

    @property
    def isc(self) -> float:
        "Nominal short-circuit current of the pv array"
        return float(self.params["Isc"])

    @property
    def pmax(self) -> float:
        "Nominal maximum power output of the pv array"
        return self.voc * self.isc

    @property
    def params(self) -> Dict:
        "Dictionary containing the parameters of the pv array"
        return self._params

    @property
    def model_name(self) -> str:
        "String containing the name of the model (for running in MATLAB)"
        return os.path.basename(self._model_path)

    @classmethod
    def from_json(cls, path: str, **kwargs):
        "Create a PV Array from a json file containing a string with the parameters"
        return cls(params=utils.load_dict(path), **kwargs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matlab.engine

    pv_params_path = os.path.join("parameters", "01_pvarray.json")
    pvarray_ckp_path = os.path.join("data", "01_pvarray_iv.json")

    engine = matlab.engine.connect_matlab()
    pvarray = PVArray.from_json(
        path=pv_params_path,
        ckp_path=pvarray_ckp_path,
        engine=engine,
    )

    p, v, i = pvarray.get_true_mpp(1000, 25)
    print(f"{p=}")
    print(f"{v=}")
    print(f"{i=}")
