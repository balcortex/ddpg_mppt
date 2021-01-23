import os
from collections import namedtuple
from functools import partial
from typing import Dict, List, Union, Optional

import numpy as np
import matlab.engine
from scipy.optimize import minimize
from tqdm import tqdm
from collections import defaultdict

import time

from src import utils
from src.logger import logger
from src.matlab_api import set_parameters, get_parameter
from src.utils import read_weather_csv

SIM_TIMES = {
    "pv_boost_rload": "0.5",
    "pv_boost_avg_rload": "1e-3",
}

PVSimResult = namedtuple(
    "PVSimResult",
    [
        "pv_power",
        "pv_voltage",
        "pv_current",
        "duty_cycle",
        "irradiance",
        "ambient_temp",
        "cell_temp",
    ],
)


class PVArray:
    def __init__(
        self,
        engine,
        model_name: str = "pv_boost_rload",
        float_precision: int = 3,
        save_every_seconds: int = 300,
        params: Optional[Dict] = None,
    ):
        """PV Array Model, interface between MATLAB and Python"""
        self.eng = engine
        self.model_name = model_name
        self.float_precision = float_precision
        self.save_every_seconds = save_every_seconds
        self.params = params

        self.model_path = os.path.join("src", "matlab", model_name)
        self.sim_time = SIM_TIMES[model_name]
        self.ckp_path = os.path.join("data", model_name + "_cache.json")
        self.ckp_mpp_path = os.path.join("data", model_name + "_mpp_cache.json")
        self.time_elapsed = time.time()

        self._init()
        self._init_history()

    def __repr__(self) -> str:
        return (
            f"PVArray {float(self.params['Im']) * float(self.params['Vm']):.0f} Watts "
            + "dc-dc converter"
        )

    def simulate(
        self,
        duty_cycle: float,
        irradiance: float,
        ambient_temp: float,
        save: bool = True,
    ) -> PVSimResult:
        """
        Simulate the simulink model

        Params:
            duty_cycle: dc-dc converter duty cycle (0.0 - 1.0)
            irradiance: solar irradiance [W/m^2]
            temperature: cell temperature [celsius]
            save: whether to save the result simulation in the file
        """
        # get_true_mpp passes a np.ndarray for the scipy optimization
        if isinstance(duty_cycle, np.ndarray):
            duty_cycle = duty_cycle[0]

        dc = round(duty_cycle, self.float_precision)
        g = int(irradiance)
        t = int(ambient_temp)

        key = f"{dc},{g},{t}"
        if self.hist[key]:
            result = PVSimResult(*self.hist[key])
        else:
            result = self._simulate(dc, g, t)

            if save:
                self.hist[key] = result
                self._check_save()

        return result

    def get_true_mpp(
        self,
        irradiance: Union[float, List[float]],
        ambient_temp: Union[float, List[float]],
        ftol: float = 1e-9,
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
            ambient_temp = [ambient_temp]
        assert len(ambient_temp) == len(
            irradiance
        ), "irradiance and ambient_temp lists must be the same length"

        pv_voltages, pv_powers, pv_currents, duty_cycles = [], [], [], []
        irradiances, ambient_temps, cell_temps = [], [], []

        float_precision = self.float_precision
        self.float_precision = 4

        for g, t in tqdm(
            list(zip(irradiance, ambient_temp)),
            desc="Calculating true MPP",
            ascii=True,
            disable=not verbose,
        ):
            g = round(g, float_precision)
            t = round(t, float_precision)

            key = f"{g},{t}"
            if self.hist_mpp[key]:
                result = PVSimResult(*self.hist_mpp[key])
            else:
                result = self._get_true_mpp(g, t, ftol)
                self.hist_mpp[key] = result
                self._check_save()
                # self._save_history(verbose=False, mpp=True)

            pv_voltages.append(round(result.pv_voltage, float_precision))
            pv_powers.append(round(result.pv_power, float_precision))
            pv_currents.append(round(result.pv_current, float_precision))
            duty_cycles.append(round(result.duty_cycle, float_precision))
            irradiances.append(round(result.irradiance, float_precision))
            ambient_temps.append(round(result.ambient_temp, float_precision))
            cell_temps.append(round(result.cell_temp, float_precision))

        self.float_precision = float_precision

        if len(pv_powers) == 1:
            return PVSimResult(
                pv_powers[0],
                pv_voltages[0],
                pv_currents[0],
                duty_cycles[0],
                irradiances[0],
                ambient_temps[0],
                cell_temps[0],
            )

        return PVSimResult(
            pv_powers,
            pv_voltages,
            pv_currents,
            duty_cycles,
            irradiances,
            ambient_temps,
            cell_temps,
        )

    def _init(self) -> None:
        "Load the model and initialize it"
        self.eng.eval("beep off", nargout=0)
        self.eng.eval(f"cd '{os.getcwd()}'", nargout=0)
        self.eng.eval('model = "{}";'.format(self.model_path), nargout=0)
        self.eng.eval("load_system(model)", nargout=0)
        set_parameters(self.eng, self.model_name, {"StopTime": self.sim_time})
        if self.params:
            logger.info("Changing parameters of the PV Array")
            set_parameters(self.eng, [self.model_name, "PV Array"], self.params)
        # logger.info("Model loaded succesfully.")

    def _init_history(self) -> None:
        if os.path.exists(self.ckp_path):
            self.hist = defaultdict(lambda: None, utils.load_dict(self.ckp_path))
        else:
            logger.info(f"Creating new dictionary at {self.ckp_path}")
            self.hist = defaultdict(lambda: None)

        if os.path.exists(self.ckp_mpp_path):
            self.hist_mpp = defaultdict(
                lambda: None, utils.load_dict(self.ckp_mpp_path)
            )
        else:
            logger.info(f"Creating new dictionary at {self.ckp_mpp_path}")
            self.hist_mpp = defaultdict(lambda: None)

    def _check_save(self) -> None:
        if time.time() - self.time_elapsed >= self.save_every_seconds:
            self.save()
            self.time_elapsed = time.time()

    def save(self, verbose: bool = True) -> None:
        utils.save_dict(self.hist_mpp, self.ckp_mpp_path, verbose=verbose)
        utils.save_dict(self.hist, self.ckp_path, verbose=verbose)

    def _get_true_mpp(
        self,
        irradiance: float,
        ambient_temp: float,
        ftol: float,
    ) -> PVSimResult:
        neg_power_fn = lambda dc, g, t: self.simulate(dc[0], g, t, save=False)[0] * -1
        min_fn = partial(neg_power_fn, g=irradiance, t=ambient_temp)
        optim_result = minimize(
            # min_fn, 1, method="SLSQP", bounds=((0.0, 1.0),), options={"ftol": ftol}
            min_fn,
            0,
            method="Powell",
            bounds=((0.0, 1.0),),
            options={"ftol": ftol},
        )
        assert optim_result.success == True
        dc = optim_result.x[0]

        return self.simulate(dc, irradiance, ambient_temp, save=False)

    def _set_cell_temp(self, cell_temp: float) -> None:
        "Auxiliar function for setting the cell temperature on the Simulink model"
        set_parameters(
            self.eng, [self.model_name, "Cell Temperature"], {"Value": str(cell_temp)}
        )

    def _set_irradiance(self, irradiance: float) -> None:
        "Auxiliar function for setting the irradiance on the Simulink model"
        set_parameters(
            self.eng, [self.model_name, "Irradiance"], {"Value": str(irradiance)}
        )

    def _set_duty_cycle(self, duty_cycle: float) -> None:
        "Auxiliar function for setting the dc-dc converter duty cycle on the Simulink model"
        set_parameters(
            self.eng, [self.model_name, "Duty Cycle"], {"Value": str(duty_cycle)}
        )

    def _simulate(
        self, duty_cycle: float, irradiance: float, ambient_temp: float
    ) -> PVSimResult:
        "Cached simulate function"
        cell_temp = round(
            self.cell_temp_from_ambient(irradiance, ambient_temp), self.float_precision
        )
        self._set_duty_cycle(duty_cycle)
        self._set_irradiance(irradiance)
        self._set_cell_temp(cell_temp)
        self._start_simulation()

        pv_voltage = self.eng.eval("V_PV(end);", nargout=1)
        pv_current = self.eng.eval("I_PV(end);", nargout=1)
        pv_power = pv_voltage * pv_current

        return PVSimResult(
            round(pv_power, self.float_precision),
            round(pv_voltage, self.float_precision),
            round(pv_current, self.float_precision),
            round(duty_cycle, self.float_precision),
            round(irradiance, self.float_precision),
            round(ambient_temp, self.float_precision),
            round(cell_temp, self.float_precision),
        )

    def _start_simulation(self) -> None:
        "Start the simulation command"
        set_parameters(self.eng, self.model_name, {"SimulationCommand": "start"})
        while True:
            status = get_parameter(self.eng, self.model_name, "SimulationStatus")
            if status in ["stopped", "compiled"]:
                break

    @staticmethod
    def mppt_eff(p_real: List[float], p: List[float]) -> float:
        assert len(p_real) == len(p)
        return (
            sum([(p1 * 60) / (p2 * 60) for p1, p2 in zip(p, p_real)])
            * 100
            / len(p_real)
        )

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

    # @property
    # def params(self) -> Dict:
    #     "Dictionary containing the parameters of the pv array"
    #     return self._params

    # @property
    # def model_name(self) -> str:
    #     "String containing the name of the model (for running in MATLAB)"
    #     return os.path.basename(self._model_path)

    @classmethod
    def from_json(cls, path: str, **kwargs):
        "Create a PV Array from a json file containing a string with the parameters"
        return cls(params=utils.load_dict(path), **kwargs)

    @staticmethod
    def cell_temp_from_ambient(
        irradiance: float, amb_temp: float, noct: float = 45.0, g_ref: float = 800.0
    ) -> float:
        "Estimate cell temperature from ambient temperature and irradiance"
        return amb_temp + (noct - 20) * (irradiance / g_ref)


if __name__ == "__main__":
    import matlab.engine

    pv_params_path = os.path.join("parameters", "01_pvarray.json")

    try:
        engine.exit()
    except NameError:
        pass
    engine = matlab.engine.connect_matlab()

    pvarray = PVArray(
        engine=engine,
        model_name="pv_boost_avg_rload",
    )

    dc, g, t = 0.1, 1000, 25
    result = pvarray.simulate(dc, g, t)
    print(result)

    dc, g, t = 0.1, 1000, 25.5
    result = pvarray.simulate(dc, g, t)
    print(result)

    result = pvarray.get_true_mpp(g, t)
    print(result)

    g, t = 1000, -6.25
    result = pvarray.get_true_mpp(g, t)
    print(result)

    pvarray.save()