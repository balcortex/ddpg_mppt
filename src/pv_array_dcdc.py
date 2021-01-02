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
        params: Dict,
        engine,
        ckp_path: str,
        f_precision: int = 3,
    ):
        """PV Array Model, interface between MATLAB and Python

        Params:
            model_params: dictionary with the PV Array parameters
            engine: MATLAB engine
            ckp_path: path of the json file to store simulation results
            float_precision: decimal places used by the model
        """
        self._params = params
        self.float_precision = f_precision
        self._model_path = os.path.join("src", "matlab_model_converter")
        self.ckp_path = ckp_path
        self._eng = engine

        self.save_state_counter = 0

        self._init()
        self._init_history()

    def __repr__(self) -> str:
        return (
            f"PVArray {float(self.params['Im']) * float(self.params['Vm']):.0f} Watts"
            + "dc-dc converter"
        )

    def simulate(
        self,
        duty_cycle: float,
        irradiance: float,
        ambient_temp: float,
        save_every: int = 10,
    ) -> PVSimResult:
        """
        Simulate the simulink model

        Params:
            duty_cycle: dc-dc converter duty cycle (0.0 - 1.0)
            irradiance: solar irradiance [W/m^2]
            temperature: cell temperature [celsius]
            save_every: append simulation results to file every 'save_every' new results
        """
        if isinstance(duty_cycle, np.ndarray):
            duty_cycle = duty_cycle[0]

        dc = round(duty_cycle, self.float_precision)
        g = round(irradiance, self.float_precision)
        t = round(ambient_temp, self.float_precision)

        key = f"{dc},{g},{t}"
        if self.hist[key]:
            result = PVSimResult(*self.hist[key])
        else:
            self.save_state_counter += 1
            result = self._simulate(dc, g, t)
            self.hist[key] = result

            if self.save_state_counter % save_every == 0:
                self.save_state_counter = 0
                self._save_history(verbose=False)

        return result

    def get_true_mpp(
        self,
        irradiance: Union[float, List[float]],
        ambient_temp: Union[float, List[float]],
        ftol: float = 1e-12,
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
        self.float_precision = 12

        for g, t in tqdm(
            list(zip(irradiance, ambient_temp)),
            desc="Calculating true MPP",
            ascii=True,
            disable=not verbose,
        ):
            result = self._get_true_mpp(g, t, ftol)
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

    # def get_po_mpp(
    #     self,
    #     irradiance: List[float],
    #     cell_temp: List[float],
    #     dc0: float = 0.0,
    #     dc_step: float = 0.01,
    #     verbose: bool = False,
    # ) -> PVSimResult:
    #     """
    #     Perform the P&O MPPT technique

    #     Params:
    #         irradiance: solar irradiance [W/m^2]
    #         temperature: pv array temperature [celsius]
    #         v0: initial voltage of the load
    #         v_step: delta voltage for incrementing/decrementing the load voltage
    #         verbose: show a progress bar
    #     """
    #     assert len(cell_temp) == len(
    #         irradiance
    #     ), "irradiance and cell_temp lists must be the same length"

    #     logger.debug(f"Running P&O, step={v_step} volts . . .")
    #     pv_voltages, pv_powers, pv_currents = [v0, v0], [0], []

    #     for g, t in tqdm(
    #         list(zip(irradiance, cell_temp)),
    #         desc="Calculating PO",
    #         ascii=True,
    #         disable=not verbose,
    #     ):
    #         sim_result = self.simulate(pv_voltages[-1], g, t)
    #         delta_v = pv_voltages[-1] - pv_voltages[-2]
    #         delta_p = sim_result.power - pv_powers[-1]
    #         pv_powers.append(sim_result.power)
    #         pv_currents.append(sim_result.current)

    #         if delta_p == 0:
    #             pv_voltages.append(pv_voltages[-1])
    #         else:
    #             if delta_p > 0:
    #                 if delta_v >= 0:
    #                     pv_voltages.append(pv_voltages[-1] + v_step)
    #                 else:
    #                     pv_voltages.append(pv_voltages[-1] - v_step)
    #             else:
    #                 if delta_v >= 0:
    #                     pv_voltages.append(pv_voltages[-1] - v_step)
    #                 else:
    #                     pv_voltages.append(pv_voltages[-1] + v_step)

    #     return PVSimResult(pv_powers[1:], pv_voltages[1:-1], pv_currents)

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
        ambient_temp: float,
        ftol: float,
    ) -> PVSimResult:
        neg_power_fn = lambda dc, g, t: self.simulate(dc[0], g, t)[0] * -1
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

        return self.simulate(dc, irradiance, ambient_temp)

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

    def _set_duty_cycle(self, duty_cycle: float) -> None:
        "Auxiliar function for setting the dc-dc converter duty cycle on the Simulink model"
        set_parameters(
            self._eng, [self.model_name, "Duty Cycle"], {"Value": str(duty_cycle)}
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

        pv_voltage = self._eng.eval("V_PV(end);", nargout=1)
        pv_current = self._eng.eval("I_PV(end);", nargout=1)
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
        set_parameters(self._eng, self.model_name, {"SimulationCommand": "start"})

    @staticmethod
    def mppt_eff(p_real: List[float], p: List[float]) -> float:
        assert len(p_real) == len(p)
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

    @staticmethod
    def cell_temp_from_ambient(
        irradiance: float, amb_temp: float, noct: float = 45.0, g_ref: float = 800.0
    ) -> float:
        "Estimate cell temperature from ambient temperature and irradiance"
        return amb_temp + (noct - 20) * (irradiance / g_ref)


if __name__ == "__main__":
    import matlab.engine

    pv_params_path = os.path.join("parameters", "01_pvarray.json")
    pvarray_ckp_path = os.path.join("data", "02_pvarray_dcdc.json")

    engine = matlab.engine.connect_matlab()
    pvarray = PVArray.from_json(
        path=pv_params_path,
        ckp_path=pvarray_ckp_path,
        engine=engine,
        f_precision=3,
    )

    for g in [100, 400, 1000]:
        for amb_t in [25, 35, 45]:
            p, v, i, dc, g_, amb_t_, cell_t = pvarray.get_true_mpp(g, amb_t)
            print(f"p={p}, v={v}, dc={dc}, g={g_}, amb_t={amb_t_}, cell_t={cell_t}")

    g, amb_t = 1000, -6.25
    p, v, i, dc, g_, amb_t_, cell_t = pvarray.get_true_mpp(g, amb_t)
    print(f"p={p}, v={v}, dc={dc}, g={g_}, amb_t={amb_t_}, cell_t={cell_t}")