from src.pv_array import PVArray
import logging
import os
import matlab.engine

logging.basicConfig(level=logging.DEBUG)


def test_output():
    PV_PARAMS_PATH = os.path.join("parameters", "01_pvarray.json")
    PVARRAY_CKP_PATH = os.path.join("data", "01_pvarray_iv.json")
    engine = matlab.engine.connect_matlab()
    pv_array = PVArray.from_json(
        PV_PARAMS_PATH, engine=engine, ckp_path=PVARRAY_CKP_PATH, f_precision=3
    )
    result = pv_array.simulate(28.459, 1000, 25)
    assert abs(result.power - 178.575) < 1e-6
    assert abs(result.voltage - 28.459) < 1e-6
    assert abs(result.current - 6.275) < 1e-6


def test_real_mpp():
    PV_PARAMS_PATH = os.path.join("parameters", "01_pvarray.json")
    PVARRAY_CKP_PATH = os.path.join("data", "01_pvarray_iv.json")
    engine = matlab.engine.connect_matlab()
    pv_array = PVArray.from_json(
        PV_PARAMS_PATH, engine=engine, ckp_path=PVARRAY_CKP_PATH, f_precision=3
    )
    result = pv_array.get_true_mpp([1000], [25])
    assert abs(result.power - 199.481) < 1e-6
    assert abs(result.voltage - 26.308) < 1e-6
    assert abs(result.current - 7.583) < 1e-6
