import matlab.engine

from src.matlab_api import compose_block_name, get_parameter, set_parameters


def test_compose_block_str():
    blocks = "hello"
    result = compose_block_name(blocks)
    expected = "hello"
    assert result == expected


def test_compose_block_one_list():
    blocks = ["hello"]
    result = compose_block_name(blocks)
    expected = "hello"
    assert result == expected


def test_compose_block_two_list():
    blocks = ["hello", "world"]
    result = compose_block_name(blocks)
    expected = "hello/world"
    assert result == expected


def test_compose_block_six_list():
    blocks = ["hello", "world"] * 3
    result = compose_block_name(blocks)
    expected = "hello/world/hello/world/hello/world"
    assert result == expected


def test_set_parameter_one_block_two_params(mocker):
    mocker.patch("matlab.engine")
    eng = matlab.engine.start_matlab()
    block = "hello"
    params = {"Np": "1", "Ns": "1"}
    set_parameters(eng, block, params)
    eng.set_param.assert_any_call("hello", "Np", "1", nargout=0)
    eng.set_param.assert_any_call("hello", "Ns", "1", nargout=0)


def test_set_parameter_two_block_two_params(mocker):
    mocker.patch("matlab.engine")
    eng = matlab.engine.start_matlab()
    block = ["hello", "world"]
    params = {"Np": "1", "Ns": "1"}
    set_parameters(eng, block, params)
    eng.set_param.assert_any_call("hello/world", "Np", "1", nargout=0)
    eng.set_param.assert_any_call("hello/world", "Ns", "1", nargout=0)


def test_set_parameter_two_block_one_list_param(mocker):
    mocker.patch("matlab.engine")
    eng = matlab.engine.start_matlab()
    block = ["hello", "world"]
    params = {"Np": ["1", "2"]}
    set_parameters(eng, block, params)
    eng.set_param.assert_called_once_with(
        "hello/world", "Np", "1", "Np", "2", nargout=0
    )


def test_get_parameter(mocker):
    mocker.patch("matlab.engine")
    eng = matlab.engine.start_matlab()
    block = ["hello", "world"]
    params = "Np"
    get_parameter(eng, block, params)
    eng.get_param.assert_called_once_with("hello/world", "Np")
