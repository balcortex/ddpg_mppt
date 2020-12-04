from typing import Sequence, Union, Any, Dict


def compose_block_name(blocks: Union[str, Sequence[str]]):
    "Return a string using '/' as a separator if a list is passed"
    if isinstance(blocks, list):
        return "/".join(blocks)
    return blocks


def set_parameters(
    eng: Any,
    blocks: Union[str, Sequence[str]],
    params: Dict[str, Any],
    nargout=0,
):
    "Helper function to call eng.set_param() from the matlab.engine module"
    block_name = compose_block_name(blocks)
    for key, value in params.items():
        if isinstance(value, list):
            param_list = []
            for val in value:
                param_list.append(key)
                param_list.append(val)
            eng.set_param(block_name, *param_list, nargout=nargout)
        else:
            eng.set_param(block_name, key, value, nargout=nargout)


def get_parameter(eng: Any, blocks: Union[str, Sequence[str]], param_to_get: str):
    "Helper function to call eng.get_param() from the matlab.engine module"
    block_name = compose_block_name(blocks)
    return eng.get_param(block_name, param_to_get)
