import torch
import copy


class TargetNet:
    "Wrapper around model which provides copy of it instead of trained weights"

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def __call__(self, *args):
        return self.target_model(*args)

    def __repr__(self):
        return str(self.target_model)

    def sync(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha: float) -> None:
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)