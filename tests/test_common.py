import torch

from src.common import TargetNet


def test_target_net():
    net = torch.nn.Sequential(torch.nn.Linear(2, 2))
    target_net = TargetNet(net)
    assert (
        str(target_net)
        == "Sequential(\n  (0): Linear(in_features=2, out_features=2, bias=True)\n)"
    )

    # Change all weights from net to 1.0
    state_dict = net.state_dict()
    for layer, weigths in state_dict.items():
        state_dict[layer] = torch.ones_like(weigths)
    net.load_state_dict(state_dict)

    # Assert that all weights of net and target_net are different
    state_dict = net.state_dict()
    tgt_state_dict = target_net.target_model.state_dict()
    for layer, weigths in state_dict.items():
        assert torch.all(torch.not_equal(weigths, tgt_state_dict[layer]))

    # Sync the weights of net and target_net and assert they are equal
    target_net.sync()
    state_dict = net.state_dict()
    tgt_state_dict = target_net.target_model.state_dict()
    for layer, weigths in state_dict.items():
        assert torch.equal(weigths, tgt_state_dict[layer])

    # Change all weights from net to 2.0
    state_dict = net.state_dict()
    for layer, weigths in state_dict.items():
        state_dict[layer] = torch.ones_like(weigths) * 2.0
    net.load_state_dict(state_dict)

    # Alpha sync the weights of net and target_net and assert the result
    target_net.alpha_sync(alpha=0.6)
    state_dict = net.state_dict()
    tgt_state_dict = target_net.target_model.state_dict()
    for layer, weigths in state_dict.items():
        # Previos target_net weights were 1.0
        expected = torch.ones_like(weigths)
        expected *= 0.6
        expected += 0.4 * weigths
        assert torch.equal(tgt_state_dict[layer], expected)