"""Tests for smartfour.selfplay — game generation and outcome assignment."""

import torch

from smartfour.config import MCTSConfig, NetworkConfig
from smartfour.encode import action_to_xyz, encode
from smartfour.game import BLACK, WHITE, apply_move, initial_state, terminal_value
from smartfour.network import ResNet
from smartfour.selfplay import play_game

SIM = 60


def tiny_net(seed=0):
    torch.manual_seed(seed)
    return ResNet(NetworkConfig(input_channels=16, blocks=1, base_channels=8,
                                policy_channels=4, value_channels=4, value_fc=8))


def cfg(**kw):
    kw.setdefault("simulations", SIM)
    kw.setdefault("c_puct", 1.0)
    kw.setdefault("dirichlet_alpha", 0.3)
    kw.setdefault("dirichlet_epsilon", 0.25)
    kw.setdefault("temperature_threshold", 12)
    kw.setdefault("batch_eval_size", 16)
    return MCTSConfig(**kw)


def test_play_game_full_and_consistent():
    torch.manual_seed(0)
    net = tiny_net()
    samples, winner = play_game(net, cfg(), temperature_threshold=12)
    assert 0 <= len(samples) <= 64
    # Every stored position is a legal, consistent position.
    z_white = 1.0 if winner == WHITE else (-1.0 if winner == BLACK else 0.0)
    for s, pi, player, z in samples:
        assert s.shape == (16, 5, 5)
        assert abs(pi.sum().item() - 1.0) < 1e-5
        assert z in (-1.0, 0.0, 1.0)
        expected_z = z_white if player == WHITE else -z_white
        assert z == expected_z


def test_play_game_replays_to_terminal():
    """Replaying the sampled moves reproduces the reported winner.

    temperature_threshold=0 makes every move the argmax, so the stored policy
    identifies the exact move played.
    """
    torch.manual_seed(1)
    net = tiny_net(seed=2)
    samples, winner = play_game(net, cfg(), temperature_threshold=0)
    state = initial_state()
    for s, pi, player, z in samples:
        # The stored encoding matches the state we replay through.
        assert torch.equal(s, encode(state))
        a = int(pi.argmax())
        x, zc, _y = action_to_xyz(a)
        state = apply_move(state, x, zc)
    assert terminal_value(state) == (1.0 if winner == WHITE else -1.0 if winner == BLACK else 0.0)


def test_terminates_quickly_with_winning_net():
    """A net that always completes an immediate win ends the game fast."""
    s = initial_state()
    for x, z in [(0, 0), (4, 4), (0, 1), (4, 3), (0, 2), (4, 2)]:
        s = apply_move(s, x, z)  # white: 3 in a row; black filler
    assert s.current == WHITE

    class WinNet:
        def __call__(self, t):
            n = t.shape[0]
            logits = torch.zeros(n, 125)
            logits[:, 0 * 25 + 3 * 5 + 0] = 10.0  # (3,0) completes the four
            return logits, torch.zeros(n, 1)

    samples, winner = play_game(WinNet(), cfg(simulations=30), temperature_threshold=0,
                                start_state=s)
    assert winner == WHITE
    assert len(samples) == 1  # white's winning move is the only played move


def test_temperature_zero_is_one_hot():
    torch.manual_seed(3)
    net = tiny_net(seed=4)
    samples, _ = play_game(net, cfg(), temperature_threshold=0)
    for s, pi, player, z in samples:
        assert pi.sum().item() == 1.0
        assert int((pi > 0).sum()) == 1  # one-hot (argmax policy)


def test_temperature_one_explores():
    torch.manual_seed(5)
    net = tiny_net(seed=6)
    samples, _ = play_game(net, cfg(simulations=100), temperature_threshold=1000)
    assert len(samples) > 0
    # Early symmetric position: visit distribution is spread over actions.
    s0, pi0, _, _ = samples[0]
    assert (pi0 > 0).sum() > 1


def test_deterministic_with_seed():
    torch.manual_seed(7)
    a, wa = play_game(tiny_net(seed=8), cfg(), temperature_threshold=12)
    torch.manual_seed(7)
    b, wb = play_game(tiny_net(seed=8), cfg(), temperature_threshold=12)
    assert wa == wb
    assert len(a) == len(b)
    for (sa, pia, pa, za), (sb, pib, pb, zb) in zip(a, b):
        assert torch.equal(sa, sb) and torch.equal(pia, pib)
        assert pa == pb and za == zb
