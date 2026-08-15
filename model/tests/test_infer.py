"""Tests for smartfour.infer — checkpoint-loaded agent and JSON state contract."""

import json

import torch

from smartfour.config import NetworkConfig
from smartfour.game import BLACK, WHITE, apply_move, initial_state, is_legal, legal_moves, state_from_json, state_to_json
from smartfour.infer import SmartFourAgent
from smartfour.network import ResNet


def make_agent(tmp_path, seed=0, device="cpu"):
    from dataclasses import asdict

    torch.manual_seed(seed)
    net_cfg = NetworkConfig(input_channels=16, blocks=1, base_channels=8,
                            policy_channels=4, value_channels=4, value_fc=8)
    net = ResNet(net_cfg)
    path = tmp_path / "net.pt"
    torch.save({"net_state": net.state_dict(), "iteration": 1, "network": asdict(net_cfg)}, path)
    return SmartFourAgent(str(path), device=device)


def test_json_state_round_trip():
    s = initial_state()
    for x, z in [(0, 0), (4, 4), (1, 1)]:
        s = apply_move(s, x, z)
    data = state_to_json(s)
    assert data["current"] == "black"
    assert data["pieces_left"] == {"white": 30, "black": 31}  # white moved twice
    s2 = state_from_json(data)
    assert s2.grid == s.grid
    assert s2.pieces_left == s.pieces_left
    assert s2.current == s.current
    assert s2.winner == s.winner


def test_choose_move_returns_legal_move(tmp_path):
    agent = make_agent(tmp_path)
    s = initial_state()
    for x, z in [(0, 0), (4, 4)]:
        s = apply_move(s, x, z)
    move = agent.choose_move(s, simulations=10)
    assert move in legal_moves(s)


def test_choose_move_accepts_json_state(tmp_path):
    agent = make_agent(tmp_path)
    s = apply_move(initial_state(), 0, 0)
    move_from_json = agent.choose_move(state_to_json(s), simulations=0)
    move_from_state = agent.choose_move(s, simulations=0)
    assert move_from_json == move_from_state
    assert move_from_json in legal_moves(s)


def test_choose_move_policy_only_is_argmax(tmp_path):
    agent = make_agent(tmp_path)
    s = initial_state()
    logits, _ = agent.net(agent._encode(s).unsqueeze(0).to(agent.device))
    mask = agent._mask(s)
    expected = int(torch.argmax(torch.where(
        mask.bool(), logits[0].cpu(), torch.full_like(logits[0].cpu(), -1e9))).item())
    from smartfour.encode import action_to_xyz

    move = agent.choose_move(s, simulations=0)
    assert action_to_xyz(expected)[:2] == move


def test_choose_move_terminal_returns_none(tmp_path):
    agent = make_agent(tmp_path)
    s = initial_state()
    for x, z in [(0, 0), (4, 4), (0, 1), (4, 3), (0, 2), (4, 2), (0, 3)]:
        s = apply_move(s, x, z)
    assert s.winner is not None
    assert agent.choose_move(s, simulations=10) is None


def test_agent_from_json_winner_parsed(tmp_path):
    agent = make_agent(tmp_path)
    s = initial_state()
    for x, z in [(0, 0), (4, 4), (0, 1), (4, 3), (0, 2), (4, 2), (0, 3)]:
        s = apply_move(s, x, z)
    data = state_to_json(s)
    assert data["winner"] == "white"
    parsed = state_from_json(data)
    assert parsed.winner == WHITE


def test_malformed_json_raises_clear_error():
    import pytest

    with pytest.raises(ValueError, match="grid"):
        state_from_json({"grid": [[[0, None, None, None, None]]], "pieces_left": {"white": 32, "black": 32}, "current": "white", "winner": None})
    with pytest.raises(ValueError, match="current"):
        state_from_json({
            "grid": [[[None for _ in range(5)] for _ in range(5)] for _ in range(5)],
            "pieces_left": {"white": 32, "black": 32}, "current": "red", "winner": None,
        })
