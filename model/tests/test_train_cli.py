"""CLI-level tests for smartfour.train: resume-by-default, --restart, and
--iterations as a target.

These drive main() end to end with a stubbed training loop, pinning the
user-facing contracts: which checkpoint resumes, when training starts and
stops, and what --restart deletes.
"""

import sys
from pathlib import Path

import pytest

from smartfour.train import Trainer, main


def write_config(tmp_path, blocks=1, **training):
    cfg_path = tmp_path / "cfg.toml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cdir = training.pop("checkpoint_dir", str(tmp_path / "checkpoints"))
    defaults = dict(
        selfplay_games=3,
        workers=1,
        train_epochs=1,
        batch_size=8,
        replay_capacity_games=10_000,
        learning_rate=0.001,
        weight_decay=0.0,
        symmetry_augment=True,
        eval_games=2,
        arena_win_ratio=0.55,
        seed=0,
        checkpoint_dir=cdir,
    )
    defaults.update(training)

    def fmt(v):
        if isinstance(v, bool):
            return str(v).lower()
        if isinstance(v, str):
            return f'"{v}"'
        return str(v)

    training_lines = "\n".join(f"{k} = {fmt(v)}" for k, v in defaults.items())
    cfg_path.write_text(
        f"""
[network]
input_channels = 16
blocks = {blocks}
base_channels = 8
policy_channels = 4
value_channels = 4
value_fc = 8

[mcts]
simulations = 20
c_puct = 1.0
dirichlet_alpha = 0.3
dirichlet_epsilon = 0.25
temperature_threshold = 12
batch_eval_size = 16

[training]
{training_lines}
"""
    )
    return cfg_path


def seed_checkpoint(cfg_path, iteration=5, name="latest.pt"):
    """Write a valid checkpoint under the config's checkpoint_dir."""
    from smartfour.config import load_config

    cfg = load_config(str(cfg_path))
    t = Trainer(cfg)
    t.iteration = iteration
    t.best_iteration = max(0, iteration - 2)
    t.save_checkpoint(Path(cfg.training.checkpoint_dir) / name)
    return t


def checkpoint_dir(cfg_path) -> Path:
    from smartfour.config import load_config

    return Path(load_config(str(cfg_path)).training.checkpoint_dir)


def noop_train(self, target):
    noop_train.calls.append((self.iteration, target))
    return []


@pytest.fixture
def stub_train(monkeypatch):
    noop_train.calls = []
    monkeypatch.setattr(Trainer, "train", noop_train)
    return noop_train


@pytest.fixture
def tty_stdin(monkeypatch):
    """Pretend stdin is a terminal so --restart reaches its input() prompt."""

    class FakeTty:
        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdin", FakeTty())


# ------------------------------------------------------------- resume is the default

def test_cli_resumes_latest_by_default(tmp_path, stub_train, capsys):
    cfg = write_config(tmp_path)
    seed_checkpoint(cfg, iteration=5)
    main(["--config", str(cfg), "--iterations", "8"])
    out = capsys.readouterr().out
    assert "Resumed from latest.pt (iteration 5)" in out
    assert "start iteration 6" in out
    assert "3 left (target 8)" in out
    assert stub_train.calls == [(5, 8)]


def test_cli_fresh_start_without_checkpoint(tmp_path, stub_train, capsys):
    cfg = write_config(tmp_path)
    main(["--config", str(cfg), "--iterations", "4"])
    out = capsys.readouterr().out
    assert "No checkpoint found; starting fresh" in out
    assert stub_train.calls == [(0, 4)]


def test_cli_never_resumes_from_best(tmp_path, stub_train, capsys):
    """best{n}.pt are inference snapshots, never resume anchors."""
    cfg = write_config(tmp_path)
    seed_checkpoint(cfg, iteration=9, name="best1.pt")
    main(["--config", str(cfg), "--iterations", "4"])
    out = capsys.readouterr().out
    assert "No checkpoint found; starting fresh" in out
    assert stub_train.calls == [(0, 4)]


def test_cli_stale_iter_files_are_ignored(tmp_path, stub_train, capsys):
    """iter_NNNN.pt from older runs are no longer read; without latest.pt
    resume starts fresh instead of falling back to an iter snapshot."""
    cfg = write_config(tmp_path)
    seed_checkpoint(cfg, iteration=7, name="iter_0007.pt")
    main(["--config", str(cfg), "--iterations", "10"])
    out = capsys.readouterr().out
    assert "No checkpoint found; starting fresh" in out
    assert stub_train.calls == [(0, 10)]


def test_cli_corrupt_latest_hard_errors(tmp_path):
    cfg = write_config(tmp_path)
    cdir = checkpoint_dir(cfg)
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "latest.pt").write_bytes(b"garbage, not a checkpoint")
    with pytest.raises(SystemExit, match="corrupt"):
        main(["--config", str(cfg), "--iterations", "4"])


def test_cli_config_mismatch_hard_errors(tmp_path):
    cfg_a = write_config(tmp_path / "a", blocks=1)
    seed_checkpoint(cfg_a, iteration=5)
    # Same checkpoint dir, but a config with a different architecture.
    cfg_b = write_config(tmp_path / "b", blocks=2,
                         checkpoint_dir=str(tmp_path / "a" / "checkpoints"))
    with pytest.raises(SystemExit, match="trained with"):
        main(["--config", str(cfg_b), "--iterations", "10"])


# ------------------------------------------------------------- --iterations as a target

def test_cli_target_already_met_exits_without_training(tmp_path, monkeypatch, capsys):
    cfg = write_config(tmp_path)
    seed_checkpoint(cfg, iteration=5)

    def boom(self, target):
        raise AssertionError("must not train when the target is already met")

    monkeypatch.setattr(Trainer, "train", boom)
    with pytest.raises(SystemExit) as e:
        main(["--config", str(cfg), "--iterations", "3"])
    assert e.value.code == 0
    assert "nothing to train" in capsys.readouterr().out


def test_cli_zero_target_exits_immediately(tmp_path, monkeypatch, capsys):
    cfg = write_config(tmp_path)

    def boom(self, target):
        raise AssertionError("must not train")

    monkeypatch.setattr(Trainer, "train", boom)
    with pytest.raises(SystemExit) as e:
        main(["--config", str(cfg), "--iterations", "0"])
    assert e.value.code == 0
    assert "nothing to train" in capsys.readouterr().out


def test_cli_negative_iterations_rejected(tmp_path):
    cfg = write_config(tmp_path)
    with pytest.raises(SystemExit) as e:
        main(["--config", str(cfg), "--iterations", "-1"])
    assert e.value.code == 2


# ------------------------------------------------------------- --restart

def test_cli_restart_confirms_and_wipes(tmp_path, stub_train, tty_stdin, monkeypatch, capsys):
    cfg = write_config(tmp_path)
    cdir = checkpoint_dir(cfg)
    seed_checkpoint(cfg, iteration=5)
    seed_checkpoint(cfg, iteration=3, name="iter_0003.pt")
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    main(["--config", str(cfg), "--restart", "--iterations", "4"])
    assert not (cdir / "latest.pt").exists()
    assert not (cdir / "iter_0003.pt").exists()
    out = capsys.readouterr().out
    assert "Deleted 2 file(s)" in out
    assert "No checkpoint found; starting fresh" in out
    assert stub_train.calls == [(0, 4)]


def test_cli_restart_declined_keeps_files(tmp_path, tty_stdin, monkeypatch, capsys):
    cfg = write_config(tmp_path)
    cdir = checkpoint_dir(cfg)
    seed_checkpoint(cfg, iteration=5)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    with pytest.raises(SystemExit) as e:
        main(["--config", str(cfg), "--restart"])
    assert e.value.code == 1
    assert (cdir / "latest.pt").exists()
    assert "Aborted" in capsys.readouterr().out


def test_cli_restart_non_tty_aborts(tmp_path, monkeypatch):
    cfg = write_config(tmp_path)
    cdir = checkpoint_dir(cfg)
    seed_checkpoint(cfg, iteration=5)

    class FakeStdin:
        def isatty(self):
            return False

    monkeypatch.setattr(sys, "stdin", FakeStdin())
    with pytest.raises(SystemExit, match="not a terminal"):
        main(["--config", str(cfg), "--restart"])
    assert (cdir / "latest.pt").exists()


def test_cli_restart_yes_skips_prompt(tmp_path, stub_train, monkeypatch):
    cfg = write_config(tmp_path)
    cdir = checkpoint_dir(cfg)
    seed_checkpoint(cfg, iteration=5)

    def no_prompt(prompt):
        raise AssertionError("--yes must skip the confirmation prompt")

    monkeypatch.setattr("builtins.input", no_prompt)
    main(["--config", str(cfg), "--restart", "--yes", "--iterations", "2"])
    assert not (cdir / "latest.pt").exists()
    assert stub_train.calls == [(0, 2)]


def test_cli_restart_empty_dir_proceeds_fresh(tmp_path, stub_train, capsys):
    cfg = write_config(tmp_path)
    checkpoint_dir(cfg).mkdir(parents=True)
    main(["--config", str(cfg), "--restart", "--iterations", "2"])
    out = capsys.readouterr().out
    assert "Nothing to delete" in out
    assert stub_train.calls == [(0, 2)]
