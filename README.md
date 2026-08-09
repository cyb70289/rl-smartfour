# Smart-Four

Advanced tic-tac-toe on a 5×5×5 board: two players place pieces on a 5×5 grid
and stack up to five high; first to line up four own pieces in 3D space wins
(horizontal planes, vertical stacks, and rising diagonals). Full rules in
[`docs/game.md`](docs/game.md).

## Layout

| Path     | Contents                                                        |
| -------- | --------------------------------------------------------------- |
| `ui/`    | Web UI — TypeScript + Vite + Three.js, includes the game engine |
| `model/` | AlphaZero-style model (Python, PyTorch): rules, resnet, MCTS, training |
| `docs/`  | Game rules, UI requirements, model spec                         |

## Model

AlphaZero-style training and inference for smart-four (resnet policy/value +
MCTS), all CPU. Self-play and the arena can parallelize across `workers`
processes (see [`docs/model.md`](docs/model.md) for full command line).

Quick steps to train the model:

```sh
cd model
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m smartfour.train --config config.toml --iterations 10
```

## UI

See [`docs/ui.md`](docs/ui.md) for design principals.

The machine player is the trained model checkpoint. Train the model
by yourself or download `best.pt` from the
[releases page](https://github.com/cyb70289/rl-smartfour/releases) and
save it as `model/checkpoints/best.pt`, then build and serve:

```sh
cd ui
npm install        # first time
npm run build
npm run preview -- --host 0.0.0.0 --port 8032
```

Play against the machine or a person on the same screen; you can revert the
last move, pick your color, and adjust machine think effort (disable = policy
only).

## CREDIT

- `DeepSeek-v4-Flash-0731` for all the coding and documentation.
- `oh-my-pi` coding agent, without additional skills.
