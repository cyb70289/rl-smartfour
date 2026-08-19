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
MCTS), CPU or GPU (Metal/CUDA) — self-play and arena leaf evaluation is
centralized on one accelerator process, optimization runs on the device.
Self-play and the arena parallelize across `workers` processes (see
[`docs/model.md`](docs/model.md) for full command line).

> [!NOTE]
> GPU is necessary to train the model in reasonable time.
It took 18 hours to run 100 training iterations on RTX-3060.

You can download model checkpoint from
[releases page](https://github.com/cyb70289/rl-smartfour/releases).
Or follow steps below to train you own model:

```sh
cd model
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m smartfour.train --config config.toml --iterations 100
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

Play against the machine or a person on the same screen; you can revert moves
(including finished games), pick your color, and adjust machine think effort
(0 = policy only) — setup changes apply immediately.

## CREDIT

- `DeepSeek-v4-Flash-0731` for design and coding.
- `GLM-5.3` for performance improvement.
- `oh-my-pi` harness, `grill me` skill.
