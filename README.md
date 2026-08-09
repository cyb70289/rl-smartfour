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
MCTS), all CPU. See [`docs/model.md`](docs/model.md) for full command line.

Quick setup and run:

```sh
cd model
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m smartfour.train --config config.toml --iterations 10
.venv/bin/python -m smartfour.infer --checkpoint checkpoints/best.pt --sims 200 --state state.json
```

## UI

See [`docs/ui.md`](docs/ui.md) for design principals.

Build and serve for remote access:

```sh
cd ui
npm run build
npm run preview -- --host 0.0.0.0 --port 8032
```

Play against the machine (currently a random dummy) or a person on the same
screen; you can revert the last move, pick your color, and adjust machine
think effort (disable = policy only).

## TODO: model integration

The machine player behind the UI is a temporary random dummy. The AlphaZero
model (see [`docs/model.md`](docs/model.md)) lives in `model/` as Python with
its own tests. `smartfour.infer.SmartFourAgent.choose_move` accepts the game
state as JSON (the `state_to_json` format) and returns `(x, z)`.

To replace the dummy, implement the `MachinePlayer` interface
([`ui/src/game/machine.ts`](ui/src/game/machine.ts)) and swap the instance in
[`ui/src/main.ts`](ui/src/main.ts):

```ts
interface MachinePlayer {
  readonly name: string;
  think(state: Readonly<GameState>, settings: ThinkSettings, signal?: AbortSignal): Promise<Move>;
}
```

- `settings.disabled` — policy-only move (no MCTS search), set from the UI's
  "Disable search" checkbox.
- `settings.effort` — MCTS search steps, set from the UI slider.
- `signal` — aborted when the move is no longer wanted (e.g. a new game
  started); the implementation should reject promptly on abort.
- The move returned must be legal for `state` (the controller validates).

## CREDIT

- `DeepSeek-v4-Flash-0731` for all the coding and documentation.
- `oh-my-pi` coding agent, without additional skills.
