# Smart-Four

Advanced tic-tac-toe on a 5×5×5 board: two players place pieces on a 5×5 grid
and stack up to five high; first to line up four own pieces in 3D space wins
(horizontal planes, vertical stacks, and rising diagonals). Full rules in
[`docs/game.md`](docs/game.md).

## Layout

| Path    | Contents                                                        |
| ------- | --------------------------------------------------------------- |
| `ui/`   | Web UI — TypeScript + Vite + Three.js, includes the game engine |
| `model/`| AlphaZero-style model (Python) — **TODO, not yet created**      |
| `docs/` | Game rules, UI requirements, model spec (TBD)                   |

## Building and running the UI

```sh
cd ui
npm install
npm run dev        # local: http://localhost:5173
```

Production build, then serve for remote access:

```sh
cd ui
npm run build
npm run preview -- --host 0.0.0.0
```

Play against the machine (currently a random dummy) or a person on the same
screen; you can revert the last move, pick your color, and adjust machine
think effort (disable = policy only).

## Tests

```sh
cd ui
npm test
```

The game engine (rules, win detection, revert semantics, machine-turn
orchestration) is unit-tested with Vitest.

## TODO: model integration

The machine player behind the UI is a temporary random dummy. The planned
AlphaZero model (resnet + MCTS, see [`docs/model.md`](docs/model.md) — spec
TBD) will live in `model/` as Python with its own tests.

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
