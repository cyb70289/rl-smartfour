smart-four UI
=============

The web UI lives in `ui/` (TypeScript + Vite + Three.js, Vitest for tests).
It renders a rotatable 3D board, enforces the game rules, and supports
human-vs-human and human-vs-machine play. The machine player is the trained
AlphaZero model (see `docs/model.md`), reached through a Vite bridge plugin.

What it provides
----------------
- 3D board, draggable to view from any angle (pieces can block each other).
- Bucket-shaped pieces stacked bottom-up, white first; piece counts and
  current player shown.
- Last move highlighted; winning line highlighted with winner shown; draw
  shown when all 64 pieces are placed without a winner.
- Revert one level: in person mode it undoes the last move; in machine mode
  it undoes the machine and the last human move together so the human can
  retry. The revert window is consumed by the revert (no double undo).
- Person or machine mode; machine color selectable; machine search can be
  disabled (policy only) or given an effort (MCTS steps) in the UI; board
  input is locked while the machine thinks.

Architecture
------------
- `src/main.ts` — composition root: creates the controller, scene and HUD,
  subscribes the scene/HUD to controller state changes.
- `src/game/` — framework-free game engine (no Three.js / DOM):
  - `types.ts` — core types: Player, Move, PlacedPiece, GameState, ThinkSettings.
  - `rules.ts` — board construction, legality, applying moves, 3D win
    detection (all line geometries incl. stacks and rising diagonals), draw.
  - `engine.ts` — pure reducer over GameState for human moves, machine moves,
    revert and reset; enforces turn/thinking/game-over rules.
  - `machine.ts` — `MachinePlayer` interface, the `ModelMachinePlayer`
    adapter (state JSON mapping, settings → simulations, legality check) and
    the `RandomMachinePlayer` reference/test fixture.
  - `controller.ts` — owns the live state, kicks off async machine thinks,
    guards against stale results (generation counter + abort signal).
- `src/ui/` — rendering layer:
  - `scene.ts` — Three.js scene: board, pieces, last-move ring, win beam,
    hover ghost, stack-height-aware column picking, orbit camera.
  - `hud.ts` — DOM side panel: status, piece counts, last move, revert/new
    game buttons, mode/color/think-effort setup.

Design notes
------------
- Game state is a plain immutable object; every change flows through the
  engine reducer and the controller notifies subscribers, which re-sync the
  scene and HUD. The board and moves are the only source of truth.
- Win detection reports the full winning run for highlighting and is
  exhaustively unit-tested over every 4-in-a-row geometry in 5x5x5.
- Column picking is sized to the visible stack: empty columns are thin
  board-level targets, so far columns stay clickable until a real stack
  blocks the view (then rotate).
- Machine turns are asynchronous: while thinking, board clicks and revert are
  disabled; starting a new game aborts the in-flight move.

Machine integration
-------------------
The machine player is the trained AlphaZero model (`docs/model.md`) reached
through a bridge:

    browser (src/game/machine.ts — ModelMachinePlayer)
       │  POST /api/think  {"state": <state_to_json>, "simulations": n}
       ▼
    vite dev/preview server (plugins/model-bridge.ts)
       │  spawns once and drives a persistent Python worker
       ▼
    model/smartfour/worker.py → SmartFourAgent.choose_move

Where things live
- `src/game/machine.ts` — `ModelMachinePlayer` implements `MachinePlayer`
  and is wired up in `src/main.ts`. It converts the state to the model's
  JSON contract (`stateToJson`: grid colors 0/1/null, `pieces_left`,
  `current`, `winner`), maps settings to MCTS steps (`simulationsOf`:
  disabled or effort < 1 → 0 = policy-only, else `floor(effort)`), passes
  the abort signal to `fetch` (aborts reject promptly) and validates the
  returned move against the snapshot before resolving. `RandomMachinePlayer`
  remains as a reference/test fixture.
- `plugins/model-bridge.ts` — Vite plugin serving `POST /api/think` on both
  dev and preview servers; spawns the worker eagerly, restarts it on demand
  after a failure, kills it with the server.
- `plugins/worker-client.ts` — process client: requests are strictly
  serialized and matched by id; an aborted request's late response is
  discarded; protocol violations or worker death fail in-flight requests.
- `model/smartfour/worker.py` — persistent newline-JSON worker: loads the
  checkpoint once, prints a ready line (including the detected device —
  CUDA/MPS/CPU, auto-selected), then answers one request per line
  (errors in-band, loop keeps serving).

Behavior
- The model is expected to be present: if the worker cannot start (missing
  venv/checkpoint), the failure is logged loudly and every think returns 503
  with the reason, shown in the UI banner. Machine play refuses until the
  model is available — there is no silent fallback.
- Override the worker paths with `SMARTFOUR_PYTHON` / `SMARTFOUR_CHECKPOINT`.
- The controller catches illegal machine moves, reports them via `onError`
  and releases the `machineThinking` lock, so a broken checkpoint can never
  wedge the UI.
- `/api/think` is served by Vite only (dev/preview); serving `dist/` with a
  plain static file server loses it.

Commands
--------
    cd ui
    npm install              # first time
    npm run dev              # local dev server
    npm test                 # vitest unit tests (game engine)
    npm run build            # typecheck + production bundle (dist/)
    npm run preview -- --host 0.0.0.0   # serve the build; remote access
