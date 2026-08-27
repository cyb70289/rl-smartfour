smart-four UI
=============

The web UI lives in `ui/` (TypeScript + Vite + Three.js, Vitest for tests).
It renders a rotatable 3D board, enforces the game rules, and supports
human-vs-human, human-vs-model and model-vs-model (auto play) games. Models
are the trained AlphaZero checkpoints (see `docs/model.md`), reached through
a Vite bridge plugin.

What it provides
----------------
- 3D board, draggable to view from any angle (pieces can block each other).
  The default view frames the board in the upper part of the window; the last
  camera position/zoom/angle is remembered and restored on the next load.
- Bucket-shaped pieces stacked bottom-up, white first; the top bar shows the
  current player and their remaining pieces.
- Last move highlighted; winning line highlighted with winner shown; draw
  shown when all 64 pieces are placed without a winner.
- Two independent player slots — **White player** and **Black player** — each
  either *Human* or *Model* (radio). A model slot has a dropdown of every
  `best{n}.pt` checkpoint in `model/checkpoints` (largest n = strongest,
  listed largest first, and that largest one is selected by default).
  Defaults: white human, black model. Changing a player slot or its
  checkpoint keeps the current board: outside running auto play the change
  takes effect immediately (a model owing a move thinks at once), in auto
  play it waits for Play.
- Game modes fall out of the slots:
  - both human — classic hot-seat play;
  - one model — play against the model (either color);
  - both model — auto play: **Play/Pause** and **Step** buttons. Play
    starts/resumes the match on the CURRENT board state, Pause aborts the
    in-flight think, Step plays exactly one move from the current state and
    pauses. A finished game needs **Reset** — Play/Step are no-ops on it.
    Auto play moves start at least 2s apart; a think that takes longer adds
    no extra delay. While auto play is running the player slots and think
    effort are locked; pause to change them.
- Revert one level: in person mode it undoes the last move; with a model
  involved it undoes the model and the last human move together so the human
  can retry. The revert window is consumed by the revert (no double undo); a
  finished game can be reverted as well.
- **Reset** (always enabled) clears the board and starts a fresh game with
  the selected players, stopping any auto play or in-flight think; player
  selections and effort are kept. It replaces the old New Game button.
- Think effort selectable as three radios — Entry (0 = policy only), Medium
  (500), High (2000), default Medium — applied immediately to every model
  move; disabled when neither side is a model. The chosen effort is persisted
  (`localStorage`) and restored on the next launch; player slots always reset
  to their defaults.
Opening-book viewer
-------------------
Starting the server with `SMARTFOUR_VIEW=openbook` (`npm run dev` or
`npm run preview`) boots into a read-only viewer instead of the game: play
controls are hidden, the side panel becomes an "Open states" list (no.1 …
no.N) with Prev/Next buttons (arrow keys work too; ends clamp), and the top
bar shows the selection ("no.42 · White to move"). Selecting a state renders
it on the board — pieces only, no placement, no hover ghost; drag/zoom still
work. The states come from `model/openbook.json` (symlinked into
`ui/public/`); no model workers are started and no `/api/think` calls are
made. Without the env var everything behaves as before.

Architecture
------------
- `src/main.ts` — composition root: loads the persisted effort, creates the
  controller with per-slot model players, subscribes the scene/HUD to
  controller state changes.
- `src/game/` — framework-free game engine (no Three.js / DOM):
  - `types.ts` — core types: Player, Move, PlacedPiece, PlayerSlot
    (human | model + checkpoint), derived Mode (person/machine/autoplay),
    GameState, ThinkSettings.
  - `rules.ts` — board construction, legality, applying moves, 3D win
    detection (all line geometries incl. stacks and rising diagonals), draw.
  - `engine.ts` — pure reducer over GameState for human moves, model moves,
    revert and reset; enforces turn/thinking/game-over rules; the thinking
    flag stays set between auto play moves.
  - `machine.ts` — `MachinePlayer` interface, the `ModelMachinePlayer`
    adapter (state JSON mapping, settings → simulations, checkpoint in the
    request, legality check) and the `RandomMachinePlayer` reference/test
    fixture.
  - `controller.ts` — owns the live state, kicks off async model thinks for
    the color that owes a move, guards against stale results (generation
    counter + abort signal), and runs the auto play scheduler (2s minimum gap
    between think starts, measured with an injectable clock).
- `src/ui/` — rendering layer:
  - `scene.ts` — Three.js scene: board, pieces, last-move ring, win beam,
    hover ghost, stack-height-aware column picking, orbit camera.
  - `src/ui/hud.ts` — top-bar status (with remaining pieces) and DOM side
    panel: action buttons (renamed per mode), white/black player rows with
    checkpoint dropdowns, effort radios, effort persistence, checkpoint-list
    loading.

Design notes
------------
- Game state is a plain immutable object; every change flows through the
  engine reducer and the controller notifies subscribers, which re-sync the
  scene and HUD. The board and moves are the only source of truth.
- Win detection reports the full winning run for highlighting and is
  exhaustively unit-tested over every 4-in-a-row geometry in 5x5x5.
- Column picking targets a piece-sized disc on the placement surface only
  (the base, or the top of the stack). Only a pointer inside the disc selects
  the candidate, and the ghost preview appears only then — pointing at a
  stack's side face, its lower body, the air above it, or the base beside a
  stack is a no-op; placed pieces also block picking rays, so aiming "through"
  a stack never lands on a column behind it. Full columns have no target.
  Small targets keep far columns clickable through gaps between stacks
  (rotate if a real stack blocks the view).
- Model turns are asynchronous: while thinking, board clicks and revert are
  disabled; a new game (or Pause/Step) aborts the in-flight move.

Machine integration
-------------------
The model players are the trained AlphaZero checkpoints (`docs/model.md`)
reached through a bridge:

    browser (src/game/machine.ts — ModelMachinePlayer)
       │  POST /api/think  {"state": <state_to_json>, "simulations": n, "checkpoint": "best3.pt"}
       ▼
    vite dev/preview server (plugins/model-bridge.ts)
       │  per-checkpoint worker cache (LRU, max 2), routes by checkpoint
       ▼
    model/smartfour/worker.py → SmartFourAgent.choose_move

    browser (src/ui/hud.ts)
       │  GET /api/checkpoints  →  {"checkpoints": ["best3.pt", "best2.pt", "best1.pt"]}
       ▼
    vite dev/preview server → lists model/checkpoints/best{n}.pt, biggest n first

Where things live
- `src/game/machine.ts` — `ModelMachinePlayer` implements `MachinePlayer`
  and is wired up in `src/main.ts` (one instance per model slot). It converts
  the state to the model's JSON contract (`stateToJson`: grid colors 0/1/null,
  `pieces_left`, `current`, `winner`), maps settings to MCTS steps
  (`simulationsOf`: effort < 1 → 0 = policy-only, else `floor(effort)`),
  sends the checkpoint name, passes the abort signal to `fetch` (aborts
  reject promptly) and validates the returned move against the snapshot
  before resolving. `RandomMachinePlayer` remains as a reference/test fixture.
- `plugins/model-bridge.ts` — Vite plugin serving `POST /api/think` and
  `GET /api/checkpoints` on both dev and preview servers. `CheckpointWorkerPool`
  caches workers per checkpoint (LRU, default max 2 — one per player slot);
  the default checkpoint worker (biggest `best{n}.pt`) is started eagerly, workers are
  restarted on demand after a failure, and all are killed with the server.
- `plugins/worker-client.ts` — process client: requests are strictly
  serialized per worker and matched by id; an aborted request's late response
  is discarded; protocol violations or worker death fail in-flight requests.
- `model/smartfour/worker.py` — persistent newline-JSON worker: loads the
  checkpoint once, prints a ready line, then answers one request per line
  (errors in-band, loop keeps serving).

Behavior
- Checkpoints are expected to be present: if a worker cannot start (missing
  venv/checkpoint), the failure is logged loudly and every think for that
  checkpoint returns 503 with the reason, shown in the UI banner. Model play
  refuses until the model is available — there is no silent fallback.
- Override the interpreter path with `SMARTFOUR_PYTHON`. The old
  `SMARTFOUR_CHECKPOINT` override is gone: checkpoints are chosen in the UI.
- The controller catches illegal model moves, reports them via `onError`
  and releases the `machineThinking` lock, so a broken checkpoint can never
  wedge the UI.
- `/api/think` and `/api/checkpoints` are served by Vite only (dev/preview);
  serving `dist/` with a plain static file server loses them.

Commands
--------
    cd ui
    npm install              # first time
    npm run dev              # local dev server
    npm test                 # vitest unit tests (game engine)
    npm run build            # typecheck + production bundle (dist/)
    npm run preview -- --host 0.0.0.0   # serve the build; remote access
