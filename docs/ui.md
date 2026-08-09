smart-four UI
=============

The web UI lives in `ui/` (TypeScript + Vite + Three.js, Vitest for tests).
It renders a rotatable 3D board, enforces the game rules, and supports
human-vs-human and human-vs-machine play. The machine player is currently a
random dummy; the AlphaZero model (see `docs/model.md`) will replace it.

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
  - `machine.ts` — `MachinePlayer` interface and the `RandomMachinePlayer`
    dummy.
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
`RandomMachinePlayer` implements `MachinePlayer`: given an immutable state
snapshot, settings (`disabled` = policy only, `effort` = MCTS search steps)
and an abort signal, it must resolve with a legal move. The AlphaZero model
in `model/` (`smartfour.infer.SmartFourAgent.choose_move`) consumes the game
state as JSON and returns `(x, z)`; a real `MachinePlayer` adapter around it
goes in `src/game/machine.ts` and is swapped in at `src/main.ts`.

Commands
--------
    cd ui
    npm install              # first time
    npm run dev              # local dev server
    npm test                 # vitest unit tests (game engine)
    npm run build            # typecheck + production bundle (dist/)
    npm run preview -- --host 0.0.0.0   # serve the build; remote access
