Smart-four AlphaZero model
==========================

AlphaZero-style training and inference (ResNet + MCTS) for the smart-four
game (rules in [`game.md`](game.md)), implemented in `model/` with PyTorch,
CPU only. All network inputs and outputs are from the perspective of the
*current player* (the player to move), which leverages the color symmetry and
keeps the design simple; MCTS keeps the same convention.

Rules recap (see [`game.md`](game.md) and `smartfour/game.py`, a faithful
port of the UI engine): 5x5 grid, stacks up to 5 high; win by lining 4+ own
pieces along any 3D line (4 flat directions, 1 vertical, 8 rising diagonals —
runs of 5 also win); draw when both players exhaust their 32 pieces (ply 64).
The game always terminates: 64 plies < 125 cells, and piece counts of the two
players never differ by more than one.

Network input
-------------
16 channels of 5x5 (one plane per height level), all from the current
player's perspective:

- channels 0-4: current player's pieces per plane (1 = present)
- channels 5-9: opponent's pieces per plane
- channels 10-14: legality per plane — 1 at the stack top of each column
  where the current player may place (empty board ⇒ level 0), 0 elsewhere
- channel 15: constant plane = total pieces remaining / 64, i.e. how close
  the game is to the ply-64 draw cap

The 15-channel layout follows the original draft; channel 16 is an extension:
the network cannot otherwise tell how many plies remain, and the *sum* of both
players' remaining pieces is perspective-invariant and exact.

Actions are indexed `a = y * 25 + x * 5 + z` (125 logits, plane-major,
matching `encode.xyz_to_action`). A move is a column `(x, z)`; the piece lands
at the current stack height, so exactly one level per column is legal at any
time (~25 legal actions per position). Policy logits are masked to legal
actions (softmax over the legal subset), per standard AlphaZero practice.

Network
-------
`smartfour/network.py` — ResNet with configurable parameters in a TOML file:

- stem: 3x3 conv → batch norm → ReLU
- `blocks` residual blocks (3x3 conv, BN, ReLU, skip connection)
- policy head: 1x1 convs → 5 height planes (125 logits)
- value head: 1x1 conv → flatten → MLP → scalar in (-1, 1)

Defaults (`config.toml`): 5 blocks, 64 base channels. `config_small.toml` is
a reduced profile (3 blocks, 32 channels) for quick CPU proof runs.

MCTS
----
`smartfour/mcts.py` — standard AlphaZero MCTS, CPU-only, configurable in the
same TOML file:

- PUCT selection: `Q + c_puct * P * sqrt(N_parent) / (1 + N_child)`; Q is
  negated at each level because every node stores values from its own
  perspective (backprop flips the sign per level)
- dirichlet noise at the root during self-play (`alpha`, `epsilon`)
- temperature schedule: `temperature_threshold` plies with tau=1 (sample from
  visit counts), then tau=0 (argmax)
- batched leaf evaluation (`batch_eval_size` positions per net forward) with
  random tie-breaking among equal UCB scores; a queued leaf is skipped until
  evaluated, and the search drains the batch and goes deeper instead of
  piling duplicate visits onto one child
- terminal leaves short-circuit the net (no forward on game-over positions)

Training
--------
`smartfour/train.py` — per iteration:

1. self-play `selfplay_games` games (MCTS with root noise and temperature)
2. store `(state, pi, z)` in a replay buffer; `z` is the outcome from each
   stored position's own perspective (+1 win, -1 loss, 0 draw)
3. optimize for `train_epochs` over sampled batches: policy cross-entropy +
   value MSE + L2 (AdamW); samples are augmented with a random D4 symmetry
   transform of the 5x5 board (policy permuted identically, value invariant)
4. arena: greedy MCTS vs the current best checkpoint (alternating colors,
   `eval_games` games, `eval_simulations` per move); the candidate replaces
   the best when its win ratio reaches `arena_win_ratio` (default 0.55)

Checkpoints live in `checkpoints/`: `latest.pt` every iteration,
`best.pt` + `best_iter_XXXX.pt` on improvement. Each checkpoint carries the
network weights, optimizer state, replay buffer, and iteration counter, so
training resumes with `--resume`.

Inference
---------
`smartfour/infer.py` — `SmartFourAgent` loads a checkpoint and returns a move
`(x, z)` for a game state given as a `GameState` or as JSON in the UI
interchange format (`game.state_to_json`): `grid[x][z][y]` with 0/1/null,
`pieces_left`, `current`, `winner`. `simulations=0` is policy-only (no
search), which is what the UI's "disable search" mode uses.

`smartfour/worker.py` — the persistent bridge worker for the UI. It loads the
checkpoint once, prints `{"ready": true, ...}`, then serves one request per
line on stdin/stdout as newline-delimited JSON: requests are
`{"id": n, "state": <state_to_json>, "simulations": m}`, responses are
`{"id": n, "move": {"x", "z"}}` (or `{"move": null}` on a terminal state);
errors are reported in-band as `{"id": n, "error": "..."}` and the loop keeps
serving. The Vite plugin in `ui/plugins/` spawns it and exposes
`POST /api/think` to the browser.

Running
-------
```sh
cd model
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu pytest

.venv/bin/python -m pytest tests/        # full TDD suite (375 tests)

.venv/bin/python -m smartfour.train --config config.toml --iterations 10
.venv/bin/python -m smartfour.train --config config.toml --iterations 10 --resume
.venv/bin/python -m smartfour.infer --checkpoint checkpoints/best.pt --sims 200 --state state.json
```

Verification
------------
- 375 unit tests: exhaustive win-line coverage over all 3D lines, encoding
  and D4 symmetry, MCTS value propagation and terminal handling, replay
  buffer augmentation, checkpoint round-trip, best-model selection.
- Proof run (`config_small.toml`, 4 iterations): loss 5.24 → 4.07; the
  candidate beat the previous best in the arena at iterations 1, 3 and 4.
- Trained `best.pt`: 60/60 (100%) vs a uniform-random opponent with 100-sim
  search across 3 seeds; 13/7 (65%) policy-only vs 7/13 (35%) for an
  untrained policy — the learned policy beats random even without search.
