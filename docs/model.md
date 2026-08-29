Smart-four AlphaZero model
==========================

AlphaZero-style training and inference (ResNet + MCTS) for the smart-four
game (rules in [`game.md`](game.md)), implemented in `model/` with PyTorch.
All network inputs and outputs are from the perspective of the *current
player* (the player to move), which leverages the color symmetry and
keeps the design simple; MCTS keeps the same convention.

Devices and the inference server
--------------------------------

`--device {auto,cpu,mps,cuda}` (or `[device] name` in the TOML, which the
flag overrides) selects where nets run. `auto` picks cuda → mps → cpu;
requesting an unavailable device is a hard error, never a silent fallback.

- **cpu**: every worker process keeps its own local net copy.
- **mps/cuda**: a single central *inference server* process owns the
  accelerator nets. Workers run MCTS tree logic on CPU and ship encoded
  leaf batches to the server over pipes; the server greedily drains all
  queued requests into one large net forward per slot (so GPU batch size
  scales with the number of busy workers, well past one search's
  `batch_eval_size`), masks+softmaxes priors over legal actions on device,
  and returns (priors, values). The server hosts two net slots: slot 0 =
  candidate/current net, slot 1 = best net (arena). It lives for the whole
  training run and receives fresh weights at phase boundaries, when no
  workers are connected.

The optimization phase runs on the selected device (batches are copied per
step; the replay buffer stays on CPU). Checkpoints (`latest.pt`, `best{n}.pt`)
normalize every tensor to CPU on save, so a checkpoint written on mps/cuda
resumes or serves on any device. Inference (`smartfour.infer`, the UI
worker) stays CPU-only by design.

Rules recap (see [`game.md`](game.md) and `smartfour/game.py`, a faithful
port of the UI engine): 5x5 grid, stacks up to 5 high; win by lining 4+ own
pieces along any 3D line (4 flat directions, 1 vertical, 8 rising diagonals —
runs of 5 also win); draw when both players exhaust their 32 pieces (ply 64).
The game always terminates: 64 plies < 125 cells, and piece counts of the two
players never differ by more than one.

Network input
-------------
15 channels of 5x5 (one plane per height level), all from the current
player's perspective:

- channels 0-4: current player's pieces per plane (1 = present)
- channels 5-9: opponent's pieces per plane
- channels 10-14: legality per plane — 1 at the stack top of each column
  where the current player may place (empty board ⇒ level 0), 0 elsewhere

Policy actions are indexed `a = y * 25 + x * 5 + z` (125 logits, plane-major,
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

Config files:
- `config.toml` is for actual model training and inference
- `config_small.toml` is a small network for testing only 

MCTS
----
`smartfour/mcts.py` — standard AlphaZero MCTS, configurable in the same TOML
file. Evaluation goes through an injectable `evaluator` (local net or the
central server's RemoteEvaluator):

- PUCT selection: `Q + c_puct * P * sqrt(N_parent) / (1 + N_child)`; Q is
  negated at each level because every node stores values from its own
  perspective (backprop flips the sign per level)
- dirichlet noise at the root during self-play (`alpha`, `epsilon`)
- temperature schedule: `temperature_threshold` plies with tau=1 (sample from
  visit counts), then tau=0 (argmax)
- virtual-loss batched evaluation: each pass descends up to
  `batch_eval_size` leaves (128 by default), charging a temporary loss
  along pending paths so PUCT explores different lines, then evaluates
  all of them in ONE net forward; random tie-breaking among equal UCB
  scores; one searcher serves training and inference
- terminal leaves short-circuit the net (no forward on game-over positions)

Training
--------
`smartfour/train.py` — per iteration:

1. self-play `selfplay_games` games (MCTS with root noise and temperature)
2. store `(state, pi, z)` in a replay buffer that keeps the last
   `replay_capacity_games` *whole games* (AlphaZero-style games window —
   games are evicted FIFO, never split, so no sample is older than
   `replay_capacity_games / selfplay_games` iterations); `z` is the outcome
   from each stored position's own perspective (+1 win, -1 loss, 0 draw)
3. optimize for `train_epochs` over sampled batches: policy cross-entropy +
   value MSE + L2 (AdamW); the net is switched back to train mode first so
   BatchNorm running statistics actually update (MCTS leaves it in eval mode);
   samples are augmented with a random D4 symmetry transform of the 5x5 board
   (policy permuted identically, value invariant)
4. arena: greedy MCTS vs the current best net (`eval_games` games,
   `[mcts]` `simulations` per move); games start from opening-book states
   when a book exists (see "Opening book" below), otherwise colors
   alternate from the initial board; the candidate replaces the best when
   its win ratio reaches `arena_win_ratio` (default 0.55)
5. before iteration 1 of a fresh start, when `pretrain_games > 0`
   (`[training]`), the value head is bootstrapped on random-rollout outcomes
   (see "Value bootstrap" below); resumes skip it.

Opening book (`model/openbook.json`)
------------------------------------
A greedy arena (no dirichlet noise, temperature 0) from the initial board
reaches the same game twice — once per color — so head-to-head results are
nearly binary. Following Leela Chess Zero's opening-book idea, the arena
instead starts each game pair from a pre-played state:
`tools/make_openbook.py` greedily self-plays a checkpoint (`--sims` MCTS
simulations per move, seeded tie-breaks; White's first move forced to each
of the 25 columns per iteration). Games are capped at 12 plies: a still-open
game stops there, since every ply-4 state then already has 8 later plies.
Only ply-2..4 states are harvested, and only those still open: at least
`--m` (default 8) plies from the game end — a game that ends before ply 12
keeps a ply-p state only if it has `--m` plies left, so early one-sided
wins contribute nothing. States are deduplicated exactly (bitboards + side
to move) and written atomically to `model/openbook.json` (default
`--target` 100 entries, shallowest first; entries are human-readable 5x5
arrays of bottom-to-top stack strings like `"wbbb."`, side to move inferred
from piece counts). Generation stops at the target or when an iteration
adds nothing new; falling short of the target is a warning, not an error.
At arena time book states are taken in sequence, wrapping around; each
state is played twice with roles swapped (candidate to move vs best to
move); ply statistics report both raw (incl. skipped book plies) and
played counts. With an empty or absent book the arena behaves exactly
as before.

Value bootstrap (`smartfour/pretrain.py`)
----------------------------------------
A freshly initialized value head predicts ~0 everywhere, which gives PUCT low
q-signal. This can cause trouble for low MCTS depth like in config_small.toml.

When `pretrain_games > 0`, before iteration 1: `pretrain.py` plays that many
random games and then trains `pretrain_epochs` epochs of MSE against those
soft labels. The result is a value function that already knows live threats,
so MCTS concentrates visits and the policy can learn real tactics from the very
first iteration.

Diagnostics
-----------
Every iteration appends one JSON row to `checkpoint_dir/diagnostics.jsonl`
after the checkpoint is saved (so the log covers exactly the completed
iterations) and prints a `[diag it N]` block during self-play. Key fields:

- `plies` (mean/median/hist): self-play game length.
- `depth` (mean/max): MCTS search depth. Frozen at ~2-3 is the signature of
  the breadth-first fill described above.
- `root_width`, `root_entropy`, `net_policy_entropy`: exploration health;
  a collapse into a few moves shows up as width << 25 or entropy -> 0.
- `root_value`, `value_alignment`/`value_sign_match`: value-head quality.
  Alignment near 0 means the value head has no predictive power;
  `value_sign_match` is the fraction of positions (decisive games only)
  whose value predicts the right winner (`v*z > 0`).
- `states_per_game`, `cross_game_redundancy`, `novel vs buffer`: how much
  distinct state space self-play covers each iteration.
- buffer `dup`/`distinct` and `pi entropy`/`one-hot`: replay-buffer
  duplication and how peaked the stored targets are.
- `loss pol/val`: the policy and value loss components separately.

Tactical probes (`tools/probe.py`)
---------------------------------
`tools/probe.py --checkpoint checkpoints/best{n}.pt --sims 200` tests a
checkpoint on reachable synthetic positions: immediate wins, one-ply blocks,
and 4-ply fork positions (prevent/execute), plus a vs-random win rate.
Checkpoints live in `checkpoint_dir` (`checkpoints/`):

- `latest.pt` — the only per-iteration checkpoint: net, optimizer, best net,
  replay buffer, iteration counter. Written atomically after every completed
  iteration, so it always holds the last completed one; an interrupt or
  crash discards the in-flight iteration and never touches it. Resume loads
  this file or starts fresh.
- `best{n}.pt` — slim inference snapshot of the arena-best net (weights + the
  iteration it won), one per arena promotion; the biggest n is the strongest
  model. The UI lists these (biggest first) and defaults to the biggest.
  Never used for resume.

Parallel self-play and arena
----------------------------
With `workers > 1` in `[training]`, each iteration spawns that many
processes. On cpu each worker rebuilds the net from the current weights and
evaluates locally; on mps/cuda the workers instead connect to the central
inference server (see above) and evaluate remotely — tree logic local,
forward passes centralized. Either way the worker plays its share of the
self-play games and the trainer collects exactly `eval_games` arena games.
The same virtual-loss searcher serves self-play, arena, and UI inference.

Inference (UI path, GPU-accelerated)
------------------------------------
`smartfour/infer.py` — `SmartFourAgent` loads a checkpoint and returns a move
`(x, z)` for a game state given as a `GameState` or as JSON in the UI
interchange format (`game.state_to_json`): `grid[x][z][y]` with 0/1/null,
`pieces_left`, `current`, `winner`. `simulations=0` is policy-only (no
search), which is what the UI's "disable search" mode uses.

The device comes from `--device {auto,cpu,mps,cuda}` or `$SMARTFOUR_DEVICE`
(default auto: cuda -> mps -> cpu); requesting an unavailable device is a
hard error. The net loads onto the device once.

For `simulations > 0` the agent runs the virtual-loss searcher (above):
one net forward per ~128-leaf pass, states stacked and masked on device.
Measured on an M4, single-move latency drops ~2.4-4x vs the old
sequential searcher (e.g. 2000 sims: 1237ms CPU-sequential -> 312ms MPS),
and at 400 sims it was also stronger (62-38 over 100 games).

`smartfour/worker.py` — the persistent bridge worker for the UI. It loads the
checkpoint once (on the selected device), prints `{"ready": true, ...,
"device": "..."}`, then serves one request per
line on stdin/stdout as newline-delimited JSON: requests are
`{"id": n, "state": <state_to_json>, "simulations": m}`, responses are
`{"id": n, "move": {"x", "z"}}` (or `{"move": null}` on a terminal state);
errors are reported in-band as `{"id": n, "error": "..."}` and the loop keeps
serving. The Vite plugin in `ui/plugins/` spawns it and exposes
`POST /api/think` to the browser; set `SMARTFOUR_DEVICE` in the server
environment to pin its device (the bridge forwards the environment).

Running
-------
```sh
cd model
.venv/bin/pip install -r requirements.txt   # torch (CPU) + tqdm
.venv/bin/pip install pytest                # dev only: run the test suite

.venv/bin/python -m pytest tests/        # full TDD suite (440+ tests)

.venv/bin/python -m smartfour.train --config config.toml --iterations 10
                                            # resume is automatic; 10 is a target
.venv/bin/python -m smartfour.train --config config.toml          # train forever until Ctrl-C
.venv/bin/python -m smartfour.train --config config.toml --device mps --iterations 10
                                            # explicit device (overrides [device]);
                                            # auto uses cuda > mps > cpu
.venv/bin/python -m smartfour.train --config config.toml --restart --yes
                                            # wipe checkpoints/ (confirmation prompt
                                            # without --yes) and start from iteration 1
.venv/bin/python -m smartfour.infer --checkpoint checkpoints/best{n}.pt --sims 200 --state state.json

# per-phase benchmarks (net throughput, self-play, optimize, arena)
.venv/bin/python tools/bench.py --config config.toml --device mps
```
