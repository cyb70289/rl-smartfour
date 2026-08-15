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

- **cpu**: every worker process keeps its own local net copy (8 parallel
  CPU inference streams beat one shared stream — threads do not scale this
  workload on Apple silicon).
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
step; the replay buffer stays on CPU). Checkpoints (`latest.pt`, `best.pt`)
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
16 channels of 5x5 (one plane per height level), all from the current
player's perspective:

- channels 0-4: current player's pieces per plane (1 = present)
- channels 5-9: opponent's pieces per plane
- channels 10-14: legality per plane — 1 at the stack top of each column
  where the current player may place (empty board ⇒ level 0), 0 elsewhere
- channel 15: constant plane = total pieces remaining / 64, i.e. how close
  the game is to the ply-64 draw cap

The 15-channel layout follows the original draft; the 16th channel (index 15)
is an extension: the network cannot otherwise tell how many plies remain, and
the *sum* of both players' remaining pieces is perspective-invariant and exact.

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

Defaults (`config.toml`): 16 blocks, 128 base channels, 500 sims, batch_eval
128. `config_small.toml` is a reduced profile (3 blocks, 32 channels) for
quick CPU proof runs.

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
   value MSE + L2 (AdamW); the net is switched back to train mode first so
   BatchNorm running statistics actually update (MCTS leaves it in eval mode);
   samples are augmented with a random D4 symmetry transform of the 5x5 board
   (policy permuted identically, value invariant)
4. arena: greedy MCTS vs the current best net (alternating colors,
   `eval_games` games, `[mcts]` `simulations` per move); the candidate
   replaces the best when its win ratio reaches `arena_win_ratio`
   (default 0.55)
5. before iteration 1 of a fresh start, when `pretrain_games > 0`
   (`[training]`), the value head is bootstrapped on random-rollout outcomes
   (see "Value bootstrap" below); resumes skip it.

Value bootstrap (`smartfour/pretrain.py`)
----------------------------------------
A freshly initialized value head predicts ~0 everywhere, which gives PUCT no
q-signal: unvisited children always outscore visited ones, so every node's
~25 children fill breadth-first before any descends, and the search freezes
at ~3 plies of depth at ANY simulation budget (depth ~ log_25(sims)). Deep
tactics (forks, 2-ply threats) then never enter the training data and the
value never improves — a self-reinforcing collapse into short races.

`pretrain_games > 0` breaks the cycle once, before iteration 1: `pretrain.py`
plays that many random games and labels the states in the last `tail_plies`
plies of each game with the average of `rollouts` (=20) random completions.
The value head (plus the shared trunk; the policy head's weights are frozen)
then trains `pretrain_epochs` epochs of MSE against those soft labels. The
result is a value function that already knows live threats, so MCTS
concentrates visits and the policy can learn real tactics from the very
first iteration.

Performance: rollout collection is pure-Python game play, so it parallelizes
across `workers` processes (independent sub-seed streams — labels are
distribution-equal to a sequential run, not bit-equal); the training loop
runs on the selected device (samples move once, batches slice on-device,
D4 augmentation stays the per-batch CPU transform). At the default
`pretrain_games = 20000` / `pretrain_epochs = 64` this is roughly
collection ~1 min + training ~20 min on an M4 (MPS) vs ~1.8 h all-CPU
before.

Diagnostics
-----------
Every iteration appends one JSON row to `checkpoint_dir/diagnostics.jsonl`
and prints a `[diag it N]` block. Key fields (see
`smartfour/diagnostics.py` and `tools/analyze_diag.py`):

- `plies` (mean/median/hist): self-play game length. At 100 sims with a
  random net this starts ~17 and collapses to ~12 as the value/policy race;
  the ply histogram shows games ending at 7-11 (fast races).
- `depth` (mean/max): MCTS search depth. Frozen at ~2-3 is the signature of
  the breadth-first fill described above.
- `root_width`, `root_entropy`, `net_policy_entropy`: exploration health;
  a collapse into a few moves shows up as width << 25 or entropy -> 0.
- `root_value` and `value_alignment`/`value_calibration`: value-head quality.
  Alignment near 0 means the value head has no predictive power.
- `states_per_game`, `cross_game_redundancy`, `novel vs buffer`: how much
  distinct state space self-play covers each iteration.
- buffer `dup`/`distinct` and `pi entropy`/`one-hot`: replay-buffer
  duplication and how peaked the stored targets are.
- `loss pol/val`: the policy and value loss components separately.

Tactical probes (`tools/probe.py`)
---------------------------------
`tools/probe.py --checkpoint checkpoints/best.pt --sims 200` tests a
checkpoint on reachable synthetic positions: immediate wins, one-ply blocks,
and 4-ply fork positions (prevent/execute), plus a vs-random win rate.
Checkpoints live in `checkpoint_dir` (`checkpoints/`):

- `latest.pt` — the only per-iteration checkpoint: net, optimizer, best net,
  replay buffer, iteration counter. Written atomically after every completed
  iteration, so it always holds the last completed one; an interrupt or
  crash discards the in-flight iteration and never touches it. Resume loads
  this file or starts fresh.
- `best.pt` — slim inference snapshot of the arena-best net (weights + the
  iteration it won). The UI loads this. Never used for resume.

Parallel self-play and arena
----------------------------
With `workers > 1` in `[training]`, each iteration spawns that many
processes. On cpu each worker rebuilds the net from the current weights and
evaluates locally; on mps/cuda the workers instead connect to the central
inference server (see above) and evaluate remotely — tree logic local,
forward passes centralized. Either way the worker plays its share of the
self-play games and ships the samples back over a queue, and the trainer
collects exactly `selfplay_games` games. The arena parallelizes the same way
with the same `workers` count: slot 0 = candidate, slot 1 = best, colors
alternate by global game index, per-game results ship over a queue, and the
trainer collects exactly `eval_games` games. Inference remains
single-process and CPU-only.

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
.venv/bin/python -m smartfour.infer --checkpoint checkpoints/best.pt --sims 200 --state state.json

# per-phase benchmarks (net throughput, self-play, optimize, arena)
.venv/bin/python tools/bench.py --config config.toml --device mps
```
