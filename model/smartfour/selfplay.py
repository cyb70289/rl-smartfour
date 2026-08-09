"""Self-play: one game = MCTS at every ply + outcome labeling.

Returns (samples, winner): samples are (state_tensor, pi, player, z) tuples
where z is the game outcome from `player`'s perspective, winner is
WHITE / BLACK / 'draw'.
"""

from .encode import action_to_xyz, encode
from .game import BLACK, WHITE, GameState, apply_move, initial_state, is_terminal, terminal_value
from .mcts import MCTS


def play_game(net, mcts_cfg, temperature_threshold: int, start_state: GameState | None = None):
    """Play one self-play game with dirichlet noise and temperature scheduling.

    temperature_threshold: plies below it use tau=1 (sample from visit
    counts), the rest use tau=0 (argmax).
    """
    state = start_state if start_state is not None else initial_state()
    mcts = MCTS(net, mcts_cfg)
    samples = []
    while not is_terminal(state):
        ply = len(samples)
        temperature = 1.0 if ply < temperature_threshold else 0.0
        pi, chosen, _ = mcts.root_policy(state, root_noise=True, temperature=temperature)
        samples.append((encode(state), pi, state.current, 0.0))
        x, z, _y = action_to_xyz(chosen)
        state = apply_move(state, x, z)

    # Outcome from white's perspective, then flip per stored player.
    winner = state.winner
    z_white = 1.0 if winner == WHITE else (-1.0 if winner == BLACK else 0.0)
    labeled = []
    for s, pi, player, _ in samples:
        z = z_white if player == WHITE else -z_white
        labeled.append((s, pi, player, z))
    return labeled, winner
