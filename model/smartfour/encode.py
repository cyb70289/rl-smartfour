"""State encoding — 16 x 5 x 5 input from the CURRENT player's perspective.

Channels (per model.md, plus one extension):
  0-4   current player's pieces, one plane per height level
  5-9   opponent player's pieces, one plane per height level
  10-14 legality for the current player: 1 at the stack top of each legal column
  15    constant plane: total pieces remaining / 64 (plies left to the draw cap)

Policy actions are indexed a = y * 25 + x * 5 + z (plane-major, 125 total).
"""

import torch

from .game import _HEIGHTS, BOARD_SIZE, STACK_HEIGHT, other, stack_height, WHITE

N_CHANNELS = 16
TOTAL_PIECES = 64
PLANES = BOARD_SIZE * BOARD_SIZE  # 25


def encode(state) -> torch.Tensor:
    """Encode a game state as a (16, 5, 5) float tensor."""
    cur = state.current
    t = torch.zeros((N_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32)
    # Player planes by height: t[y][x][z] = 1 where that player occupies the cell.
    own = state.white if cur == WHITE else state.black
    opp = state.black if cur == WHITE else state.white
    for x in range(BOARD_SIZE):
        for z in range(BOARD_SIZE):
            shift = x * 25 + z * 5
            own_col = (own >> shift) & 0b11111
            opp_col = (opp >> shift) & 0b11111
            for y in range(STACK_HEIGHT):
                b = 1 << y
                if own_col & b:
                    t[y][x][z] = 1.0
                elif opp_col & b:
                    t[5 + y][x][z] = 1.0
            h = _HEIGHTS[(own_col | opp_col)]
            if h < STACK_HEIGHT:
                t[10 + h][x][z] = 1.0
    remaining = state.pieces_left[cur] + state.pieces_left[other(cur)]
    t[15].fill_(remaining / TOTAL_PIECES)
    return t


def xyz_to_action(x: int, z: int, y: int) -> int:
    return y * PLANES + x * BOARD_SIZE + z


def action_to_xyz(a: int):
    y, r = divmod(a, PLANES)
    x, z = divmod(r, BOARD_SIZE)
    return x, z, y

def legal_actions(state) -> list:
    """Action indices of every legal move (one per legal column, at its top)."""
    if state.winner is not None or state.pieces_left[state.current] <= 0:
        return []
    occ = state.white | state.black
    out = []
    base = 0
    for x in range(BOARD_SIZE):
        for z in range(BOARD_SIZE):
            h = _HEIGHTS[(occ >> base) & 0b11111]
            if h < STACK_HEIGHT:
                # action = y*25 + x*5 + z (plane-major)
                out.append(h * PLANES + x * BOARD_SIZE + z)
            base += 5
    return out


def action_mask(state) -> torch.Tensor:
    """(125,) float tensor, 1.0 on legal actions."""
    mask = torch.zeros(PLANES * STACK_HEIGHT, dtype=torch.float32)  # 125
    for a in legal_actions(state):
        mask[a] = 1.0
    return mask

# --------------------------------------------------------------------- D4 group
_D4_PERMS = None  # cached output of d4_perms(); constant after first call
_PERM_TENSOR_CACHE: dict = {}  # perm tuple -> long tensor (reused across calls)

def d4_perms() -> list:
    """The 8 dihedral permutations of the 25 column cells.

    perm[j] = old linear index (x*5+z) that lands at new index j. Applying a
    perm to a plane: out[j] = inp[perm[j]]. Cached after first call.
    """
    global _D4_PERMS
    if _D4_PERMS is None:
        _D4_PERMS = _compute_d4_perms()
    return _D4_PERMS

def _compute_d4_perms() -> list:
    transforms = [
        lambda x, z: (x, z),
        lambda x, z: (z, 4 - x),      # rot90
        lambda x, z: (4 - x, 4 - z),  # rot180
        lambda x, z: (4 - z, x),      # rot270
        lambda x, z: (4 - x, z),      # reflect across vertical axis
        lambda x, z: (x, 4 - z),      # reflect across horizontal axis
        lambda x, z: (z, x),          # reflect across main diagonal
        lambda x, z: (4 - z, 4 - x),  # reflect across anti-diagonal
    ]
    perms = []
    for f in transforms:
        perm = [0] * PLANES
        for x in range(BOARD_SIZE):
            for z in range(BOARD_SIZE):
                nx, nz = f(x, z)
                perm[nx * BOARD_SIZE + nz] = x * BOARD_SIZE + z
        perms.append(perm)
    return perms

def _perm_tensor(perm) -> torch.Tensor:
    """Cached (perm tuple -> long tensor) so repeated D4 apply reuses tensors."""
    key = tuple(perm)
    t = _PERM_TENSOR_CACHE.get(key)
    if t is None:
        t = torch.tensor(perm, dtype=torch.long)
        _PERM_TENSOR_CACHE[key] = t
    return t


def apply_d4(t: torch.Tensor, perm) -> torch.Tensor:
    """Apply a D4 permutation to the (x, z) plane of a (C, 5, 5) tensor."""
    c, h, w = t.shape
    flat = t.reshape(c, h * w)
    return flat.index_select(1, _perm_tensor(perm)).reshape(c, h, w)


def apply_d4_policy(pi: torch.Tensor, perm) -> torch.Tensor:
    """Apply a D4 permutation to a (125,) policy (5 planes of 25 cells)."""
    planes = pi.reshape(5, PLANES)
    return planes.index_select(1, _perm_tensor(perm)).reshape(-1)
