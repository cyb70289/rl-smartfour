import { BOARD_SIZE, STACK_HEIGHT, DEFAULT_PIECES, otherPlayer } from './types';
import type { GameState, Move, Player, PlayerSlot, PlacedPiece, ThinkSettings, Winner } from './types';

export class IllegalMoveError extends Error {}

/**
 * Canonical line directions as [xDelta, zDelta, yDelta] with dy >= 0:
 * 4 flat (same plane), 1 vertical, 8 rising diagonals ("stepping up").
 * Reverses are covered because wins are counted in both directions from a cell.
 */
const DIRS: ReadonlyArray<readonly [number, number, number]> = [
  [1, 0, 0],
  [0, 0, 1],
  [1, 0, 1],
  [1, 0, -1],
  [0, 1, 0],
  [1, 1, 0],
  [0, 1, 1],
  [1, 1, 1],
  [1, 1, -1],
  [-1, 1, 0],
  [0, 1, -1],
  [-1, 1, -1],
  [-1, 1, 1],
];

export function createInitialState(
  piecesPerPlayer: number = DEFAULT_PIECES,
  white: PlayerSlot = { kind: 'human' },
  black: PlayerSlot = { kind: 'human' },
  settings: ThinkSettings = { effort: 500 },
): GameState {
  const grid: (Player | null)[][][] = [];
  for (let x = 0; x < BOARD_SIZE; x++) {
    const col: (Player | null)[][] = [];
    for (let z = 0; z < BOARD_SIZE; z++) {
      col.push(new Array<Player | null>(STACK_HEIGHT).fill(null));
    }
    grid.push(col);
  }
  return {
    grid,
    piecesLeft: { white: piecesPerPlayer, black: piecesPerPlayer },
    piecesPerPlayer,
    current: 'white',
    history: [],
    lastPlaced: null,
    winner: null,
    winningCells: null,
    revertAvailable: false,
    machineThinking: false,
    thinking: false,
    autoplay: false,
    white,
    black,
    settings,
  };
}

export function cellAt(grid: (Player | null)[][][], x: number, z: number, y: number): Player | null {
  const col = grid[x];
  if (!col) return null;
  const stack = col[z];
  if (!stack) return null;
  return stack[y] ?? null;
}

export function stackHeight(grid: (Player | null)[][][], x: number, z: number): number {
  const col = grid[x];
  if (!col) return 0;
  const stack = col[z];
  if (!stack) return 0;
  let h = 0;
  while (h < STACK_HEIGHT && stack[h] != null) h++;
  return h;
}

export function isLegal(state: GameState, move: Move): boolean {
  if (state.winner !== null) return false;
  if (!Number.isInteger(move.x) || !Number.isInteger(move.z)) return false;
  if (move.x < 0 || move.x >= BOARD_SIZE || move.z < 0 || move.z >= BOARD_SIZE) return false;
  if (state.piecesLeft[state.current] <= 0) return false;
  return stackHeight(state.grid, move.x, move.z) < STACK_HEIGHT;
}

export function legalMoves(state: GameState): Move[] {
  const moves: Move[] = [];
  for (let x = 0; x < BOARD_SIZE; x++) {
    for (let z = 0; z < BOARD_SIZE; z++) {
      const move = { x, z };
      if (isLegal(state, move)) moves.push(move);
    }
  }
  return moves;
}

/**
 * Returns the full same-color run through (x, z, y) if it contains 4+ cells,
 * or null. Runs are counted in both directions along each canonical line.
 */
export function findWinRun(
  grid: (Player | null)[][][],
  x: number,
  z: number,
  y: number,
  player: Player,
): PlacedPiece[] | null {
  for (const [dx, dz, dy] of DIRS) {
    const run: PlacedPiece[] = [{ x, z, y, player }];
    let cx = x + dx;
    let cz = z + dz;
    let cy = y + dy;
    while (
      cx >= 0 && cx < BOARD_SIZE && cz >= 0 && cz < BOARD_SIZE && cy >= 0 && cy < STACK_HEIGHT &&
      cellAt(grid, cx, cz, cy) === player
    ) {
      run.push({ x: cx, z: cz, y: cy, player });
      cx += dx;
      cz += dz;
      cy += dy;
    }
    cx = x - dx;
    cz = z - dz;
    cy = y - dy;
    while (
      cx >= 0 && cx < BOARD_SIZE && cz >= 0 && cz < BOARD_SIZE && cy >= 0 && cy < STACK_HEIGHT &&
      cellAt(grid, cx, cz, cy) === player
    ) {
      run.unshift({ x: cx, z: cz, y: cy, player });
      cx -= dx;
      cz -= dz;
      cy -= dy;
    }
    if (run.length >= 4) return run;
  }
  return null;
}

/** Applies `move` for the player whose turn it is. Throws `IllegalMoveError` on illegal moves. */
export function applyMove(state: GameState, move: Move): GameState {
  if (!isLegal(state, move)) {
    throw new IllegalMoveError(`illegal move (${move.x}, ${move.z})`);
  }
  const y = stackHeight(state.grid, move.x, move.z);
  const player = state.current;

  const grid = state.grid.map((col) => col.map((stack) => [...stack]));
  grid[move.x]![move.z]![y] = player;

  const piecesLeft: Record<Player, number> = {
    ...state.piecesLeft,
    [player]: state.piecesLeft[player] - 1,
  };

  const winningCells = findWinRun(grid, move.x, move.z, y, player);
  const winner: Winner = winningCells
    ? player
    : piecesLeft.white === 0 && piecesLeft.black === 0
      ? 'draw'
      : null;

  return {
    ...state,
    grid,
    piecesLeft,
    current: otherPlayer(player),
    history: [...state.history, { x: move.x, z: move.z, player }],
    lastPlaced: { x: move.x, z: move.z, y, player },
    winner,
    winningCells,
    revertAvailable: winner === null,
  };
}
