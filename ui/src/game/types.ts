export type Player = 'white' | 'black';

export interface Move {
  x: number;
  z: number;
}

export interface PlacedPiece {
  x: number;
  z: number;
  y: number;
  player: Player;
}

export type Winner = Player | 'draw' | null;

export type Mode = 'person' | 'machine';

/** Machine search settings. `effort` = MCTS steps; 0 = policy only (no search). */
export interface ThinkSettings {
  effort: number;
}

export const BOARD_SIZE = 5;
export const STACK_HEIGHT = 5;
export const DEFAULT_PIECES = 32;

export function otherPlayer(p: Player): Player {
  return p === 'white' ? 'black' : 'white';
}

export interface GameState {
  /** grid[x][z][y] — the piece at column (x, z), height y, or null. */
  grid: (Player | null)[][][];
  piecesLeft: Record<Player, number>;
  piecesPerPlayer: number;
  /** Whose turn it is to place a piece. */
  current: Player;
  /** Full move log; the only source of truth for reverting. */
  history: Array<Move & { player: Player }>;
  lastPlaced: PlacedPiece | null;
  winner: Winner;
  /** Full same-color run that won the game (for highlighting), or null. */
  winningCells: PlacedPiece[] | null;
  /** True right after a move; consumed by a revert. */
  revertAvailable: boolean;
  /** True while a machine move is owed (machine mode only). */
  machineThinking: boolean;
  mode: Mode;
  /** The human player's color; ignored in person mode. */
  humanColor: Player;
  settings: ThinkSettings;
}
