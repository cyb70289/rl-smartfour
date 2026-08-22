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

/** A player slot: a human, or the model loaded from a checkpoint file name. */
export type PlayerSlot = { kind: 'human' } | { kind: 'model'; checkpoint: string };

/**
 * Derived game mode:
 * - `person` — both players human,
 * - `machine` — one human, one model,
 * - `autoplay` — two models (model-vs-model auto play).
 */
export type Mode = 'person' | 'machine' | 'autoplay';

/** Model search settings. `effort` = MCTS steps; 0 = policy only (no search). */
export interface ThinkSettings {
  effort: number;
}

export const BOARD_SIZE = 5;
export const STACK_HEIGHT = 5;
export const DEFAULT_PIECES = 32;

export function otherPlayer(p: Player): Player {
  return p === 'white' ? 'black' : 'white';
}

/** The mode implied by the two player slots. */
export function modeOf(white: PlayerSlot, black: PlayerSlot): Mode {
  const whiteModel = white.kind === 'model';
  const blackModel = black.kind === 'model';
  return whiteModel && blackModel ? 'autoplay' : whiteModel || blackModel ? 'machine' : 'person';
}

/** True when `player` is played by a model in this game. */
export function isModel(slots: { white: PlayerSlot; black: PlayerSlot }, player: Player): boolean {
  return (player === 'white' ? slots.white : slots.black).kind === 'model';
}

/** The human player's color, or null in auto-play (both sides are models). */
export function humanColorOf(slots: { white: PlayerSlot; black: PlayerSlot }): Player | null {
  if (slots.white.kind === 'human') return 'white';
  if (slots.black.kind === 'human') return 'black';
  return null;
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
  /** True while a model move is owed (machine and autoplay modes). */
  machineThinking: boolean;
  /** True while a model think is actually in flight; an owed move that has
   * not started (paused auto play, the between-move gap) stays false. */
  thinking: boolean;
  /** True while model-vs-model auto play is running (autoplay mode only). */
  autoplay: boolean;
  /** The white player slot. */
  white: PlayerSlot;
  /** The black player slot. */
  black: PlayerSlot;
  settings: ThinkSettings;
}
