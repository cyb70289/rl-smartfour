import { createInitialState, applyMove } from './rules';
import type { GameState, Mode, Move, Player, ThinkSettings } from './types';
import { DEFAULT_PIECES, otherPlayer } from './types';

export type { Mode };

export interface GameConfig {
  mode: Mode;
  humanColor: Player;
  settings: ThinkSettings;
}

export type GameAction =
  | { type: 'move'; move: Move }
  | { type: 'machine-move'; move: Move }
  | { type: 'revert' }
  | { type: 'reset'; config: GameConfig };

export class IllegalActionError extends Error {}

export function machineColorOf(state: GameState): Player {
  return otherPlayer(state.humanColor);
}

/** A fresh game. In machine mode, the machine owes a move when it is white. */
export function newGame(config: GameConfig): GameState {
  const s = createInitialState(DEFAULT_PIECES, config.mode, config.humanColor, config.settings);
  if (config.mode === 'machine' && otherPlayer(config.humanColor) === 'white') {
    s.machineThinking = true;
  }
  return s;
}

export function reduce(state: GameState, action: GameAction): GameState {
  switch (action.type) {
    case 'move':
      return humanMove(state, action.move);
    case 'machine-move':
      return machineMove(state, action.move);
    case 'revert':
      return revert(state);
    case 'reset':
      return newGame(action.config);
  }
}

function humanMove(state: GameState, move: Move): GameState {
  if (state.winner !== null) throw new IllegalActionError('game is over');
  if (state.machineThinking) throw new IllegalActionError('machine is thinking');
  if (state.mode === 'machine' && state.current === machineColorOf(state)) {
    throw new IllegalActionError('machine must move');
  }
  const next = applyMove(state, move);
  if (state.mode === 'machine' && next.winner === null && next.current === machineColorOf(next)) {
    next.machineThinking = true;
  }
  return next;
}

function machineMove(state: GameState, move: Move): GameState {
  if (state.mode !== 'machine') throw new IllegalActionError('not machine mode');
  if (state.winner !== null) throw new IllegalActionError('game is over');
  if (state.current !== machineColorOf(state)) throw new IllegalActionError('not the machine\'s turn');
  const next = applyMove(state, move);
  next.machineThinking = false;
  return next;
}

function revert(state: GameState): GameState {
  if (state.winner !== null) throw new IllegalActionError('game is over');
  if (state.machineThinking) throw new IllegalActionError('machine is thinking');
  if (!state.revertAvailable || state.history.length === 0) {
    throw new IllegalActionError('nothing to revert');
  }
  // Person mode: undo one move. Machine mode: undo machine + human move together.
  const pop = state.mode === 'machine' ? Math.min(2, state.history.length) : 1;
  const kept = state.history.slice(0, state.history.length - pop);

  let s = createInitialState(state.piecesPerPlayer, state.mode, state.humanColor, state.settings);
  for (const h of kept) s = applyMove(s, { x: h.x, z: h.z });
  s.machineThinking = false;
  s.revertAvailable = false;
  return s;
}
