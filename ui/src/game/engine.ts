import { createInitialState, applyMove } from './rules';
import type { GameState, Mode, Move, PlayerSlot, ThinkSettings } from './types';
import { DEFAULT_PIECES, modeOf, isModel } from './types';

export type { Mode };

export interface GameConfig {
  white: PlayerSlot;
  black: PlayerSlot;
  settings: ThinkSettings;
}

export type GameAction =
  | { type: 'move'; move: Move }
  | { type: 'machine-move'; move: Move }
  | { type: 'revert' }
  | { type: 'reset'; config: GameConfig };

export class IllegalActionError extends Error {}

/** The mode implied by a config's player slots. */
export function modeOfConfig(config: GameConfig): Mode {
  return modeOf(config.white, config.black);
}

/** A fresh game. A model owes the first move when white is a model. */
export function newGame(config: GameConfig): GameState {
  const s = createInitialState(DEFAULT_PIECES, config.white, config.black, config.settings);
  s.machineThinking = isModel(s, s.current);
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
  if (isModel(state, state.current)) throw new IllegalActionError('model must move');
  const next = applyMove(state, move);
  if (next.winner === null && isModel(next, next.current)) next.machineThinking = true;
  return next;
}

function machineMove(state: GameState, move: Move): GameState {
  if (!isModel(state, state.current)) throw new IllegalActionError('not a model\'s turn');
  if (state.winner !== null) throw new IllegalActionError('game is over');
  const next = applyMove(state, move);
  // Auto-play hands the turn to the other model, which owes a move right away.
  next.machineThinking = next.winner === null && isModel(next, next.current);
  return next;
}

function revert(state: GameState): GameState {
  if (state.machineThinking) throw new IllegalActionError('machine is thinking');
  if (state.history.length === 0) throw new IllegalActionError('nothing to revert');
  // A finished game (win/draw) is revertable too; mid-game the one-move
  // revert window applies.
  if (!state.revertAvailable && state.winner === null) {
    throw new IllegalActionError('nothing to revert');
  }
  // Person mode: undo one move. Machine/autoplay: undo the last two together
  // (model + human, or the two models).
  const pop = modeOf(state.white, state.black) === 'person' ? 1 : Math.min(2, state.history.length);
  const kept = state.history.slice(0, state.history.length - pop);

  let s = createInitialState(state.piecesPerPlayer, state.white, state.black, state.settings);
  for (const h of kept) s = applyMove(s, { x: h.x, z: h.z });
  // Reverting a finished game can hand the turn back to a model
  // (e.g. the human's winning move is undone); the controller then thinks.
  s.machineThinking = s.winner === null && isModel(s, s.current);
  s.revertAvailable = false;
  return s;
}
