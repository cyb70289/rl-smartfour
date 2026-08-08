import { legalMoves } from './rules';
import type { GameState, Move, ThinkSettings } from './types';

/**
 * The interface a real model must implement. The board is handed over as an
 * immutable game-state snapshot; the model derives its own input encoding.
 * `settings.disabled` = policy only (no MCTS search), `settings.effort` = MCTS
 * search steps. `signal` is aborted when the move is no longer wanted (e.g. a
 * new game was started); implementations SHOULD abort and reject promptly.
 */
export interface MachinePlayer {
  readonly name: string;
  think(state: Readonly<GameState>, settings: ThinkSettings, signal?: AbortSignal): Promise<Move>;
}

/**
 * Dummy machine player: uniform random among legal moves, with a short delay so
 * the UI's "thinking" state is actually observable. Replaced by the real model.
 */
export class RandomMachinePlayer implements MachinePlayer {
  readonly name = 'random';

  async think(state: Readonly<GameState>, _settings: ThinkSettings, signal?: AbortSignal): Promise<Move> {
    if (signal?.aborted) throw new DOMException('aborted', 'AbortError');
    const moves = legalMoves(state);
    if (moves.length === 0) throw new Error('no legal moves');
    const move = moves[Math.floor(Math.random() * moves.length)]!;
    await new Promise((resolve) => setTimeout(resolve, 120));
    if (signal?.aborted) throw new DOMException('aborted', 'AbortError');
    return move;
  }
}
