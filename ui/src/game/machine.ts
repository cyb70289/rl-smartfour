import { isLegal, legalMoves } from './rules';
import type { GameState, Move, Player, ThinkSettings, Winner } from './types';

/**
 * The interface a real model must implement. The board is handed over as an
 * immutable game-state snapshot; the model derives its own input encoding.
 * `settings.effort` = MCTS search steps (0 = policy only). `signal` is
 * aborted when the move is no longer wanted (e.g. a new game was started);
 * implementations SHOULD abort and reject promptly.
 */
export interface MachinePlayer {
  readonly name: string;
  think(state: Readonly<GameState>, settings: ThinkSettings, signal?: AbortSignal): Promise<Move>;
}

/**
 * Dummy machine player: uniform random among legal moves, with a short delay so
 * the UI's "thinking" state is actually observable. Kept as a reference
 * implementation and test fixture; the real machine player is
 * `ModelMachinePlayer`.
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

/**
 * Game state in the model's interchange format (model/smartfour/game.py
 * `state_from_json`): grid values are 0 (white) / 1 (black) / null, plus
 * pieces_left, current and winner (null | color | "draw").
 */
export interface ModelRequestState {
  grid: (0 | 1 | null)[][][];
  pieces_left: { white: number; black: number };
  current: Player;
  winner: Winner;
}

export function stateToJson(state: Readonly<GameState>): ModelRequestState {
  return {
    grid: state.grid.map((cols) => cols.map((stack) => stack.map((p) => (p === 'white' ? 0 : p === 'black' ? 1 : null)))),
    pieces_left: { white: state.piecesLeft.white, black: state.piecesLeft.black },
    current: state.current,
    winner: state.winner,
  };
}

/**
 * MCTS search steps for the given settings. Policy-only (simulations=0) when
 * effort < 1: the UI slider's minimum is 0, and running MCTS with 0
 * simulations is not the same as a policy-only argmax.
 */
export function simulationsOf(settings: ThinkSettings): number {
  if (!Number.isFinite(settings.effort) || settings.effort < 1) return 0;
  return Math.floor(settings.effort);
}

/**
 * Machine player backed by the AlphaZero model, reached through the Vite
 * bridge at `endpoint` (POST /api/think by default, served by
 * plugins/model-bridge.ts). Rejects promptly on abort, surfaces bridge/model
 * errors, and validates the returned move against the snapshot so a broken
 * checkpoint can never wedge the controller.
 */
export class ModelMachinePlayer implements MachinePlayer {
  readonly name: string;

  constructor(
    /** Checkpoint file name under model/checkpoints, e.g. "best3.pt". */
    readonly checkpoint: string,
    private endpoint = '/api/think',
  ) {
    this.name = `model:${checkpoint}`;
  }

  async think(state: Readonly<GameState>, settings: ThinkSettings, signal?: AbortSignal): Promise<Move> {
    if (signal?.aborted) throw new DOMException('aborted', 'AbortError');
    let res: Response;
    try {
      res = await fetch(this.endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          state: stateToJson(state),
          simulations: simulationsOf(settings),
          checkpoint: this.checkpoint,
        }),
        signal,
      });
    } catch (err) {
      if (signal?.aborted) throw new DOMException('aborted', 'AbortError');
      throw err;
    }
    const move = await parseMove(res);
    if (move === null) throw new Error('model returned no move (game over)');
    if (!isLegal(state, move)) throw new Error(`model returned illegal move (${move.x}, ${move.z})`);
    return move;
  }
}

/** Parses the bridge response into a Move, or null for a terminal "no move". */
async function parseMove(res: Response): Promise<Move | null> {
  let body: unknown;
  try {
    body = await res.json();
  } catch {
    throw new Error(`model bridge returned a non-JSON response (HTTP ${res.status})`);
  }
  if (typeof body !== 'object' || body === null) throw new Error('malformed model response');
  if ('error' in body) {
    const err = body.error;
    throw new Error(typeof err === 'string' ? err : 'malformed model response');
  }
  if (!res.ok) throw new Error(`model bridge error (HTTP ${res.status})`);
  if (!('move' in body)) throw new Error('malformed model response');
  const move = body.move;
  if (move === null) return null;
  if (typeof move !== 'object' || move === null || !('x' in move) || !('z' in move)) {
    throw new Error('malformed model response');
  }
  const { x, z } = move;
  if (typeof x !== 'number' || typeof z !== 'number') throw new Error('malformed model response');
  return { x, z };
}
