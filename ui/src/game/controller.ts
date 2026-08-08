import { reduce, newGame } from './engine';
import type { GameConfig } from './engine';
import type { GameState, Move } from './types';
import type { MachinePlayer } from './machine';

/**
 * Owns the game state and the asynchronous machine-turn orchestration:
 * - kicks off `machine.think` whenever the state owes a machine move,
 * - guards against stale machine results (generation counter incremented on
 *   every reset, plus an AbortSignal for the in-flight think),
 * - exposes `machineThinking` through the state so the UI can lock controls.
 */
export class GameController {
  state: GameState;
  private generation = 0;
  private abortCtrl: AbortController | null = null;
  private listeners = new Set<() => void>();

  constructor(
    private machine: MachinePlayer,
    initialConfig: GameConfig,
    private onError?: (err: unknown) => void,
  ) {
    this.state = newGame(initialConfig);
    this.kickMachine();
  }

  subscribe(fn: () => void): () => void {
    this.listeners.add(fn);
    return () => {
      this.listeners.delete(fn);
    };
  }

  humanMove(move: Move): void {
    if (this.state.machineThinking) throw new Error('machine is thinking');
    this.setState(reduce(this.state, { type: 'move', move }));
    this.kickMachine();
  }

  revert(): void {
    this.setState(reduce(this.state, { type: 'revert' }));
  }

  reset(config: GameConfig): void {
    this.generation++;
    this.abortCtrl?.abort();
    this.abortCtrl = null;
    this.setState(newGame(config));
    this.kickMachine();
  }

  private setState(next: GameState): void {
    this.state = next;
    for (const fn of this.listeners) fn();
  }

  private kickMachine(): void {
    if (!this.state.machineThinking || this.state.winner !== null) return;
    const gen = this.generation;
    const ac = new AbortController();
    this.abortCtrl = ac;
    const snapshot = this.state;
    this.machine.think(snapshot, snapshot.settings, ac.signal).then(
      (move) => {
        if (gen !== this.generation) return;
        if (this.state.machineThinking && this.state.winner === null) {
          this.setState(reduce(this.state, { type: 'machine-move', move }));
        }
      },
      (err) => {
        if (gen !== this.generation) return;
        this.setState({ ...this.state, machineThinking: false });
        this.onError?.(err);
      },
    );
  }
}
