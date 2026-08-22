import { reduce, newGame } from './engine';
import type { GameConfig } from './engine';
import type { GameState, Move, Player, ThinkSettings } from './types';
import { modeOf } from './types';
import type { MachinePlayer } from './machine';

/** The model player per color; null = that side is human. */
export interface PlayerMachines {
  white: MachinePlayer | null;
  black: MachinePlayer | null;
}

const DEFAULT_AUTOPLAY_GAP_MS = 2000;

/** Handle for the scheduled auto play timer. */
type TimerId = ReturnType<typeof setTimeout>;

/** Controller tuning, overridable in tests for deterministic timing. */
export interface ControllerOptions {
  /** Minimum gap between auto play think starts (default 2000ms). */
  autoplayGapMs?: number;
  /** Clock for measuring think time (default performance.now). */
  now?: () => number;
}

/**
 * Owns the game state and the asynchronous model-turn orchestration:
 * - kicks off `think` on the model player of the color that owes a move
 *   whenever the state owes one (machine and autoplay modes),
 * - guards against stale model results (generation counter incremented on
 *   every reset/abort, plus an AbortSignal for the in-flight think),
 * - runs the model-vs-model auto play loop: after a move lands, the next
 *   think starts no sooner than 2s after the previous one started; if the
 *   think itself took longer, no extra delay is added.
 */
export class GameController {
  /** Assigned in the constructor via reset(). */
  state!: GameState;
  private generation = 0;
  private abortCtrl: AbortController | null = null;
  private stepTimer: TimerId | null = null;
  private thinkStart = 0;
  private listeners = new Set<() => void>();

  constructor(
    private players: PlayerMachines,
    initialConfig: GameConfig,
    private onError?: (err: unknown) => void,
    private options: ControllerOptions = {},
  ) {
    this.reset(initialConfig, players);
  }

  private get autoplayGapMs(): number {
    return this.options.autoplayGapMs ?? DEFAULT_AUTOPLAY_GAP_MS;
  }

  private now(): number {
    return this.options.now ? this.options.now() : performance.now();
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
    // Reverting a finished game can hand the turn to a model.
    this.kickMachine();
  }

  /**
   * Starts or resumes model-vs-model auto play. When the game is over, starts
   * a fresh game first (Play is the only restart path in auto play). No-op in
   * person/machine modes.
   */
  play(): void {
    if (modeOf(this.state.white, this.state.black) !== 'autoplay') return;
    if (this.state.winner !== null) {
      this.reset(
        { white: this.state.white, black: this.state.black, settings: this.state.settings },
        this.players,
      );
      this.setState({ ...this.state, autoplay: true });
      this.kickMachine();
      return;
    }
    if (this.state.autoplay) return; // already running
    this.setState({ ...this.state, autoplay: true, machineThinking: true });
    this.kickMachine();
  }

  /** Stops auto play and discards any in-flight think immediately. */
  pause(): void {
    if (modeOf(this.state.white, this.state.black) !== 'autoplay') return;
    if (!this.state.autoplay && !this.state.machineThinking) return;
    this.abortInFlight();
    this.setState({ ...this.state, autoplay: false, machineThinking: false });
  }

  /**
   * Plays exactly one move: stops auto play, then starts a single think (a
   * fresh game when the previous one is over).
   */
  step(): void {
    if (modeOf(this.state.white, this.state.black) !== 'autoplay') return;
    this.abortInFlight();
    if (this.state.winner !== null) {
      this.reset(
        { white: this.state.white, black: this.state.black, settings: this.state.settings },
        this.players,
      );
    }
    this.setState({ ...this.state, autoplay: false, machineThinking: true });
    this.startThink();
  }

  /**
   * Applies new think settings without restarting the game. If a model is
   * mid-think, the in-flight search is aborted and restarted with the new
   * settings so the change takes effect immediately.
   */
  updateSettings(settings: ThinkSettings): void {
    if (this.state.settings.effort === settings.effort) return;
    if (this.state.machineThinking) this.abortInFlight();
    this.setState({ ...this.state, settings });
    this.kickMachine();
  }

  reset(config: GameConfig, players: PlayerMachines): void {
    this.abortInFlight();
    this.players = players;
    const s = newGame(config);
    s.autoplay = false;
    this.setState(s);
    this.kickMachine();
  }

  private abortInFlight(): void {
    this.generation++;
    this.abortCtrl?.abort();
    this.abortCtrl = null;
    if (this.stepTimer !== null) {
      clearTimeout(this.stepTimer);
      this.stepTimer = null;
    }
  }

  private setState(next: GameState): void {
    this.state = next;
    for (const fn of this.listeners) fn();
  }

  private kickMachine(): void {
    // A fresh auto-play game stays paused (Play starts it); machine-mode
    // games kick immediately whenever a model owes a move.
    if (modeOf(this.state.white, this.state.black) === 'autoplay' && !this.state.autoplay) return;
    if (!this.state.machineThinking || this.state.winner !== null) return;
    this.startThink();
  }

  private startThink(): void {
    const machine = this.players[this.state.current];
    if (!machine) {
      // machineThinking implies the current side is a model; a missing
      // player means the controller was wired wrong — fail loudly.
      this.setState({ ...this.state, machineThinking: false, autoplay: false });
      this.onError?.(new Error('no model player for the current side'));
      return;
    }
    const gen = this.generation;
    const ac = new AbortController();
    this.abortCtrl = ac;
    this.thinkStart = this.now();
    const snapshot = this.state;
    machine.think(snapshot, snapshot.settings, ac.signal).then(
      (move) => {
        if (gen !== this.generation) return;
        if (this.state.machineThinking && this.state.winner === null) {
          try {
            this.setState(reduce(this.state, { type: 'machine-move', move }));
          } catch (err) {
            // A broken model (e.g. a bad checkpoint returning an illegal
            // move) must surface as an error and release the lock, not throw
            // out of the promise chain and wedge the UI.
            this.setState({ ...this.state, machineThinking: false, autoplay: false });
            this.onError?.(err);
            return;
          }
          this.scheduleAutoplay();
        }
      },
      (err) => {
        if (gen !== this.generation) return;
        this.setState({ ...this.state, machineThinking: false, autoplay: false });
        this.onError?.(err);
      },
    );
  }

  /**
   * Schedules the next auto play move: at least `autoplayGapMs` after the
   * previous think started (a slower think adds no extra delay). Clears the
   * running flag when the game ended so the Play button can restart.
   */
  private scheduleAutoplay(): void {
    const s = this.state;
    if (s.winner !== null) {
      if (s.autoplay) this.setState({ ...s, autoplay: false });
      return;
    }
    if (!s.autoplay || !s.machineThinking) return;
    const elapsed = this.now() - this.thinkStart;
    const delay = Math.max(0, this.autoplayGapMs - elapsed);
    const gen = this.generation;
    this.stepTimer = setTimeout(() => {
      if (gen !== this.generation) return;
      this.stepTimer = null;
      this.kickMachine();
    }, delay);
  }
}
