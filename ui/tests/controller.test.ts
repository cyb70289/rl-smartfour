import { describe, it, expect, vi } from 'vitest';
import { GameController } from '../src/game/controller';
import type { PlayerMachines } from '../src/game/controller';
import { RandomMachinePlayer } from '../src/game/machine';
import { legalMoves } from '../src/game/rules';
import type { GameConfig } from '../src/game/engine';
import type { GameState, Move, ThinkSettings } from '../src/game/types';

const person: GameConfig = { white: { kind: 'human' }, black: { kind: 'human' }, settings: { effort: 100 } };
const machineHumanWhite: GameConfig = { white: { kind: 'human' }, black: { kind: 'model', checkpoint: 'best1.pt' }, settings: { effort: 100 } };
const machineHumanBlack: GameConfig = { white: { kind: 'model', checkpoint: 'best1.pt' }, black: { kind: 'human' }, settings: { effort: 100 } };
const autoplay: GameConfig = { white: { kind: 'model', checkpoint: 'best1.pt' }, black: { kind: 'model', checkpoint: 'best2.pt' }, settings: { effort: 100 } };

/** Fake machine with manually controlled promise resolution. */
class FakeMachine {
  calls: Array<{ state: Readonly<GameState>; settings: ThinkSettings; signal?: AbortSignal }> = [];
  private pending: Array<{ resolve: (m: Move) => void; reject: (e: unknown) => void }> = [];
  constructor(public readonly name = 'fake') {}
  think(state: Readonly<GameState>, settings: ThinkSettings, signal?: AbortSignal): Promise<Move> {
    this.calls.push({ state, settings, signal });
    const { promise, resolve, reject } = Promise.withResolvers<Move>();
    this.pending.push({ resolve, reject });
    return promise;
  }
  resolveNext(move: Move): void {
    this.pending.shift()!.resolve(move);
  }
  rejectNext(err: unknown): void {
    this.pending.shift()!.reject(err);
  }
}

async function flush(): Promise<void> {
  const { promise, resolve } = Promise.withResolvers<void>();
  setTimeout(resolve, 0);
  await promise;
}

/** Like flush, but yields the landing microtask first so a zero-delay
 * scheduler timer created by that microtask runs before we resume. */
async function settle(): Promise<void> {
  await Promise.resolve();
  const { promise, resolve } = Promise.withResolvers<void>();
  setTimeout(resolve, 0);
  await promise;
}

describe('RandomMachinePlayer', () => {
  it('returns a legal move for a given state', async () => {
    const machine = new RandomMachinePlayer();
    const ctrl = new GameController({ white: null, black: machine }, machineHumanWhite, () => {});
    ctrl.humanMove({ x: 2, z: 2 });
    const move = await machine.think(ctrl.state, { effort: 50 });
    expect(legalMoves(ctrl.state).some((m) => m.x === move.x && m.z === move.z)).toBe(true);
  });
});

describe('GameController: machine turn orchestration', () => {
  it('machine responds to a human move and clears the thinking flag', async () => {
    const machine = new FakeMachine();
    const ctrl = new GameController({ white: null, black: machine }, machineHumanWhite, () => {});
    expect(ctrl.state.machineThinking).toBe(false);

    ctrl.humanMove({ x: 2, z: 2 });
    expect(ctrl.state.machineThinking).toBe(true);
    expect(ctrl.state.thinking).toBe(true); // think in flight
    expect(machine.calls).toHaveLength(1);
    expect(machine.calls[0]!.state.current).toBe('black');

    machine.resolveNext({ x: 0, z: 0 });
    await flush();
    expect(ctrl.state.machineThinking).toBe(false);
    expect(ctrl.state.thinking).toBe(false);
    expect(ctrl.state.history).toEqual([
      { x: 2, z: 2, player: 'white' },
      { x: 0, z: 0, player: 'black' },
    ]);
  });

  it('blocks human moves while the machine thinks', () => {
    const machine = new FakeMachine();
    const ctrl = new GameController({ white: null, black: machine }, machineHumanWhite, () => {});
    ctrl.humanMove({ x: 2, z: 2 });
    expect(() => ctrl.humanMove({ x: 3, z: 3 })).toThrow();
  });

  it('revert pops the machine and human moves after the machine replied', async () => {
    const machine = new FakeMachine();
    const ctrl = new GameController({ white: null, black: machine }, machineHumanWhite, () => {});
    ctrl.humanMove({ x: 2, z: 2 });
    machine.resolveNext({ x: 0, z: 0 });
    await settle();
    expect(ctrl.state.history).toHaveLength(2);

    ctrl.revert();
    expect(ctrl.state.history).toEqual([]);
    expect(ctrl.state.current).toBe('white');
    expect(ctrl.state.revertAvailable).toBe(false);
  });

  it('machine (white) moves first when the human plays black', async () => {
    const machine = new FakeMachine();
    const ctrl = new GameController({ white: machine, black: null }, machineHumanBlack, () => {});
    expect(ctrl.state.machineThinking).toBe(true);
    expect(machine.calls).toHaveLength(1);

    machine.resolveNext({ x: 1, z: 1 });
    await settle();
    expect(ctrl.state.machineThinking).toBe(false);
    expect(ctrl.state.current).toBe('black');
    expect(ctrl.state.history).toEqual([{ x: 1, z: 1, player: 'white' }]);

    ctrl.humanMove({ x: 4, z: 4 });
    expect(ctrl.state.machineThinking).toBe(true);
  });

  it('a stale machine result from a reset game is discarded', async () => {
    const machine = new FakeMachine();
    const ctrl = new GameController({ white: null, black: machine }, machineHumanWhite, () => {});
    ctrl.humanMove({ x: 2, z: 2 });
    expect(machine.calls).toHaveLength(1);

    ctrl.reset(machineHumanBlack, { white: machine, black: null }); // new game, machine is white → new think starts
    expect(machine.calls).toHaveLength(2);

    // Resolve the FIRST (stale) think; it must not touch the new game.
    machine.resolveNext({ x: 0, z: 0 });
    await settle();
    expect(ctrl.state.history).toEqual([]);
    expect(ctrl.state.machineThinking).toBe(true);

    // The second (current) think resolves normally.
    machine.resolveNext({ x: 3, z: 3 });
    await settle();
    expect(ctrl.state.history).toEqual([{ x: 3, z: 3, player: 'white' }]);
    expect(ctrl.state.machineThinking).toBe(false);
  });

  it('aborts the in-flight think on reset', async () => {
    const machine = new FakeMachine();
    const ctrl = new GameController({ white: null, black: machine }, machineHumanWhite, () => {});
    ctrl.humanMove({ x: 2, z: 2 });
    expect(machine.calls[0]!.signal?.aborted).toBe(false);
    ctrl.reset(machineHumanWhite, { white: null, black: machine });
    expect(machine.calls[0]!.signal?.aborted).toBe(true);
  });

  it('no think is scheduled when the game is over', async () => {
    const machine = new FakeMachine();
    const ctrl = new GameController({ white: null, black: machine }, machineHumanWhite, () => {});
    // Simulate a game where the human wins on the move that would trigger think.
    const seq: Array<{ x: number; z: number; kind: 'human' | 'machine' }> = [
      { x: 0, z: 0, kind: 'human' },
      { x: 4, z: 4, kind: 'machine' },
      { x: 1, z: 0, kind: 'human' },
      { x: 4, z: 3, kind: 'machine' },
      { x: 2, z: 0, kind: 'human' },
      { x: 4, z: 2, kind: 'machine' },
      { x: 3, z: 0, kind: 'human' },
    ];
    for (const step of seq) {
      if (step.kind === 'human') ctrl.humanMove(step);
      else {
        machine.resolveNext(step);
        await flush();
      }
    }
    expect(ctrl.state.winner).toBe('white');
    expect(ctrl.state.machineThinking).toBe(false);
    expect(machine.calls).toHaveLength(3); // no think after the winning move
  });

  it('reports machine failures via onError and releases the lock', async () => {
    const machine = new FakeMachine();
    const onError = vi.fn();
    const ctrl = new GameController({ white: null, black: machine }, machineHumanWhite, onError);
    ctrl.humanMove({ x: 2, z: 2 });
    machine.rejectNext(new Error('model exploded'));
    await settle();
    expect(onError).toHaveBeenCalledOnce();
    expect(ctrl.state.machineThinking).toBe(false);
  });

  it('an illegal machine move is reported via onError and releases the lock', async () => {
    const machine = new FakeMachine();
    const onError = vi.fn();
    const ctrl = new GameController({ white: null, black: machine }, machineHumanWhite, onError);
    ctrl.humanMove({ x: 2, z: 2 });
    machine.resolveNext({ x: 9, z: 9 }); // out of bounds — must not wedge the UI
    await settle();
    expect(onError).toHaveBeenCalledOnce();
    expect(onError.mock.calls[0]![0]).toBeInstanceOf(Error);
    expect(ctrl.state.machineThinking).toBe(false);
    expect(ctrl.state.history).toEqual([{ x: 2, z: 2, player: 'white' }]); // human move kept
  });

  it('notifies subscribers on every state change', () => {
    const machine = new FakeMachine();
    const ctrl = new GameController({ white: null, black: machine }, machineHumanWhite, () => {});
    const seen: string[] = [];
    ctrl.subscribe(() => seen.push(`${ctrl.state.history.length}:${ctrl.state.thinking}`));
    ctrl.humanMove({ x: 2, z: 2 });
    // Two transitions: the move lands, then the think is flagged in flight.
    expect(seen).toEqual(['1:false', '1:true']);
    machine.resolveNext({ x: 0, z: 0 });
    return flush().then(() => expect(seen).toEqual(['1:false', '1:true', '2:false']));
  });

  it('reverting a finished machine game re-kicks the machine when it owes a move', async () => {
    const machine = new FakeMachine();
    const ctrl = new GameController({ white: null, black: machine }, machineHumanWhite, () => {});
    // White (human) wins on the 7th ply; machine black replies harmlessly.
    const humans: Array<[number, number]> = [
      [0, 0],
      [1, 0],
      [2, 0],
      [3, 0],
    ];
    const machines: Array<[number, number]> = [
      [4, 4],
      [4, 3],
      [4, 2],
    ];
    for (let i = 0; i < 3; i++) {
      ctrl.humanMove({ x: humans[i]![0], z: humans[i]![1] });
      machine.resolveNext({ x: machines[i]![0], z: machines[i]![1] });
      await flush();
    }
    ctrl.humanMove({ x: humans[3]![0], z: humans[3]![1] });
    expect(ctrl.state.winner).toBe('white');
    expect(ctrl.state.machineThinking).toBe(false);
    expect(machine.calls).toHaveLength(3);

    ctrl.revert();
    expect(ctrl.state.winner).toBeNull();
    expect(ctrl.state.history).toHaveLength(5);
    expect(ctrl.state.machineThinking).toBe(true); // machine owes a move
    expect(machine.calls).toHaveLength(4); // think re-kicked

    machine.resolveNext({ x: 2, z: 2 });
    await settle();
    expect(ctrl.state.history).toHaveLength(6);
    expect(ctrl.state.current).toBe('white');
    expect(ctrl.state.machineThinking).toBe(false);
  });
});

describe('GameController: think settings', () => {
  it('applies new settings immediately without restarting the game', () => {
    const machine = new FakeMachine();
    const ctrl = new GameController({ white: null, black: machine }, machineHumanWhite, () => {});
    ctrl.updateSettings({ effort: 1500 });
    expect(ctrl.state.settings).toEqual({ effort: 1500 });
    expect(ctrl.state.history).toEqual([]);
  });

  it('restarts the in-flight think with the new settings', async () => {
    const machine = new FakeMachine();
    const ctrl = new GameController({ white: null, black: machine }, machineHumanWhite, () => {});
    ctrl.humanMove({ x: 2, z: 2 });
    expect(machine.calls).toHaveLength(1);
    expect(machine.calls[0]!.settings).toEqual({ effort: 100 });
    const oldSignal = machine.calls[0]!.signal;

    ctrl.updateSettings({ effort: 2000 });
    expect(ctrl.state.machineThinking).toBe(true);
    expect(machine.calls).toHaveLength(2);
    expect(machine.calls[1]!.settings).toEqual({ effort: 2000 });
    expect(oldSignal?.aborted).toBe(true);

    // The stale (aborted) think resolving must not corrupt the state.
    machine.resolveNext({ x: 1, z: 1 });
    await settle();
    expect(ctrl.state.history).toEqual([{ x: 2, z: 2, player: 'white' }]);
    expect(ctrl.state.machineThinking).toBe(true);

    machine.resolveNext({ x: 0, z: 0 });
    await settle();
    expect(ctrl.state.history).toEqual([
      { x: 2, z: 2, player: 'white' },
      { x: 0, z: 0, player: 'black' },
    ]);
  });

  it('ignores a settings update that does not change the effort', () => {
    const machine = new FakeMachine();
    const ctrl = new GameController({ white: null, black: machine }, machineHumanWhite, () => {});
    ctrl.humanMove({ x: 2, z: 2 });
    const calls = machine.calls.length;
    ctrl.updateSettings({ effort: 100 });
    expect(machine.calls).toHaveLength(calls);
  });
});

describe('GameController: auto play (model vs model)', () => {
  function makeCtrl(opts: { autoplayGapMs?: number; now?: () => number } = { autoplayGapMs: 0 }): {
    ctrl: GameController;
    white: FakeMachine;
    black: FakeMachine;
  } {
    const white = new FakeMachine('white');
    const black = new FakeMachine('black');
    const ctrl = new GameController({ white, black }, autoplay, () => {}, opts);
    return { ctrl, white, black };
  }

  /** Drives an auto-played white win: white 4-in-a-column, black fills. */
  async function playToWhiteWin(ctrl: GameController, white: FakeMachine, black: FakeMachine): Promise<void> {
    const whites: Array<[number, number]> = [
      [0, 0],
      [1, 0],
      [2, 0],
      [3, 0],
    ];
    const blacks: Array<[number, number]> = [
      [4, 4],
      [4, 3],
      [4, 2],
    ];
    for (let i = 0; i < 3; i++) {
      white.resolveNext({ x: whites[i]![0], z: whites[i]![1] });
      await settle();
      black.resolveNext({ x: blacks[i]![0], z: blacks[i]![1] });
      await settle();
    }
    white.resolveNext({ x: whites[3]![0], z: whites[3]![1] });
    await settle();
  }

  it('a fresh auto play game starts paused: no think until Play', () => {
    const { ctrl, white } = makeCtrl();
    expect(ctrl.state.autoplay).toBe(false);
    expect(ctrl.state.machineThinking).toBe(true); // white owes a move
    expect(ctrl.state.thinking).toBe(false); // nothing in flight while paused
    expect(white.calls).toHaveLength(0);
  });

  it('Play starts the loop; moves alternate and keep the thinking flag', async () => {
    const { ctrl, white, black } = makeCtrl();
    ctrl.play();
    expect(ctrl.state.autoplay).toBe(true);
    expect(ctrl.state.thinking).toBe(true); // first think in flight
    expect(white.calls).toHaveLength(1);

    white.resolveNext({ x: 0, z: 0 });
    await settle();
    expect(ctrl.state.history).toHaveLength(1);
    expect(ctrl.state.current).toBe('black');
    expect(ctrl.state.machineThinking).toBe(true); // black owes a move
    // Zero-gap loop: black's think already started, so it is in flight.
    expect(ctrl.state.thinking).toBe(true);
    expect(black.calls).toHaveLength(1);

    black.resolveNext({ x: 4, z: 4 });
    await settle();
    expect(ctrl.state.history).toHaveLength(2);
    expect(ctrl.state.current).toBe('white');
    expect(ctrl.state.machineThinking).toBe(true);
    expect(white.calls).toHaveLength(2);
  });

  it('a second Play while running is a no-op', async () => {
    const { ctrl, white } = makeCtrl();
    ctrl.play();
    const calls = white.calls.length;
    ctrl.play();
    expect(white.calls).toHaveLength(calls);
  });

  it('Play is a no-op outside auto play', () => {
    const machine = new FakeMachine();
    const ctrl = new GameController({ white: null, black: machine }, machineHumanWhite, () => {});
    ctrl.play();
    expect(ctrl.state.autoplay).toBe(false);
    expect(machine.calls).toHaveLength(0);
  });

  it('Pause aborts the in-flight think and discards its result', async () => {
    const { ctrl, white, black } = makeCtrl();
    ctrl.play();
    expect(white.calls).toHaveLength(1);
    const signal = white.calls[0]!.signal;

    ctrl.pause();
    expect(signal?.aborted).toBe(true);
    expect(ctrl.state.autoplay).toBe(false);
    expect(ctrl.state.machineThinking).toBe(false);
    expect(ctrl.state.thinking).toBe(false);

    // The stale think resolving must not move the board or start the loop.
    white.resolveNext({ x: 0, z: 0 });
    await settle();
    expect(ctrl.state.history).toEqual([]);
    expect(black.calls).toHaveLength(0);
  });

  it('Pause during the between-move gap stops further moves', async () => {
    vi.useFakeTimers();
    try {
      let now = 0;
      const { ctrl, white, black } = makeCtrl({ autoplayGapMs: 1000, now: () => now });
      ctrl.play();
      white.resolveNext({ x: 0, z: 0 });
      await vi.advanceTimersByTimeAsync(0); // move lands; gap timer pending
      expect(black.calls).toHaveLength(0);

      ctrl.pause();
      vi.advanceTimersByTime(5000); // past the would-be gap
      expect(black.calls).toHaveLength(0);
      expect(ctrl.state.machineThinking).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });

  it('Step plays one move and pauses auto play', async () => {
    const { ctrl, white, black } = makeCtrl();
    ctrl.step();
    expect(ctrl.state.autoplay).toBe(false);
    expect(ctrl.state.machineThinking).toBe(true); // white owes a move
    expect(ctrl.state.thinking).toBe(true); // step think in flight
    expect(white.calls).toHaveLength(1);

    white.resolveNext({ x: 0, z: 0 });
    await settle();
    expect(ctrl.state.history).toHaveLength(1);
    expect(black.calls).toHaveLength(0); // no auto continuation
    expect(ctrl.state.machineThinking).toBe(true); // black still owes a move
    expect(ctrl.state.thinking).toBe(false); // paused again, nothing in flight
  });

  it('Step also pauses an auto-playing game and plays a single move', async () => {
    const { ctrl, white, black } = makeCtrl();
    ctrl.play();
    expect(white.calls).toHaveLength(1);

    ctrl.step(); // stops the loop, starts one move for the current side
    expect(ctrl.state.autoplay).toBe(false);
    expect(ctrl.state.machineThinking).toBe(true);

    white.resolveNext({ x: 0, z: 0 });
    await settle();
    expect(black.calls).toHaveLength(0);
  });

  it('auto play stops when the game ends', async () => {
    const { ctrl, white, black } = makeCtrl();
    ctrl.play();
    await playToWhiteWin(ctrl, white, black);
    expect(ctrl.state.winner).toBe('white');
    expect(ctrl.state.autoplay).toBe(false);
    expect(ctrl.state.machineThinking).toBe(false);
  });

  it('Play after game over is a no-op: Reset is the only restart', async () => {
    const { ctrl, white, black } = makeCtrl();
    ctrl.play();
    await playToWhiteWin(ctrl, white, black);
    const callsBefore = white.calls.length;
    const history = [...ctrl.state.history];

    ctrl.play();
    expect(ctrl.state.winner).toBe('white');
    expect(ctrl.state.history).toEqual(history); // board untouched
    expect(ctrl.state.autoplay).toBe(false);
    expect(white.calls).toHaveLength(callsBefore); // no fresh think
  });

  it('Step after game over is a no-op', async () => {
    const { ctrl, white, black } = makeCtrl();
    ctrl.play();
    await playToWhiteWin(ctrl, white, black);
    const whiteCallsBefore = white.calls.length;
    const blackCallsBefore = black.calls.length;
    const history = [...ctrl.state.history];

    ctrl.step();
    expect(ctrl.state.winner).toBe('white');
    expect(ctrl.state.history).toEqual(history);
    expect(white.calls).toHaveLength(whiteCallsBefore);
    expect(black.calls).toHaveLength(blackCallsBefore);
  });

  it('respects the minimum gap between think starts', async () => {
    vi.useFakeTimers();
    try {
      let now = 0;
      const { ctrl, white, black } = makeCtrl({ autoplayGapMs: 1000, now: () => now });
      ctrl.play();
      white.resolveNext({ x: 0, z: 0 });
      await vi.advanceTimersByTimeAsync(0); // move lands; next think due at +1000
      expect(black.calls).toHaveLength(0);

      vi.advanceTimersByTime(999);
      expect(black.calls).toHaveLength(0);
      vi.advanceTimersByTime(1);
      expect(black.calls).toHaveLength(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it('a think slower than the gap adds no extra delay', async () => {
    vi.useFakeTimers();
    try {
      let now = 0;
      const { ctrl, white, black } = makeCtrl({ autoplayGapMs: 1000, now: () => now });
      ctrl.play();
      now = 1500; // the think outlived the gap
      white.resolveNext({ x: 0, z: 0 });
      await vi.advanceTimersByTimeAsync(0); // zero-delay timer fires immediately
      expect(black.calls).toHaveLength(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it('updateSettings while paused does not start a move', () => {
    const { ctrl, white } = makeCtrl();
    ctrl.updateSettings({ effort: 2000 });
    expect(ctrl.state.settings).toEqual({ effort: 2000 });
    expect(white.calls).toHaveLength(0);
  });
});

describe('GameController: setConfig keeps the board', () => {
  function makeSetCtrl(opts: { autoplayGapMs?: number; now?: () => number } = { autoplayGapMs: 0 }): {
    ctrl: GameController;
    white: FakeMachine;
    black: FakeMachine;
  } {
    const white = new FakeMachine('white');
    const black = new FakeMachine('black');
    const ctrl = new GameController({ white, black }, autoplay, () => {}, opts);
    return { ctrl, white, black };
  }

  it('a mid-game slot change keeps the position; Play resumes with new players', async () => {
    const { ctrl, white, black } = makeSetCtrl();
    ctrl.play();
    white.resolveNext({ x: 0, z: 0 });
    await settle();
    black.resolveNext({ x: 4, z: 4 });
    await settle();
    expect(ctrl.state.history).toHaveLength(2);

    ctrl.pause();
    // Swap black to a different model (same setup semantics in tests).
    const switched = { ...autoplay, black: { kind: 'model', checkpoint: 'best3.pt' } as const };
    ctrl.setConfig(switched, { white, black });
    expect(ctrl.state.history).toHaveLength(2); // board kept
    expect((ctrl.state.black as { kind: 'model'; checkpoint: string }).checkpoint).toBe('best3.pt');
    expect(white.calls).toHaveLength(black.calls.length + 1); // nothing started while paused
    ctrl.play();
    expect(ctrl.state.autoplay).toBe(true);
    expect(black.calls).toHaveLength(1); // the NEW black model owes the move
  });

  it('switching a model slot to human mid-think aborts and hands over control', async () => {
    const machine = new FakeMachine();
    const ctrl = new GameController({ white: null, black: machine }, machineHumanWhite, () => {});
    ctrl.humanMove({ x: 2, z: 2 });
    expect(machine.calls).toHaveLength(1);

    ctrl.setConfig(
      { white: { kind: 'human' }, black: { kind: 'human' }, settings: { effort: 100 } },
      { white: null, black: null },
    );
    expect(machine.calls[0]!.signal?.aborted).toBe(true);
    expect(ctrl.state.machineThinking).toBe(false);
    expect(ctrl.state.history).toEqual([{ x: 2, z: 2, player: 'white' }]);

    // The stale result must not land on the human-controlled board.
    machine.resolveNext({ x: 0, z: 0 });
    await settle();
    expect(ctrl.state.history).toHaveLength(1);
  });

  it('switching a human slot to model makes it think immediately (machine mode)', async () => {
    const whiteM = new FakeMachine('w');
    const machine = new FakeMachine('b');
    const ctrl = new GameController({ white: null, black: machine }, machineHumanWhite, () => {});
    ctrl.humanMove({ x: 2, z: 2 });
    machine.resolveNext({ x: 4, z: 4 });
    await settle();
    expect(ctrl.state.current).toBe('white');
    expect(ctrl.state.machineThinking).toBe(false);

    ctrl.setConfig(
      { white: { kind: 'model', checkpoint: 'best9.pt' }, black: { kind: 'human' }, settings: { effort: 100 } },
      { white: whiteM, black: null },
    );
    expect((ctrl.state.white as { kind: 'model'; checkpoint: string }).checkpoint).toBe('best9.pt');
    expect(ctrl.state.machineThinking).toBe(true);
    expect(whiteM.calls).toHaveLength(1); // instant response
    expect(ctrl.state.history).toHaveLength(2); // untouched
  });
});
