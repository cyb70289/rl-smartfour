import { describe, it, expect, vi } from 'vitest';
import { GameController } from '../src/game/controller';
import { RandomMachinePlayer } from '../src/game/machine';
import { legalMoves } from '../src/game/rules';
import type { GameConfig } from '../src/game/engine';
import type { GameState, Move, ThinkSettings } from '../src/game/types';

const person: GameConfig = { mode: 'person', humanColor: 'white', settings: { effort: 100 } };
const machineHumanWhite: GameConfig = { mode: 'machine', humanColor: 'white', settings: { effort: 100 } };
const machineHumanBlack: GameConfig = { mode: 'machine', humanColor: 'black', settings: { effort: 100 } };

/** Fake machine with manually controlled promise resolution. */
class FakeMachine {
  calls: Array<{ state: Readonly<GameState>; settings: ThinkSettings; signal?: AbortSignal }> = [];
  private pending: Array<{ resolve: (m: Move) => void; reject: (e: unknown) => void }> = [];
  constructor(public readonly name = 'fake') {}
  think(state: Readonly<GameState>, settings: ThinkSettings, signal?: AbortSignal): Promise<Move> {
    this.calls.push({ state, settings, signal });
    return new Promise<Move>((resolve, reject) => {
      this.pending.push({ resolve, reject });
    });
  }
  resolveNext(move: Move): void {
    this.pending.shift()!.resolve(move);
  }
  rejectNext(err: unknown): void {
    this.pending.shift()!.reject(err);
  }
}

async function flush(): Promise<void> {
  await new Promise((r) => setTimeout(r, 0));
}

describe('RandomMachinePlayer', () => {
  it('returns a legal move for a given state', async () => {
    const machine = new RandomMachinePlayer();
    const ctrl = new GameController(machine, machineHumanWhite, () => {});
    ctrl.humanMove({ x: 2, z: 2 });
    const move = await machine.think(ctrl.state, { effort: 50 });
    expect(legalMoves(ctrl.state).some((m) => m.x === move.x && m.z === move.z)).toBe(true);
  });
});

describe('GameController: machine turn orchestration', () => {
  it('machine responds to a human move and clears the thinking flag', async () => {
    const machine = new FakeMachine();
    const ctrl = new GameController(machine, machineHumanWhite, () => {});
    expect(ctrl.state.machineThinking).toBe(false);

    ctrl.humanMove({ x: 2, z: 2 });
    expect(ctrl.state.machineThinking).toBe(true);
    expect(machine.calls).toHaveLength(1);
    expect(machine.calls[0]!.state.current).toBe('black');

    machine.resolveNext({ x: 0, z: 0 });
    await flush();
    expect(ctrl.state.machineThinking).toBe(false);
    expect(ctrl.state.history).toEqual([
      { x: 2, z: 2, player: 'white' },
      { x: 0, z: 0, player: 'black' },
    ]);
  });

  it('blocks human moves while the machine thinks', () => {
    const machine = new FakeMachine();
    const ctrl = new GameController(machine, machineHumanWhite, () => {});
    ctrl.humanMove({ x: 2, z: 2 });
    expect(() => ctrl.humanMove({ x: 3, z: 3 })).toThrow();
  });

  it('revert pops the machine and human moves after the machine replied', async () => {
    const machine = new FakeMachine();
    const ctrl = new GameController(machine, machineHumanWhite, () => {});
    ctrl.humanMove({ x: 2, z: 2 });
    machine.resolveNext({ x: 0, z: 0 });
    await flush();
    expect(ctrl.state.history).toHaveLength(2);

    ctrl.revert();
    expect(ctrl.state.history).toEqual([]);
    expect(ctrl.state.current).toBe('white');
    expect(ctrl.state.revertAvailable).toBe(false);
  });

  it('machine (white) moves first when the human plays black', async () => {
    const machine = new FakeMachine();
    const ctrl = new GameController(machine, machineHumanBlack, () => {});
    expect(ctrl.state.machineThinking).toBe(true);
    expect(machine.calls).toHaveLength(1);

    machine.resolveNext({ x: 1, z: 1 });
    await flush();
    expect(ctrl.state.machineThinking).toBe(false);
    expect(ctrl.state.current).toBe('black');
    expect(ctrl.state.history).toEqual([{ x: 1, z: 1, player: 'white' }]);

    ctrl.humanMove({ x: 4, z: 4 });
    expect(ctrl.state.machineThinking).toBe(true);
  });

  it('a stale machine result from a reset game is discarded', async () => {
    const machine = new FakeMachine();
    const ctrl = new GameController(machine, machineHumanWhite, () => {});
    ctrl.humanMove({ x: 2, z: 2 });
    expect(machine.calls).toHaveLength(1);

    ctrl.reset(machineHumanBlack); // new game, machine is white → new think starts
    expect(machine.calls).toHaveLength(2);

    // Resolve the FIRST (stale) think; it must not touch the new game.
    machine.resolveNext({ x: 0, z: 0 });
    await flush();
    expect(ctrl.state.history).toEqual([]);
    expect(ctrl.state.machineThinking).toBe(true);

    // The second (current) think resolves normally.
    machine.resolveNext({ x: 3, z: 3 });
    await flush();
    expect(ctrl.state.history).toEqual([{ x: 3, z: 3, player: 'white' }]);
    expect(ctrl.state.machineThinking).toBe(false);
  });

  it('aborts the in-flight think on reset', async () => {
    const machine = new FakeMachine();
    const ctrl = new GameController(machine, machineHumanWhite, () => {});
    ctrl.humanMove({ x: 2, z: 2 });
    expect(machine.calls[0]!.signal?.aborted).toBe(false);
    ctrl.reset(machineHumanWhite);
    expect(machine.calls[0]!.signal?.aborted).toBe(true);
  });

  it('no think is scheduled when the game is over', async () => {
    const machine = new FakeMachine();
    const ctrl = new GameController(machine, machineHumanWhite, () => {});
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
    const ctrl = new GameController(machine, machineHumanWhite, onError);
    ctrl.humanMove({ x: 2, z: 2 });
    machine.rejectNext(new Error('model exploded'));
    await flush();
    expect(onError).toHaveBeenCalledOnce();
    expect(ctrl.state.machineThinking).toBe(false);
  });

  it('an illegal machine move is reported via onError and releases the lock', async () => {
    const machine = new FakeMachine();
    const onError = vi.fn();
    const ctrl = new GameController(machine, machineHumanWhite, onError);
    ctrl.humanMove({ x: 2, z: 2 });
    machine.resolveNext({ x: 9, z: 9 }); // out of bounds — must not wedge the UI
    await flush();
    expect(onError).toHaveBeenCalledOnce();
    expect(onError.mock.calls[0]![0]).toBeInstanceOf(Error);
    expect(ctrl.state.machineThinking).toBe(false);
    expect(ctrl.state.history).toEqual([{ x: 2, z: 2, player: 'white' }]); // human move kept
  });

  it('notifies subscribers on every state change', () => {
    const machine = new FakeMachine();
    const ctrl = new GameController(machine, machineHumanWhite, () => {});
    const seen: string[] = [];
    ctrl.subscribe(() => seen.push(`${ctrl.state.history.length}`));
    ctrl.humanMove({ x: 2, z: 2 });
    expect(seen).toEqual(['1']);
    machine.resolveNext({ x: 0, z: 0 });
    return flush().then(() => expect(seen).toEqual(['1', '2']));
  });

  it('reverting a finished machine game re-kicks the machine when it owes a move', async () => {
    const machine = new FakeMachine();
    const ctrl = new GameController(machine, machineHumanWhite, () => {});
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
    await flush();
    expect(ctrl.state.history).toHaveLength(6);
    expect(ctrl.state.current).toBe('white');
    expect(ctrl.state.machineThinking).toBe(false);
  });
});

describe('GameController: think settings', () => {
  it('applies new settings immediately without restarting the game', () => {
    const machine = new FakeMachine();
    const ctrl = new GameController(machine, machineHumanWhite, () => {});
    ctrl.updateSettings({ effort: 1500 });
    expect(ctrl.state.settings).toEqual({ effort: 1500 });
    expect(ctrl.state.history).toEqual([]);
  });

  it('restarts the in-flight think with the new settings', async () => {
    const machine = new FakeMachine();
    const ctrl = new GameController(machine, machineHumanWhite, () => {});
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
    await flush();
    expect(ctrl.state.history).toEqual([{ x: 2, z: 2, player: 'white' }]);
    expect(ctrl.state.machineThinking).toBe(true);

    machine.resolveNext({ x: 0, z: 0 });
    await flush();
    expect(ctrl.state.history).toEqual([
      { x: 2, z: 2, player: 'white' },
      { x: 0, z: 0, player: 'black' },
    ]);
  });

  it('ignores a settings update that does not change the effort', () => {
    const machine = new FakeMachine();
    const ctrl = new GameController(machine, machineHumanWhite, () => {});
    ctrl.humanMove({ x: 2, z: 2 });
    const calls = machine.calls.length;
    ctrl.updateSettings({ effort: 100 });
    expect(machine.calls).toHaveLength(calls);
  });
});
