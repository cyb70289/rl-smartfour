import { describe, it, expect, vi } from 'vitest';
import { GameController } from '../src/game/controller';
import { RandomMachinePlayer } from '../src/game/machine';
import { legalMoves } from '../src/game/rules';
import type { GameConfig } from '../src/game/engine';
import type { GameState, Move, ThinkSettings } from '../src/game/types';

const person: GameConfig = { mode: 'person', humanColor: 'white', settings: { disabled: false, effort: 100 } };
const machineHumanWhite: GameConfig = { mode: 'machine', humanColor: 'white', settings: { disabled: false, effort: 100 } };
const machineHumanBlack: GameConfig = { mode: 'machine', humanColor: 'black', settings: { disabled: false, effort: 100 } };

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
    const move = await machine.think(ctrl.state, { disabled: false, effort: 50 });
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
});
