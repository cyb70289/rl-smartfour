import { describe, it, expect } from 'vitest';
import { Hud } from '../src/ui/hud';
import { GameController } from '../src/game/controller';
import type { GameConfig } from '../src/game/engine';
import type { GameState, Move, ThinkSettings } from '../src/game/types';

// Minimal DOM shims so the real Hud binds its listeners; assertions read the
// same properties sync() writes (textContent/disabled/hidden).
class FakeEl {
  id = '';
  style: Record<string, string> = {};
  children: FakeEl[] = [];
  disabled = false;
  hidden = false;
  textContent = '';
  value = '';
  name = '';
  checked = false;
  listeners: Record<string, Array<() => void>> = {};

  appendChild(child: FakeEl): void {
    this.children.push(child);
  }
  replaceChildren(): void {
    this.children = [];
  }
  addEventListener(type: string, fn: () => void): void {
    (this.listeners[type] ??= []).push(fn);
  }
  fire(type: string): void {
    for (const fn of this.listeners[type] ?? []) fn();
  }
  walk(): FakeEl[] {
    const out: FakeEl[] = [];
    for (const c of this.children) out.push(c, ...c.walk());
    return out;
  }
  querySelector(sel: string): FakeEl | null {
    if (!sel.startsWith('#')) throw new Error('unsupported selector ' + sel);
    return this.walk().find((el) => el.id === sel.slice(1)) ?? null;
  }
  querySelectorAll(sel: string): FakeEl[] {
    const m = /^input\[name="(.+)"\]$/.exec(sel);
    if (!m) throw new Error('unsupported selector ' + sel);
    return this.walk().filter((el) => el.name === m[1]);
  }
}

function radio(root: FakeEl, name: string, value: string): FakeEl {
  const r = new FakeEl();
  r.name = name;
  r.value = value;
  root.appendChild(r);
  return r;
}

function buildApp(domWhite: 'human' | 'model', domBlack: 'human' | 'model'): {
  app: FakeEl;
  radios: Record<string, FakeEl>;
  effort: FakeEl[];
  groups: Record<string, FakeEl[]>;
} {
  const app = new FakeEl();
  for (const id of ['status', 'revert-btn', 'play-btn', 'reset-btn', 'white-checkpoint', 'black-checkpoint']) {
    const el = new FakeEl();
    el.id = id;
    app.appendChild(el);
  }
  const radios: Record<string, FakeEl> = {};
  const groups: Record<string, FakeEl[]> = {};
  for (const color of ['white', 'black']) {
    const domKind = color === 'white' ? domWhite : domBlack;
    for (const kind of ['human', 'model']) {
      const r = radio(app, `${color}-kind`, kind);
      radios[`${color}-${kind}`] = r;
      (groups[`${color}-kind`] ??= []).push(r);
    }
    radios[`${color}-${domKind}`]!.checked = true;
    const select = app.querySelector(`#${color}-checkpoint`) as FakeEl;
    select.value = 'best1.pt';
  }
  const effort = ['0', '500', '2000'].map((v) => radio(app, 'effort', v));
  groups['effort'] = [...effort];
  effort[1]!.checked = true;
  return { app, radios, effort, groups };
}

// localStorage is unavailable under node; hide the global so restore/saveEffort
// take their catch paths.
delete (globalThis as { localStorage?: unknown }).localStorage;
(globalThis as { document?: unknown }).document = {
  createElement: () => new FakeEl(),
  getElementById: () => null,
};
// The checkpoint API answers immediately so the dropdowns populate and the
// boot-align path in loadCheckpoints runs deterministically.
(globalThis as { fetch?: unknown }).fetch = () =>
  Promise.resolve({ json: async () => ({ checkpoints: ['best2.pt', 'best1.pt'] }) });

describe('Hud ↔ controller behaviour (DOM stubs)', () => {
  function setup(config: GameConfig) {
    // Align the panel's visible selection with the boot config so the
    // boot-align path in loadCheckpoints becomes a no-op instead of a flip.
    const domWhite = config.white.kind === 'model' ? 'model' : 'human';
    const { app, radios, effort } = buildApp(domWhite, config.black.kind === 'model' ? 'model' : 'human');
    type Slot = 'white' | 'black';
    interface FakeMachine {
      calls: number;
      signals: Array<AbortSignal | undefined>;
      pending: Array<(m: Move) => void>;
    }
    const machines: Record<Slot, FakeMachine> = {
      white: { calls: 0, signals: [], pending: [] },
      black: { calls: 0, signals: [], pending: [] },
    };
    const mkPlayer = (slot: Slot) =>
      ({
        think(_s: Readonly<GameState>, _st: ThinkSettings, signal?: AbortSignal): Promise<Move> {
          machines[slot].calls++;
          machines[slot].signals.push(signal);
          const { promise, resolve } = Promise.withResolvers<Move>();
          machines[slot].pending.push(resolve);
          return promise;
        },
      }) as never;
    const playersOf = (cfg: GameConfig) => ({
      white: cfg.white.kind === 'model' ? mkPlayer('white') : null,
      black: cfg.black.kind === 'model' ? mkPlayer('black') : null,
    });
    const ctrl = new GameController(playersOf(config), config, undefined, { autoplayGapMs: 0 });
    const hud = new Hud(app as unknown as HTMLElement, {
      onRevert: () => ctrl.revert(),
      onReset: (c) => ctrl.reset(c, playersOf(c)),
      onPlayersChange: (c) => {
        ctrl.setConfig(c, playersOf(c));
      },
      onPlayPause: () => (ctrl.state.autoplay ? ctrl.pause() : ctrl.play()),
      onStep: () => ctrl.step(),
      onSettingsChange: (settings) => ctrl.updateSettings(settings),
    });

    // Two-step yield (as controller.test.ts settle): drains the landing
    // microtask first so a zero-gap scheduler timer it creates fires too.
    const settle = async (): Promise<void> => {
      await Promise.resolve();
      const promise = new Promise<void>((r) => setTimeout(r, 0));
      await promise;
    };
    /** Resolves the oldest outstanding think of that color (FIFO). */
    const resolveNext = (slot: Slot, move: Move): void => {
      machines[slot].pending.shift()!(move);
    };
    const sync = (): void => hud.sync(ctrl.state);
    sync();

    const btn = (id: string): FakeEl => app.querySelector(`#${id}`)!;
    const select = (id: string): FakeEl => app.querySelector(`#${id}`)!;
    const pickRadio = (key: string): void => {
      const dash = key.lastIndexOf('-');
      const name = key.slice(0, dash) + '-kind';
      for (const other of buildAppGroups(app)[name]!) other.checked = false;
      radios[key]!.checked = true;
      radios[key]!.fire('change');
    };
    return { app, radios, effort, ctrl, sync, btn, select, pickRadio, resolveNext, settle };
  }

  function buildAppGroups(app: FakeEl): Record<string, FakeEl[]> {
    const groups: Record<string, FakeEl[]> = {};
    for (const el of app.walk()) if (el.name) (groups[el.name] ??= []).push(el);
    return groups;
  }

  const machineHumanWhite: GameConfig = {
    white: { kind: 'human' },
    black: { kind: 'model', checkpoint: 'best1.pt' },
    settings: { effort: 100 },
  };
  const bothModels: GameConfig = {
    white: { kind: 'model', checkpoint: 'best1.pt' },
    black: { kind: 'model', checkpoint: 'best2.pt' },
    settings: { effort: 100 },
  };

  it('person/machine modes show [Revert][Reset], Play hidden, nothing locked', async () => {
    const t = setup(machineHumanWhite);
    await t.settle();
    await t.settle(); // let the (stubbed) checkpoint list load fully
    expect(t.btn('play-btn').hidden).toBe(true);
    expect(t.btn('revert-btn').textContent).toBe('Revert');
    expect(t.radios['white-human']!.disabled).toBe(false);
    expect(t.radios['black-model']!.disabled).toBe(false);
    expect(t.select('black-checkpoint').disabled).toBe(false);
    expect(t.effort.every((r) => !r.disabled)).toBe(true);
  });

  it('switching the second slot to a model pauses mid-game; Play continues with it', async () => {
    const t = setup(machineHumanWhite);
    await t.settle();
    await t.settle();
    t.ctrl.humanMove({ x: 2, z: 2 });
    t.resolveNext('black', { x: 4, z: 4 });
    await t.settle();
    expect(t.ctrl.state.history).toHaveLength(2);

    // Flip white to a model -> autoplay mode, board kept, still paused.
    t.pickRadio('white-model');
    t.sync();
    expect(t.ctrl.state.history).toHaveLength(2); // board untouched
    expect(t.ctrl.state.autoplay).toBe(false); // waits for Play
    expect(t.btn('play-btn').hidden).toBe(false); // now an auto play game

    t.btn('play-btn').fire('click'); // continue on the current position
    t.sync();
    expect(t.ctrl.state.autoplay).toBe(true);
    expect(t.ctrl.state.machineThinking).toBe(true); // white owes the move now
  });

  it('machine-mode checkpoint switch re-thinks instantly on the same board', async () => {
    const humanBlack: GameConfig = {
      white: { kind: 'model', checkpoint: 'best1.pt' },
      black: { kind: 'human' },
      settings: { effort: 100 },
    };
    const t = setup(humanBlack);
    await t.settle();
    await t.settle();
    expect(t.ctrl.state.thinking).toBe(true); // machine moves first here

    t.select('white-checkpoint').value = 'best2.pt';
    t.select('white-checkpoint').fire('change');
    expect(t.ctrl.state.history).toHaveLength(0); // board untouched
    expect(t.ctrl.state.machineThinking).toBe(true); // still owes the move
  });

  it('running auto play locks slots and effort; Pause unlocks', async () => {
    const t = setup(bothModels);
    await t.settle();
    await t.settle();
    expect(t.btn('play-btn').hidden).toBe(false);
    expect(t.btn('play-btn').textContent).toBe('Play');

    t.btn('play-btn').fire('click'); // Play
    t.sync();
    expect(t.ctrl.state.autoplay).toBe(true);
    expect(t.btn('play-btn').textContent).toBe('Pause');
    for (const key of ['white-human', 'white-model', 'black-human', 'black-model'])
      expect(t.radios[key]!.disabled, key).toBe(true);
    expect(t.select('white-checkpoint').disabled).toBe(true);
    expect(t.effort.every((r) => r.disabled)).toBe(true);

    t.btn('play-btn').fire('click'); // Pause
    t.sync();
    expect(t.ctrl.state.autoplay).toBe(false);
    for (const key of ['white-human', 'white-model', 'black-human', 'black-model'])
      expect(t.radios[key]!.disabled, key).toBe(false);
    expect(t.effort.every((r) => !r.disabled)).toBe(true);
  });

  it('a finished game disables Play/Step; Reset clears the board and restarts', async () => {
    const t = setup(bothModels);
    await t.settle();
    await t.settle();
    t.btn('play-btn').fire('click');
    const whites: Array<[number, number]> = [[0, 0], [1, 0], [2, 0], [3, 0]];
    const blacks: Array<[number, number]> = [[4, 4], [4, 3], [4, 2]];
    for (let i = 0; i < 3; i++) {
      t.resolveNext('white', { x: whites[i]![0], z: whites[i]![1] });
      await t.settle();
      t.resolveNext('black', { x: blacks[i]![0], z: blacks[i]![1] });
      await t.settle();
    }
    t.resolveNext('white', { x: whites[3]![0], z: whites[3]![1] });
    await t.settle();
    t.sync();
    expect(t.ctrl.state.winner).toBe('white');
    expect(t.btn('play-btn').textContent).toBe('Play');
    expect(t.btn('play-btn').disabled).toBe(true); // Reset is the only restart
    expect(t.btn('revert-btn').textContent).toBe('Step');
    expect(t.btn('revert-btn').disabled).toBe(true);
    const pliesBefore = t.ctrl.state.history.length;

    t.btn('play-btn').fire('click'); // no-op on finished game
    expect(t.ctrl.state.history).toHaveLength(pliesBefore);

    t.btn('reset-btn').fire('click'); // explicit restart, same players
    t.sync();
    expect(t.ctrl.state.winner).toBeNull();
    expect(t.ctrl.state.history).toHaveLength(0);
    expect(t.ctrl.state.autoplay).toBe(false);
    expect(t.btn('play-btn').disabled).toBe(false);
    expect(t.btn('revert-btn').textContent).toBe('Step'); // still both models
  });
});
