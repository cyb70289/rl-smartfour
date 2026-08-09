import { afterEach, describe, expect, it, vi } from 'vitest';
import { ModelMachinePlayer, simulationsOf, stateToJson } from '../src/game/machine';
import { newGame, reduce } from '../src/game/engine';
import type { GameConfig } from '../src/game/engine';
import type { GameState, ThinkSettings } from '../src/game/types';

const person: GameConfig = { mode: 'person', humanColor: 'white', settings: { disabled: false, effort: 100 } };

function gameState(): GameState {
  return newGame(person);
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
}

function fetchMock(): ReturnType<typeof vi.fn> {
  const mock = vi.fn(async (_url: unknown, _init?: RequestInit) => jsonResponse({ move: { x: 1, z: 1 } }));
  vi.stubGlobal('fetch', mock);
  return mock;
}

afterEach(() => vi.unstubAllGlobals());

describe('stateToJson (UI state → model JSON contract)', () => {
  it('maps a fresh state to the interchange format', () => {
    expect(stateToJson(gameState())).toEqual({
      grid: Array.from({ length: 5 }, () =>
        Array.from({ length: 5 }, () => [null, null, null, null, null]),
      ),
      pieces_left: { white: 32, black: 32 },
      current: 'white',
      winner: null,
    });
  });

  it('maps colors to 0/1 per stack level and carries counts and current player', () => {
    let s = gameState();
    s = reduce(s, { type: 'move', move: { x: 0, z: 0 } }); // white
    s = reduce(s, { type: 'move', move: { x: 0, z: 0 } }); // black
    s = reduce(s, { type: 'move', move: { x: 0, z: 0 } }); // white
    const j = stateToJson(s);
    expect(j.grid[0]![0]).toEqual([0, 1, 0, null, null]);
    expect(j.grid[4]![4]).toEqual([null, null, null, null, null]);
    expect(j.pieces_left).toEqual({ white: 30, black: 31 });
    expect(j.current).toBe('black');
    expect(j.winner).toBeNull();
  });

  it('reports the winner as a color string', () => {
    let s = gameState();
    for (const [x, z] of [[0, 0], [4, 4], [1, 0], [4, 3], [2, 0], [4, 2], [3, 0]] as const) {
      s = reduce(s, { type: 'move', move: { x, z } });
    }
    expect(s.winner).toBe('white');
    expect(stateToJson(s).winner).toBe('white');
  });

  it('reports a draw winner as "draw"', () => {
    const s: GameState = { ...gameState(), winner: 'draw' };
    expect(stateToJson(s).winner).toBe('draw');
  });
});

describe('simulationsOf (settings → MCTS steps)', () => {
  const cases: Array<[ThinkSettings, number]> = [
    [{ disabled: false, effort: 100 }, 100],
    [{ disabled: true, effort: 100 }, 0], // policy only
    [{ disabled: true, effort: 0 }, 0],
    [{ disabled: false, effort: 0 }, 0], // slider min is 0; MCTS(0) is not policy-only
    [{ disabled: false, effort: 50.7 }, 50],
    [{ disabled: false, effort: -3 }, 0],
    [{ disabled: false, effort: NaN }, 0],
  ];
  it('maps every settings combination', () => {
    for (const [settings, expected] of cases) {
      expect(simulationsOf(settings), JSON.stringify(settings)).toBe(expected);
    }
  });
});

describe('ModelMachinePlayer', () => {
  it('posts the state and simulations and returns the move', async () => {
    const mock = fetchMock();
    const player = new ModelMachinePlayer();
    const state = gameState();
    const move = await player.think(state, { disabled: false, effort: 50 });
    expect(move).toEqual({ x: 1, z: 1 });
    const [url, init] = mock.mock.calls[0]!;
    expect(url).toBe('/api/think');
    expect(init?.method).toBe('POST');
    expect(init?.headers).toMatchObject({ 'Content-Type': 'application/json' });
    const body = JSON.parse(String(init?.body));
    expect(body).toEqual({ state: stateToJson(state), simulations: 50 });
  });

  it('sends simulations=0 when search is disabled', async () => {
    const mock = fetchMock();
    const player = new ModelMachinePlayer();
    await player.think(gameState(), { disabled: true, effort: 200 });
    const body = JSON.parse(String(mock.mock.calls[0]![1]?.body));
    expect(body.simulations).toBe(0);
  });

  it('rejects with AbortError when the signal is already aborted, without fetching', async () => {
    const mock = fetchMock();
    const player = new ModelMachinePlayer();
    const ac = new AbortController();
    ac.abort();
    await expect(player.think(gameState(), { disabled: false, effort: 10 }, ac.signal)).rejects.toThrow(/aborted/i);
    expect(mock).not.toHaveBeenCalled();
  });

  it('rejects promptly with AbortError when the fetch aborts mid-flight', async () => {
    const mock = vi.fn((_url: unknown, init?: RequestInit) => {
      const { promise, reject } = Promise.withResolvers<Response>();
      init?.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
      return promise;
    });
    vi.stubGlobal('fetch', mock);
    const player = new ModelMachinePlayer();
    const ac = new AbortController();
    const pending = player.think(gameState(), { disabled: false, effort: 10 }, ac.signal);
    ac.abort();
    await expect(pending).rejects.toThrow(/aborted/i);
  });

  it('surfaces the bridge error message from an error response', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse({ error: 'model unavailable: no checkpoint' }, 503)));
    const player = new ModelMachinePlayer();
    await expect(player.think(gameState(), { disabled: false, effort: 10 })).rejects.toThrow(
      'model unavailable: no checkpoint',
    );
  });

  it('rejects on a non-JSON error response', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('Internal Server Error', { status: 500 })));
    const player = new ModelMachinePlayer();
    await expect(player.think(gameState(), { disabled: false, effort: 10 })).rejects.toThrow(/non-JSON/);
  });

  it('surfaces an in-band worker error from a 200 response', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse({ error: 'ValueError: missing state' })));
    const player = new ModelMachinePlayer();
    await expect(player.think(gameState(), { disabled: false, effort: 10 })).rejects.toThrow('ValueError: missing state');
  });

  it('rejects a null move (game over) defensively', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse({ move: null })));
    const player = new ModelMachinePlayer();
    await expect(player.think(gameState(), { disabled: false, effort: 10 })).rejects.toThrow(/no move/);
  });

  it('rejects a malformed move shape', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse({ move: { x: '2', z: 3 } })));
    const player = new ModelMachinePlayer();
    await expect(player.think(gameState(), { disabled: false, effort: 10 })).rejects.toThrow(/malformed/);
  });

  it('rejects an out-of-bounds move as illegal', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse({ move: { x: 9, z: 9 } })));
    const player = new ModelMachinePlayer();
    await expect(player.think(gameState(), { disabled: false, effort: 10 })).rejects.toThrow(/illegal/);
  });

  it('rejects a move onto a full column as illegal', async () => {
    let s = gameState();
    for (let i = 0; i < 5; i++) s = reduce(s, { type: 'move', move: { x: 0, z: 0 } });
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse({ move: { x: 0, z: 0 } })));
    const player = new ModelMachinePlayer();
    await expect(player.think(s, { disabled: false, effort: 10 })).rejects.toThrow(/illegal/);
  });

  it('accepts a response with extra fields', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => jsonResponse({ move: { x: 2, z: 4 }, id: 7 })));
    const player = new ModelMachinePlayer();
    await expect(player.think(gameState(), { disabled: false, effort: 10 })).resolves.toEqual({ x: 2, z: 4 });
  });
});
