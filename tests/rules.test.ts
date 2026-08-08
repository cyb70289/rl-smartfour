import { describe, it, expect } from 'vitest';
import {
  createInitialState,
  applyMove,
  legalMoves,
  isLegal,
  stackHeight,
  cellAt,
  findWinRun,
} from '../src/game/rules';
import type { Player } from '../src/game/types';

const SIZE = 5;

// Canonical line directions as [xDelta, zDelta, yDelta] with dy >= 0:
// 4 flat + 1 vertical + 8 rising diagonals.
const DIRS: ReadonlyArray<readonly [number, number, number]> = [
  [1, 0, 0],
  [0, 0, 1],
  [1, 0, 1],
  [1, 0, -1],
  [0, 1, 0],
  [1, 1, 0],
  [0, 1, 1],
  [1, 1, 1],
  [1, 1, -1],
  [-1, 1, 0],
  [0, 1, -1],
  [-1, 1, -1],
  [-1, 1, 1],
];

/** All (direction, start) pairs whose 4 cells stay inside the 5x5x5 cube. */
function allWinLines(): Array<{ dir: readonly [number, number, number]; cells: Array<[number, number, number]> }> {
  const lines: Array<{ dir: readonly [number, number, number]; cells: Array<[number, number, number]> }> = [];
  for (const dir of DIRS) {
    for (let x = 0; x < SIZE; x++) {
      for (let z = 0; z < SIZE; z++) {
        for (let y = 0; y < SIZE; y++) {
          const cells: Array<[number, number, number]> = [];
          let ok = true;
          for (let i = 0; i < 4; i++) {
            const cx = x + dir[0] * i;
            const cz = z + dir[1] * i;
            const cy = y + dir[2] * i;
            if (cx < 0 || cx >= SIZE || cz < 0 || cz >= SIZE || cy < 0 || cy >= SIZE) {
              ok = false;
              break;
            }
            cells.push([cx, cz, cy]);
          }
          if (ok) lines.push({ dir, cells });
        }
      }
    }
  }
  return lines;
}

function emptyGrid(): (Player | null)[][][] {
  const grid: (Player | null)[][][] = [];
  for (let x = 0; x < SIZE; x++) {
    const col: (Player | null)[][] = [];
    for (let z = 0; z < SIZE; z++) col.push(new Array<Player | null>(SIZE).fill(null));
    grid.push(col);
  }
  return grid;
}

/** Build a game where `cells` (flat y=0 lines only) are filled by white in order; black fills a disjoint column. */
function playLine(cells: Array<[number, number, number]>) {
  const cols = new Set(cells.map(([x, z]) => `${x},${z}`));
  let filler: [number, number] = [4, 4];
  outer: for (let fx = 0; fx < SIZE; fx++) {
    for (let fz = 0; fz < SIZE; fz++) {
      if (!cols.has(`${fx},${fz}`)) {
        filler = [fx, fz];
        break outer;
      }
    }
  }
  let state = createInitialState();
  for (let i = 0; i < cells.length; i++) {
    const [x, z] = cells[i]!;
    state = applyMove(state, { x, z });
    if (i < cells.length - 1) state = applyMove(state, { x: filler[0], z: filler[1] });
  }
  return state;
}

describe('rules: initial state', () => {
  it('is empty, white to move, 32 pieces each, no winner, no revert window', () => {
    const s = createInitialState();
    expect(s.grid).toHaveLength(SIZE);
    expect(s.grid[0]).toHaveLength(SIZE);
    expect(s.grid[0]![0]).toHaveLength(SIZE);
    expect(legalMoves(s)).toHaveLength(25);
    expect(s.current).toBe('white');
    expect(s.piecesLeft).toEqual({ white: 32, black: 32 });
    expect(s.winner).toBeNull();
    expect(s.winningCells).toBeNull();
    expect(s.revertAvailable).toBe(false);
    expect(s.machineThinking).toBe(false);
    expect(s.history).toEqual([]);
  });
});

describe('rules: placement and stacking', () => {
  it('places at y=0 in an empty column and flips the turn', () => {
    let s = createInitialState();
    s = applyMove(s, { x: 2, z: 3 });
    expect(cellAt(s.grid, 2, 3, 0)).toBe('white');
    expect(stackHeight(s.grid, 2, 3)).toBe(1);
    expect(s.current).toBe('black');
    expect(s.piecesLeft.white).toBe(31);
    expect(s.history).toEqual([{ x: 2, z: 3, player: 'white' }]);
    expect(s.lastPlaced).toEqual({ x: 2, z: 3, y: 0, player: 'white' });
    expect(s.revertAvailable).toBe(true);
  });

  it('stacks on top of existing pieces, including the opponent\'s', () => {
    let s = createInitialState();
    s = applyMove(s, { x: 0, z: 0 }); // white
    s = applyMove(s, { x: 0, z: 0 }); // black on top
    s = applyMove(s, { x: 0, z: 0 }); // white
    expect(stackHeight(s.grid, 0, 0)).toBe(3);
    expect(cellAt(s.grid, 0, 0, 0)).toBe('white');
    expect(cellAt(s.grid, 0, 0, 1)).toBe('black');
    expect(cellAt(s.grid, 0, 0, 2)).toBe('white');
    expect(cellAt(s.grid, 0, 0, 3)).toBeNull();
  });

  it('never floats: pieces only appear at the current stack top', () => {
    let s = createInitialState();
    s = applyMove(s, { x: 1, z: 1 });
    expect(cellAt(s.grid, 1, 1, 1)).toBeNull();
    expect(cellAt(s.grid, 1, 1, 0)).toBe('white');
  });
});

describe('rules: legality', () => {
  it('rejects out-of-bounds columns', () => {
    const s = createInitialState();
    expect(isLegal(s, { x: -1, z: 0 })).toBe(false);
    expect(isLegal(s, { x: 5, z: 0 })).toBe(false);
    expect(isLegal(s, { x: 0, z: -1 })).toBe(false);
    expect(isLegal(s, { x: 0, z: 5 })).toBe(false);
    expect(() => applyMove(s, { x: -1, z: 0 })).toThrow();
    expect(() => applyMove(s, { x: 5, z: 5 })).toThrow();
  });

  it('rejects moves onto a full column of 5', () => {
    let s = createInitialState();
    for (let i = 0; i < 5; i++) s = applyMove(s, { x: 0, z: 0 });
    expect(stackHeight(s.grid, 0, 0)).toBe(5);
    expect(isLegal(s, { x: 0, z: 0 })).toBe(false);
    expect(() => applyMove(s, { x: 0, z: 0 })).toThrow();
    expect(legalMoves(s)).toHaveLength(24);
    expect(legalMoves(s).some((m) => m.x === 0 && m.z === 0)).toBe(false);
  });

  it('rejects moves when the player has no pieces left', () => {
    const s = createInitialState(0);
    expect(legalMoves(s)).toEqual([]);
    expect(() => applyMove(s, { x: 0, z: 0 })).toThrow();
  });
});

describe('rules: win detection — findWinRun exhaustive over all 3D 4-lines', () => {
  const lines = allWinLines();

  it('detects every possible 4-in-a-row geometry', () => {
    // 4 flat dirs x50 + vertical x50 + 8 rising dirs (20/8 starts each) = 302.
    expect(lines.length).toBe(302);
    for (const { dir, cells } of lines) {
      const grid = emptyGrid();
      for (const [x, z, y] of cells) grid[x]![z]![y] = 'white';
      const [lx, lz, ly] = cells[3]!;
      const run = findWinRun(grid, lx, lz, ly, 'white');
      expect(run, `expected run for dir ${JSON.stringify(dir)} cells ${JSON.stringify(cells)}`).not.toBeNull();
      expect(run!.map((p) => [p.x, p.z, p.y]).sort()).toEqual(cells.map((c) => [...c]).sort());
    }
  });

  it('returns null when the line is broken by another color', () => {
    for (const { cells } of lines.slice(0, 80)) {
      const grid = emptyGrid();
      for (const [x, z, y] of cells) grid[x]![z]![y] = 'white';
      const [bx, bz, by] = cells[1]!;
      grid[bx]![bz]![by] = 'black';
      const [lx, lz, ly] = cells[3]!;
      expect(findWinRun(grid, lx, lz, ly, 'white')).toBeNull();
    }
  });

  it('returns null when the run is only 3 long', () => {
    const grid = emptyGrid();
    const cells: Array<[number, number, number]> = [
      [0, 0, 0],
      [1, 0, 0],
      [2, 0, 0],
    ];
    for (const [x, z, y] of cells) grid[x]![z]![y] = 'white';
    expect(findWinRun(grid, 2, 0, 0, 'white')).toBeNull();
  });
});

describe('rules: win detection — applyMove integration', () => {
  it('detects a win only when the completing piece is placed (horizontal row)', () => {
    const state = playLine([
      [0, 0, 0],
      [1, 0, 0],
      [2, 0, 0],
      [3, 0, 0],
    ]);
    expect(state.winner).toBe('white');
    expect(state.winningCells).toHaveLength(4);
  });

  it('detects a flat diagonal win', () => {
    const state = playLine([
      [0, 0, 0],
      [1, 1, 0],
      [2, 2, 0],
      [3, 3, 0],
    ]);
    expect(state.winner).toBe('white');
    expect(state.winningCells!.map((p) => [p.x, p.z, p.y]).sort()).toEqual([
      [0, 0, 0],
      [1, 1, 0],
      [2, 2, 0],
      [3, 3, 0],
    ]);
  });

  it('detects a vertical 4-stack win', () => {
    let s = createInitialState();
    const seq: Array<[number, number]> = [
      [2, 2],
      [4, 4],
      [2, 2],
      [4, 3],
      [2, 2],
      [4, 2],
      [2, 2],
    ];
    for (const [x, z] of seq) s = applyMove(s, { x, z });
    expect(s.winner).toBe('white');
    expect(s.winningCells).toHaveLength(4);
    expect(s.winningCells!.every((p) => p.x === 2 && p.z === 2)).toBe(true);
  });

  it('detects a rising "stepping up" diagonal via legal play', () => {
    // White: (0,0,y0), (1,0,y1), (2,0,y2), (3,0,y3). Black fills the pieces
    // beneath (1,0),(2,0),(3,0); white pads with harmless (4,4) pieces while
    // waiting for the black fills.
    let s = createInitialState();
    const seq: Array<[number, number]> = [
      [0, 0],
      [1, 0],
      [4, 4],
      [2, 0],
      [1, 0],
      [2, 0],
      [4, 4],
      [3, 0],
      [2, 0],
      [3, 0],
      [4, 4],
      [3, 0],
      [3, 0],
    ];
    for (const [x, z] of seq) s = applyMove(s, { x, z });
    expect(s.winner).toBe('white');
    expect(s.winningCells!.map((p) => [p.x, p.z, p.y])).toEqual([
      [0, 0, 0],
      [1, 0, 1],
      [2, 0, 2],
      [3, 0, 3],
    ]);
  });

  it('does not detect a win for 4 non-collinear same-color pieces (2x2 block)', () => {
    let s = createInitialState();
    const white: Array<[number, number]> = [
      [0, 0],
      [1, 0],
      [0, 1],
      [1, 1],
    ];
    const black: Array<[number, number]> = [
      [4, 4],
      [4, 3],
      [4, 2],
    ];
    for (let i = 0; i < 4; i++) {
      s = applyMove(s, { x: white[i]![0], z: white[i]![1] });
      if (i < 3) s = applyMove(s, { x: black[i]![0], z: black[i]![1] });
    }
    expect(s.winner).toBeNull();
  });

  it('does not detect a win when the line has a gap', () => {
    let s = createInitialState();
    const seq: Array<[number, number]> = [
      [0, 0],
      [4, 4],
      [0, 1],
      [4, 3],
      [0, 2],
      [4, 2],
      [0, 4],
    ];
    for (const [x, z] of seq) s = applyMove(s, { x, z });
    expect(s.winner).toBeNull();
  });

  it('does not detect a win when the row mixes colors', () => {
    let s = createInitialState();
    const seq: Array<[number, number]> = [
      [0, 0],
      [1, 0],
      [2, 0],
      [4, 4],
      [3, 0],
    ];
    for (const [x, z] of seq) s = applyMove(s, { x, z });
    expect(s.winner).toBeNull();
  });

  it('does not detect a win for same-color pieces with a gap in a stack', () => {
    let s = createInitialState();
    const seq: Array<[number, number]> = [
      [0, 0],
      [0, 1],
      [0, 0],
      [4, 4],
      [0, 0],
      [4, 3],
      [0, 4],
      [4, 2],
    ];
    // white at (0,0) y0,y2,y4 and (0,1) y0; no 4-run anywhere.
    for (const [x, z] of seq) s = applyMove(s, { x, z });
    expect(s.winner).toBeNull();
  });

  it('highlights the full extended run (5 in a row)', () => {
    // Whites fill (0,0)..(0,4) in the order y0, y2, y4, y1, y3 so the game does
    // not end early (no 4 consecutive until the final piece lands in the gap).
    let s = createInitialState();
    const seq: Array<[number, number]> = [
      [0, 0],
      [4, 4],
      [0, 2],
      [3, 4],
      [0, 4],
      [4, 3],
      [0, 1],
      [3, 3],
      [0, 3],
    ];
    for (const [x, z] of seq) s = applyMove(s, { x, z });
    expect(s.winner).toBe('white');
    expect(s.winningCells).toHaveLength(5);
    expect(s.winningCells!.every((p) => p.x === 0 && p.y === 0 && p.player === 'white')).toBe(true);
    expect(new Set(s.winningCells!.map((p) => p.z))).toEqual(new Set([0, 1, 2, 3, 4]));
  });
});

describe('rules: draw', () => {
  it('declares a draw when all pieces are placed without a winner', () => {
    let s = createInitialState(1);
    s = applyMove(s, { x: 0, z: 0 });
    expect(s.winner).toBeNull();
    s = applyMove(s, { x: 0, z: 1 });
    expect(s.winner).toBe('draw');
    expect(legalMoves(s)).toEqual([]);
  });

  it('reports a win instead of a draw when the last move wins', () => {
    // 4 pieces each; black completes a vertical 4-stack on its 4th (final) move.
    let s = createInitialState(4);
    const seq: Array<[number, number]> = [
      [4, 4],
      [0, 0],
      [3, 3],
      [0, 1],
      [4, 2],
      [0, 2],
      [3, 1],
      [0, 3],
    ];
    for (const [x, z] of seq) s = applyMove(s, { x, z });
    expect(s.piecesLeft).toEqual({ white: 0, black: 0 });
    expect(s.winner).toBe('black');
    expect(s.winningCells).toHaveLength(4);
  });
});

describe('rules: no moves after game over', () => {
  it('refuses moves after a win', () => {
    const s = playLine([
      [0, 0, 0],
      [1, 1, 0],
      [2, 2, 0],
      [3, 3, 0],
    ]);
    expect(s.winner).toBe('white');
    expect(() => applyMove(s, { x: 4, z: 4 })).toThrow();
  });

  it('refuses moves after a draw', () => {
    let s = createInitialState(1);
    s = applyMove(s, { x: 0, z: 0 });
    s = applyMove(s, { x: 0, z: 1 });
    expect(s.winner).toBe('draw');
    expect(() => applyMove(s, { x: 1, z: 0 })).toThrow();
  });
});
