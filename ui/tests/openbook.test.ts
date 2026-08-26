import { describe, expect, it } from 'vitest';
import { entryToState } from '../src/game/openbook';

const EMPTY_ROW = ['.....', '.....', '.....', '.....', '.....'];

function entry(rows: string[][]): string[][] {
  return rows;
}

describe('entryToState', () => {
  it('parses stacks bottom-up and derives white to move on equal counts', () => {
    const rows = [EMPTY_ROW, EMPTY_ROW, EMPTY_ROW, EMPTY_ROW, EMPTY_ROW];
    const state = entryToState(entry(rows), 0);
    expect(state.current).toBe('white');
    expect(state.piecesLeft).toEqual({ white: 32, black: 32 });
    expect(state.winner).toBeNull();
  });

  it('stack chars map y=0..4 bottom to top with x/z row order', () => {
    // "wb..." at (x=0,z=0): white under black.
    const row0 = ['wb...', ...EMPTY_ROW.slice(1)];
    const state = entryToState(entry([row0, EMPTY_ROW, EMPTY_ROW, EMPTY_ROW, EMPTY_ROW]), 0);
    expect(state.grid[0]![0]).toEqual(['white', 'black', null, null, null]);
    expect(state.current).toBe('white'); // equal counts after white+black
    expect(state.piecesLeft).toEqual({ white: 31, black: 31 });
  });

  it('rejects malformed entries naming the index', () => {
    expect(() => entryToState([EMPTY_ROW], 3)).toThrow(/entry 3/);
    const floatRow = [...EMPTY_ROW];
    floatRow[2] = '.w...'; // white above an empty level
    expect(() => entryToState(entry([floatRow, EMPTY_ROW, EMPTY_ROW, EMPTY_ROW, EMPTY_ROW]), 1))
      .toThrow(/floating piece/);
    const badCharRow = [...EMPTY_ROW];
    badCharRow[1] = 'wz...';
    expect(() => entryToState(entry([badCharRow, EMPTY_ROW, EMPTY_ROW, EMPTY_ROW, EMPTY_ROW]), 0))
      .toThrow(/invalid char/);
  });
});
