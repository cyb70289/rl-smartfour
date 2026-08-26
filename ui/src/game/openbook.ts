import type { GameState, Player } from './types';
import { BOARD_SIZE, DEFAULT_PIECES, STACK_HEIGHT } from './types';

/** One opening-book entry: a 5x5 array of 5-char stack strings. Row index
 * is the board x, char index the stack level y bottom-to-top; 'w' = white,
 * 'b' = black, '.' = empty. Mirrors model/openbook.json. */
export type BookEntry = string[][];

/**
 * Parses one book entry into an immutable, read-only GameState for display.
 * Strict like the Python loader: raises Error naming the entry on any
 * malformation. The side to move follows from piece counts (equal -> white).
 */
export function entryToState(entry: BookEntry, index: number): GameState {
  const who = `openbook: entry ${index}: `;
  if (!Array.isArray(entry) || entry.length !== BOARD_SIZE) {
    throw new Error(`${who}expected ${BOARD_SIZE} rows`);
  }
  const grid: (Player | null)[][][] = [];
  let whites = 0;
  let blacks = 0;
  for (let x = 0; x < BOARD_SIZE; x++) {
    const row = entry[x];
    if (!Array.isArray(row) || row.length !== BOARD_SIZE) {
      throw new Error(`${who}row ${x} must have ${BOARD_SIZE} columns`);
    }
    const colPlane: (Player | null)[][] = [];
    for (let z = 0; z < BOARD_SIZE; z++) {
      const cell = row[z];
      if (typeof cell !== 'string' || cell.length !== STACK_HEIGHT) {
        throw new Error(`${who}cell (${x},${z}) must be a ${STACK_HEIGHT}-char string`);
      }
      const stack: (Player | null)[] = [];
      let gap = false;
      for (let y = 0; y < STACK_HEIGHT; y++) {
        const ch = cell[y];
        if (ch === '.') {
          gap = true;
          stack.push(null);
        } else if (ch === 'w' || ch === 'b') {
          if (gap) throw new Error(`${who}cell (${x},${z}) has a floating piece`);
          stack.push(ch === 'w' ? 'white' : 'black');
          if (ch === 'w') whites++;
          else blacks++;
        } else {
          throw new Error(`${who}cell (${x},${z}) has invalid char '${ch}'`);
        }
      }
      colPlane.push(stack);
    }
    grid.push(colPlane);
  }
  if (Math.abs(whites - blacks) > 1) {
    throw new Error(`${who}unbalanced material: ${whites} white vs ${blacks} black`);
  }

  return {
    grid,
    piecesLeft: { white: DEFAULT_PIECES - whites, black: DEFAULT_PIECES - blacks },
    piecesPerPlayer: DEFAULT_PIECES,
    current: whites === blacks ? 'white' : 'black',
    history: [],
    lastPlaced: null,
    winner: null,
    winningCells: null,
    revertAvailable: false,
    machineThinking: false,
    thinking: false,
    autoplay: false,
    white: { kind: 'human' },
    black: { kind: 'human' },
    settings: { effort: 0 },
  };
}

/** Fetches and parses every entry of /openbook.json into view states. */
export async function loadOpenStates(): Promise<GameState[]> {
  const res = await fetch('/openbook.json');
  if (!res.ok) throw new Error(`GET /openbook.json failed: HTTP ${res.status}`);
  const entries: unknown = await res.json();
  if (!Array.isArray(entries)) throw new Error('openbook.json must hold a list of entries');
  return entries.map((e, i) => entryToState(e as BookEntry, i));
}
