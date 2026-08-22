import { describe, it, expect } from 'vitest';
import { reduce, newGame, IllegalActionError } from '../src/game/engine';
import type { GameConfig } from '../src/game/engine';
import { cellAt, stackHeight } from '../src/game/rules';
import type { GameState } from '../src/game/types';
import { modeOf, humanColorOf } from '../src/game/types';

const person: GameConfig = { white: { kind: 'human' }, black: { kind: 'human' }, settings: { effort: 100 } };
const machineHumanWhite: GameConfig = { white: { kind: 'human' }, black: { kind: 'model', checkpoint: 'best1.pt' }, settings: { effort: 100 } };
const machineHumanBlack: GameConfig = { white: { kind: 'model', checkpoint: 'best1.pt' }, black: { kind: 'human' }, settings: { effort: 100 } };
const autoplay: GameConfig = { white: { kind: 'model', checkpoint: 'best1.pt' }, black: { kind: 'model', checkpoint: 'best2.pt' }, settings: { effort: 100 } };

/** Helper: apply a plain move (person mode, or human move in machine mode). */
function mv(s: GameState, x: number, z: number): GameState {
  return reduce(s, { type: 'move', move: { x, z } });
}

/** Alternating full-ply sequence of human (white) / machine (black) moves. */
const machineSeq: Array<{ x: number; z: number; kind: 'human' | 'machine' }> = [
  { x: 4, z: 4, kind: 'human' },
  { x: 0, z: 0, kind: 'machine' },
  { x: 3, z: 4, kind: 'human' },
  { x: 0, z: 0, kind: 'machine' },
  { x: 4, z: 3, kind: 'human' },
  { x: 0, z: 0, kind: 'machine' },
  { x: 3, z: 3, kind: 'human' },
  { x: 0, z: 0, kind: 'machine' },
];

/** Apply a sequence of pure machine moves (auto play). */
function autoplayMoves(s: GameState, seq: Array<{ x: number; z: number }>): GameState {
  for (const m of seq) s = reduce(s, { type: 'machine-move', move: m });
  return s;
}

describe('engine: modes and player slots', () => {
  it('derives person mode from two human slots', () => {
    const s = newGame(person);
    expect(modeOf(s.white, s.black)).toBe('person');
    expect(humanColorOf(s)).toBe('white');
    expect(s.machineThinking).toBe(false);
    expect(s.thinking).toBe(false);
    expect(s.winner).toBeNull();
  });

  it('machine mode, human white: the model does not think first', () => {
    const s = newGame(machineHumanWhite);
    expect(modeOf(s.white, s.black)).toBe('machine');
    expect(humanColorOf(s)).toBe('white');
    expect(s.machineThinking).toBe(false);
    expect(s.current).toBe('white');
  });

  it('machine mode, human black: the model (white) owes the first move', () => {
    const s = newGame(machineHumanBlack);
    expect(humanColorOf(s)).toBe('black');
    expect(s.current).toBe('white');
    expect(s.machineThinking).toBe(true);
  });

  it('auto play: white model owes the first move', () => {
    const s = newGame(autoplay);
    expect(modeOf(s.white, s.black)).toBe('autoplay');
    expect(humanColorOf(s)).toBeNull();
    expect(s.current).toBe('white');
    expect(s.machineThinking).toBe(true);
    expect(s.autoplay).toBe(false);
  });
});

describe('engine: turns and thinking flag', () => {
  it('person mode: alternating moves work and open the revert window', () => {
    let s = newGame(person);
    s = mv(s, 0, 0);
    expect(s.current).toBe('black');
    expect(s.revertAvailable).toBe(true);
    s = mv(s, 1, 1);
    expect(s.current).toBe('white');
    expect(s.revertAvailable).toBe(true);
  });

  it('machine mode: a human move makes the model owe a move', () => {
    let s = newGame(machineHumanWhite);
    s = mv(s, 2, 2);
    expect(s.machineThinking).toBe(true);
    expect(s.current).toBe('black');
  });

  it('machine mode: human move on the model\'s turn is rejected', () => {
    let s = newGame(machineHumanWhite);
    s = mv(s, 2, 2);
    expect(() => mv(s, 3, 3)).toThrow(IllegalActionError);
  });

  it('machine mode: machine-move only applies on the model\'s turn', () => {
    let s = newGame(machineHumanWhite);
    expect(() => reduce(s, { type: 'machine-move', move: { x: 0, z: 0 } })).toThrow(IllegalActionError);
    s = mv(s, 2, 2);
    s = reduce(s, { type: 'machine-move', move: { x: 0, z: 0 } });
    expect(s.machineThinking).toBe(false);
    expect(s.current).toBe('white');
    expect(cellAt(s.grid, 0, 0, 0)).toBe('black');
    expect(s.revertAvailable).toBe(true);
  });

  it('person mode: machine-move is rejected', () => {
    const s = newGame(person);
    expect(() => reduce(s, { type: 'machine-move', move: { x: 0, z: 0 } })).toThrow(IllegalActionError);
  });

  it('auto play: a human move is rejected (model must move)', () => {
    const s = newGame(autoplay);
    expect(() => mv(s, 2, 2)).toThrow(IllegalActionError);
  });

  it('auto play: a model move hands the turn to the other model, which owes a move', () => {
    let s = newGame(autoplay);
    s = reduce(s, { type: 'machine-move', move: { x: 2, z: 2 } });
    expect(s.current).toBe('black');
    expect(s.machineThinking).toBe(true);
    expect(s.history).toEqual([{ x: 2, z: 2, player: 'white' }]);
    s = reduce(s, { type: 'machine-move', move: { x: 4, z: 4 } });
    expect(s.current).toBe('white');
    expect(s.machineThinking).toBe(true);
  });

  it('machine mode: machine wins and the thinking flag clears', () => {
    let s = newGame(machineHumanWhite);
    for (const step of machineSeq) {
      s = reduce(s, step.kind === 'human' ? { type: 'move', move: step } : { type: 'machine-move', move: step });
    }
    expect(s.winner).toBe('black');
    expect(s.machineThinking).toBe(false);
    expect(s.winningCells).toHaveLength(4);
  });

  it('auto play: a model wins and the thinking flag clears', () => {
    // White (model) stacks four in column (0,0); black (model) fills the rest.
    const seq: Array<{ x: number; z: number }> = [
      { x: 0, z: 0 },
      { x: 4, z: 4 },
      { x: 0, z: 0 },
      { x: 4, z: 3 },
      { x: 0, z: 0 },
      { x: 4, z: 2 },
      { x: 0, z: 0 },
    ];
    const s = autoplayMoves(newGame(autoplay), seq);
    expect(s.winner).toBe('white');
    expect(s.machineThinking).toBe(false);
    expect(s.winningCells).toHaveLength(4);
  });

  it('human move after game over is rejected', () => {
    let s = newGame(machineHumanWhite);
    for (const step of machineSeq) {
      s = reduce(s, step.kind === 'human' ? { type: 'move', move: step } : { type: 'machine-move', move: step });
    }
    expect(s.winner).toBe('black');
    expect(() => mv(s, 1, 1)).toThrow(IllegalActionError);
  });
});

describe('engine: revert (person mode)', () => {
  it('undoes the single most recent move and returns the turn to its mover', () => {
    let s = newGame(person);
    s = mv(s, 0, 0);
    s = mv(s, 1, 1);
    expect(s.history).toHaveLength(2);
    s = reduce(s, { type: 'revert' });
    expect(s.history).toHaveLength(1);
    expect(s.current).toBe('black');
    expect(s.revertAvailable).toBe(false);
    expect(cellAt(s.grid, 1, 1, 0)).toBeNull();
    expect(cellAt(s.grid, 0, 0, 0)).toBe('white');
    expect(stackHeight(s.grid, 0, 0)).toBe(1);
    expect(s.winner).toBeNull();
  });

  it('revert window is consumed by a revert (no double revert)', () => {
    let s = newGame(person);
    s = mv(s, 0, 0);
    s = reduce(s, { type: 'revert' });
    expect(s.revertAvailable).toBe(false);
    expect(() => reduce(s, { type: 'revert' })).toThrow(IllegalActionError);
  });

  it('a fresh move reopens the revert window', () => {
    let s = newGame(person);
    s = mv(s, 0, 0);
    s = reduce(s, { type: 'revert' });
    s = mv(s, 2, 2);
    expect(s.revertAvailable).toBe(true);
    s = reduce(s, { type: 'revert' });
    expect(s.history).toEqual([]);
    expect(s.current).toBe('white');
  });

  it('reverts the most recent move even after the opponent has moved', () => {
    let s = newGame(person);
    s = mv(s, 0, 0); // white
    s = mv(s, 1, 1); // black
    s = reduce(s, { type: 'revert' }); // undo black's move
    expect(s.history).toEqual([{ x: 0, z: 0, player: 'white' }]);
    expect(s.current).toBe('black');
    expect(s.revertAvailable).toBe(false);
  });

  it('rejects revert with no moves and while thinking', () => {
    expect(() => reduce(newGame(person), { type: 'revert' })).toThrow(IllegalActionError);

    let s = newGame(machineHumanWhite);
    s = mv(s, 2, 2);
    expect(s.machineThinking).toBe(true);
    expect(() => reduce(s, { type: 'revert' })).toThrow(IllegalActionError);
  });

  it('reverts a finished game in person mode', () => {
    let s = newGame(person);
    const seq: Array<[number, number]> = [
      [0, 0],
      [4, 4],
      [1, 0],
      [4, 3],
      [2, 0],
      [4, 2],
      [3, 0],
    ];
    for (const [x, z] of seq) s = mv(s, x, z);
    expect(s.winner).toBe('white');
    s = reduce(s, { type: 'revert' });
    expect(s.winner).toBeNull();
    expect(s.winningCells).toBeNull();
    expect(s.history).toHaveLength(6);
    expect(s.current).toBe('white'); // turn returns to the mover of the undone move
    expect(cellAt(s.grid, 3, 0, 0)).toBeNull();
    expect(s.revertAvailable).toBe(false);
  });

  it('rejects moves after a revert consumed the window', () => {
    // Revert is not required for this, but guards ordering: after reverting,
    // the reverted player must move again before any further revert.
    let s = newGame(person);
    s = mv(s, 0, 0);
    s = reduce(s, { type: 'revert' });
    expect(() => reduce(s, { type: 'revert' })).toThrow(IllegalActionError);
  });
});

describe('engine: revert (machine mode)', () => {
  it('reverts both the model and the last human moves', () => {
    let s = newGame(machineHumanWhite);
    s = mv(s, 2, 2); // human white
    s = reduce(s, { type: 'machine-move', move: { x: 0, z: 0 } });
    expect(s.history).toHaveLength(2);
    s = reduce(s, { type: 'revert' });
    expect(s.history).toEqual([]);
    expect(s.current).toBe('white');
    expect(s.machineThinking).toBe(false);
    expect(s.revertAvailable).toBe(false);
    expect(stackHeight(s.grid, 0, 0)).toBe(0);
    expect(stackHeight(s.grid, 2, 2)).toBe(0);
  });

  it('after revert, the human retries and the model responds again', () => {
    let s = newGame(machineHumanWhite);
    s = mv(s, 2, 2);
    s = reduce(s, { type: 'machine-move', move: { x: 0, z: 0 } });
    s = reduce(s, { type: 'revert' });
    s = mv(s, 3, 3); // retry with a different move
    expect(s.machineThinking).toBe(true);
    s = reduce(s, { type: 'machine-move', move: { x: 1, z: 1 } });
    expect(s.history).toHaveLength(2);
    expect(s.history[0]).toEqual({ x: 3, z: 3, player: 'white' });
    expect(s.history[1]).toEqual({ x: 1, z: 1, player: 'black' });
  });

  it('reverting a human win hands the turn back to the model', () => {
    let s = newGame(machineHumanWhite);
    // White (human) wins on the 7th ply: (0,0),(1,0),(2,0),(3,0) at level 0.
    const seq: Array<[number, number]> = [
      [0, 0],
      [4, 4],
      [1, 0],
      [4, 3],
      [2, 0],
      [4, 2],
      [3, 0],
    ];
    for (let i = 0; i < seq.length; i++) {
      s = reduce(
        s,
        i % 2 === 0
          ? { type: 'move', move: { x: seq[i]![0], z: seq[i]![1] } }
          : { type: 'machine-move', move: { x: seq[i]![0], z: seq[i]![1] } },
      );
    }
    expect(s.winner).toBe('white');
    s = reduce(s, { type: 'revert' });
    expect(s.winner).toBeNull();
    expect(s.history).toHaveLength(5);
    expect(s.current).toBe('black');
    expect(s.machineThinking).toBe(true); // model owes a move again
    expect(cellAt(s.grid, 3, 0, 0)).toBeNull();
  });

  it('reverting a machine win returns the turn to the human', () => {
    let s = newGame(machineHumanBlack);
    // White (machine) wins on the 7th ply; human black fills the rest.
    const seq: Array<[number, number]> = [
      [0, 0],
      [4, 4],
      [1, 0],
      [4, 3],
      [2, 0],
      [4, 2],
      [3, 0],
    ];
    for (let i = 0; i < seq.length; i++) {
      s = reduce(
        s,
        i % 2 === 0
          ? { type: 'machine-move', move: { x: seq[i]![0], z: seq[i]![1] } }
          : { type: 'move', move: { x: seq[i]![0], z: seq[i]![1] } },
      );
    }
    expect(s.winner).toBe('white');
    s = reduce(s, { type: 'revert' });
    expect(s.winner).toBeNull();
    expect(s.history).toHaveLength(5);
    expect(s.current).toBe('black');
    expect(s.machineThinking).toBe(false);
  });

  it('reverts a finished auto play game by popping both model moves', () => {
    let s = newGame(autoplay);
    const seq: Array<{ x: number; z: number }> = [
      { x: 0, z: 0 },
      { x: 4, z: 4 },
      { x: 1, z: 0 },
      { x: 4, z: 3 },
      { x: 2, z: 0 },
      { x: 4, z: 2 },
      { x: 3, z: 0 },
    ];
    s = autoplayMoves(s, seq);
    expect(s.winner).toBe('white');
    s = reduce(s, { type: 'revert' });
    expect(s.winner).toBeNull();
    expect(s.history).toHaveLength(5);
    expect(s.current).toBe('black');
    expect(s.machineThinking).toBe(true); // black model owes a move again
    expect(cellAt(s.grid, 3, 0, 0)).toBeNull();
    expect(cellAt(s.grid, 4, 2, 0)).toBeNull();
  });
});

describe('engine: reset', () => {
  it('starts a fresh game with the given config', () => {
    let s = newGame(person);
    s = mv(s, 0, 0);
    s = reduce(s, { type: 'reset', config: machineHumanBlack });
    expect(s.history).toEqual([]);
    expect(s.current).toBe('white');
    expect(s.machineThinking).toBe(true);
  });
});
