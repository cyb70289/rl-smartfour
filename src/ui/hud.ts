import type { GameConfig } from '../game/engine';
import type { GameState, Player } from '../game/types';

export interface HudCallbacks {
  onRevert(): void;
  onNewGame(config: GameConfig): void;
}

function cap(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** Binds and updates the DOM side panel; owns the game-over / error banner. */
export class Hud {
  private statusEl: HTMLElement;
  private whiteCountEl: HTMLElement;
  private blackCountEl: HTMLElement;
  private lastMoveEl: HTMLElement;
  private revertBtn: HTMLButtonElement;
  private newGameBtn: HTMLButtonElement;
  private modeSelect: HTMLSelectElement;
  private colorField: HTMLLabelElement;
  private colorSelect: HTMLSelectElement;
  private thinkDisable: HTMLInputElement;
  private effortRange: HTMLInputElement;
  private effortValue: HTMLElement;
  private banner: HTMLDivElement;
  private errorText = '';

  constructor(root: HTMLElement, private cb: HudCallbacks) {
    this.statusEl = byId(root, 'status');
    this.whiteCountEl = byId(root, 'white-count');
    this.blackCountEl = byId(root, 'black-count');
    this.lastMoveEl = byId(root, 'last-move');
    this.revertBtn = byId<HTMLButtonElement>(root, 'revert-btn');
    this.newGameBtn = byId<HTMLButtonElement>(root, 'new-game-btn');
    this.modeSelect = byId<HTMLSelectElement>(root, 'mode-select');
    this.colorField = byId<HTMLLabelElement>(root, 'color-field');
    this.colorSelect = byId<HTMLSelectElement>(root, 'color-select');
    this.thinkDisable = byId<HTMLInputElement>(root, 'think-disable');
    this.effortRange = byId<HTMLInputElement>(root, 'effort-range');
    this.effortValue = byId(root, 'effort-value');

    this.banner = document.createElement('div');
    this.banner.className = 'banner';
    this.banner.style.display = 'none';
    const sceneContainer = document.getElementById('scene-container');
    sceneContainer?.appendChild(this.banner);

    this.revertBtn.addEventListener('click', () => this.cb.onRevert());
    this.newGameBtn.addEventListener('click', () => this.cb.onNewGame(this.pendingConfig()));
    this.modeSelect.addEventListener('change', () => this.refreshSetup());
    this.colorSelect.addEventListener('change', () => this.refreshSetup());
    this.thinkDisable.addEventListener('change', () => this.refreshSetup());
    this.effortRange.addEventListener('input', () => this.refreshSetup());

    this.refreshSetup();
  }

  /** Update every panel element from the game state. */
  sync(state: GameState): void {
    // Status line.
    let text: string;
    let cls = '';
    if (state.machineThinking) {
      text = 'Machine is thinking…';
      cls = 'thinking';
    } else if (state.winner === 'white') {
      text = 'White wins!';
      cls = 'win-white';
    } else if (state.winner === 'black') {
      text = 'Black wins!';
      cls = 'win-black';
    } else if (state.winner === 'draw') {
      text = 'Draw — no pieces left';
      cls = 'draw';
    } else {
      const you = state.mode === 'machine' && state.current === state.humanColor ? ' (you)' : '';
      text = `${cap(state.current)} to move${you}`;
    }
    this.statusEl.textContent = text;
    this.statusEl.className = `status ${cls}`.trim();

    // Counts and last move.
    this.whiteCountEl.textContent = String(state.piecesLeft.white);
    this.blackCountEl.textContent = String(state.piecesLeft.black);
    this.lastMoveEl.textContent = state.lastPlaced
      ? `(${state.lastPlaced.x}, ${state.lastPlaced.z}) · level ${state.lastPlaced.y}`
      : '—';

    // Revert.
    this.revertBtn.disabled = !state.revertAvailable || state.machineThinking || state.winner !== null;

    // Banner.
    if (this.errorText) {
      this.showBanner(this.errorText, 'error');
      this.errorText = '';
    } else if (state.winner) {
      const cls2 = state.winner === 'draw' ? 'draw' : `win-${state.winner}`;
      this.showBanner(state.winner === 'draw' ? 'Draw' : `${cap(state.winner)} wins!`, cls2);
    } else if (state.history.length === 0) {
      this.hideBanner();
    }
  }

  showError(msg: string): void {
    this.errorText = `Machine error: ${msg}`;
  }

  private pendingConfig(): GameConfig {
    return {
      mode: this.modeSelect.value === 'machine' ? 'machine' : 'person',
      humanColor: this.colorSelect.value as Player,
      settings: {
        disabled: this.thinkDisable.checked,
        effort: Number(this.effortRange.value),
      },
    };
  }

  private refreshSetup(): void {
    const machine = this.modeSelect.value === 'machine';
    this.colorField.style.display = machine ? '' : 'none';
    this.thinkDisable.disabled = !machine;
    this.effortRange.disabled = !machine || this.thinkDisable.checked;
    this.effortValue.textContent = this.effortRange.value;
  }

  private showBanner(text: string, cls: string): void {
    this.banner.textContent = text;
    this.banner.className = `banner ${cls}`;
    this.banner.style.display = 'block';
  }

  private hideBanner(): void {
    this.banner.style.display = 'none';
  }
}

function byId<T extends HTMLElement = HTMLElement>(root: HTMLElement, id: string): T {
  const el = root.querySelector<T>(`#${id}`);
  if (!el) throw new Error(`missing element #${id}`);
  return el;
}
