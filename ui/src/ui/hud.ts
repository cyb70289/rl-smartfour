import type { GameConfig } from '../game/engine';
import type { GameState, Player, ThinkSettings } from '../game/types';

export interface HudCallbacks {
  onRevert(): void;
  onNewGame(config: GameConfig): void;
  onSettingsChange(settings: ThinkSettings): void;
}

function cap(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** Binds and updates the DOM side panel; owns the game-over / error banner. */
export class Hud {
  private statusEl: HTMLElement;
  private revertBtn: HTMLButtonElement;
  private newGameBtn: HTMLButtonElement;
  private modeRadios: NodeListOf<HTMLInputElement>;
  private colorField: HTMLElement;
  private colorRadios: NodeListOf<HTMLInputElement>;
  private effortRange: HTMLInputElement;
  private effortValue: HTMLElement;
  private banner: HTMLDivElement;
  private errorText = '';

  constructor(root: HTMLElement, private cb: HudCallbacks) {
    this.statusEl = byId(root, 'status');
    this.revertBtn = byId<HTMLButtonElement>(root, 'revert-btn');
    this.newGameBtn = byId<HTMLButtonElement>(root, 'new-game-btn');
    this.modeRadios = root.querySelectorAll<HTMLInputElement>('input[name="mode"]');
    this.colorField = byId(root, 'color-field');
    this.colorRadios = root.querySelectorAll<HTMLInputElement>('input[name="color"]');
    this.effortRange = byId<HTMLInputElement>(root, 'effort-range');
    this.effortValue = byId(root, 'effort-value');

    this.banner = document.createElement('div');
    this.banner.className = 'banner';
    this.banner.style.display = 'none';
    const sceneContainer = document.getElementById('scene-container');
    sceneContainer?.appendChild(this.banner);

    this.revertBtn.addEventListener('click', () => this.cb.onRevert());
    this.newGameBtn.addEventListener('click', () => this.cb.onNewGame(this.pendingConfig()));
    // Opponent / color changes restart the game immediately.
    this.modeRadios.forEach((r) =>
      r.addEventListener('change', () => {
        this.refreshSetup();
        this.cb.onNewGame(this.pendingConfig());
      }),
    );
    this.colorRadios.forEach((r) => r.addEventListener('change', () => this.cb.onNewGame(this.pendingConfig())));
    // Think effort applies to the machine immediately, no new game needed.
    this.effortRange.addEventListener('input', () => {
      this.refreshSetup();
      this.cb.onSettingsChange({ effort: Number(this.effortRange.value) });
    });

    this.refreshSetup();
  }

  /** Update every panel element from the game state. */
  sync(state: GameState): void {
    // Top bar: status plus the human's remaining pieces.
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
    const human = state.mode === 'machine' ? state.humanColor : state.current;
    this.statusEl.textContent = `${text} · ${state.piecesLeft[human]} pieces`;
    this.statusEl.className = `status ${cls}`.trim();

    // Revert: disabled while thinking or mid-game outside the revert window;
    // a finished game can always be reverted.
    this.revertBtn.disabled = state.machineThinking || (state.winner === null && !state.revertAvailable);

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

  private selectedValue(radios: NodeListOf<HTMLInputElement>): string {
    for (const r of radios) {
      if (r.checked) return r.value;
    }
    return '';
  }

  private pendingConfig(): GameConfig {
    return {
      mode: this.selectedValue(this.modeRadios) === 'machine' ? 'machine' : 'person',
      humanColor: this.selectedValue(this.colorRadios) as Player,
      settings: {
        effort: Number(this.effortRange.value),
      },
    };
  }

  private refreshSetup(): void {
    const machine = this.selectedValue(this.modeRadios) === 'machine';
    this.colorField.style.display = machine ? '' : 'none';
    this.effortRange.disabled = !machine;
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
