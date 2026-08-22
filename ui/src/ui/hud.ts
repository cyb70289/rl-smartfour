import type { GameConfig } from '../game/engine';
import type { GameState, Mode, PlayerSlot, ThinkSettings } from '../game/types';
import { modeOf, humanColorOf } from '../game/types';

export interface HudCallbacks {
  onRevert(): void;
  onNewGame(config: GameConfig): void;
  onPlayPause(): void;
  onStep(): void;
  onSettingsChange(settings: ThinkSettings): void;
}

export const EFFORT_KEY = 'smartfour.effort';
const EFFORT_VALUES = ['0', '500', '2000'];
const DEFAULT_EFFORT = '500';

function cap(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** Binds and updates the DOM side panel; owns the game-over / error banner. */
export class Hud {
  private statusEl: HTMLElement;
  private revertBtn: HTMLButtonElement;
  private newGameBtn: HTMLButtonElement;
  private whiteKindRadios: NodeListOf<HTMLInputElement>;
  private blackKindRadios: NodeListOf<HTMLInputElement>;
  private whiteSelect: HTMLSelectElement;
  private blackSelect: HTMLSelectElement;
  private effortRadios: NodeListOf<HTMLInputElement>;
  private banner: HTMLDivElement;
  private errorText = '';
  private checkpoints: string[] = [];
  private mode: Mode = 'person';

  constructor(root: HTMLElement, private cb: HudCallbacks) {
    this.statusEl = byId(root, 'status');
    this.revertBtn = byId<HTMLButtonElement>(root, 'revert-btn');
    this.newGameBtn = byId<HTMLButtonElement>(root, 'new-game-btn');
    this.whiteKindRadios = root.querySelectorAll<HTMLInputElement>('input[name="white-kind"]');
    this.blackKindRadios = root.querySelectorAll<HTMLInputElement>('input[name="black-kind"]');
    this.whiteSelect = byId<HTMLSelectElement>(root, 'white-checkpoint');
    this.blackSelect = byId<HTMLSelectElement>(root, 'black-checkpoint');
    this.effortRadios = root.querySelectorAll<HTMLInputElement>('input[name="effort"]');

    this.banner = document.createElement('div');
    this.banner.className = 'banner';
    this.banner.style.display = 'none';
    const sceneContainer = document.getElementById('scene-container');
    sceneContainer?.appendChild(this.banner);

    // In auto play the action buttons become Play/Pause and Step; the click
    // target depends on the current mode, so both handlers dispatch on it.
    this.revertBtn.addEventListener('click', () => {
      if (this.mode === 'autoplay') this.cb.onStep();
      else this.cb.onRevert();
    });
    this.newGameBtn.addEventListener('click', () => {
      if (this.mode === 'autoplay') this.cb.onPlayPause();
      else this.cb.onNewGame(this.pendingConfig());
    });
    // Player selection changes restart the game immediately.
    const wireSlot = (radios: NodeListOf<HTMLInputElement>, select: HTMLSelectElement): void => {
      radios.forEach((r) =>
        r.addEventListener('change', () => {
          this.refreshSetup();
          this.cb.onNewGame(this.pendingConfig());
        }),
      );
      select.addEventListener('change', () => this.cb.onNewGame(this.pendingConfig()));
    };
    wireSlot(this.whiteKindRadios, this.whiteSelect);
    wireSlot(this.blackKindRadios, this.blackSelect);
    // Think effort applies to the model(s) immediately, no new game needed.
    this.effortRadios.forEach((r) =>
      r.addEventListener('change', () => {
        this.saveEffort();
        this.cb.onSettingsChange({ effort: Number(this.selectedValue(this.effortRadios)) });
      }),
    );

    this.restoreEffort();
    this.refreshSetup();
    void this.loadCheckpoints();
  }

  /** Update every panel element from the game state. */
  sync(state: GameState): void {
    this.mode = modeOf(state.white, state.black);

    // Top bar: status plus the human's remaining pieces (current player's in
    // auto play, where there is no human).
    let text: string;
    let cls = '';
    if (state.machineThinking && state.thinking) {
      text = this.mode === 'autoplay' ? `${cap(state.current)} is thinking…` : 'Machine is thinking…';
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
      const human = humanColorOf(state);
      const you = human !== null && state.current === human ? ' (you)' : '';
      text = `${cap(state.current)} to move${you}`;
    }
    const human = humanColorOf(state) ?? state.current;
    this.statusEl.textContent = `${text} · ${state.piecesLeft[human]} pieces`;
    this.statusEl.className = `status ${cls}`.trim();

    if (this.mode === 'autoplay') {
      this.newGameBtn.textContent = state.autoplay ? 'Pause' : 'Play';
      this.newGameBtn.disabled = false;
      this.revertBtn.textContent = 'Step';
      this.revertBtn.disabled = false;
    } else {
      this.newGameBtn.textContent = 'New Game';
      this.newGameBtn.disabled = false;
      this.revertBtn.textContent = 'Revert';
      this.revertBtn.disabled = state.machineThinking || (state.winner === null && !state.revertAvailable);
    }

    // Banner: errors and finished games only; anything else (including a
    // revert of a finished game) clears it.
    if (this.errorText) {
      this.showBanner(this.errorText, 'error');
      this.errorText = '';
    } else if (state.winner) {
      const cls2 = state.winner === 'draw' ? 'draw' : `win-${state.winner}`;
      this.showBanner(state.winner === 'draw' ? 'Draw' : `${cap(state.winner)} wins!`, cls2);
    } else {
      this.hideBanner();
    }
  }

  showError(msg: string): void {
    this.errorText = `Machine error: ${msg}`;
  }

  private async loadCheckpoints(): Promise<void> {
    try {
      const res = await fetch('/api/checkpoints');
      const body: unknown = await res.json();
      if (typeof body === 'object' && body !== null && 'checkpoints' in body && Array.isArray(body.checkpoints)) {
        this.checkpoints = body.checkpoints.filter((n): n is string => typeof n === 'string');
      }
    } catch {
      // The dropdowns stay empty/disabled; model play still attempts best.pt.
    }
    this.populateSelect(this.whiteSelect);
    this.populateSelect(this.blackSelect);
    this.refreshSetup();
  }

  private populateSelect(sel: HTMLSelectElement): void {
    const prev = sel.value;
    sel.replaceChildren();
    if (this.checkpoints.length === 0) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'no models';
      sel.appendChild(opt);
      sel.disabled = true;
      return;
    }
    sel.disabled = false;
    for (const name of this.checkpoints) {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name.replace(/\.pt$/, '');
      sel.appendChild(opt);
    }
    if (this.checkpoints.includes(prev)) sel.value = prev;
  }

  private selectedValue(radios: NodeListOf<HTMLInputElement>): string {
    for (const r of radios) {
      if (r.checked) return r.value;
    }
    return '';
  }

  private slotOf(kindRadios: NodeListOf<HTMLInputElement>, select: HTMLSelectElement): PlayerSlot {
    return this.selectedValue(kindRadios) === 'model'
      ? { kind: 'model', checkpoint: select.value || 'best.pt' }
      : { kind: 'human' };
  }

  private pendingConfig(): GameConfig {
    return {
      white: this.slotOf(this.whiteKindRadios, this.whiteSelect),
      black: this.slotOf(this.blackKindRadios, this.blackSelect),
      settings: {
        effort: Number(this.selectedValue(this.effortRadios) || DEFAULT_EFFORT),
      },
    };
  }

  private refreshSetup(): void {
    const whiteModel = this.selectedValue(this.whiteKindRadios) === 'model';
    const blackModel = this.selectedValue(this.blackKindRadios) === 'model';
    const noModels = this.checkpoints.length === 0;
    this.whiteSelect.disabled = !whiteModel || noModels;
    this.blackSelect.disabled = !blackModel || noModels;
    this.effortRadios.forEach((r) => {
      r.disabled = !whiteModel && !blackModel;
    });
  }

  private restoreEffort(): void {
    let value = DEFAULT_EFFORT;
    try {
      const stored = localStorage.getItem(EFFORT_KEY);
      if (stored !== null && EFFORT_VALUES.includes(stored)) value = stored;
    } catch {
      // storage unavailable — keep the default
    }
    for (const r of this.effortRadios) {
      if (r.value === value) r.checked = true;
    }
  }

  private saveEffort(): void {
    const value = this.selectedValue(this.effortRadios);
    if (!EFFORT_VALUES.includes(value)) return;
    try {
      localStorage.setItem(EFFORT_KEY, value);
    } catch {
      // storage unavailable — the choice still applies for this session
    }
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
