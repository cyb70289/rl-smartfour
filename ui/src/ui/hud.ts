import type { GameConfig } from '../game/engine';
import type { GameState, Mode, PlayerSlot, ThinkSettings } from '../game/types';
import { modeOf, humanColorOf } from '../game/types';

export interface HudCallbacks {
  onRevert(): void;
  onReset(config: GameConfig): void;
  onPlayersChange(config: GameConfig): void;
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
  private playBtn: HTMLButtonElement;
  private resetBtn: HTMLButtonElement;
  private whiteKindRadios: NodeListOf<HTMLInputElement>;
  private blackKindRadios: NodeListOf<HTMLInputElement>;
  private whiteSelect: HTMLSelectElement;
  private blackSelect: HTMLSelectElement;
  private effortRadios: NodeListOf<HTMLInputElement>;
  private banner: HTMLDivElement;
  private errorText = '';
  private checkpoints: string[] = [];
  private mode: Mode = 'person';
  /** True once the user changed any player slot; disables the default-switch. */
  private userTouched = false;
  /** True while model-vs-model auto play is running: player selection and
   * think effort are locked. */
  private autoplayRunning = false;


  constructor(root: HTMLElement, private cb: HudCallbacks) {
    this.statusEl = byId(root, 'status');
    this.revertBtn = byId<HTMLButtonElement>(root, 'revert-btn');
    this.playBtn = byId<HTMLButtonElement>(root, 'play-btn');
    this.resetBtn = byId<HTMLButtonElement>(root, 'reset-btn');
    this.whiteKindRadios = root.querySelectorAll<HTMLInputElement>('input[name="white-kind"]');
    this.blackKindRadios = root.querySelectorAll<HTMLInputElement>('input[name="black-kind"]');
    this.whiteSelect = byId<HTMLSelectElement>(root, 'white-checkpoint');
    this.blackSelect = byId<HTMLSelectElement>(root, 'black-checkpoint');
    this.effortRadios = root.querySelectorAll<HTMLInputElement>('input[name="effort"]');

    this.banner = document.createElement('div');
    this.banner.className = 'banner';
    this.banner.style.display = 'none';
    const sceneContainer = document.getElementById('scene-container');
    // In auto play the action buttons become Play/Pause and Step; the click
    // target depends on the current mode, so the revert button dispatches.
    this.revertBtn.addEventListener('click', () => {
      if (this.mode === 'autoplay') this.cb.onStep();
      else this.cb.onRevert();
    });
    this.playBtn.addEventListener('click', () => this.cb.onPlayPause());
    this.resetBtn.addEventListener('click', () => this.cb.onReset(this.pendingConfig()));
    // Player selection changes apply to the current board immediately.
    const wireSlot = (radios: NodeListOf<HTMLInputElement>, select: HTMLSelectElement): void => {
      radios.forEach((r) =>
        r.addEventListener('change', () => {
          this.userTouched = true;
          this.refreshSetup();
          this.cb.onPlayersChange(this.pendingConfig());
        }),
      );
      select.addEventListener('change', () => {
        this.userTouched = true;
        this.cb.onPlayersChange(this.pendingConfig());
      });
    };
    wireSlot(this.whiteKindRadios, this.whiteSelect);
    wireSlot(this.blackKindRadios, this.blackSelect);
    // Think effort applies to the model(s) immediately.
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
      this.autoplayRunning = state.autoplay;
      this.playBtn.hidden = false;
      this.playBtn.textContent = state.autoplay ? 'Pause' : 'Play';
      this.playBtn.disabled = state.winner !== null; // Reset is the only restart
      this.revertBtn.textContent = 'Step';
      this.revertBtn.disabled = state.winner !== null;
    } else {
      this.autoplayRunning = false;
      this.playBtn.hidden = true;
      this.revertBtn.textContent = 'Revert';
      this.revertBtn.disabled = state.machineThinking || (state.winner === null && !state.revertAvailable);
    }
    this.resetBtn.disabled = false;

    // Re-apply slot/select/radio enablement: locking follows mode+running,
    // which can change without any panel input.
    this.refreshSetup();

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
      // The dropdowns stay empty/disabled; model play still falls back to the
      // server's default checkpoint (the biggest best{n}.pt) via an empty name.
    }
    this.populateSelect(this.whiteSelect);
    this.populateSelect(this.blackSelect);
    this.refreshSetup();
    // The boot config starts before the list arrives; if the user has not
    // touched the panel, adopt the real default (biggest best{n}.pt) so the
    // running game matches the dropdown.
    if (!this.userTouched && this.checkpoints.length > 0) {
      this.cb.onPlayersChange(this.pendingConfig());
    }
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
    // An empty selection (no checkpoints listed) sends '' so the bridge falls
    // back to its default; without any checkpoint that is a 503, never a move.
    return this.selectedValue(kindRadios) === 'model'
      ? { kind: 'model', checkpoint: select.value || '' }
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
    const lockPlayerControls = this.mode === 'autoplay' && this.autoplayRunning;
    this.whiteSelect.disabled = !whiteModel || noModels || lockPlayerControls;
    this.blackSelect.disabled = !blackModel || noModels || lockPlayerControls;
    this.effortRadios.forEach((r) => {
      r.disabled = (!whiteModel && !blackModel) || lockPlayerControls;
    });
    this.whiteKindRadios.forEach((r) => {
      r.disabled = lockPlayerControls;
    });
    this.blackKindRadios.forEach((r) => {
      r.disabled = lockPlayerControls;
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
