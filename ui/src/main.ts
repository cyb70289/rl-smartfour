import './style.css';
import { GameController } from './game/controller';
import { ModelMachinePlayer } from './game/machine';
import { loadOpenStates } from './game/openbook';
import { GameScene } from './ui/scene';
import { Hud } from './ui/hud';
import { OpenStateViewer } from './ui/viewer';
import type { GameConfig } from './game/engine';
import type { GameState, PlayerSlot, ThinkSettings } from './game/types';

const container = document.getElementById('scene-container')!;
if (!container) throw new Error('missing #scene-container');

/** Opening-book viewer mode (SMARTFOUR_VIEW=openbook): no players, no play
 * controls — just the board rendering whichever state is selected. */
async function startOpenbookViewer(): Promise<void> {
  const statusEl = document.getElementById('status')!;
  const hintbar = document.getElementById('hintbar');
  if (hintbar) hintbar.textContent = 'Read-only view · drag to rotate · scroll to zoom';

  let states: GameState[];
  try {
    states = await loadOpenStates();
  } catch (err) {
    statusEl.textContent = `Failed to load open states: ${String(err)}`;
    return;
  }

  const scene = new GameScene(container, { onColumnClick: () => {} });
  const viewer = new OpenStateViewer(document.getElementById('panel')!, states.length, (i) => {
    const s = states[i]!;
    scene.sync(s, false);
    const name = s.current.charAt(0).toUpperCase() + s.current.slice(1);
    statusEl.textContent = `no.${i + 1} · ${name} to move`;
  });
  viewer.select(0);
}

function startGame(): void {
  const EFFORT_KEY = 'smartfour.effort';
  const EFFORT_VALUES = [0, 500, 2000];
  const DEFAULT_EFFORT = 500;

  function loadEffort(): number {
    try {
      const value = Number(localStorage.getItem(EFFORT_KEY));
      if (EFFORT_VALUES.includes(value)) return value;
    } catch {
      // storage unavailable — fall through to the default
    }
    return DEFAULT_EFFORT;
  }

  const DEFAULT_WHITE: PlayerSlot = { kind: 'human' };
  // Placeholder checkpoint until the list loads: the bridge falls back to the
  // biggest best{n}.pt, and loadCheckpoints adopts it when untouched.
  const DEFAULT_BLACK: PlayerSlot = { kind: 'model', checkpoint: '' };
  /** The model player per slot; null = human. */
  function playersOf(config: GameConfig): { white: ModelMachinePlayer | null; black: ModelMachinePlayer | null } {
    return {
      white: config.white.kind === 'model' ? new ModelMachinePlayer(config.white.checkpoint) : null,
      black: config.black.kind === 'model' ? new ModelMachinePlayer(config.black.checkpoint) : null,
    };
  }

  let config: GameConfig = {
    white: DEFAULT_WHITE,
    black: DEFAULT_BLACK,
    settings: { effort: loadEffort() },
  };

  function canHumanMove(s: GameState): boolean {
    if (s.winner !== null || s.machineThinking) return false;
    return (s.current === 'white' ? s.white : s.black).kind === 'human';
  }

  const controller = new GameController(playersOf(config), config, (err) => {
    hud.showError(String(err));
  });

  const scene = new GameScene(container, {
    onColumnClick: (x, z) => {
      if (canHumanMove(controller.state)) controller.humanMove({ x, z });
    },
  });

  const hud = new Hud(document.getElementById('app')!, {
    onRevert: () => controller.revert(),
    onReset: (cfg) => {
      config = cfg;
      controller.reset(cfg, playersOf(cfg));
    },
    onPlayersChange: (cfg) => {
      config = cfg;
      controller.setConfig(cfg, playersOf(cfg));
    },
    onPlayPause: () => {
      if (controller.state.autoplay) controller.pause();
      else controller.play();
    },
    onStep: () => controller.step(),
    onSettingsChange: (settings: ThinkSettings) => controller.updateSettings(settings),
  });

  function sync(): void {
    const s = controller.state;
    scene.sync(s, canHumanMove(s));
    hud.sync(s);
  }

  controller.subscribe(sync);
  sync();
}


async function boot(): Promise<void> {
  let openbookView = false;
  try {
    const res = await fetch('/api/viewmode');
    const mode: unknown = (await res.json()).openbookView;
    openbookView = res.ok && mode === true;
  } catch {
    // endpoint absent (e.g. static hosting) — fall back to the game
  }
  if (openbookView) void startOpenbookViewer();
  else startGame();
}

void boot();
