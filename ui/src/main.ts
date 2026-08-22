import './style.css';
import { GameController } from './game/controller';
import { ModelMachinePlayer } from './game/machine';
import { GameScene } from './ui/scene';
import { Hud } from './ui/hud';
import type { GameConfig } from './game/engine';
import type { GameState, PlayerSlot, ThinkSettings } from './game/types';

const container = document.getElementById('scene-container')!;
if (!container) throw new Error('missing #scene-container');

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
const DEFAULT_BLACK: PlayerSlot = { kind: 'model', checkpoint: 'best.pt' };

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
  onNewGame: (cfg) => {
    config = cfg;
    controller.reset(cfg, playersOf(cfg));
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
