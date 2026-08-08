import './style.css';
import { GameController } from './game/controller';
import { RandomMachinePlayer } from './game/machine';
import { GameScene } from './ui/scene';
import { Hud } from './ui/hud';
import type { GameConfig } from './game/engine';
import type { GameState } from './game/types';

const container = document.getElementById('scene-container')!;
if (!container) throw new Error('missing #scene-container');

const initialConfig: GameConfig = {
  mode: 'machine',
  humanColor: 'white',
  settings: { disabled: false, effort: 100 },
};

function canHumanMove(s: GameState): boolean {
  if (s.winner !== null || s.machineThinking) return false;
  if (s.mode === 'machine' && s.current !== s.humanColor) return false;
  return true;
}

const controller = new GameController(new RandomMachinePlayer(), initialConfig, (err) => {
  hud.showError(String(err));
});

const scene = new GameScene(container, {
  onColumnClick: (x, z) => {
    if (canHumanMove(controller.state)) controller.humanMove({ x, z });
  },
});

const hud = new Hud(document.getElementById('app')!, {
  onRevert: () => controller.revert(),
  onNewGame: (config) => controller.reset(config),
});

function sync(): void {
  const s = controller.state;
  scene.sync(s, canHumanMove(s));
  hud.sync(s);
}

controller.subscribe(sync);
sync();
