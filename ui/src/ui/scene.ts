import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { stackHeight } from '../game/rules';
import { BOARD_SIZE, STACK_HEIGHT } from '../game/types';
import type { GameState, Player } from '../game/types';

const CELL = 1.15 * Math.SQRT2; // board 2x in area
const BASE_Y = 0.18;
const PIECE_H = 0.45; // 1.5x taller
const PIECE_R = 0.36 / Math.SQRT2; // 1/2 footprint area

const BODY_H = PIECE_H / 2;

/** Default framing: close, natural play view. */
const FOV = 50;
const CAMERA_DEFAULT = new THREE.Vector3(5.6, 5.2, 5.6);
const TARGET_DEFAULT = new THREE.Vector3(0, 1.1, 0);
/** Board center sits 1/5 of the container height above the viewport center
 * (2/5 of the vertical half-FOV, in tan space). */
const FRAME_UP = 0.4 * Math.tan((FOV * Math.PI) / 360);

const VIEW_KEY = 'smartfour.boardView.v1';

interface BoardView {
  camera: [number, number, number];
  target: [number, number, number];
}

function loadBoardView(): BoardView | null {
  try {
    const raw = localStorage.getItem(VIEW_KEY);
    if (!raw) return null;
    const v = JSON.parse(raw) as Partial<BoardView>;
    if (!Array.isArray(v.camera) || v.camera.length !== 3 || !Array.isArray(v.target) || v.target.length !== 3) {
      return null;
    }
    const nums = [...v.camera, ...v.target];
    if (!nums.every((n) => typeof n === 'number' && Number.isFinite(n))) return null;
    return { camera: v.camera as [number, number, number], target: v.target as [number, number, number] };
  } catch {
    return null; // corrupt payload or unavailable storage → default framing
  }
}

function saveBoardView(view: BoardView): void {
  try {
    localStorage.setItem(VIEW_KEY, JSON.stringify(view));
  } catch {
    // storage unavailable (private mode / quota): the view just won't persist
  }
}

function wx(x: number): number {
  return (x - 2) * CELL;
}
function wz(z: number): number {
  return (z - 2) * CELL;
}
function wy(y: number): number {
  // Stack pitch == piece height: stacked pieces touch, no floating gap.
  return BASE_Y + y * PIECE_H;
}

export interface SceneCallbacks {
  onColumnClick(x: number, z: number): void;
}

/** Renders the 5x5 board, pieces, highlights, and handles column picking. */
export class GameScene {
  private renderer: THREE.WebGLRenderer;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private rafId = 0;
  private resizeObserver: ResizeObserver;

  private pieceGroup = new THREE.Group();
  private marker: THREE.Object3D | null = null;
  private beamGroup = new THREE.Group();
  private ghost: THREE.Mesh;
  private pickGroup = new THREE.Group();
  private columnBoxes: THREE.Mesh[] = [];

  private state: GameState | null = null;
  private inputEnabled = false;
  private hovered: { x: number; z: number } | null = null;
  private downPos: { x: number; y: number } | null = null;
  private saveTimer = 0;

  private persistViewNow = (): void => {
    window.clearTimeout(this.saveTimer);
    saveBoardView({
      camera: [this.camera.position.x, this.camera.position.y, this.camera.position.z],
      target: [this.controls.target.x, this.controls.target.y, this.controls.target.z],
    });
  };

  private persistViewDebounced = (): void => {
    window.clearTimeout(this.saveTimer);
    this.saveTimer = window.setTimeout(this.persistViewNow, 500);
  };

  constructor(private container: HTMLElement, private cb: SceneCallbacks) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x3b4354); // light slate, black pieces stay visible

    this.camera = new THREE.PerspectiveCamera(FOV, container.clientWidth / container.clientHeight, 0.1, 100);

    // Default: close to the board, framed 1/5 container height above center.
    const shift = FRAME_UP * CAMERA_DEFAULT.distanceTo(TARGET_DEFAULT);
    this.camera.position.copy(CAMERA_DEFAULT).add(new THREE.Vector3(0, -shift, 0));

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.copy(TARGET_DEFAULT).add(new THREE.Vector3(0, -shift, 0));
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 3.5;
    this.controls.maxDistance = 22;
    this.controls.maxPolarAngle = Math.PI * 0.48;

    // Restore the exact last view, verbatim; no tweaks even if the stored
    // values sit outside today's orbit limits (they clamp on next interaction).
    const saved = loadBoardView();
    if (saved) {
      this.camera.position.set(saved.camera[0], saved.camera[1], saved.camera[2]);
      this.controls.target.set(saved.target[0], saved.target[1], saved.target[2]);
    }

    this.controls.addEventListener('change', this.persistViewDebounced);
    this.controls.addEventListener('end', this.persistViewNow);
    window.addEventListener('pagehide', this.persistViewNow);

    this.scene.add(new THREE.AmbientLight(0xffffff, 1.4));
    const key = new THREE.DirectionalLight(0xffffff, 1.9);
    key.position.set(7, 11, 5);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0x8899bb, 0.7);
    fill.position.set(-6, 3, -6);
    this.scene.add(fill);

    this.buildBoard();
    this.buildPickers();

    this.ghost = new THREE.Mesh(
      new THREE.CylinderGeometry(PIECE_R * 0.92, PIECE_R * 0.82, PIECE_H, 24),
      ghostMaterial('white'),
    );
    this.ghost.position.y = wy(0) + BODY_H;
    this.ghost.visible = false;
    this.scene.add(this.ghost);

    this.scene.add(this.pieceGroup);
    this.scene.add(this.beamGroup);

    this.renderer.domElement.addEventListener('pointerdown', this.onPointerDown);
    this.renderer.domElement.addEventListener('pointerup', this.onPointerUp);
    this.renderer.domElement.addEventListener('pointermove', this.onPointerMove);

    this.resizeObserver = new ResizeObserver(() => this.onResize());
    this.resizeObserver.observe(container);

    this.loop();
  }

  /** Rebuilds all scene content from the game state. */
  sync(state: GameState, inputEnabled: boolean): void {
    this.state = state;
    this.inputEnabled = inputEnabled;

    // Pieces.
    this.pieceGroup.clear();
    for (let x = 0; x < BOARD_SIZE; x++) {
      for (let z = 0; z < BOARD_SIZE; z++) {
        for (let y = 0; y < STACK_HEIGHT; y++) {
          const player = state.grid[x]![z]![y];
          if (!player) continue;
          const piece = makePiece(player);
          piece.position.set(wx(x), wy(y), wz(z));
          this.pieceGroup.add(piece);
        }
      }
    }

    this.updatePickers(state);

    // Last-move marker: a flat ring on top of the last placed piece.
    if (this.marker) {
      this.scene.remove(this.marker);
      this.marker = null;
    }
    if (state.lastPlaced) {
      const { x, z, y } = state.lastPlaced;
      // Ring hugs the piece: same 1.45x piece-radius ratio as the original
      // design, so it stays proportional to the now-thinner pieces.
      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(PIECE_R * 1.45, 0.03, 8, 32),
        new THREE.MeshStandardMaterial({ color: 0xffd54f, emissive: 0xffb300, emissiveIntensity: 0.9 }),
      );
      ring.rotation.x = Math.PI / 2;
      ring.position.set(wx(x), wy(y) + PIECE_H + 0.02, wz(z));
      this.scene.add(ring);
      this.marker = ring;
    }

    // Winning beam.
    this.beamGroup.clear();
    if (state.winningCells) {
      const cells = state.winningCells;
      const mat = new THREE.MeshStandardMaterial({ color: 0xffd54f, emissive: 0xffaa00, emissiveIntensity: 1.6 });
      const beamY = PIECE_H + 0.06;
      for (let i = 0; i < cells.length - 1; i++) {
        this.beamGroup.add(
          beamBetween(
            new THREE.Vector3(wx(cells[i]!.x), wy(cells[i]!.y) + beamY, wz(cells[i]!.z)),
            new THREE.Vector3(wx(cells[i + 1]!.x), wy(cells[i + 1]!.y) + beamY, wz(cells[i + 1]!.z)),
            mat,
          ),
        );
      }
      // Glow spheres above the winning cells.
      for (const c of cells) {
        const orb = new THREE.Mesh(new THREE.SphereGeometry(0.15, 12, 12), mat);
        orb.position.set(wx(c.x), wy(c.y) + beamY + 0.05, wz(c.z));
        this.beamGroup.add(orb);
      }
    }

    if (!inputEnabled) this.hovered = null;
    this.updateGhost();
  }

  dispose(): void {
    cancelAnimationFrame(this.rafId);
    window.clearTimeout(this.saveTimer);
    window.removeEventListener('pagehide', this.persistViewNow);
    this.controls.removeEventListener('change', this.persistViewDebounced);
    this.controls.removeEventListener('end', this.persistViewNow);
    this.resizeObserver.disconnect();
    this.renderer.domElement.removeEventListener('pointerdown', this.onPointerDown);
    this.renderer.domElement.removeEventListener('pointerup', this.onPointerUp);
    this.renderer.domElement.removeEventListener('pointermove', this.onPointerMove);
    this.controls.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  private buildBoard(): void {
    const base = new THREE.Mesh(
      new THREE.BoxGeometry(5 * CELL + 0.25, 0.34, 5 * CELL + 0.25),
      new THREE.MeshStandardMaterial({ color: 0x4a5266, roughness: 0.8 }),
    );
    base.position.y = BASE_Y - 0.17;
    this.scene.add(base);

    // Mesh-based grid lines. LineSegments are avoided because line rendering is
    // unreliable in some GL environments (e.g. software rasterizers).
    const span = (BOARD_SIZE - 1) * CELL;
    const gridMat = new THREE.MeshBasicMaterial({ color: 0x79829a });
    const lineY = BASE_Y + 0.006;
    const alongX = new THREE.BoxGeometry(span, 0.012, 0.02);
    const alongZ = new THREE.BoxGeometry(0.02, 0.012, span);
    for (let i = 0; i < BOARD_SIZE; i++) {
      const off = (i - 2) * CELL;
      const a = new THREE.Mesh(alongX, gridMat);
      a.position.set(0, lineY, off);
      this.scene.add(a);
      const b = new THREE.Mesh(alongZ, gridMat);
      b.position.set(off, lineY, 0);
      this.scene.add(b);
    }
  }

  private buildPickers(): void {
    const mat = new THREE.MeshBasicMaterial({ visible: false });
    const geo = new THREE.BoxGeometry(CELL * 0.92, 1, CELL * 0.92); // unit height; scaled per stack in sync()
    for (let x = 0; x < BOARD_SIZE; x++) {
      for (let z = 0; z < BOARD_SIZE; z++) {
        const box = new THREE.Mesh(geo, mat);
        box.userData = { x, z };
        this.pickGroup.add(box);
        this.columnBoxes.push(box);
      }
    }
    this.scene.add(this.pickGroup);
  }

  /** Match each picker box to its visible stack so near stacks only occlude what they actually cover. */
  private updatePickers(state: GameState): void {
    for (const box of this.columnBoxes) {
      const { x, z } = box.userData as { x: number; z: number };
      const h = Math.max(stackHeight(state.grid, x, z) * PIECE_H + 0.05, 0.05);
      box.scale.y = h;
      box.position.set(wx(x), BASE_Y + h / 2, wz(z));
    }
  }

  private onPointerDown = (e: PointerEvent): void => {
    if (e.button !== 0) return;
    this.downPos = { x: e.clientX, y: e.clientY };
  };

  private onPointerUp = (e: PointerEvent): void => {
    if (e.button !== 0 || !this.downPos) return;
    const dx = e.clientX - this.downPos.x;
    const dy = e.clientY - this.downPos.y;
    this.downPos = null;
    if (dx * dx + dy * dy > 36) return; // was a drag
    const hit = this.pick(e.clientX, e.clientY);
    if (hit && this.inputEnabled) {
      this.cb.onColumnClick(hit.x, hit.z);
    }
  };

  private onPointerMove = (e: PointerEvent): void => {
    if (!this.inputEnabled) return;
    const hit = this.pick(e.clientX, e.clientY);
    this.hovered = hit ? { x: hit.x, z: hit.z } : null;
    this.updateGhost();
  };

  private pick(clientX: number, clientY: number): { x: number; z: number } | null {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1,
    );
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(ndc, this.camera);
    const hits = raycaster.intersectObjects(this.columnBoxes, false);
    if (hits.length === 0) return null;
    return { x: hits[0]!.object.userData.x, z: hits[0]!.object.userData.z };
  }

  private updateGhost(): void {
    const state = this.state;
    if (!this.inputEnabled || !this.hovered || !state) {
      this.ghost.visible = false;
      return;
    }
    const { x, z } = this.hovered;
    const stack = state.grid[x]![z]!;
    const height = stack.filter((p) => p !== null).length;
    if (height >= STACK_HEIGHT) {
      this.ghost.visible = false;
      return;
    }
    this.ghost.material = ghostMaterial(state.current);
    // Anchor the ghost's bottom on the stack top so it matches the real
    // piece's height exactly (center would bury half of it in the stack).
    this.ghost.position.set(wx(x), wy(height) + BODY_H, wz(z));
    this.ghost.visible = true;
  }

  private onResize(): void {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (w === 0 || h === 0) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  private loop = (): void => {
    this.rafId = requestAnimationFrame(this.loop);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };
}

function makePiece(player: Player): THREE.Group {
  const g = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.CylinderGeometry(PIECE_R, PIECE_R * 0.84, PIECE_H, 24),
    player === 'white' ? pieceMaterial('white') : pieceMaterial('black'),
  );
  body.position.y = BODY_H;
  const rim = new THREE.Mesh(
    new THREE.TorusGeometry(PIECE_R * 0.99, 0.03, 8, 24),
    player === 'white' ? rimMaterial('white') : rimMaterial('black'),
  );
  rim.rotation.x = Math.PI / 2;
  // Rim fully inside the piece height: with pieces stacked flush it must not
  // poke into the seam above.
  rim.position.y = PIECE_H - 0.045;
  g.add(body, rim);
  return g;
}

function ghostMaterial(player: Player): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({
    color: player === 'white' ? 0xffffff : 0x0a0a0c,
    transparent: true,
    opacity: 0.4,
    roughness: 0.4,
  });
}

function pieceMaterial(player: Player): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial(
    player === 'white'
      ? { color: 0xececec, roughness: 0.35, metalness: 0.05 }
      : { color: 0x1b1c22, roughness: 0.45, metalness: 0.12 },
  );
}

function rimMaterial(player: Player): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial(
    player === 'white'
      ? { color: 0xb9bcc6, roughness: 0.4 }
      : { color: 0x050507, roughness: 0.5 },
  );
}

function beamBetween(a: THREE.Vector3, b: THREE.Vector3, mat: THREE.Material): THREE.Mesh {
  const dir = new THREE.Vector3().subVectors(b, a);
  const len = dir.length();
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, len, 8), mat);
  mesh.position.copy(a).add(b).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
  return mesh;
}
