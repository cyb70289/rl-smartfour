/** Celebration overlay: confetti rain (color strips + petals) over the 3D
 * board when the human beats the model. Self-contained: creates a 2D canvas
 * on top of the scene container, animates ~6s, fades out, and removes
 * itself. Pointer-events pass through; the win banner stays clickable-free
 * anyway (it already ignores pointers). */

const DURATION_MS = 6000;
const FADE_MS = 1200;
const PARTICLE_COUNT = 160;

/** Festive palette: brights for strips, softs for petals. */
const STRIP_COLORS = ['#f6416c', '#ffd54f', '#4fc3f7', '#66bb6a', '#b388ff', '#ff8a65', '#ffffff'];
const PETAL_COLORS = ['#ffb7c5', '#ffd9e3', '#ffe9a8', '#e6d4f7', '#ffffff'];

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  rot: number;
  vr: number;
  /** 'strip' = rectangle, 'petal' = ellipse. */
  shape: 'strip' | 'petal';
  size: number;
  color: string;
  /** Horizontal flutter amplitude and phase (per-particle sine). */
  sway: number;
  phase: number;
  /** Flip speed for the 3D "tumbling paper" look of strips. */
  flip: number;
}

export function celebrate(container: HTMLElement): void {
  const canvas = document.createElement('canvas');
  const w = container.clientWidth;
  const h = container.clientHeight;
  canvas.width = w;
  canvas.height = h;
  Object.assign(canvas.style, {
    position: 'absolute',
    left: '0',
    top: '0',
    width: '100%',
    height: '100%',
    pointerEvents: 'none',
    zIndex: '10',
  });
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    canvas.remove();
    return;
  }

  const rand = (lo: number, hi: number): number => lo + Math.random() * (hi - lo);
  const makeParticle = (): Particle => {
    const petal = Math.random() < 0.35;
    return {
      x: rand(0, w),
      y: rand(-h * 0.7, -20), // staggered above the frame → streams in
      vx: rand(-0.4, 0.4),
      vy: rand(1.6, 3.6),
      rot: rand(0, Math.PI * 2),
      vr: rand(-0.12, 0.12),
      shape: petal ? 'petal' : 'strip',
      size: petal ? rand(6, 10) : rand(9, 15),
      color: petal
        ? PETAL_COLORS[Math.floor(Math.random() * PETAL_COLORS.length)]!
        : STRIP_COLORS[Math.floor(Math.random() * STRIP_COLORS.length)]!,
      sway: rand(0.4, 1.4),
      phase: rand(0, Math.PI * 2),
      flip: rand(2, 5),
    };
  };
  const particles: Particle[] = Array.from({ length: PARTICLE_COUNT }, makeParticle);

  const start = performance.now();
  let rafId = 0;

  const frame = (): void => {
    const t = performance.now() - start;
    ctx.clearRect(0, 0, w, h);

    // Fade the whole overlay out near the end.
    const alpha = t > DURATION_MS - FADE_MS ? Math.max(0, (DURATION_MS - t) / FADE_MS) : 1;
    ctx.globalAlpha = alpha;

    for (const p of particles) {
      const s = t / 1000;
      p.y += p.vy;
      p.x += p.vx + Math.sin(s * p.flip + p.phase) * p.sway;
      p.rot += p.vr;
      // 3D tumble: squash vertically around the particle's own axis.
      const squash = 0.35 + 0.65 * Math.abs(Math.sin(s * p.flip + p.phase));

      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.rotate(p.rot);
      ctx.fillStyle = p.color;
      if (p.shape === 'strip') {
        ctx.scale(1, squash);
        ctx.fillRect(-p.size * 0.25, -p.size * 0.8, p.size * 0.5, p.size * 1.6);
      } else {
        ctx.scale(1, 0.6 + 0.4 * squash);
        ctx.beginPath();
        ctx.ellipse(0, 0, p.size * 0.65, p.size, 0, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }

    if (t < DURATION_MS) {
      rafId = requestAnimationFrame(frame);
    } else {
      cancelAnimationFrame(rafId);
      canvas.remove();
    }
  };
  rafId = requestAnimationFrame(frame);
}
