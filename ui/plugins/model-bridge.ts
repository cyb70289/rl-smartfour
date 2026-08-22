import { readdirSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { ServerResponse } from 'node:http';
import type { Plugin } from 'vite';
import { WorkerClient } from './worker-client';
import type { WorkerResponse } from './worker-client';

export interface ModelBridgeOptions {
  /** Python interpreter running the worker (default: model/.venv/bin/python). */
  python?: string;
  /** HTTP path the bridge answers on (default: /api/think). */
  endpoint?: string;
  readyTimeoutMs?: number;
  /** Max concurrently resident model workers (default: 2 — one per player). */
  maxWorkers?: number;
}

export interface ThinkHandlerDeps {
  ensureStarted(checkpoint: string): Promise<void>;
  request(state: unknown, simulations: number, checkpoint: string, signal?: AbortSignal): Promise<WorkerResponse>;
  /** The default checkpoint when the request omits one: the biggest best{n}.pt. */
  defaultCheckpoint?(): string | undefined;
}

/** The parts of http.IncomingMessage the handler needs (structural, testable). */
export interface ThinkRequest {
  method?: string;
  on(event: string, listener: (...args: unknown[]) => void): unknown;
  destroy?(): void;
}

/** The parts of http.ServerResponse the handler needs (structural, testable). */
export interface ThinkResponse {
  statusCode: number;
  writableEnded: boolean;
  setHeader(name: string, value: string | number | readonly string[]): unknown;
  end(chunk?: unknown): unknown;
  on(event: string, listener: (...args: unknown[]) => void): unknown;
}

/** Paths are resolved relative to this file (ui/plugins → repo root). */
const MODEL_DIR = fileURLToPath(new URL('../../model', import.meta.url));
const DEFAULT_PYTHON = fileURLToPath(new URL('../../model/.venv/bin/python', import.meta.url));
const CHECKPOINTS_DIR = fileURLToPath(new URL('../../model/checkpoints', import.meta.url));

/** A checkpoint is a bare .pt file name; the regex rules out path separators,
 * so resolving against CHECKPOINTS_DIR can never escape it. */
const CHECKPOINT_RE = /^[\w.-]+\.pt$/;

/** A versioned best checkpoint: best{n}.pt (the only files listed). */
const BEST_RE = /^best(\d+)\.pt$/;

/** Numerical version n of a best{n}.pt file name, or -1 when not one. */
function bestVersion(name: string): number {
  return Number(BEST_RE.exec(name)?.[1] ?? -1);
}

/** Absolute path of a checkpoint file inside the checkpoints dir. */
function checkpointPath(name: string): string {
  if (!CHECKPOINT_RE.test(name)) throw new Error(`invalid checkpoint name: ${name}`);
  return resolve(CHECKPOINTS_DIR, name);
}

/**
 * Lists the checkpoints dir: every best{n}.pt file, biggest n first (largest
 * n = strongest model). Plain best.pt, other .pt files and directories are
 * ignored. Empty on read errors (missing dir, permission).
 */
export function listCheckpoints(dir: string = CHECKPOINTS_DIR): string[] {
  let names: string[];
  try {
    names = readdirSync(dir).filter((n) => BEST_RE.test(n) && statSync(join(dir, n)).isFile());
  } catch {
    return [];
  }
  return names.sort((a, b) => bestVersion(b) - bestVersion(a));
}

/** The worker surface the pool needs (WorkerClient satisfies it). */
export interface PoolWorker {
  readonly alive: boolean;
  start(): Promise<void>;
  close(): void;
  request(state: unknown, simulations: number, signal?: AbortSignal): Promise<WorkerResponse>;
}

/**
 * LRU cache of model workers keyed by checkpoint file name, capped at
 * `maxSize`. `ensure` spawns a worker on first use (re-spawning dead ones),
 * keeps accessed entries MRU, and evicts+closes the least-recently-used
 * worker when over the cap.
 */
export class CheckpointWorkerPool {
  private clients = new Map<string, PoolWorker>();

  constructor(
    private factory: (checkpoint: string) => PoolWorker,
    private maxSize: number,
  ) {}

  get size(): number {
    return this.clients.size;
  }

  /** Returns the live worker for `checkpoint`, or null when not started. */
  get(checkpoint: string): PoolWorker | null {
    const client = this.clients.get(checkpoint);
    if (client?.alive) {
      this.touch(checkpoint);
      return client;
    }
    return null;
  }

  /** Spawns (or reuses) and starts the worker for `checkpoint`. */
  async ensure(checkpoint: string): Promise<PoolWorker> {
    const existing = this.clients.get(checkpoint);
    if (existing?.alive) {
      this.touch(checkpoint);
      return existing;
    }
    if (existing) {
      existing.close();
      this.clients.delete(checkpoint);
    }
    const client = this.factory(checkpoint);
    this.clients.set(checkpoint, client);
    try {
      await client.start();
    } catch (err) {
      if (this.clients.get(checkpoint) === client) this.clients.delete(checkpoint);
      throw err;
    }
    this.touch(checkpoint);
    this.evict();
    return client;
  }

  closeAll(): void {
    for (const client of this.clients.values()) client.close();
    this.clients.clear();
  }

  private touch(checkpoint: string): void {
    const client = this.clients.get(checkpoint);
    if (!client) return;
    this.clients.delete(checkpoint);
    this.clients.set(checkpoint, client);
  }

  private evict(): void {
    while (this.clients.size > this.maxSize) {
      const oldest = this.clients.keys().next().value;
      if (oldest === undefined) return;
      this.clients.get(oldest)?.close();
      this.clients.delete(oldest);
    }
  }
}

/**
 * POST /api/think middleware. Body: {"state": <state_to_json>, "simulations": n,
 * "checkpoint": <file name, optional, default the biggest best{n}.pt>}.
 * Response: 200 {"move": {"x","z"}} | {"move": null}, 400 bad body, 500 worker
 * error, 502 worker failure, 503 model unavailable. Aborting the request
 * (client disconnect) aborts the in-flight worker request; the late response
 * is discarded by the WorkerClient, never leaking into the next request.
 */
export function createThinkHandler(deps: ThinkHandlerDeps): (req: ThinkRequest, res: ThinkResponse) => Promise<void> {
  return async (req, res) => {
    if (req.method !== 'POST') {
      sendJson(res, 405, { error: 'method not allowed' });
      return;
    }
    const body = await readBody(req);
    await handleBody(body, res, deps);
  };
}

async function handleBody(body: string, res: ThinkResponse, deps: ThinkHandlerDeps): Promise<void> {
  let payload: unknown;
  try {
    payload = JSON.parse(body);
  } catch {
    sendJson(res, 400, { error: 'invalid JSON body' });
    return;
  }
  if (typeof payload !== 'object' || payload === null) {
    sendJson(res, 400, { error: 'expected a JSON object' });
    return;
  }
  if (!('state' in payload) || !('simulations' in payload)) {
    sendJson(res, 400, { error: "missing 'state' or 'simulations'" });
    return;
  }
  const simulations = payload.simulations;
  if (typeof simulations !== 'number' || !Number.isInteger(simulations) || simulations < 0) {
    sendJson(res, 400, { error: "'simulations' must be an integer >= 0" });
    return;
  }
  const checkpoint = 'checkpoint' in payload ? payload.checkpoint : undefined;
  let checkpointName: string;
  if (checkpoint === undefined || checkpoint === null || checkpoint === '') {
    // Unspecified checkpoint: default to the biggest best{n}.pt.
    const fallback = (deps.defaultCheckpoint ?? defaultCheckpoint)();
    if (fallback === undefined) {
      sendJson(res, 503, { error: 'model unavailable: no best{n}.pt checkpoints' });
      return;
    }
    checkpointName = fallback;
  } else if (typeof checkpoint !== 'string' || !CHECKPOINT_RE.test(checkpoint)) {
    sendJson(res, 400, { error: `'checkpoint' must be a .pt file name, got ${JSON.stringify(checkpoint)}` });
    return;
  } else {
    checkpointName = checkpoint;
  }
  const state = payload.state;

  try {
    await deps.ensureStarted(checkpointName);
  } catch (err) {
    sendJson(res, 503, { error: `model unavailable: ${messageOf(err)}` });
    return;
  }

  const ac = new AbortController();
  res.on('close', () => {
    if (!res.writableEnded) ac.abort();
  });

  let result: WorkerResponse;
  try {
    result = await deps.request(state, simulations, checkpointName, ac.signal);
  } catch (err) {
    if (res.writableEnded || ac.signal.aborted) return; // client is gone
    sendJson(res, 502, { error: `model worker failed: ${messageOf(err)}` });
    return;
  }
  if (res.writableEnded) return;
  if ('error' in result) sendJson(res, 500, { error: result.error });
  else sendJson(res, 200, { move: result.move });
}

function readBody(req: ThinkRequest): Promise<string> {
  const { promise, resolve } = Promise.withResolvers<string>();
  let body = '';
  req.on('data', (chunk: unknown) => {
    body += String(chunk);
    if (body.length > 1_000_000) req.destroy?.();
  });
  req.on('end', () => resolve(body));
  req.on('close', () => resolve(body)); // abort mid-body: settle with what we have
  return promise;
}

/** The default checkpoint: the biggest best{n}.pt in the checkpoints dir. */
function defaultCheckpoint(): string | undefined {
  return listCheckpoints()[0];
}

function sendJson(res: ThinkResponse, status: number, payload: unknown): void {
  if (res.writableEnded) return;
  res.statusCode = status;
  res.setHeader('Content-Type', 'application/json');
  res.end(JSON.stringify(payload));
}

function messageOf(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/**
 * Serves the model on the dev and preview servers:
 * - `POST /api/think` — model moves, routed to the worker of the requested
 *   checkpoint; workers are cached per checkpoint (LRU, max 2).
 * - `GET /api/checkpoints` — the list of best{n}.pt files in
 *   model/checkpoints, biggest n first.
 * The default checkpoint worker (the biggest best{n}.pt) is started
 * eagerly; if it cannot start (missing venv/checkpoint), the failure is
 * logged loudly and every think returns 503 — model play refuses until the
 * model is available, with a retry on each request. Override the
 * interpreter with SMARTFOUR_PYTHON.
 */
export function modelBridge(options: ModelBridgeOptions = {}): Plugin {
  const python = process.env.SMARTFOUR_PYTHON ?? options.python ?? DEFAULT_PYTHON;
  const endpoint = options.endpoint ?? '/api/think';
  const readyTimeoutMs = options.readyTimeoutMs ?? 120_000;
  const maxWorkers = options.maxWorkers ?? 2;

  const pool = new CheckpointWorkerPool(
    (checkpoint) =>
      new WorkerClient({
        command: python,
        args: ['-m', 'smartfour.worker', '--checkpoint', checkpointPath(checkpoint)],
        cwd: MODEL_DIR,
        readyTimeoutMs,
      }),
    maxWorkers,
  );

  const ensureStarted = (checkpoint: string): Promise<void> => pool.ensure(checkpoint).then(() => undefined);

  const request = (state: unknown, simulations: number, checkpoint: string, signal?: AbortSignal): Promise<WorkerResponse> => {
    const client = pool.get(checkpoint);
    if (!client) return Promise.reject(new Error('worker not started'));
    return client.request(state, simulations, signal);
  };

  const handle = createThinkHandler({ ensureStarted, request });

  const handleList = (req: ThinkRequest, res: ThinkResponse): void => {
    if (req.method !== 'GET') {
      sendJson(res, 405, { error: 'method not allowed' });
      return;
    }
    sendJson(res, 200, { checkpoints: listCheckpoints() });
  };

  const startEager = (): void => {
    const first = listCheckpoints()[0];
    if (first === undefined) {
      console.error('[smartfour] no best{n}.pt checkpoints found in model/checkpoints;');
      console.error('[smartfour] model play returns HTTP 503 until a best{n}.pt checkpoint is added.');
      return;
    }
    ensureStarted(first).catch((err: unknown) => {
      console.error(`[smartfour] model unavailable: ${messageOf(err)}`);
      console.error('[smartfour] model play returns HTTP 503 until the model starts. Set SMARTFOUR_PYTHON to override the default interpreter path.');
    });
  };

  const stop = (): void => {
    pool.closeAll();
  };

  return {
    name: 'smartfour-model-bridge',
    configureServer(server) {
      if (process.env.VITEST) return; // vitest boots a dev server; no model needed there
      server.httpServer?.once('close', stop);
      startEager();
      server.middlewares.use(endpoint, handle);
      server.middlewares.use('/api/checkpoints', handleList);
    },
    configurePreviewServer(server) {
      // Register in the hook body so the API route runs BEFORE preview's
      // static + SPA-fallback middlewares; in the post-setup closure the
      // fallback would swallow non-POST requests to /api/think.
      server.httpServer?.once('close', stop);
      startEager();
      server.middlewares.use(endpoint, handle);
      server.middlewares.use('/api/checkpoints', handleList);
    },
  };
}
