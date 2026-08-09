import { fileURLToPath } from 'node:url';
import type { ServerResponse } from 'node:http';
import type { Plugin } from 'vite';
import { WorkerClient } from './worker-client';
import type { WorkerResponse } from './worker-client';

export interface ModelBridgeOptions {
  /** Python interpreter running the worker (default: model/.venv/bin/python). */
  python?: string;
  /** Model checkpoint the worker loads (default: model/checkpoints/best.pt). */
  checkpoint?: string;
  /** HTTP path the bridge answers on (default: /api/think). */
  endpoint?: string;
  readyTimeoutMs?: number;
}

export interface ThinkHandlerDeps {
  ensureStarted(): Promise<void>;
  request(state: unknown, simulations: number, signal?: AbortSignal): Promise<WorkerResponse>;
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
const DEFAULT_CHECKPOINT = fileURLToPath(new URL('../../model/checkpoints/best.pt', import.meta.url));

/**
 * POST /api/think middleware. Body: {"state": <state_to_json>, "simulations": n}.
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
  const state = payload.state;

  try {
    await deps.ensureStarted();
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
    result = await deps.request(state, simulations, ac.signal);
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
 * Spawns the persistent smart-four model worker and serves its moves at
 * `endpoint` on both the dev and preview servers. The worker is started
 * eagerly; if it cannot start (missing venv/checkpoint), the failure is
 * logged loudly and every think returns 503 — machine play refuses until the
 * model is available, with a retry on each request. Override paths with
 * SMARTFOUR_PYTHON / SMARTFOUR_CHECKPOINT env vars.
 */
export function modelBridge(options: ModelBridgeOptions = {}): Plugin {
  const python = process.env.SMARTFOUR_PYTHON ?? options.python ?? DEFAULT_PYTHON;
  const checkpoint = process.env.SMARTFOUR_CHECKPOINT ?? options.checkpoint ?? DEFAULT_CHECKPOINT;
  const endpoint = options.endpoint ?? '/api/think';
  const readyTimeoutMs = options.readyTimeoutMs ?? 120_000;

  let client: WorkerClient | null = null;
  let starting: Promise<void> | null = null;

  const ensureStarted = (): Promise<void> => {
    if (client?.alive) return Promise.resolve();
    if (starting) return starting;
    const c = new WorkerClient({
      command: python,
      args: ['-m', 'smartfour.worker', '--checkpoint', checkpoint],
      cwd: MODEL_DIR,
      readyTimeoutMs,
    });
    client = c;
    starting = c
      .start()
      .catch((err: unknown) => {
        if (client === c) client = null;
        throw err;
      })
      .finally(() => {
        starting = null;
      });
    return starting;
  };

  const handle = createThinkHandler({
    ensureStarted,
    request: (state, simulations, signal) => {
      if (!client) return Promise.reject(new Error('worker not started'));
      return client.request(state, simulations, signal);
    },
  });

  const startEager = (): void => {
    ensureStarted().catch((err: unknown) => {
      console.error(`[smartfour] model unavailable: ${messageOf(err)}`);
      console.error('[smartfour] machine play returns HTTP 503 until the model starts. Set SMARTFOUR_PYTHON / SMARTFOUR_CHECKPOINT to override the default paths.');
    });
  };

  const stop = (): void => {
    client?.close();
    client = null;
  };

  return {
    name: 'smartfour-model-bridge',
    configureServer(server) {
      if (process.env.VITEST) return; // vitest boots a dev server; no model needed there
      server.httpServer?.once('close', stop);
      startEager();
      server.middlewares.use(endpoint, handle);
    },
    configurePreviewServer(server) {
      // Register in the hook body so the API route runs BEFORE preview's
      // static + SPA-fallback middlewares; in the post-setup closure the
      // fallback would swallow non-POST requests to /api/think.
      server.httpServer?.once('close', stop);
      startEager();
      server.middlewares.use(endpoint, handle);
    },
  };
}
