import { describe, expect, it, vi } from 'vitest';
import type { Mock } from 'vitest';
import { EventEmitter } from 'node:events';
import { createThinkHandler } from '../plugins/model-bridge';
import type { ThinkHandlerDeps } from '../plugins/model-bridge';
import type { WorkerResponse } from '../plugins/worker-client';

class FakeReq extends EventEmitter {
  method: string;
  constructor(method = 'POST') {
    super();
    this.method = method;
  }
  /** Emits asynchronously so the handler's body listeners are registered first,
   * like a real socket where the body cannot arrive before the middleware runs. */
  send(body: string): void {
    queueMicrotask(() => {
      this.emit('data', body);
      this.emit('end');
    });
  }
}

class FakeRes extends EventEmitter {
  statusCode = 200;
  writableEnded = false;
  headers: Record<string, string> = {};
  body = '';
  setHeader(k: string, v: string): void {
    this.headers[k] = v;
  }
  end(payload?: string): void {
    this.writableEnded = true;
    this.body = payload ?? '';
  }
}

interface DepsOverrides {
  ensureStarted?: () => Promise<void>;
  request?: (state: unknown, simulations: number, signal?: AbortSignal) => Promise<WorkerResponse>;
}

function makeDeps(overrides: DepsOverrides = {}): { deps: ThinkHandlerDeps; ensureStarted: Mock; request: Mock } {
  const ensureStarted = vi.fn(async () => {});
  const request = vi.fn(async (_state: unknown, _simulations: number, _signal?: AbortSignal): Promise<WorkerResponse> => ({ move: { x: 1, z: 2 } }));
  return {
    deps: { ensureStarted, request, ...overrides },
    ensureStarted,
    request,
  };
}

const BODY = JSON.stringify({ state: { grid: [] }, simulations: 10 });

describe('createThinkHandler', () => {
  it('rejects non-POST requests with 405 and never touches the worker', async () => {
    const { deps, request } = makeDeps();
    const handler = createThinkHandler(deps);
    const req = new FakeReq('GET');
    const res = new FakeRes();
    await handler(req, res);
    expect(res.statusCode).toBe(405);
    expect(JSON.parse(res.body)).toEqual({ error: 'method not allowed' });
    expect(request).not.toHaveBeenCalled();
  });

  it('rejects an invalid JSON body with 400', async () => {
    const { deps, request } = makeDeps();
    const handler = createThinkHandler(deps);
    const req = new FakeReq();
    const res = new FakeRes();
    req.send('{not json');
    await handler(req, res);
    expect(res.statusCode).toBe(400);
    expect(request).not.toHaveBeenCalled();
  });

  it('rejects a missing state or simulations with 400', async () => {
    const { deps, request } = makeDeps();
    const handler = createThinkHandler(deps);
    const req = new FakeReq();
    const res = new FakeRes();
    req.send(JSON.stringify({ state: {} }));
    await handler(req, res);
    expect(res.statusCode).toBe(400);
    expect(request).not.toHaveBeenCalled();
  });

  it('rejects non-integer or negative simulations with 400', async () => {
    for (const bad of [1.5, -1, '10']) {
      const { deps, request } = makeDeps();
      const handler = createThinkHandler(deps);
      const req = new FakeReq();
      const res = new FakeRes();
      req.send(JSON.stringify({ state: {}, simulations: bad }));
      await handler(req, res);
      expect(res.statusCode).toBe(400);
      expect(request).not.toHaveBeenCalled();
    }
  });

  it('returns 503 with the startup reason when the model cannot start', async () => {
    const { deps, request } = makeDeps({
      ensureStarted: () => Promise.reject(new Error('no checkpoint at /nope/best.pt')),
    });
    const handler = createThinkHandler(deps);
    const req = new FakeReq();
    const res = new FakeRes();
    req.send(BODY);
    await handler(req, res);
    expect(res.statusCode).toBe(503);
    expect(JSON.parse(res.body)).toEqual({ error: 'model unavailable: no checkpoint at /nope/best.pt' });
    expect(request).not.toHaveBeenCalled();
  });

  it('forwards state and simulations and returns the move as 200', async () => {
    const { deps, request } = makeDeps();
    const handler = createThinkHandler(deps);
    const req = new FakeReq();
    const res = new FakeRes();
    req.send(BODY);
    await handler(req, res);
    expect(request).toHaveBeenCalledWith({ grid: [] }, 10, expect.any(AbortSignal));
    expect(res.statusCode).toBe(200);
    expect(JSON.parse(res.body)).toEqual({ move: { x: 1, z: 2 } });
  });

  it('returns 502 when the worker fails mid-request', async () => {
    const { deps } = makeDeps({
      request: () => Promise.reject(new Error('worker exited (code=3)')),
    });
    const handler = createThinkHandler(deps);
    const req = new FakeReq();
    const res = new FakeRes();
    req.send(BODY);
    await handler(req, res);
    expect(res.statusCode).toBe(502);
    expect(JSON.parse(res.body)).toEqual({ error: 'model worker failed: worker exited (code=3)' });
  });

  it('returns the worker error message as 500', async () => {
    const { deps } = makeDeps({
      request: () => Promise.resolve({ error: 'ValueError: missing state' }),
    });
    const handler = createThinkHandler(deps);
    const req = new FakeReq();
    const res = new FakeRes();
    req.send(BODY);
    await handler(req, res);
    expect(res.statusCode).toBe(500);
    expect(JSON.parse(res.body)).toEqual({ error: 'ValueError: missing state' });
  });

  it('passes a null move through as 200', async () => {
    const { deps } = makeDeps({
      request: () => Promise.resolve({ move: null }),
    });
    const handler = createThinkHandler(deps);
    const req = new FakeReq();
    const res = new FakeRes();
    req.send(BODY);
    await handler(req, res);
    expect(res.statusCode).toBe(200);
    expect(JSON.parse(res.body)).toEqual({ move: null });
  });

  it('aborts the in-flight request and writes nothing when the client disconnects', async () => {
    const requestStarted = Promise.withResolvers<void>();
    const request = vi.fn((_state: unknown, _simulations: number, signal?: AbortSignal) => {
      requestStarted.resolve();
      const { promise, reject } = Promise.withResolvers<WorkerResponse>();
      signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
      return promise;
    });
    const { deps } = makeDeps({ request });
    const handler = createThinkHandler(deps);
    const req = new FakeReq();
    const res = new FakeRes();
    const done = handler(req, res);
    req.send(BODY);
    await requestStarted.promise; // the think is in flight and abort-wired
    res.emit('close'); // client disconnects mid-think
    await done; // the handler settles without writing when the client is gone
    expect(res.writableEnded).toBe(false);
    expect(res.body).toBe('');
  });
});
