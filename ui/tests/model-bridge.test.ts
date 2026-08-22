import { describe, expect, it, vi, afterEach } from 'vitest';
import type { Mock } from 'vitest';
import { EventEmitter } from 'node:events';
import { mkdtempSync, writeFileSync, mkdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createThinkHandler, listCheckpoints, CheckpointWorkerPool } from '../plugins/model-bridge';
import type { ThinkHandlerDeps } from '../plugins/model-bridge';
import type { PoolWorker } from '../plugins/model-bridge';
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
  ensureStarted?: (checkpoint: string) => Promise<void>;
  request?: (state: unknown, simulations: number, checkpoint: string, signal?: AbortSignal) => Promise<WorkerResponse>;
  defaultCheckpoint?: () => string | undefined;
}

function makeDeps(overrides: DepsOverrides = {}): { deps: ThinkHandlerDeps; ensureStarted: Mock; request: Mock } {
  const ensureStarted = vi.fn(async (_checkpoint: string) => {});
  const request = vi.fn(
    async (_state: unknown, _simulations: number, _checkpoint: string, _signal?: AbortSignal): Promise<WorkerResponse> => ({
      move: { x: 1, z: 2 },
    }),
  );
  return {
    deps: { ensureStarted, request, ...overrides },
    ensureStarted,
    request,
  };
}

const BODY = JSON.stringify({ state: { grid: [] }, simulations: 10, checkpoint: 'best1.pt' });

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

  it('defaults the checkpoint to the biggest best{n}.pt when absent', async () => {
    const { deps, ensureStarted, request } = makeDeps({ defaultCheckpoint: () => 'best7.pt' });
    const handler = createThinkHandler(deps);
    const req = new FakeReq();
    const res = new FakeRes();
    req.send(JSON.stringify({ state: { grid: [] }, simulations: 10 }));
    await handler(req, res);
    expect(ensureStarted).toHaveBeenCalledWith('best7.pt');
    expect(request).toHaveBeenCalledWith({ grid: [] }, 10, 'best7.pt', expect.any(AbortSignal));
  });

  it('treats null and an empty string as the default checkpoint', async () => {
    for (const missing of [null, '']) {
      const { deps, request } = makeDeps({ defaultCheckpoint: () => 'best7.pt' });
      const handler = createThinkHandler(deps);
      const req = new FakeReq();
      const res = new FakeRes();
      req.send(JSON.stringify({ state: {}, simulations: 10, checkpoint: missing }));
      await handler(req, res);
      expect(request).toHaveBeenCalledWith({}, 10, 'best7.pt', expect.any(AbortSignal));
    }
  });

  it('returns 503 when no default checkpoint exists', async () => {
    const { deps, request } = makeDeps({ defaultCheckpoint: () => undefined });
    const handler = createThinkHandler(deps);
    const req = new FakeReq();
    const res = new FakeRes();
    req.send(JSON.stringify({ state: { grid: [] }, simulations: 10 }));
    await handler(req, res);
    expect(res.statusCode).toBe(503);
    expect(JSON.parse(res.body)).toEqual({ error: 'model unavailable: no best{n}.pt checkpoints' });
    expect(request).not.toHaveBeenCalled();
  });

  it('rejects a non-string checkpoint with 400 (null and empty string mean default)', async () => {
    for (const bad of [42, { name: 'best.pt' }]) {
      const { deps, request } = makeDeps();
      const handler = createThinkHandler(deps);
      const req = new FakeReq();
      const res = new FakeRes();
      req.send(JSON.stringify({ state: {}, simulations: 10, checkpoint: bad }));
      await handler(req, res);
      expect(res.statusCode).toBe(400);
      expect(request).not.toHaveBeenCalled();
    }
  });

  it('rejects a checkpoint that is not a bare .pt name with 400 (path traversal guard)', async () => {
    for (const bad of ['../best.pt', 'a/b.pt', 'best.pt.exe', 'best']) {
      const { deps, request } = makeDeps();
      const handler = createThinkHandler(deps);
      const req = new FakeReq();
      const res = new FakeRes();
      req.send(JSON.stringify({ state: {}, simulations: 10, checkpoint: bad }));
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

  it('forwards state, simulations and checkpoint and returns the move as 200', async () => {
    const { deps, request } = makeDeps();
    const handler = createThinkHandler(deps);
    const req = new FakeReq();
    const res = new FakeRes();
    req.send(BODY);
    await handler(req, res);
    expect(request).toHaveBeenCalledWith({ grid: [] }, 10, 'best1.pt', expect.any(AbortSignal));
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
    const request = vi.fn((_state: unknown, _simulations: number, _checkpoint: string, signal?: AbortSignal) => {
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

describe('listCheckpoints', () => {
  let dir: string;

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  it('lists best{n}.pt files in descending n, ignoring everything else', () => {
    dir = mkdtempSync(join(tmpdir(), 'sf-checkpoints-'));
    for (const name of ['best2.pt', 'best.pt', 'random.pt', 'best1.pt', 'best10.pt', 'best0.pt']) {
      writeFileSync(join(dir, name), 'x');
    }
    writeFileSync(join(dir, 'notes.txt'), 'x'); // not a checkpoint
    writeFileSync(join(dir, 'best.pt.bak'), 'x'); // extension mismatch
    mkdirSync(join(dir, 'fake.pt')); // a directory, not a file

    expect(listCheckpoints(dir)).toEqual(['best10.pt', 'best2.pt', 'best1.pt', 'best0.pt']);
  });

  it('returns an empty list when only best.pt or unrelated files exist', () => {
    dir = mkdtempSync(join(tmpdir(), 'sf-checkpoints-'));
    writeFileSync(join(dir, 'best.pt'), 'x');
    writeFileSync(join(dir, 'random.pt'), 'x');
    expect(listCheckpoints(dir)).toEqual([]);
  });

  it('returns an empty list for a missing directory', () => {
    expect(listCheckpoints(join(tmpdir(), 'sf-no-such-dir-xyz'))).toEqual([]);
  });
});

describe('CheckpointWorkerPool', () => {
  class FakeWorker implements PoolWorker {
    alive = false;
    started = false;
    closed = false;
    constructor(
      public readonly checkpoint: string,
      private failStart = false,
    ) {}
    async start(): Promise<void> {
      if (this.failStart) throw new Error(`cannot load ${this.checkpoint}`);
      this.started = true;
      this.alive = true;
    }
    close(): void {
      this.closed = true;
      this.alive = false;
    }
    request(): Promise<WorkerResponse> {
      return Promise.resolve({ move: { x: 1, z: 1 } });
    }
  }

  function makePool(maxSize: number): {
    pool: CheckpointWorkerPool;
    made: FakeWorker[];
    factory: (checkpoint: string) => FakeWorker;
  } {
    const made: FakeWorker[] = [];
    const factory = (checkpoint: string): FakeWorker => {
      const w = new FakeWorker(checkpoint);
      made.push(w);
      return w;
    };
    return { pool: new CheckpointWorkerPool(factory, maxSize), made, factory };
  }

  it('reuses a live worker for the same checkpoint', async () => {
    const { pool, made } = makePool(2);
    const a1 = await pool.ensure('a.pt');
    const a2 = await pool.ensure('a.pt');
    expect(a1).toBe(a2);
    expect(made).toHaveLength(1);
  });

  it('evicts the least-recently-used worker when over the cap', async () => {
    const { pool, made } = makePool(2);
    await pool.ensure('a.pt');
    await pool.ensure('b.pt');
    await pool.ensure('c.pt');
    expect(pool.size).toBe(2);
    expect(made[0]!.closed).toBe(true); // a evicted
    expect(made[1]!.closed).toBe(false);
    expect(made[2]!.closed).toBe(false);
    expect(pool.get('a.pt')).toBeNull();
  });

  it('touching a worker protects it from eviction', async () => {
    const { pool, made } = makePool(2);
    await pool.ensure('a.pt');
    await pool.ensure('b.pt');
    pool.get('a.pt'); // a is now MRU
    await pool.ensure('c.pt');
    expect(made[1]!.closed).toBe(true); // b evicted instead
    expect(made[0]!.closed).toBe(false);
    expect(made[2]!.closed).toBe(false);
  });

  it('recreates an evicted checkpoint on demand', async () => {
    const { pool, made, factory } = makePool(2);
    await pool.ensure('a.pt');
    await pool.ensure('b.pt');
    await pool.ensure('c.pt'); // evicts a
    const a2 = await pool.ensure('a.pt');
    expect(a2).toBe(made[3]!); // a new worker was spawned
    expect(pool.size).toBe(2);
  });

  it('replaces a dead worker with a fresh one', async () => {
    const { pool, made } = makePool(2);
    const a = (await pool.ensure('a.pt')) as FakeWorker;
    a.alive = false; // simulate worker death
    const a2 = await pool.ensure('a.pt');
    expect(a2).not.toBe(a);
    expect(made).toHaveLength(2);
  });

  it('a failed start leaves no entry behind', async () => {
    const failFactory = (checkpoint: string): FakeWorker => new FakeWorker(checkpoint, true);
    const failing = new CheckpointWorkerPool(failFactory, 2);
    await expect(failing.ensure('bad.pt')).rejects.toThrow('cannot load bad.pt');
    expect(failing.size).toBe(0);
  });
});
