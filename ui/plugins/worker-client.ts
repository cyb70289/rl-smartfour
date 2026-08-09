import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import type { ChildProcessWithoutNullStreams } from 'node:child_process';

export interface WorkerSpec {
  command: string;
  args: string[];
  cwd?: string;
  env?: Record<string, string>;
  /** How long to wait for the worker's {"ready": true} line before giving up. */
  readyTimeoutMs?: number;
}

export type WorkerResponse = { move: { x: number; z: number } | null } | { error: string };

interface Entry {
  id: number;
  body: string;
  aborted: boolean;
  resolve: (res: WorkerResponse) => void;
  reject: (err: Error) => void;
}

const DEFAULT_READY_TIMEOUT_MS = 120_000;

function isMoveShape(v: unknown): v is { x: number; z: number } {
  if (typeof v !== 'object' || v === null) return false;
  if (!('x' in v) || !('z' in v)) return false;
  return typeof v.x === 'number' && typeof v.z === 'number';
}

/**
 * Talks to the persistent smart-four model worker over newline-delimited JSON
 * on stdin/stdout. Requests are strictly serialized (one in flight): each
 * response is matched to the in-flight request by id, and the response of an
 * aborted request is discarded so it can never leak into the next one.
 * Protocol violations and worker death fail in-flight and queued requests and
 * mark the client dead (the bridge restarts it on the next request).
 */
export class WorkerClient {
  private proc: ChildProcessWithoutNullStreams | null = null;
  private ready = false;
  private dead = false;
  private nextId = 1;
  private inFlight: Entry | null = null;
  private queue: Entry[] = [];
  private stderrTail = '';
  private starting: Promise<void> | null = null;
  private stderrListeners = new Set<(chunk: string) => void>();

  constructor(private spec: WorkerSpec) {}

  /** True when the worker is ready and its process is still running. */
  get alive(): boolean {
    return this.ready && !this.dead && this.proc !== null && this.proc.exitCode === null;
  }

  /** Subscribe to raw stderr chunks (test hooks, diagnostics). Returns unsubscribe. */
  onStderr(cb: (chunk: string) => void): () => void {
    this.stderrListeners.add(cb);
    return () => this.stderrListeners.delete(cb);
  }

  /** Spawn the worker and wait for its ready line. Rejects on spawn failure,
   * exit-before-ready (stderr included), unexpected pre-ready output, or timeout. */
  start(): Promise<void> {
    if (this.starting) return this.starting;
    this.starting = this.doStart().finally(() => {
      this.starting = null;
    });
    return this.starting;
  }

  /**
   * Enqueue a think. Resolves with the worker's response, or rejects when the
   * signal aborts (AbortError), the worker dies, or the protocol breaks.
   */
  request(state: unknown, simulations: number, signal?: AbortSignal): Promise<WorkerResponse> {
    if (!this.ready || this.dead) return Promise.reject(new Error('worker is not running'));
    if (signal?.aborted) return Promise.reject(new DOMException('aborted', 'AbortError'));
    const { promise, resolve, reject } = Promise.withResolvers<WorkerResponse>();
    const id = this.nextId++;
    const entry: Entry = {
      id,
      body: JSON.stringify({ id, state, simulations }),
      aborted: false,
      resolve,
      reject,
    };
    if (signal) {
      signal.addEventListener(
        'abort',
        () => {
          entry.aborted = true;
          reject(new DOMException('aborted', 'AbortError'));
        },
        { once: true },
      );
    }
    this.queue.push(entry);
    this.pump();
    return promise;
  }

  /** Kill the worker process (if any). Pending requests were already failed by exit. */
  close(): void {
    this.dead = true;
    this.ready = false;
    if (this.proc && this.proc.exitCode === null) this.proc.kill('SIGKILL');
    this.proc = null;
  }

  private doStart(): Promise<void> {
    const { promise, resolve, reject } = Promise.withResolvers<void>();
    const { command, args, cwd, env } = this.spec;
    const timeoutMs = this.spec.readyTimeoutMs ?? DEFAULT_READY_TIMEOUT_MS;
    const proc = spawn(command, args, {
      cwd,
      env: { ...process.env, PYTHONUNBUFFERED: '1', ...env },
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    this.proc = proc;
    this.stderrTail = '';
    proc.stderr.on('data', (chunk: Buffer) => {
      const text = chunk.toString();
      this.stderrTail = (this.stderrTail + text).slice(-4096);
      for (const cb of this.stderrListeners) cb(text);
    });

    let settled = false;
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      this.dead = true;
      proc.kill('SIGKILL');
      reject(new Error(`worker did not report ready within ${timeoutMs}ms: ${this.stderrTail.trim() || 'no stderr'}`));
    }, timeoutMs);

    const onExit = (code: number | null, signal: string | null): void => {
      clearTimeout(timer);
      if (!this.ready) {
        if (settled) return;
        settled = true;
        reject(new Error(`worker exited before ready (code=${code}, signal=${signal}): ${this.stderrTail.trim() || 'no stderr'}`));
      } else {
        this.dead = true;
        this.failAll(new Error(`worker exited (code=${code}, signal=${signal}): ${this.stderrTail.trim() || 'no stderr'}`));
      }
    };
    const onError = (err: Error): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      this.dead = true;
      reject(new Error(`failed to spawn worker: ${err.message}`));
    };
    proc.once('exit', onExit);
    proc.once('error', onError);

    const rl = createInterface({ input: proc.stdout });
    rl.on('line', (line) => {
      if (!this.ready) {
        try {
          const parsed: unknown = JSON.parse(line);
          if (parsed !== null && typeof parsed === 'object' && 'ready' in parsed && parsed.ready === true) {
            this.ready = true;
            if (!settled) {
              settled = true;
              clearTimeout(timer);
              resolve();
            }
            return;
          }
        } catch {
          // not JSON — fall through to the protocol error below
        }
        if (!settled) {
          settled = true;
          clearTimeout(timer);
          this.dead = true;
          proc.kill('SIGKILL');
          reject(new Error(`worker emitted unexpected output before ready: ${line.slice(0, 200)}`));
        }
        return;
      }
      this.onWorkerLine(line);
    });

    return promise;
  }

  private pump(): void {
    if (this.inFlight || this.dead || !this.ready || !this.proc) return;
    while (this.queue.length > 0) {
      const entry = this.queue.shift()!;
      if (entry.aborted) continue; // never send a request nobody wants
      this.inFlight = entry;
      this.proc.stdin.write(entry.body + '\n');
      return;
    }
  }

  private onWorkerLine(line: string): void {
    if (line.trim() === '') return;
    const entry = this.inFlight;
    if (!entry) {
      this.killOnProtocolError(`unexpected worker output with no pending request: ${line.slice(0, 200)}`);
      return;
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(line);
    } catch {
      this.killOnProtocolError(`non-JSON output: ${line.slice(0, 200)}`);
      return;
    }
    const p = parsed as { id?: unknown; move?: unknown; error?: unknown };
    if (typeof p.id !== 'number' || p.id !== entry.id) {
      this.killOnProtocolError(`response id mismatch: expected ${entry.id}, got ${p.id}`);
      return;
    }
    if (entry.aborted) {
      this.inFlight = null;
      this.pump(); // discard the late response; the promise already rejected on abort
      return;
    }
    if (typeof p.error === 'string') {
      this.inFlight = null;
      entry.resolve({ error: p.error });
    } else if (p.move === null || isMoveShape(p.move)) {
      this.inFlight = null;
      entry.resolve(p.move === null ? { move: null } : { move: p.move });
    } else {
      this.killOnProtocolError(`malformed response: ${line.slice(0, 200)}`);
      return;
    }
    this.pump();
  }

  /** Protocol violation: fail everything, kill the worker, mark dead. */
  private killOnProtocolError(detail: string): void {
    this.dead = true;
    this.failAll(new Error(`worker protocol error: ${detail}`));
    this.proc?.kill('SIGKILL');
  }

  private failAll(err: Error): void {
    if (this.inFlight) {
      this.inFlight.reject(err);
      this.inFlight = null;
    }
    for (const e of this.queue.splice(0)) e.reject(err);
  }
}
