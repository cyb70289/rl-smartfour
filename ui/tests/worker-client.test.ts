// Integration tests: they drive a REAL child-process protocol (stdin/stdout/
// stderr JSON lines), so ordering is anchored to real events (ready line,
// stderr "GOT" markers) rather than test-side sleeps. The only wall-clock
// delays live inside the fixture scripts (the worker's own response timing),
// which deterministic fake timers cannot control across a process boundary.
import { afterEach, describe, expect, it } from 'vitest';
import { WorkerClient } from '../plugins/worker-client';
import type { WorkerSpec } from '../plugins/worker-client';

/** Replies immediately, echoing the request id back as the move x. */
const ECHO_WORKER = `
  const rl = require('node:readline').createInterface({ input: process.stdin });
  console.log(JSON.stringify({ ready: true }));
  rl.on('line', (line) => {
    const req = JSON.parse(line);
    console.log(JSON.stringify({ id: req.id, move: { x: req.id, z: 0 } }));
  });
`;

/** Signals receipt on stderr (GOT <id>), then replies after a delay. */
const GOT_THEN_SLOW_REPLY = `
  const rl = require('node:readline').createInterface({ input: process.stdin });
  console.log(JSON.stringify({ ready: true }));
  rl.on('line', (line) => {
    const req = JSON.parse(line);
    process.stderr.write('GOT ' + req.id + '\\n');
    setTimeout(() => console.log(JSON.stringify({ id: req.id, move: { x: req.id, z: 0 } })), 200);
  });
`;

/** Signals the received (state, simulations) on stderr, replies immediately. */
const ECHO_STDERR = `
  const rl = require('node:readline').createInterface({ input: process.stdin });
  console.log(JSON.stringify({ ready: true }));
  rl.on('line', (line) => {
    const req = JSON.parse(line);
    process.stderr.write('GOT ' + JSON.stringify({ state: req.state, simulations: req.simulations }) + '\\n');
    console.log(JSON.stringify({ id: req.id, move: { x: 0, z: 0 } }));
  });
`;

const NEVER_READY = `
  setInterval(() => {}, 1000);
`;

const EXIT_BEFORE_READY = `
  console.error('torch exploded');
  process.exit(1);
`;

const EXIT_MID_REQUEST = `
  const rl = require('node:readline').createInterface({ input: process.stdin });
  console.log(JSON.stringify({ ready: true }));
  rl.on('line', () => process.exit(3));
`;

const GARBAGE_OUTPUT = `
  const rl = require('node:readline').createInterface({ input: process.stdin });
  console.log(JSON.stringify({ ready: true }));
  rl.on('line', () => console.log('this is not json'));
`;

const WRONG_ID = `
  const rl = require('node:readline').createInterface({ input: process.stdin });
  console.log(JSON.stringify({ ready: true }));
  rl.on('line', (line) => {
    const req = JSON.parse(line);
    console.log(JSON.stringify({ id: req.id + 1, move: { x: 0, z: 0 } }));
  });
`;

const clients: WorkerClient[] = [];

function makeClient(script: string, readyTimeoutMs = 2000): WorkerClient {
  const spec: WorkerSpec = { command: process.execPath, args: ['-e', script], readyTimeoutMs };
  const client = new WorkerClient(spec);
  clients.push(client);
  return client;
}

/** Resolves with the first stderr chunk containing `marker`. */
function onceStderr(client: WorkerClient, marker: string): Promise<string> {
  const { promise, resolve } = Promise.withResolvers<string>();
  const off = client.onStderr((chunk) => {
    if (chunk.includes(marker)) {
      off();
      resolve(chunk);
    }
  });
  return promise;
}

afterEach(() => {
  for (const c of clients.splice(0)) c.close();
});

describe('WorkerClient: readiness', () => {
  it('start resolves once the worker reports ready, then serves a request', async () => {
    const client = makeClient(ECHO_WORKER);
    await client.start();
    expect(client.alive).toBe(true);
    const res = await client.request({ grid: [] }, 100);
    expect(res).toEqual({ move: { x: 1, z: 0 } });
  });

  it('start rejects when the worker exits before ready, with stderr', async () => {
    const client = makeClient(EXIT_BEFORE_READY);
    await expect(client.start()).rejects.toThrow(/torch exploded/);
    expect(client.alive).toBe(false);
  });

  it('start rejects on the ready timeout', async () => {
    const client = makeClient(NEVER_READY, 150);
    await expect(client.start()).rejects.toThrow(/ready/);
    expect(client.alive).toBe(false);
  });

  it('start rejects when the worker emits unexpected output before ready', async () => {
    const client = makeClient(`console.log('hello before ready'); setInterval(() => {}, 1000);`, 500);
    await expect(client.start()).rejects.toThrow(/before ready/);
    expect(client.alive).toBe(false);
  });
});

describe('WorkerClient: request flow', () => {
  it('serializes concurrent requests and matches responses by id', async () => {
    const client = makeClient(ECHO_WORKER);
    await client.start();
    const [a, b] = await Promise.all([client.request({}, 1), client.request({}, 1)]);
    // Serialized: the first request gets id 1, the second id 2 — no cross-talk.
    expect(a).toEqual({ move: { x: 1, z: 0 } });
    expect(b).toEqual({ move: { x: 2, z: 0 } });
  });

  it('passes the state and simulations through in the request body', async () => {
    const client = makeClient(ECHO_STDERR);
    await client.start();
    const got = onceStderr(client, 'GOT');
    const pending = client.request({ board: [1] }, 42);
    const chunk = await got;
    expect(chunk).toContain('"simulations":42');
    expect(chunk).toContain('"board":[1]');
    await pending;
  });

  it('rejects with AbortError when the signal aborts mid-flight and discards the late response', async () => {
    const client = makeClient(GOT_THEN_SLOW_REPLY);
    await client.start();
    const got = onceStderr(client, 'GOT 1'); // worker holds the request; reply is 200ms away
    const ac = new AbortController();
    const pending = client.request({}, 1, ac.signal);
    await got; // abort strictly after send, strictly before the reply
    ac.abort();
    await expect(pending).rejects.toThrow(/aborted/i);
    // The discarded reply must not leak into the next request.
    const next = await client.request({}, 1);
    expect(next).toEqual({ move: { x: 2, z: 0 } });
    expect(client.alive).toBe(true);
  });

  it('rejects immediately for an already-aborted signal without sending', async () => {
    const client = makeClient(ECHO_WORKER);
    await client.start();
    const ac = new AbortController();
    ac.abort();
    await expect(client.request({}, 1, ac.signal)).rejects.toThrow(/aborted/i);
    const next = await client.request({}, 1);
    expect(next).toEqual({ move: { x: 1, z: 0 } }); // id 1: the aborted request was never sent
  });

  it('rejects when the worker exits mid-request and marks the client dead', async () => {
    const client = makeClient(EXIT_MID_REQUEST);
    await client.start();
    await expect(client.request({}, 1)).rejects.toThrow(/worker exited/);
    expect(client.alive).toBe(false);
  });

  it('treats non-JSON worker output as a protocol error and kills the worker', async () => {
    const client = makeClient(GARBAGE_OUTPUT);
    await client.start();
    await expect(client.request({}, 1)).rejects.toThrow(/protocol error/);
    expect(client.alive).toBe(false);
  });

  it('treats a mismatched response id as a protocol error', async () => {
    const client = makeClient(WRONG_ID);
    await client.start();
    await expect(client.request({}, 1)).rejects.toThrow(/id mismatch/);
    expect(client.alive).toBe(false);
  });

  it('rejects requests made before start or after close', async () => {
    const client = makeClient(ECHO_WORKER);
    await expect(client.request({}, 1)).rejects.toThrow(/not running/);
    await client.start();
    client.close();
    await expect(client.request({}, 1)).rejects.toThrow(/not running/);
  });
});
