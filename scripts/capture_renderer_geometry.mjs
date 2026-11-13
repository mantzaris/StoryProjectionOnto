#!/usr/bin/env node
/** Capture frozen Cytoscape geometry through a real local browser renderer. */

import { spawn } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import net from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";

const [chromeExecutable, applicationUrl, ...projectionIds] = process.argv.slice(2);
if (!chromeExecutable || !applicationUrl || projectionIds.length === 0) {
  throw new Error(
    "usage: capture_renderer_geometry.mjs CHROME_EXECUTABLE APPLICATION_URL PROJECTION_ID...",
  );
}

const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

const reservePort = async () => {
  const server = net.createServer();
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const address = server.address();
  const port = typeof address === "object" && address ? address.port : null;
  await new Promise((resolve, reject) =>
    server.close((error) => (error ? reject(error) : resolve())),
  );
  if (!port) throw new Error("could not reserve a DevTools port");
  return port;
};

const pollJson = async (url, attempts = 100) => {
  let lastError;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      const response = await fetch(url, { method: "GET" });
      if (response.ok) return await response.json();
      lastError = new Error(`${response.status} ${response.statusText}`);
    } catch (error) {
      lastError = error;
    }
    await delay(100);
  }
  throw new Error(`DevTools endpoint did not become ready: ${lastError}`);
};

const connect = async (url) => {
  const socket = new WebSocket(url);
  await new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, { once: true });
    socket.addEventListener("error", reject, { once: true });
  });
  let nextId = 1;
  const pending = new Map();
  const listeners = new Map();
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      const { resolve, reject } = pending.get(message.id);
      pending.delete(message.id);
      if (message.error) reject(new Error(JSON.stringify(message.error)));
      else resolve(message.result);
      return;
    }
    const waiting = listeners.get(message.method);
    if (waiting) {
      listeners.delete(message.method);
      waiting(message.params);
    }
  });
  const send = (method, params = {}) => {
    const id = nextId;
    nextId += 1;
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject });
      socket.send(JSON.stringify({ id, method, params }));
    });
  };
  const event = (method, timeoutMilliseconds = 15000) =>
    new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        listeners.delete(method);
        reject(new Error(`timed out waiting for ${method}`));
      }, timeoutMilliseconds);
      listeners.set(method, (params) => {
        clearTimeout(timer);
        resolve(params);
      });
    });
  return { socket, send, event };
};

const evaluate = async (session, expression) => {
  const response = await session.send("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
  if (response.exceptionDetails) {
    throw new Error(response.exceptionDetails.exception?.description || "browser evaluation failed");
  }
  return response.result.value;
};

const waitFor = async (session, expression, description, attempts = 150) => {
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    if (await evaluate(session, expression)) return;
    await delay(100);
  }
  throw new Error(`timed out waiting for ${description}`);
};

const devtoolsPort = await reservePort();
const profileDirectory = mkdtempSync(join(tmpdir(), "story-projection-geometry-"));
const chrome = spawn(
  chromeExecutable,
  [
    "--headless=new",
    "--no-sandbox",
    "--disable-background-networking",
    "--disable-component-update",
    "--disable-default-apps",
    "--disable-extensions",
    "--disable-sync",
    "--disable-features=Translate",
    "--force-device-scale-factor=1",
    "--metrics-recording-only",
    "--no-first-run",
    "--window-size=1400,1000",
    `--remote-debugging-port=${devtoolsPort}`,
    `--user-data-dir=${profileDirectory}`,
    "about:blank",
  ],
  { stdio: ["ignore", "ignore", "pipe"] },
);
let chromeError = "";
chrome.stderr.on("data", (chunk) => {
  chromeError += chunk.toString();
});

try {
  await pollJson(`http://127.0.0.1:${devtoolsPort}/json/version`);
  const targetResponse = await fetch(
    `http://127.0.0.1:${devtoolsPort}/json/new?${encodeURIComponent("about:blank")}`,
    { method: "PUT" },
  );
  if (!targetResponse.ok) {
    throw new Error(`could not create browser target: ${targetResponse.status}`);
  }
  const target = await targetResponse.json();
  const session = await connect(target.webSocketDebuggerUrl);
  try {
    await session.send("Page.enable");
    await session.send("Runtime.enable");
    for (const projectionId of projectionIds) {
      const url = new URL(applicationUrl);
      url.searchParams.set("geometry_capture", "1");
      url.searchParams.set("projection_id", projectionId);
      const loaded = session.event("Page.loadEventFired");
      await session.send("Page.navigate", { url: url.toString() });
      await loaded;
      await waitFor(
        session,
        `typeof window.storyProjectionCaptureGeometry === "function" &&
         document.querySelector("#projection-select")?.value === ${JSON.stringify(projectionId)} &&
         document.querySelectorAll("#graph canvas").length > 0`,
        `frozen Cytoscape renderer for ${projectionId}`,
      );
      const capture = await evaluate(session, "window.storyProjectionCaptureGeometry()");
      process.stdout.write(`${JSON.stringify(capture)}\n`);
    }
  } finally {
    session.socket.close();
  }
} catch (error) {
  throw new Error(`${error.message}\nChrome diagnostics:\n${chromeError.slice(-4000)}`);
} finally {
  chrome.kill("SIGTERM");
  await Promise.race([
    new Promise((resolve) => chrome.once("exit", resolve)),
    delay(2000),
  ]);
  if (chrome.exitCode === null) chrome.kill("SIGKILL");
  rmSync(profileDirectory, { recursive: true, force: true });
}
