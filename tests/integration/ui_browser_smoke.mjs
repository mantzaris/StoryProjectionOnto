import { spawn } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import net from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";

const [chromeExecutable, applicationUrl] = process.argv.slice(2);
if (!chromeExecutable || !applicationUrl) {
  throw new Error("usage: ui_browser_smoke.mjs CHROME_EXECUTABLE APPLICATION_URL");
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
  await new Promise((resolve, reject) => server.close((error) => (error ? reject(error) : resolve())));
  if (!port) throw new Error("could not reserve a DevTools port");
  return port;
};

const pollJson = async (url, attempts = 80) => {
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
  const event = (method, timeoutMilliseconds = 10000) =>
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

const waitFor = async (session, expression, description, attempts = 100) => {
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    if (await evaluate(session, expression)) return;
    await delay(100);
  }
  throw new Error(`timed out waiting for ${description}`);
};

const assertBrowserState = async (session, expression, message) => {
  if (!(await evaluate(session, expression))) throw new Error(message);
};

const devtoolsPort = await reservePort();
const profileDirectory = mkdtempSync(join(tmpdir(), "story-projection-browser-"));
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
    "--metrics-recording-only",
    "--no-first-run",
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
    `http://127.0.0.1:${devtoolsPort}/json/new?${encodeURIComponent(applicationUrl)}`,
    { method: "PUT" },
  );
  if (!targetResponse.ok) throw new Error(`could not create browser target: ${targetResponse.status}`);
  const target = await targetResponse.json();
  const session = await connect(target.webSocketDebuggerUrl);
  try {
    await session.send("Page.enable");
    await session.send("Runtime.enable");
    const loaded = session.event("Page.loadEventFired");
    await session.send("Page.navigate", { url: applicationUrl });
    await loaded;
    await waitFor(
      session,
      `document.querySelectorAll("#projection-select option").length > 0 &&
       document.querySelector("#asset-status").textContent.includes("verified locally")`,
      "verified Cytoscape projection",
    );
    await assertBrowserState(
      session,
      `document.querySelectorAll("#graph canvas").length > 0`,
      "Cytoscape did not create a canvas",
    );
    await assertBrowserState(
      session,
      `document.querySelector("#detail-panel").textContent.includes("qualified assertions")`,
      "initial rich projection summary was not rendered",
    );

    await evaluate(
      session,
      `(() => {
        document.querySelector("#story-point").value = "1";
        document.querySelector("#discourse-horizon").value = "1";
        document.querySelector("#revelation-horizon").value = "1";
        document.querySelector("#apply-filter").click();
        return true;
      })()`,
    );
    await waitFor(
      session,
      `performance.getEntriesByType("resource").some(
        (entry) => entry.name.includes("/filter")
      )`,
      "story/spoiler filter request",
    );

    const submit = async (action, evidence, mentions, signature, lens = "") => {
      await evaluate(
        session,
        `(() => {
          document.querySelector("#revision-action").value = ${JSON.stringify(action)};
          document.querySelector("#revision-evidence").value = ${JSON.stringify(evidence)};
          document.querySelector("#revision-mentions").value = ${JSON.stringify(mentions)};
          document.querySelector("#revision-signature").value = ${JSON.stringify(signature)};
          document.querySelector("#revision-lens").value = ${JSON.stringify(lens)};
          document.querySelector("#revision-rationale").value =
            "System-browser validation of the bounded typed action.";
          document.querySelector("#revision-form").requestSubmit();
          return true;
        })()`,
      );
      await waitFor(
        session,
        `document.querySelector("#revision-status").textContent.includes(
          "no regeneration was claimed"
        )`,
        `${action} capability-safe response`,
      );
    };
    await submit("REFINE_CONTEXT", "ev-a", "m-a", "focus responsibility", "responsibility");
    await submit("REQUEST_MERGE_SPLIT", "ev-a, ev-b", "m-a | m-b", "merge identity");

    const evidenceMetadata = await evaluate(
      session,
      `fetch("/api/evidence/ev-a").then((response) => response.json())`,
    );
    if (evidenceMetadata.evidence_id !== "ev-a") {
      throw new Error("browser could not resolve evidence detail through the local API");
    }
  } finally {
    session.socket.close();
  }
  process.stdout.write("system browser smoke passed\n");
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
