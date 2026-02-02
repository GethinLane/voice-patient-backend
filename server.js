// server.js (Vertex Gemini Live API proxy + Airtable case selector)
// npm i express cors ws dotenv google-auth-library node-fetch
// Ensure Node >= 18 (recommended). If not, node-fetch fallback will be used.

require("dotenv").config();

const express = require("express");
const cors = require("cors");
const http = require("http");
const WebSocket = require("ws");
const { GoogleAuth } = require("google-auth-library");

// ---- fetch compatibility (Node 18 has global fetch) ----
let fetchFn = global.fetch;
async function getFetch() {
  if (fetchFn) return fetchFn;
  const mod = await import("node-fetch");
  fetchFn = mod.default;
  return fetchFn;
}

const app = express();
app.use(cors());
app.use(express.json({ limit: "2mb" }));

app.get("/", (_req, res) => res.status(200).send("OK"));
app.get("/health", (_req, res) => res.json({ ok: true }));

const PORT = process.env.PORT || 3001;

// ----------------------- AIRTABLE CONFIG -----------------------
const AIRTABLE_API_KEY = process.env.AIRTABLE_API_KEY;
const AIRTABLE_BASE_ID = process.env.AIRTABLE_BASE_ID;
const AIRTABLE_VIEW = process.env.AIRTABLE_VIEW || ""; // e.g. "AI"

function assertAirtableEnv() {
  if (!AIRTABLE_API_KEY || !AIRTABLE_BASE_ID) {
    throw new Error("Missing AIRTABLE_API_KEY or AIRTABLE_BASE_ID in environment.");
  }
}

function safeJsonParse(s) {
  try {
    return JSON.parse(s);
  } catch {
    return null;
  }
}

async function airtableFetch(url) {
  assertAirtableEnv();
  const fetch = await getFetch();

  const res = await fetch(url, {
    headers: { Authorization: `Bearer ${AIRTABLE_API_KEY}` },
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Airtable error ${res.status}: ${text.slice(0, 400)}`);
  }
  return res.json();
}

// List tables in base and extract Case numbers from names like "Case 194"
async function listCaseNumbersFromMeta() {
  const url = `https://api.airtable.com/v0/meta/bases/${AIRTABLE_BASE_ID}/tables`;
  const json = await airtableFetch(url);

  const tableNames = (json.tables || []).map((t) => t.name);
  const nums = tableNames
    .map((name) => {
      const m = /^Case\s+(\d+)$/i.exec(String(name).trim());
      return m ? Number(m[1]) : null;
    })
    .filter((n) => Number.isFinite(n))
    .sort((a, b) => a - b);

  return nums;
}

async function fetchAllCaseRows(caseNumber) {
  const tableName = `Case ${caseNumber}`;

  let offset = null;
  const records = [];

  do {
    const params = new URLSearchParams();
    params.set("pageSize", "100");
    if (AIRTABLE_VIEW) params.set("view", AIRTABLE_VIEW);
    if (offset) params.set("offset", offset);

    const url =
      `https://api.airtable.com/v0/${AIRTABLE_BASE_ID}/` +
      `${encodeURIComponent(tableName)}?${params.toString()}`;

    const json = await airtableFetch(url);
    records.push(...(json.records || []));
    offset = json.offset || null;
  } while (offset);

  if (!records.length) {
    throw new Error(`No records found in Airtable table "${tableName}".`);
  }

  return { tableName, records };
}

function combineFieldAcrossRows(records, fieldName) {
  const parts = [];
  for (const r of records) {
    const v = r?.fields?.[fieldName];
    if (typeof v === "string") {
      const t = v.trim();
      if (t) parts.push(t);
    } else if (v != null) {
      const t = String(v).trim();
      if (t) parts.push(t);
    }
  }
  return parts.join("\n\n");
}

function buildSystemTextFromCase(records) {
  const opening = combineFieldAcrossRows(records, "Opening Sentence");
  const divulgeFreely = combineFieldAcrossRows(records, "Divulge freely");
  const divulgeAsked = combineFieldAcrossRows(records, "Divulge Asked");
  const pmhx = combineFieldAcrossRows(records, "PMHx RP");
  const social = combineFieldAcrossRows(records, "Social History");

  const family =
    combineFieldAcrossRows(records, "Family Hiostory") ||
    combineFieldAcrossRows(records, "Family History");

  const ice = combineFieldAcrossRows(records, "ICE");
  const reaction = combineFieldAcrossRows(records, "Reaction");

  const RULES = `
CRITICAL:
- You MUST NOT invent details.
- Only use information explicitly present in the CASE DETAILS below.
- If something is not stated, say: "I'm not sure" / "I don't remember" / "I haven't been told".
- NEVER substitute another symptom.
- NEVER create symptoms.
- NEVER swap relatives. If relationship is not explicit, say you're not sure.
- Answer only what the clinician asks.
`.trim();

  const CASE = `
CASE DETAILS (THIS IS YOUR ENTIRE MEMORY):

OPENING SENTENCE:
${opening || "[Not provided]"}

DIVULGE FREELY:
${divulgeFreely || "[Not provided]"}

DIVULGE ONLY IF ASKED:
${divulgeAsked || "[Not provided]"}

PAST MEDICAL HISTORY:
${pmhx || "[Not provided]"}

SOCIAL HISTORY:
${social || "[Not provided]"}

FAMILY HISTORY:
${family || "[Not provided]"}

ICE (Ideas / Concerns / Expectations):
${ice || "[Not provided]"}

REACTION / AFFECT:
${reaction || "[Not provided]"}
`.trim();

  return `${CASE}\n\n${RULES}`;
}

// ----------------------- VERTEX LIVE CONFIG -----------------------
const VERTEX_PROJECT_ID =
  process.env.VERTEX_PROJECT_ID || process.env.GOOGLE_CLOUD_PROJECT;

const VERTEX_LOCATION = process.env.VERTEX_LOCATION || "us-central1";
const VERTEX_MODEL_ID =
  process.env.VERTEX_MODEL_ID || "gemini-live-2.5-flash-native-audio";

const VERTEX_WS_ENDPOINT =
  `wss://${VERTEX_LOCATION}-aiplatform.googleapis.com/ws/` +
  `google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent`;

const INPUT_MIME = "audio/pcm;rate=16000";
const OUTPUT_EXPECTED_RATE = 24000;

// ----------------------- AUTH HELPERS -----------------------
function toBase64(buf) {
  return Buffer.from(buf).toString("base64");
}

function loadServiceAccountCredentials() {
  if (process.env.GOOGLE_CREDENTIALS_JSON) {
    const obj = safeJsonParse(process.env.GOOGLE_CREDENTIALS_JSON);
    if (obj) return obj;
  }

  if (process.env.GOOGLE_APPLICATION_CREDENTIALS_JSON_B64) {
    const decoded = Buffer.from(
      process.env.GOOGLE_APPLICATION_CREDENTIALS_JSON_B64.replace(/^"+|"+$/g, ""),
      "base64"
    ).toString("utf8");
    const obj = safeJsonParse(decoded);
    if (obj) return obj;
  }

  return null;
}

async function getAccessToken() {
  const creds = loadServiceAccountCredentials();

  const auth = creds
    ? new GoogleAuth({
        credentials: creds,
        scopes: ["https://www.googleapis.com/auth/cloud-platform"],
      })
    : new GoogleAuth({
        scopes: ["https://www.googleapis.com/auth/cloud-platform"],
      });

  const client = await auth.getClient();
  const tokenResponse = await client.getAccessToken();

  const token =
    typeof tokenResponse === "string" ? tokenResponse : tokenResponse?.token;
  if (!token) throw new Error("Failed to obtain Google OAuth access token.");
  return token;
}

function fullyQualifiedModelName() {
  if (!VERTEX_PROJECT_ID) return null;
  return `projects/${VERTEX_PROJECT_ID}/locations/${VERTEX_LOCATION}/publishers/google/models/${VERTEX_MODEL_ID}`;
}

// Debug endpoint so you can verify env is loaded on Render
app.get("/vertex", (_req, res) => {
  res.json({
    ok: true,
    project: VERTEX_PROJECT_ID ? "[set]" : null,
    location: VERTEX_LOCATION,
    modelId: VERTEX_MODEL_ID,
    modelFqn: fullyQualifiedModelName(),
    wsEndpoint: VERTEX_WS_ENDPOINT,
    credsJsonPresent: !!process.env.GOOGLE_CREDENTIALS_JSON,
    credsB64Present: !!process.env.GOOGLE_APPLICATION_CREDENTIALS_JSON_B64,
    outputRate: OUTPUT_EXPECTED_RATE,
    airtable: {
      baseIdPresent: !!AIRTABLE_BASE_ID,
      apiKeyPresent: !!AIRTABLE_API_KEY,
      view: AIRTABLE_VIEW || null,
    },
  });
});

// ----------------------- LIST AVAILABLE CASES -----------------------
app.get("/cases", async (_req, res) => {
  try {
    const cases = await listCaseNumbersFromMeta();
    res.json({ ok: true, cases });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message || String(e) });
  }
});

// ----------------------- WS PROXY -----------------------
const server = http.createServer(app);
const wss = new WebSocket.Server({ server, path: "/ws" });

wss.on("connection", async (clientWs) => {
  let vertexWs = null;
  let vertexReady = false;

  let selectedCaseNumber = null;
  let initReceived = false;

  const CLIENT_PING_MS = 15000;
  const VERTEX_PING_MS = 15000;

  function send(type, obj) {
    try {
      clientWs.send(JSON.stringify({ type, ...obj }));
    } catch {}
  }

  function sendErrorAndClose(message) {
    send("error", { message });
    try { clientWs.close(); } catch {}
    try { vertexWs?.close(); } catch {}
  }

  const clientPingTimer = setInterval(() => {
    try {
      if (clientWs.readyState === WebSocket.OPEN) clientWs.ping();
    } catch {}
  }, CLIENT_PING_MS);

  let vertexPingTimer = null;

  async function startVertexWithCase(caseNumber) {
    const modelFqn = fullyQualifiedModelName();
    if (!modelFqn) throw new Error("Missing VERTEX_PROJECT_ID/GOOGLE_CLOUD_PROJECT.");

    // Fetch case from Airtable
    const { tableName, records } = await fetchAllCaseRows(caseNumber);
    const systemText = buildSystemTextFromCase(records);

    // OAuth token
    const token = await getAccessToken();

    // Connect to Vertex Live WS
    vertexWs = new WebSocket(VERTEX_WS_ENDPOINT, {
      headers: { Authorization: `Bearer ${token}` },
    });

    vertexWs.on("open", () => {
      const setupMsg = {
        setup: {
          model: modelFqn,
          generation_config: {
            // Option 4: include TEXT too for debugging
            response_modalities: ["AUDIO"],
            temperature: 0.2,
            max_output_tokens: 512,
          },
          system_instruction: {
            parts: [{ text: systemText }],
          },
          input_audio_transcription: {},
          output_audio_transcription: {},
          realtime_input_config: {
            automatic_activity_detection: {
              disabled: false,
              start_of_speech_sensitivity: "START_SENSITIVITY_LOW",
              end_of_speech_sensitivity: "END_SENSITIVITY_HIGH",
              prefix_padding_ms: 50,
              silence_duration_ms: 250,
            },
          },
        },
      };

      vertexWs.send(JSON.stringify(setupMsg));

      // Keep Vertex alive too
      vertexPingTimer = setInterval(() => {
        try {
          if (vertexWs && vertexWs.readyState === WebSocket.OPEN) vertexWs.ping?.();
        } catch {}
      }, VERTEX_PING_MS);
    });

    vertexWs.on("message", (data) => {
      const msgStr = Buffer.isBuffer(data) ? data.toString("utf8") : String(data);
      const msg = safeJsonParse(msgStr);

      if (!msg) {
        send("debug", { raw: msgStr.slice(0, 800) });
        return;
      }

      if (msg.setupComplete) {
        vertexReady = true;
        send("ready", {
          model: modelFqn,
          outputRate: OUTPUT_EXPECTED_RATE,
          caseId: caseNumber,
          caseTable: tableName,
        });
        return;
      }

      // Forward useful non-content signals
      if (msg.goAway || msg.go_away) send("goaway", { goAway: msg.goAway || msg.go_away });
      if (msg.usageMetadata || msg.usage_metadata) send("usage", { usage: msg.usageMetadata || msg.usage_metadata });

      const serverContent = msg.serverContent || msg.server_content;

      // Optional transcriptions
      const inTr =
        serverContent?.input_transcription?.text ||
        serverContent?.inputTranscription?.text;
      if (inTr) send("transcript", { text: inTr });

      const outTr =
        serverContent?.output_transcription?.text ||
        serverContent?.outputTranscription?.text;
      if (outTr) send("ai_transcript", { text: outTr });

      // Main streamed parts
      const modelTurn = serverContent?.model_turn || serverContent?.modelTurn;
      const parts = modelTurn?.parts || [];
      for (const part of parts) {
        if (part.text) send("ai_text", { text: part.text });

        const inline = part.inline_data || part.inlineData;
        if (inline?.data) {
          send("audio", {
            mimeType:
              inline.mime_type ||
              inline.mimeType ||
              `audio/pcm;rate=${OUTPUT_EXPECTED_RATE}`,
            data: inline.data,
          });
        }
      }

      const interrupted = serverContent?.interrupted;
      const turnComplete = serverContent?.turn_complete || serverContent?.turnComplete;
      if (interrupted) send("interrupted", {});
      if (turnComplete) send("turn_complete", {});
    });

    vertexWs.on("close", (code, reason) => {
      if (vertexPingTimer) clearInterval(vertexPingTimer);
      vertexPingTimer = null;

      send("closed", {
        message: `Vertex WS closed. code=${code} reason=${reason?.toString?.() || ""}`,
      });
      try { clientWs.close(); } catch {}
    });

    vertexWs.on("error", (err) => {
      if (vertexPingTimer) clearInterval(vertexPingTimer);
      vertexPingTimer = null;
      sendErrorAndClose(`Vertex WS error: ${err.message}`);
    });
  }

  // Browser -> Server
  clientWs.on("message", async (payload, isBinary) => {
    if (isBinary) {
      // Block binary audio until init & vertex ready
      if (!initReceived || !vertexWs || vertexWs.readyState !== WebSocket.OPEN || !vertexReady) return;

      const audioMsg = {
        realtime_input: {
          media_chunks: [
            {
              mime_type: INPUT_MIME,
              data: toBase64(payload),
            },
          ],
        },
      };
      try {
        vertexWs.send(JSON.stringify(audioMsg));
      } catch {}
      return;
    }

    const text = payload.toString("utf8");
    const msg = safeJsonParse(text);
    if (!msg) return;

    if (msg.type === "init") {
      if (initReceived) return;

      const caseId = Number(msg.caseId);
      if (!Number.isFinite(caseId) || caseId <= 0) {
        sendErrorAndClose(`Invalid caseId in init: ${msg.caseId}`);
        return;
      }

      initReceived = true;
      selectedCaseNumber = caseId;

      try {
        await startVertexWithCase(selectedCaseNumber);
      } catch (e) {
        sendErrorAndClose(e.message || String(e));
      }
      return;
    }

    if (msg.type === "ping") {
      send("pong", { t: Date.now() });
      return;
    }

    if (msg.type === "stop_audio") {
      try { vertexWs?.close(); } catch {}
      return;
    }

    if (msg.type === "text" && typeof msg.text === "string") {
      if (!initReceived || !vertexWs || vertexWs.readyState !== WebSocket.OPEN || !vertexReady) return;

      const clientContentMsg = {
        client_content: {
          turns: [{ role: "user", parts: [{ text: msg.text }] }],
          turn_complete: true,
        },
      };
      try {
        vertexWs.send(JSON.stringify(clientContentMsg));
      } catch {}
      return;
    }
  });

  clientWs.on("close", () => {
    clearInterval(clientPingTimer);
    if (vertexPingTimer) clearInterval(vertexPingTimer);
    try { vertexWs?.close(); } catch {}
  });

  clientWs.on("error", () => {
    clearInterval(clientPingTimer);
    if (vertexPingTimer) clearInterval(vertexPingTimer);
    try { vertexWs?.close(); } catch {}
  });

  // reminder if frontend forgets to init
  setTimeout(() => {
    if (!initReceived) {
      send("error", {
        message: "No init received. Send {type:'init', caseId:<number>} right after WS open.",
      });
      try { clientWs.close(); } catch {}
    }
  }, 15000);
});

server.listen(PORT, () => {
  console.log(`Backend listening on port ${PORT}`);
});
