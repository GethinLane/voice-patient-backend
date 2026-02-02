// server.js (Vertex Gemini Live API proxy + Airtable case selector)
// npm i express cors ws dotenv google-auth-library

require("dotenv").config();

const express = require("express");
const cors = require("cors");
const http = require("http");
const WebSocket = require("ws");
const { GoogleAuth } = require("google-auth-library");

const app = express();
app.use(cors());
app.use(express.json({ limit: "2mb" }));

app.get("/", (_req, res) => res.status(200).send("OK"));

const PORT = process.env.PORT || 3001;

// ----------------------- AIRTABLE CONFIG -----------------------
const AIRTABLE_API_KEY = process.env.AIRTABLE_API_KEY;
const AIRTABLE_BASE_ID = process.env.AIRTABLE_BASE_ID;

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

  const res = await fetch(url, {
    headers: {
      Authorization: `Bearer ${AIRTABLE_API_KEY}`,
    },
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

// Fetch first record from table "Case X"
async function fetchCaseRecord(caseNumber) {
  const tableName = `Case ${caseNumber}`;
  const url =
    `https://api.airtable.com/v0/${AIRTABLE_BASE_ID}/` +
    `${encodeURIComponent(tableName)}?pageSize=1`;

  const json = await airtableFetch(url);
  const rec = (json.records || [])[0];
  if (!rec || !rec.fields) {
    throw new Error(`No records found in Airtable table "${tableName}".`);
  }
  return { tableName, recordId: rec.id, fields: rec.fields };
}

function fieldStr(fields, name) {
  const v = fields?.[name];
  if (v === undefined || v === null) return "";
  if (typeof v === "string") return v.trim();
  // If someone later turns a field into array/rich text etc, stringify gently:
  return String(v).trim();
}

// Build dynamic patient system text from Airtable fields
function buildSystemTextFromCase(fields) {
  // Supports your typo and a corrected version, just in case
  const family =
    fieldStr(fields, "Family Hiostory") || fieldStr(fields, "Family History");

  const opening = fieldStr(fields, "Opening Sentence");
  const divulgeFreely = fieldStr(fields, "Divulge freely");
  const divulgeAsked = fieldStr(fields, "Divulge Asked");
  const pmhx = fieldStr(fields, "PMHx RP");
  const social = fieldStr(fields, "Social History");
  const ice = fieldStr(fields, "ICE");
  const reaction = fieldStr(fields, "Reaction");

  const SHARED_BEHAVIOUR_RULES = `
GLOBAL BEHAVIOUR RULES (APPLY THROUGHOUT THE CONSULTATION):

1. Asking questions:
   - You NEVER ask the clinician questions unless you are explicitly told to do so in your system instructions.
   - You do not ask "What do you think is going on?", "What tests will you do?", "Should I be worried?", etc.
   - You never speak as though you are the clinician or give advice or instructions to the clinician.
   - “At the start, always begin with the Opening Sentence”
   - you will divulge the information in 'Divulge Freely' section quickly if the doctor asks you to expand on your opening sentence

2. Worries and concerns:
   - If you mention a worry or concern and the clinician clearly acknowledges and addresses it,
     you consider that concern handled.
   - After it has been addressed once, you do NOT bring that worry up again unless the clinician directly asks you about it.

3. How you give information:
   - You ONLY give information in direct response to questions the clinician asks.
   - You do NOT volunteer extra information unprompted.
   - Your answers are brief, focused monologues: usually 1–3 sentences, directly answering the question.
   - If the clinician asks a very broad question (like "Tell me more about that"), you can expand slightly but still stay concise.

   IMPORTANT SPECIAL RULE FOR THIS SIMULATION:
   - If the clinician asks an OPEN question at the start (e.g. "Tell me what's been happening"),
     you may include BOTH:
       (a) the Opening Sentence, and
       (b) key points from "Divulge freely".
   - Otherwise, stick to only answering what was asked.

4. Role boundaries:
   - You are a patient, not a clinician.
   - You never give medical explanations, diagnoses, or management plans.
   - You do not ask questions unless specifically instructed to do so.
   - If the clinician asks you for medical advice, you say you are not qualified and just describe your own experience.

5. Use ONLY the case information (no invention):
   - You have a fixed set of case details provided in these instructions. Treat these as your entire memory.
   - You MUST NOT invent or guess new medical facts, investigations, timelines, or personal history beyond what is written.
   - If the clinician asks for information that is NOT specified, you reply:
       "I'm not sure," or "I don't remember that," or "I haven't been told that."
   - If the clinician asks a rude, sexual, offensive, or clearly inappropriate question, you reply with a boundary such as:
       "I'm not here to discuss that. I'd like to focus on my health problem."

6. If you are unsure:
   - If you are ever unsure whether something is in the case details, you assume it is NOT and you say you are not sure,
     rather than inventing or guessing.
`.trim();

  // Keep your persona simple; your "Reaction" field can override behaviour/tone
  const PERSONA = `
You are the patient in a medical consultation.
You speak naturally (UK English).
You sound like a real person: not robotic, not overly verbose.
`.trim();

  const CASE_DETAILS = `
CASE DETAILS (THIS IS YOUR ENTIRE MEMORY – DO NOT INVENT ANYTHING ELSE):

OPENING SENTENCE:
${opening || "[Not provided]"}

DIVULGE FREELY (can be included when asked broad/open questions):
${divulgeFreely || "[Not provided]"}

DIVULGE ONLY IF ASKED SPECIFICALLY:
${divulgeAsked || "[Not provided]"}

PAST MEDICAL HISTORY (ONLY IF ASKED):
${pmhx || "[Not provided]"}

SOCIAL HISTORY (ONLY IF ASKED):
${social || "[Not provided]"}

FAMILY HISTORY (ONLY IF ASKED):
${family || "[Not provided]"}

ICE (Ideas / Concerns / Expectations):
${ice || "[Not provided]"}

REACTION / AFFECT / HOW TO ACT:
${reaction || "[Not provided]"}
`.trim();

  return `${PERSONA}\n\n${CASE_DETAILS}\n\n${SHARED_BEHAVIOUR_RULES}`;
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
    },
  });
});

// ----------------------- NEW: LIST AVAILABLE CASES -----------------------
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

  // We now wait for init before connecting to Vertex
  let selectedCaseNumber = null;
  let initReceived = false;

  function sendErrorAndClose(message) {
    try {
      clientWs.send(JSON.stringify({ type: "error", message }));
    } catch {}
    try { clientWs.close(); } catch {}
    try { vertexWs?.close(); } catch {}
  }

  async function startVertexWithCase(caseNumber) {
    const modelFqn = fullyQualifiedModelName();
    if (!modelFqn) throw new Error("Missing VERTEX_PROJECT_ID/GOOGLE_CLOUD_PROJECT.");

    // Fetch case from Airtable
    const { tableName, fields } = await fetchCaseRecord(caseNumber);
    const systemText = buildSystemTextFromCase(fields);

    // OAuth token (service account)
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
            response_modalities: ["AUDIO"],
            temperature: 0.7,
            max_output_tokens: 512,
          },
          system_instruction: {
            parts: [{ text: systemText }],
          },
          input_audio_transcription: {},
          output_audio_transcription: {},
          realtime_input_config: {},
        },
      };

      vertexWs.send(JSON.stringify(setupMsg));
    });

    vertexWs.on("message", (data) => {
      const msgStr = Buffer.isBuffer(data) ? data.toString("utf8") : String(data);
      const msg = safeJsonParse(msgStr);

      if (!msg) {
        clientWs.send(JSON.stringify({ type: "debug", raw: msgStr.slice(0, 500) }));
        return;
      }

      if (msg.setupComplete) {
        vertexReady = true;
        clientWs.send(JSON.stringify({
          type: "ready",
          model: modelFqn,
          outputRate: OUTPUT_EXPECTED_RATE,
          caseId: caseNumber,
          caseTable: tableName,
        }));
        return;
      }

      const serverContent = msg.serverContent || msg.server_content;

      // Optional transcriptions
      const inTr =
        serverContent?.input_transcription?.text ||
        serverContent?.inputTranscription?.text;
      if (inTr) clientWs.send(JSON.stringify({ type: "transcript", text: inTr }));

      const outTr =
        serverContent?.output_transcription?.text ||
        serverContent?.outputTranscription?.text;
      if (outTr) clientWs.send(JSON.stringify({ type: "ai_transcript", text: outTr }));

      // Main streamed parts
      const modelTurn = serverContent?.model_turn || serverContent?.modelTurn;
      const parts = modelTurn?.parts || [];
      for (const part of parts) {
        if (part.text) {
          clientWs.send(JSON.stringify({ type: "ai_text", text: part.text }));
        }

        const inline = part.inline_data || part.inlineData;
        if (inline?.data) {
          clientWs.send(JSON.stringify({
            type: "audio",
            mimeType:
              inline.mime_type ||
              inline.mimeType ||
              `audio/pcm;rate=${OUTPUT_EXPECTED_RATE}`,
            data: inline.data,
          }));
        }
      }

      // Turn markers
      const interrupted = serverContent?.interrupted;
      const turnComplete = serverContent?.turn_complete || serverContent?.turnComplete;
      if (interrupted) clientWs.send(JSON.stringify({ type: "interrupted" }));
      if (turnComplete) clientWs.send(JSON.stringify({ type: "turn_complete" }));
    });

    vertexWs.on("close", (code, reason) => {
      clientWs.send(JSON.stringify({
        type: "closed",
        message: `Vertex WS closed. code=${code} reason=${reason?.toString?.() || ""}`,
      }));
      try { clientWs.close(); } catch {}
    });

    vertexWs.on("error", (err) => {
      sendErrorAndClose(`Vertex WS error: ${err.message}`);
    });
  }

  // Browser -> Server
  clientWs.on("message", async (payload, isBinary) => {
    // Block binary audio until init & vertex ready
    if (isBinary) {
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
      vertexWs.send(JSON.stringify(audioMsg));
      return;
    }

    const text = payload.toString("utf8");
    const msg = safeJsonParse(text);
    if (!msg) return;

    // NEW: init message selects case
    if (msg.type === "init") {
      if (initReceived) return; // ignore repeats

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

    // Allow stop_audio even before init
    if (msg.type === "stop_audio") {
      try { vertexWs?.close(); } catch {}
      return;
    }

    // Text chat support (optional) after ready
    if (msg.type === "text" && typeof msg.text === "string") {
      if (!initReceived || !vertexWs || vertexWs.readyState !== WebSocket.OPEN || !vertexReady) return;

      const clientContentMsg = {
        client_content: {
          turns: [{ role: "user", parts: [{ text: msg.text }] }],
          turn_complete: true,
        },
      };
      vertexWs.send(JSON.stringify(clientContentMsg));
      return;
    }
  });

  clientWs.on("close", () => {
    try { vertexWs?.close(); } catch {}
  });

  clientWs.on("error", () => {
    try { vertexWs?.close(); } catch {}
  });

  // Gentle reminder if frontend forgets to init
  setTimeout(() => {
    if (!initReceived) {
      try {
        clientWs.send(JSON.stringify({
          type: "error",
          message: "No init received. Send {type:'init', caseId:<number>} right after WS open.",
        }));
      } catch {}
      try { clientWs.close(); } catch {}
    }
  }, 15000);
});

server.listen(PORT, () => {
  console.log(`Backend listening on port ${PORT}`);
});
