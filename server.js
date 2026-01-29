// server.js (Vertex AI Gemini Live WebSocket proxy — Render-friendly)
require("dotenv").config();
const fs = require("fs");
const path = require("path");
const express = require("express");
const cors = require("cors");
const http = require("http");
const WebSocket = require("ws");
const { GoogleAuth } = require("google-auth-library");

const app = express();
app.use(cors());
app.use(express.json());

// -------------------- Health + Debug --------------------
app.get("/", (_req, res) => res.status(200).send("OK"));

// Version/debug so you can confirm you’re on the right build
app.get("/version", (_req, res) => {
  res.json({ backend: "VERTEX_LIVE", time: new Date().toISOString() });
});

const PORT = process.env.PORT || 3001;

// -------------------- Scenario / System Instructions --------------------
const SHARED_BEHAVIOUR_RULES = `
GLOBAL BEHAVIOUR RULES (APPLY THROUGHOUT THE CONSULTATION):

1. Asking questions:
   - You NEVER ask the clinician questions unless you are explicitly told to do so in your system instructions.
   - You do not ask "What do you think is going on?", "What tests will you do?", "Should I be worried?", etc.
   - You never speak as though you are the clinician or give advice or instructions to the clinician.

2. Worries and concerns:
   - If you mention a worry or concern (for example, fear of cancer or heart attack) and the clinician clearly acknowledges and addresses it,
     you consider that concern handled.
   - After it has been addressed once, you do NOT bring that worry up again unless the clinician directly asks you about it.

3. How you give information:
   - You ONLY give information in direct response to questions the clinician asks.
   - You do NOT volunteer extra information unprompted.
   - Your answers are brief, focused monologues: usually 1–3 sentences, directly answering the question.
   - If you are not asked about something, you do not mention it.
   - If the clinician asks a very broad question (like "Tell me more about that"), you can expand slightly but still stay concise.

4. Role boundaries:
   - You are a patient, not a clinician.
   - You never give medical explanations, diagnoses, or management plans.
   - If the clinician asks you for medical advice, you say you are not qualified and just describe your own experience.

5. Use ONLY the case information (no invention):
   - You have a fixed set of case details provided in these instructions (symptoms, history, background, etc.). Treat these as your entire memory.
   - You MUST NOT invent or guess new medical facts, investigations, timelines, or personal history beyond what is written in the case.
   - If the clinician asks for information that is NOT specified in the case details, you reply with something like:
       "I'm not sure," or "I don't remember that," or "I haven't been told that."
   - If the clinician asks a rude, sexual, offensive, or clearly inappropriate question, you reply with a boundary such as:
       "I'm not here to discuss that. I'd like to focus on my health problem."

6. If you are unsure:
   - If you are ever unsure whether something is in the case details, you assume it is NOT and you say you are not sure,
     rather than inventing or guessing.
   - These behaviour rules are CRITICAL. If you are unsure whether to say something extra, it is safer to say nothing unless asked directly.
`.trim();

const PERSONA = `
You are a 42-year-old patient called Sam.
You speak with a soft Northern English accent.
Your tone is anxious but not aggressive; you sound worried and a bit breathless.
You are attending a consultation because of chest discomfort.
You are polite and cooperative.
`.trim();

const CASE_DETAILS = `
CASE DETAILS (THIS IS YOUR ENTIRE MEMORY – DO NOT INVENT ANYTHING ELSE):

- Presenting complaint:
  - Central chest tightness for the last 2 hours.
  - Came on at rest while you were watching TV.
  - Pain has been fairly constant since, maybe slightly easing now.

- Character of pain:
  - Feels like a tight band across the centre of your chest.
  - Does not clearly radiate to your arm or jaw.

- Associated symptoms:
  - Mild shortness of breath because you feel anxious.
  - No sweating.
  - No nausea or vomiting.
  - No palpitations.

- Aggravating/relieving factors:
  - Not clearly worse on exertion.
  - Not obviously related to breathing or movement.
  - You have not tried any medication yet.

- Past medical history:
  - Mild asthma.
  - No known heart disease.
  - No previous heart attacks or angina.
  - No known high blood pressure, no known high cholesterol (unless specifically tested in the past and told normal).

- Medications:
  - Salbutamol inhaler as needed.
  - No regular cardiac medications.
  - No recent changes in medication.

- Allergies:
  - None known.

- Social history:
  - Non-smoker.
  - Drinks alcohol socially, about 6 units per week.
  - Desk job, generally sedentary but not completely inactive.
  - Lives with partner.

- Family history:
  - Father had a heart attack in his late 60s.
  - No known sudden cardiac deaths in younger relatives.

You MUST ONLY use information from these case details when answering questions.
If something is not written here, you do not know it.
`.trim();

const SYSTEM_TEXT = `${PERSONA}\n\n${CASE_DETAILS}\n\n${SHARED_BEHAVIOUR_RULES}`;

// -------------------- Env / Config --------------------
// IMPORTANT: these are the env var names this server expects:
const PROJECT = process.env.GOOGLE_CLOUD_PROJECT || null;
const LOCATION = process.env.GOOGLE_CLOUD_LOCATION || "us-central1";
const MODEL_ID = process.env.VERTEX_MODEL || "gemini-2.0-flash-live-preview-04-09";

// Vertex Live WS endpoint (v1beta1)
const VERTEX_WS_URL =
  `wss://${LOCATION}-aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent`;

// Audio formats
const INPUT_MIME = "audio/pcm;rate=16000";
const OUTPUT_EXPECTED_RATE = 24000;

// -------------------- Helpers --------------------
function toBase64(buf) {
  return Buffer.from(buf).toString("base64");
}
function safeJsonParse(s) {
  try { return JSON.parse(s); } catch { return null; }
}

// -------------------- Credentials handling (RAW JSON preferred) --------------------
// Prefer putting the RAW JSON into GOOGLE_CREDENTIALS_JSON in Render.
// If you *must* use base64, set GOOGLE_APPLICATION_CREDENTIALS_JSON_B64.
function ensureCredsFile() {
  const rawJson = process.env.GOOGLE_CREDENTIALS_JSON || process.env.GOOGLE_SERVICE_ACCOUNT_JSON;
  const b64 = process.env.GOOGLE_APPLICATION_CREDENTIALS_JSON_B64;

  let jsonStr = null;

  if (rawJson && rawJson.trim()) {
    jsonStr = rawJson;
  } else if (b64 && b64.trim()) {
    jsonStr = Buffer.from(b64.trim(), "base64").toString("utf8");
  } else {
    throw new Error(
      "Missing credentials: set GOOGLE_CREDENTIALS_JSON (preferred) or GOOGLE_APPLICATION_CREDENTIALS_JSON_B64."
    );
  }

  const trimmed = jsonStr.trim();

  // Quick sanity check
  if (!trimmed.startsWith("{")) {
    throw new Error("Service account JSON is malformed (does not start with '{').");
  }

  // Validate JSON so we fail loudly with useful errors
  JSON.parse(trimmed);

  const credPath = path.join("/tmp", "gcp-sa.json");
  fs.writeFileSync(credPath, trimmed, "utf8");
  process.env.GOOGLE_APPLICATION_CREDENTIALS = credPath;
  return credPath;
}

async function getAccessToken() {
  ensureCredsFile();

  const auth = new GoogleAuth({
    scopes: ["https://www.googleapis.com/auth/cloud-platform"],
  });

  const client = await auth.getClient();
  const tokenResponse = await client.getAccessToken();
  const token = typeof tokenResponse === "string" ? tokenResponse : tokenResponse?.token;

  if (!token) throw new Error("Failed to acquire Google OAuth access token.");
  return token;
}

// Debug endpoint (does NOT reveal secrets)
app.get("/vertex", (_req, res) => {
  let credsFileWritten = false;
  let credsError = null;

  try {
    ensureCredsFile();
    credsFileWritten = true;
  } catch (e) {
    credsError = e.message;
  }

  res.json({
    ok: true,
    project: PROJECT ? "[set]" : null,
    location: LOCATION,
    model: MODEL_ID,
    wsEndpoint: VERTEX_WS_URL,
    credsFromRawJson: !!(process.env.GOOGLE_CREDENTIALS_JSON || process.env.GOOGLE_SERVICE_ACCOUNT_JSON),
    credsFromEnvB64: !!process.env.GOOGLE_APPLICATION_CREDENTIALS_JSON_B64,
    credsFileWritten,
    credsPathSet: !!process.env.GOOGLE_APPLICATION_CREDENTIALS,
    credsError
  });
});

// -------------------- Server + Browser WS --------------------
const server = http.createServer(app);
const wss = new WebSocket.Server({ server, path: "/ws" });

wss.on("connection", async (clientWs) => {
  if (!PROJECT) {
    clientWs.send(JSON.stringify({ type: "error", message: "Missing GOOGLE_CLOUD_PROJECT env var." }));
    clientWs.close();
    return;
  }

  let vertexWs = null;
  let vertexReady = false;

  try {
    const accessToken = await getAccessToken();

    vertexWs = new WebSocket(VERTEX_WS_URL, {
      headers: {
        Authorization: `Bearer ${accessToken}`,
        "x-goog-user-project": PROJECT
      }
    });

    // If the WS handshake fails (401/403/etc), this event gives the real reason
    vertexWs.on("unexpected-response", (_req, resp) => {
      const chunks = [];
      resp.on("data", (c) => chunks.push(c));
      resp.on("end", () => {
        const body = Buffer.concat(chunks).toString("utf8").slice(0, 2000);
        clientWs.send(JSON.stringify({
          type: "error",
          message: `Vertex WS handshake failed: HTTP ${resp.statusCode}. Body: ${body}`
        }));
        try { clientWs.close(); } catch {}
      });
    });

    vertexWs.on("open", () => {
      // IMPORTANT: systemInstruction must be a Content object, not a raw string.
      const setupMsg = {
        setup: {
          model: MODEL_ID,
          generationConfig: {
            responseModalities: ["AUDIO"],
            temperature: 0.7,
            maxOutputTokens: 512
          },
          systemInstruction: {
            role: "system",
            parts: [{ text: SYSTEM_TEXT }]
          },
          inputAudioTranscription: {},
          outputAudioTranscription: {},
          realtimeInputConfig: {}
        }
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

      // Setup complete
      if (msg.setupComplete) {
        vertexReady = true;
        clientWs.send(JSON.stringify({
          type: "ready",
          model: MODEL_ID,
          outputRate: OUTPUT_EXPECTED_RATE
        }));
        return;
      }

      const serverContent = msg.serverContent;

      // Transcripts (optional)
      if (serverContent?.inputTranscription?.text) {
        clientWs.send(JSON.stringify({ type: "transcript", text: serverContent.inputTranscription.text }));
      }
      if (serverContent?.outputTranscription?.text) {
        clientWs.send(JSON.stringify({ type: "ai_transcript", text: serverContent.outputTranscription.text }));
      }

      // Main output parts
      if (serverContent?.modelTurn?.parts?.length) {
        for (const part of serverContent.modelTurn.parts) {
          if (part.text) {
            clientWs.send(JSON.stringify({ type: "ai_text", text: part.text }));
          }

          const inline = part.inlineData || part.inline_data;
          if (inline?.data) {
            clientWs.send(JSON.stringify({
              type: "audio",
              mimeType: inline.mimeType || inline.mime_type || `audio/pcm;rate=${OUTPUT_EXPECTED_RATE}`,
              data: inline.data
            }));
          }
        }
      }

      if (serverContent?.interrupted) clientWs.send(JSON.stringify({ type: "interrupted" }));
      if (serverContent?.turnComplete) clientWs.send(JSON.stringify({ type: "turn_complete" }));
    });

    vertexWs.on("close", (code, reasonBuf) => {
      const reason = Buffer.isBuffer(reasonBuf) ? reasonBuf.toString("utf8") : String(reasonBuf || "");
      clientWs.send(JSON.stringify({
        type: "closed",
        message: `Vertex WS closed. code=${code} reason=${reason}`
      }));
      try { clientWs.close(); } catch {}
    });

    vertexWs.on("error", (err) => {
      clientWs.send(JSON.stringify({ type: "error", message: `Vertex WS error: ${err.message}` }));
      try { clientWs.close(); } catch {}
    });

    // Browser -> Vertex
    clientWs.on("message", (payload, isBinary) => {
      if (!vertexWs || vertexWs.readyState !== WebSocket.OPEN) return;
      if (!vertexReady) return;

      if (isBinary) {
        const b64 = toBase64(payload);

        // Vertex Live “realtimeInput.audio” format (matches your original client->server design)
        vertexWs.send(JSON.stringify({
          realtimeInput: {
            audio: { mimeType: INPUT_MIME, data: b64 }
          }
        }));
        return;
      }

      const text = payload.toString("utf8");
      const msg = safeJsonParse(text);
      if (!msg) return;

      if (msg.type === "text" && typeof msg.text === "string") {
        vertexWs.send(JSON.stringify({ realtimeInput: { text: msg.text } }));
      }

      if (msg.type === "stop_audio") {
        vertexWs.send(JSON.stringify({ realtimeInput: { audioStreamEnd: true } }));
      }
    });

    clientWs.on("close", () => {
      try { vertexWs?.close(); } catch {}
    });
    clientWs.on("error", () => {
      try { vertexWs?.close(); } catch {}
    });

  } catch (err) {
    clientWs.send(JSON.stringify({ type: "error", message: err.message || String(err) }));
    clientWs.close();
  }
});

server.listen(PORT, () => {
  console.log(`Backend listening on port ${PORT}`);
});
