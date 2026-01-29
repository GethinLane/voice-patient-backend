// server.js (Vertex Gemini Live API proxy)
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

// ----------------------- YOUR PATIENT PROMPT -----------------------
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

// ----------------------- VERTEX LIVE CONFIG -----------------------
const VERTEX_PROJECT_ID =
  process.env.VERTEX_PROJECT_ID || process.env.GOOGLE_CLOUD_PROJECT;
const VERTEX_LOCATION = process.env.VERTEX_LOCATION || "us-central1";

// Use a current Live model ID (Vertex docs list these; example below)
const VERTEX_MODEL_ID =
  process.env.VERTEX_MODEL_ID || "gemini-live-2.5-flash-native-audio";

const VERTEX_WS_ENDPOINT =
  `wss://${VERTEX_LOCATION}-aiplatform.googleapis.com/ws/` +
  `google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent`;

const INPUT_MIME = "audio/pcm;rate=16000";
const OUTPUT_EXPECTED_RATE = 24000;

// ----------------------- AUTH HELPERS -----------------------
function safeJsonParse(s) {
  try {
    return JSON.parse(s);
  } catch {
    return null;
  }
}

function toBase64(buf) {
  return Buffer.from(buf).toString("base64");
}

function loadServiceAccountCredentials() {
  // Prefer full JSON (no base64 headaches)
  if (process.env.GOOGLE_CREDENTIALS_JSON) {
    const obj = safeJsonParse(process.env.GOOGLE_CREDENTIALS_JSON);
    if (obj) return obj;
  }

  // If you truly want base64, it must decode to VALID JSON starting with "{"
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

  const token = typeof tokenResponse === "string" ? tokenResponse : tokenResponse?.token;
  if (!token) throw new Error("Failed to obtain Google OAuth access token.");
  return token;
}

function fullyQualifiedModelName() {
  if (!VERTEX_PROJECT_ID) return null;

  // Vertex Live requires publisher-model FQN
  // Format: projects/{project}/locations/{location}/publishers/*/models/*
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
  });
});

// ----------------------- WS PROXY -----------------------
const server = http.createServer(app);
const wss = new WebSocket.Server({ server, path: "/ws" });

wss.on("connection", async (clientWs) => {
  let vertexWs = null;
  let vertexReady = false;

  try {
    const modelFqn = fullyQualifiedModelName();
    if (!modelFqn) {
      clientWs.send(JSON.stringify({ type: "error", message: "Missing VERTEX_PROJECT_ID/GOOGLE_CLOUD_PROJECT." }));
      clientWs.close();
      return;
    }

    // OAuth token (service account)
    const token = await getAccessToken();

    // Connect to Vertex Live WS with Authorization header
    vertexWs = new WebSocket(VERTEX_WS_ENDPOINT, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });

    vertexWs.on("open", () => {
      // IMPORTANT:
      // - model must be fully-qualified publisher model name
      // - system_instruction must be a Content object (parts[].text)
      const setupMsg = {
        setup: {
          model: modelFqn,
          generation_config: {
            response_modalities: ["AUDIO"],
            temperature: 0.7,
            max_output_tokens: 512,
            // speech_config optional (voices etc)
          },
          system_instruction: {
            parts: [{ text: SYSTEM_TEXT }],
          },
          input_audio_transcription: {},
          output_audio_transcription: {},
          realtime_input_config: {
            // defaults are fine (server-side VAD)
          },
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
        }));
        return;
      }

      const serverContent = msg.serverContent || msg.server_content;

      // Optional transcriptions
      const inTr = serverContent?.input_transcription?.text || serverContent?.inputTranscription?.text;
      if (inTr) clientWs.send(JSON.stringify({ type: "transcript", text: inTr }));

      const outTr = serverContent?.output_transcription?.text || serverContent?.outputTranscription?.text;
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
            mimeType: inline.mime_type || inline.mimeType || `audio/pcm;rate=${OUTPUT_EXPECTED_RATE}`,
            data: inline.data, // base64
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
      clientWs.send(JSON.stringify({ type: "error", message: `Vertex WS error: ${err.message}` }));
      try { clientWs.close(); } catch {}
    });

    // Browser -> Vertex
    clientWs.on("message", (payload, isBinary) => {
      if (!vertexWs || vertexWs.readyState !== WebSocket.OPEN) return;
      if (!vertexReady) return;

      if (isBinary) {
        // Vertex Live expects realtime_input.media_chunks[]
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

      if (msg.type === "text" && typeof msg.text === "string") {
        // Send as client_content update
        const clientContentMsg = {
          client_content: {
            turns: [{ role: "user", parts: [{ text: msg.text }] }],
            turn_complete: true,
          },
        };
        vertexWs.send(JSON.stringify(clientContentMsg));
      }

      if (msg.type === "stop_audio") {
        // Safe: just end connection from our side
        try { vertexWs.close(); } catch {}
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
    try { clientWs.close(); } catch {}
    try { vertexWs?.close(); } catch {}
  }
});

server.listen(PORT, () => {
  console.log(`Backend listening on port ${PORT}`);
});
