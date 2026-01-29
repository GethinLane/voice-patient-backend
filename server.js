// server.js (Vertex AI Gemini Live via WebSockets + OAuth)
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

// Health check
app.get("/", (_req, res) => res.status(200).send("OK"));

const PORT = process.env.PORT || 3001;

// ===================== Your patient scenario =====================
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

const SYSTEM_INSTRUCTIONS = `${PERSONA}\n\n${CASE_DETAILS}\n\n${SHARED_BEHAVIOUR_RULES}`;

// ===================== Audio settings =====================
const INPUT_MIME = "audio/pcm;rate=16000";
const OUTPUT_EXPECTED_RATE = 24000;

// ===================== Vertex config =====================
const VERTEX_PROJECT_ID = process.env.VERTEX_PROJECT_ID;
const VERTEX_LOCATION = process.env.VERTEX_LOCATION || "us-central1";
const VERTEX_MODEL_ID =
  process.env.VERTEX_MODEL_ID || "gemini-2.0-flash-live-preview-04-09";

// Vertex Live WS endpoint (regional)
function vertexWsUrl(location) {
  // Vertex Live uses aiplatform regional host + /ws/.../BidiGenerateContent
  return `wss://${location}-aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent`;
}

function toBase64(buf) {
  return Buffer.from(buf).toString("base64");
}
function safeJsonParse(s) {
  try { return JSON.parse(s); } catch { return null; }
}

// Write service account JSON to disk (Render-friendly) if provided inline
function ensureGoogleCredsFile() {
  const raw = process.env.GOOGLE_SERVICE_ACCOUNT_JSON;
  if (!raw) return null;

  const credsPath = path.join("/tmp", "gcp-sa.json");
  if (!fs.existsSync(credsPath)) {
    fs.writeFileSync(credsPath, raw, "utf8");
  }
  process.env.GOOGLE_APPLICATION_CREDENTIALS = credsPath;
  return credsPath;
}

async function getAccessToken() {
  ensureGoogleCredsFile();

  const auth = new GoogleAuth({
    scopes: ["https://www.googleapis.com/auth/cloud-platform"],
  });
  const client = await auth.getClient();
  const tokenResponse = await client.getAccessToken();
  const token = typeof tokenResponse === "string" ? tokenResponse : tokenResponse?.token;
  if (!token) throw new Error("Failed to obtain Google OAuth access token.");
  return token;
}

// Debug endpoint so you can confirm env vars are set
app.get("/vertex", (_req, res) => {
  res.json({
    project: !!VERTEX_PROJECT_ID ? VERTEX_PROJECT_ID : null,
    location: VERTEX_LOCATION,
    modelId: VERTEX_MODEL_ID,
    hasServiceAccountJson: !!process.env.GOOGLE_SERVICE_ACCOUNT_JSON,
  });
});

const server = http.createServer(app);
const wss = new WebSocket.Server({ server, path: "/ws" });

wss.on("connection", async (clientWs) => {
  try {
    if (!VERTEX_PROJECT_ID) {
      clientWs.send(JSON.stringify({ type: "error", message: "Missing VERTEX_PROJECT_ID env var." }));
      clientWs.close();
      return;
    }

    const accessToken = await getAccessToken();
    const url = vertexWsUrl(VERTEX_LOCATION);

    // Connect to Vertex Live with Authorization header
    const geminiWs = new WebSocket(url, {
      headers: {
        Authorization: `Bearer ${accessToken}`,
        // This can help in some org setups (safe to include):
        "x-goog-user-project": VERTEX_PROJECT_ID,
      },
    });

    let geminiReady = false;

    geminiWs.on("open", () => {
      // IMPORTANT: system_instruction must be a Content object (parts[]), not a string.
      const setupMsg = {
        setup: {
          model: `projects/${VERTEX_PROJECT_ID}/locations/${VERTEX_LOCATION}/publishers/google/models/${VERTEX_MODEL_ID}`,
          generation_config: {
            response_modalities: ["AUDIO"],
            temperature: 0.7,
            max_output_tokens: 512,
          },
          system_instruction: {
            parts: [{ text: SYSTEM_INSTRUCTIONS }],
          },
          input_audio_transcription: {},
          output_audio_transcription: {},
        },
      };

      geminiWs.send(JSON.stringify(setupMsg));
    });

    geminiWs.on("message", (data) => {
      const msgStr = Buffer.isBuffer(data) ? data.toString("utf8") : String(data);
      const msg = safeJsonParse(msgStr);

      if (!msg) {
        clientWs.send(JSON.stringify({ type: "debug", raw: msgStr.slice(0, 500) }));
        return;
      }

      if (msg.setupComplete) {
        geminiReady = true;
        clientWs.send(JSON.stringify({
          type: "ready",
          model: `projects/${VERTEX_PROJECT_ID}/locations/${VERTEX_LOCATION}/publishers/google/models/${VERTEX_MODEL_ID}`,
          outputRate: OUTPUT_EXPECTED_RATE,
        }));
        return;
      }

      // Transcripts (optional)
      const inputTx = msg.serverContent?.inputTranscription?.text || msg.server_content?.input_transcription?.text;
      if (inputTx) clientWs.send(JSON.stringify({ type: "transcript", text: inputTx }));

      const outputTx = msg.serverContent?.outputTranscription?.text || msg.server_content?.output_transcription?.text;
      if (outputTx) clientWs.send(JSON.stringify({ type: "ai_transcript", text: outputTx }));

      // Main streamed content
      const serverContent = msg.serverContent || msg.server_content;
      const modelTurn = serverContent?.modelTurn || serverContent?.model_turn;

      const parts = modelTurn?.parts || [];
      for (const part of parts) {
        if (part.text) {
          clientWs.send(JSON.stringify({ type: "ai_text", text: part.text }));
        }

        const inline = part.inlineData || part.inline_data;
        if (inline?.data) {
          clientWs.send(JSON.stringify({
            type: "audio",
            mimeType: inline.mimeType || inline.mime_type || `audio/pcm;rate=${OUTPUT_EXPECTED_RATE}`,
            data: inline.data,
          }));
        }
      }

      if (serverContent?.interrupted) {
        clientWs.send(JSON.stringify({ type: "interrupted" }));
      }
      if (serverContent?.turnComplete || serverContent?.turn_complete) {
        clientWs.send(JSON.stringify({ type: "turn_complete" }));
      }
    });

    geminiWs.on("close", (e) => {
      clientWs.send(JSON.stringify({ type: "closed", message: `Vertex WS closed (${e?.code || ""}) ${e?.reason || ""}` }));
      clientWs.close();
    });

    geminiWs.on("error", (err) => {
      clientWs.send(JSON.stringify({ type: "error", message: `Vertex WS error: ${err.message}` }));
      clientWs.close();
    });

    clientWs.on("message", (payload, isBinary) => {
      if (geminiWs.readyState !== WebSocket.OPEN) return;
      if (!geminiReady) return;

      if (isBinary) {
        // realtime_input/media_chunks is the Vertex Live schema
        const audioMsg = {
          realtime_input: {
            media_chunks: [{
              mime_type: INPUT_MIME,
              data: toBase64(payload),
            }],
          },
        };
        geminiWs.send(JSON.stringify(audioMsg));
        return;
      }

      const text = payload.toString("utf8");
      const msg = safeJsonParse(text);
      if (!msg) return;

      if (msg.type === "text" && typeof msg.text === "string") {
        geminiWs.send(JSON.stringify({
          client_content: {
            turns: [{ role: "user", parts: [{ text: msg.text }] }],
            turn_complete: true,
          },
        }));
      }

      if (msg.type === "stop_audio") {
        geminiWs.send(JSON.stringify({
          realtime_input: { audio_stream_end: true },
        }));
      }
    });

    clientWs.on("close", () => {
      try { geminiWs.close(); } catch {}
    });
    clientWs.on("error", () => {
      try { geminiWs.close(); } catch {}
    });

  } catch (err) {
    clientWs.send(JSON.stringify({ type: "error", message: err.message || String(err) }));
    clientWs.close();
  }
});

server.listen(PORT, () => {
  console.log(`Backend listening on port ${PORT}`);
});
