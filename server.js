// server.js
require("dotenv").config();
const express = require("express");
const cors = require("cors");
const http = require("http");
const WebSocket = require("ws");

const app = express();
app.use(cors());
app.use(express.json());

// Health check so Render "/" shows OK
app.get("/", (_req, res) => res.status(200).send("OK"));

const PORT = process.env.PORT || 3001;

// --- Your patient scenario (same content you had) ----------------
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

// --- Gemini Live API connection details --------------------------
// Official endpoint (v1beta) from docs :contentReference[oaicite:1]{index=1}
const GEMINI_WS_BASE =
  "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent";

// Model example from Live docs :contentReference[oaicite:2]{index=2}
const DEFAULT_MODEL = process.env.GEMINI_MODEL || "gemini-2.5-flash-native-audio-preview-12-2025";

// Input audio should be PCM16 @ 16kHz; output audio is PCM @ 24kHz per docs :contentReference[oaicite:3]{index=3}
const INPUT_MIME = "audio/pcm;rate=16000";
const OUTPUT_EXPECTED_RATE = 24000;

// Helpers
function toBase64(buf) {
  return Buffer.from(buf).toString("base64");
}

function safeJsonParse(s) {
  try { return JSON.parse(s); } catch { return null; }
}

// Create HTTP server so WS can share same port (Render)
const server = http.createServer(app);

// WS server: browser connects here
const wss = new WebSocket.Server({ server, path: "/ws" });

wss.on("connection", (clientWs) => {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    clientWs.send(JSON.stringify({ type: "error", message: "Server missing GEMINI_API_KEY env var." }));
    clientWs.close();
    return;
  }

  // Auth: In practice many clients pass the key as a query param for the Live WS endpoint.
  // (If Google changes this, you may need ephemeral tokens later.) :contentReference[oaicite:4]{index=4}
  const geminiUrl = `${GEMINI_WS_BASE}?key=${encodeURIComponent(apiKey)}`;

  const geminiWs = new WebSocket(geminiUrl);

  let geminiReady = false;

  geminiWs.on("open", () => {
    // First message must be setup/config :contentReference[oaicite:5]{index=5}
    const setupMsg = {
      setup: {
        model: DEFAULT_MODEL,
        generationConfig: {
          responseModalities: ["AUDIO"],
          // Optional knobs:
          temperature: 0.7,
          maxOutputTokens: 512
          // speechConfig can be added here if you want voice/language settings
        },
        systemInstruction: SYSTEM_INSTRUCTIONS,
        // Optional: transcripts from input/output audio :contentReference[oaicite:6]{index=6}
        inputAudioTranscription: {},
        outputAudioTranscription: {},
        realtimeInputConfig: {
          // Leave defaults; Live API supports activity detection/barge-in :contentReference[oaicite:7]{index=7}
        }
      }
    };

    geminiWs.send(JSON.stringify(setupMsg));
  });

  geminiWs.on("message", (data) => {
    const msgStr = Buffer.isBuffer(data) ? data.toString("utf8") : String(data);
    const msg = safeJsonParse(msgStr);

    // Some servers send non-JSON (rare) — forward as debug
    if (!msg) {
      clientWs.send(JSON.stringify({ type: "debug", raw: msgStr.slice(0, 500) }));
      return;
    }

    // Setup complete
    if (msg.setupComplete) {
      geminiReady = true;
      clientWs.send(JSON.stringify({ type: "ready", model: DEFAULT_MODEL, outputRate: OUTPUT_EXPECTED_RATE }));
      return;
    }

    // Transcripts (optional)
    if (msg.serverContent?.inputTranscription?.text) {
      clientWs.send(JSON.stringify({ type: "transcript", text: msg.serverContent.inputTranscription.text }));
    }
    if (msg.serverContent?.outputTranscription?.text) {
      clientWs.send(JSON.stringify({ type: "ai_transcript", text: msg.serverContent.outputTranscription.text }));
    }

    // Main content stream
    const serverContent = msg.serverContent;
    if (serverContent?.modelTurn?.parts?.length) {
      for (const part of serverContent.modelTurn.parts) {
        // Text part
        if (part.text) {
          clientWs.send(JSON.stringify({ type: "ai_text", text: part.text }));
        }

        // Audio part (inlineData base64 PCM) :contentReference[oaicite:8]{index=8}
        const inline = part.inlineData || part.inline_data;
        if (inline?.data) {
          clientWs.send(JSON.stringify({
            type: "audio",
            mimeType: inline.mimeType || inline.mime_type || `audio/pcm;rate=${OUTPUT_EXPECTED_RATE}`,
            data: inline.data // already base64 from Gemini
          }));
        }
      }
    }

    // Turn markers
    if (serverContent?.interrupted) {
      clientWs.send(JSON.stringify({ type: "interrupted" }));
    }
    if (serverContent?.turnComplete) {
      clientWs.send(JSON.stringify({ type: "turn_complete" }));
    }
  });

  geminiWs.on("close", (e) => {
    clientWs.send(JSON.stringify({ type: "closed", message: `Gemini WS closed (${e})` }));
    clientWs.close();
  });

  geminiWs.on("error", (err) => {
    clientWs.send(JSON.stringify({ type: "error", message: `Gemini WS error: ${err.message}` }));
    clientWs.close();
  });

  // Browser -> Server: binary PCM16 16kHz chunks, or JSON control messages
  clientWs.on("message", (payload, isBinary) => {
    if (geminiWs.readyState !== WebSocket.OPEN) return;

    if (!geminiReady) {
      // allow buffering? simplest: ignore until setupComplete
      return;
    }

    if (isBinary) {
      // Forward audio chunk as realtimeInput.audio blob
      const audioMsg = {
        realtimeInput: {
          audio: {
            mimeType: INPUT_MIME,
            data: toBase64(payload)
          }
        }
      };
      geminiWs.send(JSON.stringify(audioMsg));
      return;
    }

    // Text JSON messages from client (optional)
    const text = payload.toString("utf8");
    const msg = safeJsonParse(text);
    if (!msg) return;

    if (msg.type === "text" && typeof msg.text === "string") {
      geminiWs.send(JSON.stringify({
        realtimeInput: { text: msg.text }
      }));
    }

    if (msg.type === "stop_audio") {
      // Notify Gemini audio stream ended :contentReference[oaicite:9]{index=9}
      geminiWs.send(JSON.stringify({ realtimeInput: { audioStreamEnd: true } }));
    }
  });

  clientWs.on("close", () => {
    try { geminiWs.close(); } catch {}
  });

  clientWs.on("error", () => {
    try { geminiWs.close(); } catch {}
  });
});

server.listen(PORT, () => {
  console.log(`Backend listening on port ${PORT}`);
});
