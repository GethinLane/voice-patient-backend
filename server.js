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

// --- Patient scenario -------------------------------------------------
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

// --- Gemini Live WS config ------------------------------------------
const GEMINI_WS_BASE =
  "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent";

const DEFAULT_MODEL =
  process.env.GEMINI_MODEL || "gemini-1.5-flash";






const INPUT_MIME = "audio/pcm;rate=16000";
const OUTPUT_EXPECTED_RATE = 24000;

// Helpers
function toBase64(buf) {
  return Buffer.from(buf).toString("base64");
}
function safeJsonParse(s) {
  try {
    return JSON.parse(s);
  } catch {
    return null;
  }
}
function wsSendSafe(ws, obj) {
  try {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  } catch {}
}

// Create HTTP server so WS can share same port (Render)
const server = http.createServer(app);

// WS server: browser connects here
const wss = new WebSocket.Server({ server, path: "/ws" });

wss.on("connection", (clientWs, req) => {
  console.log("Browser WS connected", req?.socket?.remoteAddress);

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    wsSendSafe(clientWs, { type: "error", message: "Server missing GEMINI_API_KEY env var." });
    clientWs.close();
    return;
  }

  const geminiUrl = `${GEMINI_WS_BASE}?key=${encodeURIComponent(apiKey)}`;
  const geminiWs = new WebSocket(geminiUrl);

  let geminiReady = false;

  geminiWs.on("open", () => {
    console.log("Gemini WS open -> sending setup");

    const setupMsg = {
      setup: {
        model: DEFAULT_MODEL,
        generationConfig: {
          responseModalities: ["AUDIO"],
          temperature: 0.7,
          maxOutputTokens: 512,
        },
        systemInstruction: {
  role: "system",
  parts: [{ text: SYSTEM_INSTRUCTIONS }]
},

        inputAudioTranscription: {},
        outputAudioTranscription: {},
        realtimeInputConfig: {},
      },
    };

    try {
      geminiWs.send(JSON.stringify(setupMsg));
    } catch (e) {
      console.error("Failed to send setup:", e);
      wsSendSafe(clientWs, { type: "error", message: "Failed to send setup to Gemini." });
      try { clientWs.close(); } catch {}
    }
  });

  geminiWs.on("message", (data) => {
    const msgStr = Buffer.isBuffer(data) ? data.toString("utf8") : String(data);
    const msg = safeJsonParse(msgStr);

    // Sometimes non-JSON; forward a snippet for debugging
    if (!msg) {
      console.log("Gemini non-JSON:", msgStr.slice(0, 200));
      wsSendSafe(clientWs, { type: "debug", raw: msgStr.slice(0, 500) });
      return;
    }

    // Gemini error payloads (very important to surface)
    if (msg.error) {
      console.log("Gemini error payload:", msg.error);
      wsSendSafe(clientWs, { type: "error", message: JSON.stringify(msg.error) });
      return;
    }

    // Setup complete
    if (msg.setupComplete) {
      geminiReady = true;
      console.log("Gemini setupComplete");
      wsSendSafe(clientWs, {
        type: "ready",
        model: DEFAULT_MODEL,
        outputRate: OUTPUT_EXPECTED_RATE,
      });
      return;
    }

    // Transcripts (optional)
    if (msg.serverContent?.inputTranscription?.text) {
      wsSendSafe(clientWs, { type: "transcript", text: msg.serverContent.inputTranscription.text });
    }
    if (msg.serverContent?.outputTranscription?.text) {
      wsSendSafe(clientWs, { type: "ai_transcript", text: msg.serverContent.outputTranscription.text });
    }

    // Main content stream
    const serverContent = msg.serverContent;
    if (serverContent?.modelTurn?.parts?.length) {
      for (const part of serverContent.modelTurn.parts) {
        if (part.text) {
          wsSendSafe(clientWs, { type: "ai_text", text: part.text });
        }

        const inline = part.inlineData || part.inline_data;
        if (inline?.data) {
          wsSendSafe(clientWs, {
            type: "audio",
            mimeType: inline.mimeType || inline.mime_type || `audio/pcm;rate=${OUTPUT_EXPECTED_RATE}`,
            data: inline.data, // base64 from Gemini
          });
        }
      }
    }

    // Turn markers
    if (serverContent?.interrupted) wsSendSafe(clientWs, { type: "interrupted" });
    if (serverContent?.turnComplete) wsSendSafe(clientWs, { type: "turn_complete" });
  });

  geminiWs.on("close", (code, reasonBuf) => {
    const reason = Buffer.isBuffer(reasonBuf) ? reasonBuf.toString("utf8") : String(reasonBuf || "");
    console.log("Gemini WS closed:", { code, reason });
    wsSendSafe(clientWs, {
      type: "closed",
      message: `Gemini WS closed. code=${code} reason=${reason}`,
    });
    try { clientWs.close(); } catch {}
  });

  geminiWs.on("error", (err) => {
    console.error("Gemini WS error:", err);
    wsSendSafe(clientWs, { type: "error", message: `Gemini WS error: ${err.message}` });
    try { clientWs.close(); } catch {}
  });

  // Browser -> Server: binary PCM16 16kHz chunks, or JSON control messages
  clientWs.on("message", (payload, isBinary) => {
    if (geminiWs.readyState !== WebSocket.OPEN) return;

    // Don’t forward audio/text until setupComplete
    if (!geminiReady) return;

    if (isBinary) {
      const audioMsg = {
        realtimeInput: {
          audio: {
            mimeType: INPUT_MIME,
            data: toBase64(payload),
          },
        },
      };
      try { geminiWs.send(JSON.stringify(audioMsg)); } catch {}
      return;
    }

    const text = payload.toString("utf8");
    const msg = safeJsonParse(text);
    if (!msg) return;

    if (msg.type === "text" && typeof msg.text === "string") {
      try {
        geminiWs.send(JSON.stringify({ realtimeInput: { text: msg.text } }));
      } catch {}
    }

    if (msg.type === "stop_audio") {
      try {
        geminiWs.send(JSON.stringify({ realtimeInput: { audioStreamEnd: true } }));
      } catch {}
    }
  });

  clientWs.on("close", () => {
    console.log("Browser WS closed");
    try { geminiWs.close(); } catch {}
  });

  clientWs.on("error", (e) => {
    console.log("Browser WS error:", e?.message);
    try { geminiWs.close(); } catch {}
  });
});

server.listen(PORT, () => {
  console.log(`Backend listening on port ${PORT}`);
});
