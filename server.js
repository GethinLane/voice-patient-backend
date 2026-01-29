// server.js
require("dotenv").config();
const express = require("express");
const cors = require("cors");
const http = require("http");
const WebSocket = require("ws");

const app = express();
app.use(cors());
app.use(express.json());

app.get("/", (_req, res) => res.status(200).send("OK"));

const PORT = process.env.PORT || 3001;

// ---------------- Patient scenario ----------------
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
   - You have a fixed set of case details provided in these instructions. Treat these as your entire memory.
   - You MUST NOT invent or guess new medical facts beyond what is written.
   - If asked for info NOT specified, say "I'm not sure" / "I don't remember" / "I haven't been told that."

6. If you are unsure:
   - If unsure whether something is in the case details, assume it is NOT and say you are not sure.
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
  - No known high blood pressure, no known high cholesterol.

- Medications:
  - Salbutamol inhaler as needed.

- Allergies:
  - None known.

- Social history:
  - Non-smoker.
  - Drinks alcohol socially, about 6 units per week.
  - Desk job, generally sedentary.
  - Lives with partner.

- Family history:
  - Father had a heart attack in his late 60s.
  - No known sudden cardiac deaths in younger relatives.
`.trim();

const SYSTEM_INSTRUCTIONS = `${PERSONA}\n\n${CASE_DETAILS}\n\n${SHARED_BEHAVIOUR_RULES}`;

// ---------------- Gemini Live WS ----------------
// Live WS endpoint (v1beta) :contentReference[oaicite:3]{index=3}
const GEMINI_WS_BASE =
  "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent";

const INPUT_MIME = "audio/pcm;rate=16000";
const OUTPUT_EXPECTED_RATE = 24000;

function toBase64(buf) {
  return Buffer.from(buf).toString("base64");
}
function safeJsonParse(s) {
  try { return JSON.parse(s); } catch { return null; }
}
function wsSendSafe(ws, obj) {
  try {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  } catch {}
}

// ---- Model discovery (no curl needed) ----
// Uses Models API models.list :contentReference[oaicite:4]{index=4}
async function listModels(apiKey) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models?key=${encodeURIComponent(apiKey)}`;
  const res = await fetch(url);
  const json = await res.json();
  if (!res.ok) {
    throw new Error(`models.list failed: HTTP ${res.status} ${JSON.stringify(json)}`);
  }
  return json.models || [];
}

function pickLiveModel(models) {
  // supportedGenerationMethods includes method names like "generateContent"
  // Live API method name is "BidiGenerateContent" (Pascal case per docs) :contentReference[oaicite:5]{index=5}
  const live = models.filter(m =>
    Array.isArray(m.supportedGenerationMethods) &&
    m.supportedGenerationMethods.includes("BidiGenerateContent")
  );

  // Prefer native audio model if present; otherwise first live-capable model.
  const preferred = [
    "models/gemini-2.5-flash-native-audio-preview-12-2025",
    "models/gemini-2.0-flash-live-001",
  ];

  for (const want of preferred) {
    const hit = live.find(m => m.name === want);
    if (hit) return hit.name;
  }

  return live[0]?.name || null;
}

let cached = { at: 0, models: [], chosen: null, error: null };
async function refreshModelCache(apiKey) {
  const now = Date.now();
  if (cached.chosen && now - cached.at < 10 * 60 * 1000) return cached; // 10 min cache

  try {
    const models = await listModels(apiKey);
    const chosen = pickLiveModel(models);
    cached = { at: now, models, chosen, error: null };
  } catch (e) {
    cached = { at: now, models: [], chosen: null, error: e.message };
  }
  return cached;
}

// Debug endpoint to see what your key supports
app.get("/models", async (_req, res) => {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) return res.status(500).json({ error: "Missing GEMINI_API_KEY" });

  const data = await refreshModelCache(apiKey);
  const live = data.models
    .filter(m => (m.supportedGenerationMethods || []).includes("BidiGenerateContent"))
    .map(m => ({ name: m.name, baseModelId: m.baseModelId, methods: m.supportedGenerationMethods }));

  res.json({
    chosenLiveModel: data.chosen,
    error: data.error,
    liveModels: live,
  });
});

// HTTP server so WS shares same port (Render)
const server = http.createServer(app);
const wss = new WebSocket.Server({ server, path: "/ws" });

wss.on("connection", async (clientWs) => {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    wsSendSafe(clientWs, { type: "error", message: "Server missing GEMINI_API_KEY env var." });
    clientWs.close();
    return;
  }

  // If you set GEMINI_MODEL explicitly, we'll try it, but the cache tells you what's valid.
  const data = await refreshModelCache(apiKey);

  const envModel = process.env.GEMINI_MODEL && process.env.GEMINI_MODEL.trim();
  const modelToUse = envModel || data.chosen;

  console.log("BOOT/CONN modelToUse =", modelToUse, "envModel =", envModel, "cacheChosen =", data.chosen, "cacheError =", data.error);

  if (!modelToUse) {
    wsSendSafe(clientWs, {
      type: "error",
      message:
        "No Live-capable model found for this API key. Open /models on your backend to see what models support BidiGenerateContent.",
    });
    clientWs.close();
    return;
  }

  const geminiUrl = `${GEMINI_WS_BASE}?key=${encodeURIComponent(apiKey)}`;
  const geminiWs = new WebSocket(geminiUrl);

  let geminiReady = false;

  geminiWs.on("open", () => {
    console.log("Gemini WS open -> sending setup with model:", modelToUse);

    const setupMsg = {
      setup: {
        model: modelToUse,
        generationConfig: {
          responseModalities: ["AUDIO"],
          temperature: 0.7,
          maxOutputTokens: 512,
          // speechConfig can be added later once audio is flowing reliably
        },

        // Keep this Content form because it matched the validator you hit earlier
        systemInstruction: {
          role: "system",
          parts: [{ text: SYSTEM_INSTRUCTIONS }],
        },

        inputAudioTranscription: {},
        outputAudioTranscription: {},
        realtimeInputConfig: {},
      },
    };

    geminiWs.send(JSON.stringify(setupMsg));
  });

  geminiWs.on("message", (data) => {
    const msgStr = Buffer.isBuffer(data) ? data.toString("utf8") : String(data);
    const msg = safeJsonParse(msgStr);

    if (!msg) {
      wsSendSafe(clientWs, { type: "debug", raw: msgStr.slice(0, 500) });
      return;
    }

    if (msg.error) {
      wsSendSafe(clientWs, { type: "error", message: JSON.stringify(msg.error) });
      return;
    }

    if (msg.setupComplete) {
      geminiReady = true;
      wsSendSafe(clientWs, { type: "ready", model: modelToUse, outputRate: OUTPUT_EXPECTED_RATE });
      return;
    }

    const serverContent = msg.serverContent;

    if (serverContent?.inputTranscription?.text) {
      wsSendSafe(clientWs, { type: "transcript", text: serverContent.inputTranscription.text });
    }
    if (serverContent?.outputTranscription?.text) {
      wsSendSafe(clientWs, { type: "ai_transcript", text: serverContent.outputTranscription.text });
    }

    if (serverContent?.modelTurn?.parts?.length) {
      for (const part of serverContent.modelTurn.parts) {
        if (part.text) wsSendSafe(clientWs, { type: "ai_text", text: part.text });

        const inline = part.inlineData || part.inline_data;
        if (inline?.data) {
          wsSendSafe(clientWs, {
            type: "audio",
            mimeType: inline.mimeType || inline.mime_type || `audio/pcm;rate=${OUTPUT_EXPECTED_RATE}`,
            data: inline.data,
          });
        }
      }
    }

    if (serverContent?.interrupted) wsSendSafe(clientWs, { type: "interrupted" });
    if (serverContent?.turnComplete) wsSendSafe(clientWs, { type: "turn_complete" });
  });

  geminiWs.on("close", (code, reasonBuf) => {
    const reason = Buffer.isBuffer(reasonBuf) ? reasonBuf.toString("utf8") : String(reasonBuf || "");
    wsSendSafe(clientWs, { type: "closed", message: `Gemini WS closed. code=${code} reason=${reason}` });
    try { clientWs.close(); } catch {}
  });

  geminiWs.on("error", (err) => {
    wsSendSafe(clientWs, { type: "error", message: `Gemini WS error: ${err.message}` });
    try { clientWs.close(); } catch {}
  });

  clientWs.on("message", (payload, isBinary) => {
    if (geminiWs.readyState !== WebSocket.OPEN) return;
    if (!geminiReady) return;

    if (isBinary) {
      geminiWs.send(JSON.stringify({
        realtimeInput: {
          audio: { mimeType: INPUT_MIME, data: toBase64(payload) },
        },
      }));
      return;
    }

    const text = payload.toString("utf8");
    const msg = safeJsonParse(text);
    if (!msg) return;

    if (msg.type === "text" && typeof msg.text === "string") {
      geminiWs.send(JSON.stringify({ realtimeInput: { text: msg.text } }));
    }
    if (msg.type === "stop_audio") {
      geminiWs.send(JSON.stringify({ realtimeInput: { audioStreamEnd: true } }));
    }
  });

  clientWs.on("close", () => { try { geminiWs.close(); } catch {} });
  clientWs.on("error", () => { try { geminiWs.close(); } catch {} });
});

server.listen(PORT, () => console.log(`Backend listening on port ${PORT}`));
