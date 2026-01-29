// server.js
import "dotenv/config";
import fs from "fs";
import http from "http";
import express from "express";
import cors from "cors";
import { WebSocketServer } from "ws";

import { SpeechClient } from "@google-cloud/speech";
import textToSpeech from "@google-cloud/text-to-speech";
import { GoogleGenerativeAI } from "@google/generative-ai";

const PORT = process.env.PORT || 3001;

// ---- Load Google service account JSON from env ----
if (!process.env.GOOGLE_CREDENTIALS_JSON) {
  throw new Error("Missing GOOGLE_CREDENTIALS_JSON env var (paste your service account JSON).");
}
fs.writeFileSync("/tmp/google-creds.json", process.env.GOOGLE_CREDENTIALS_JSON);
process.env.GOOGLE_APPLICATION_CREDENTIALS = "/tmp/google-creds.json";

// ---- Gemini key (AI Studio key) ----
if (!process.env.GEMINI_API_KEY) {
  throw new Error("Missing GEMINI_API_KEY env var (from Google AI Studio).");
}

const app = express();
app.use(cors());
app.use(express.json());
app.get("/", (_req, res) => res.send("OK"));

const server = http.createServer(app);
const wss = new WebSocketServer({ server, path: "/ws" });

// Clients
const speechClient = new SpeechClient();
const ttsClient = new textToSpeech.TextToSpeechClient();

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
// Use a fast model. If your account doesn’t have this exact name, switch to the one you see in AI Studio.
const gemini = genAI.getGenerativeModel({ model: "gemini-2.0-flash-lite" });

// --- Shared behaviour rules (your existing prompt, reused) ---
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

function sendJSON(ws, obj) {
  if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(obj));
}

function buildPrompt(history, scenarioName) {
  const system = `${PERSONA}\n\n${CASE_DETAILS}\n\n${SHARED_BEHAVIOUR_RULES}`.trim();
  const convo = history
    .slice(-12)
    .map((h) => `${h.role.toUpperCase()}: ${h.text}`)
    .join("\n");

  return `
SYSTEM:
${system}

SCENARIO:
${scenarioName}

CONVERSATION:
${convo}

ASSISTANT:
`.trim();
}

wss.on("connection", (ws) => {
  let recognizeStream = null;
  let started = false;

  // Config defaults (can be overridden by client "start" message)
  let scenario = "anxious-chest-pain";
  let sttLanguageCode = "en-GB";
  let ttsLanguageCode = "en-GB";
  let ttsVoiceName = "en-GB-Neural2-B"; // you can swap to WaveNet/Neural2 etc.

  const history = [];

  function startSTT() {
    recognizeStream = speechClient
      .streamingRecognize({
        config: {
          encoding: "LINEAR16",
          sampleRateHertz: 16000,
          languageCode: sttLanguageCode,
          enableAutomaticPunctuation: true
        },
        interimResults: true
      })
      .on("error", (err) => {
        sendJSON(ws, { type: "error", message: "STT error: " + (err.message || err) });
      })
      .on("data", async (data) => {
        const result = data.results?.[0];
        const alt = result?.alternatives?.[0];
        const text = alt?.transcript?.trim();
        if (!text) return;

        if (!result.isFinal) {
          sendJSON(ws, { type: "partial_transcript", text });
          return;
        }

        // final transcript
        sendJSON(ws, { type: "final_transcript", text });
        history.push({ role: "user", text });

        // LLM response
        let aiText = "";
        try {
          const prompt = buildPrompt(history, scenario);
          const resp = await gemini.generateContent(prompt);
          aiText = (resp.response.text() || "").trim();
          if (!aiText) aiText = "I'm not sure.";
        } catch (e) {
          sendJSON(ws, { type: "error", message: "Gemini error: " + (e.message || e) });
          return;
        }

        history.push({ role: "assistant", text: aiText });
        sendJSON(ws, { type: "ai_text", text: aiText });

        // TTS (MP3)
        try {
          const [ttsResp] = await ttsClient.synthesizeSpeech({
            input: { text: aiText },
            voice: {
              languageCode: ttsLanguageCode,
              name: ttsVoiceName
            },
            audioConfig: {
              audioEncoding: "MP3"
            }
          });

          if (ttsResp.audioContent) {
            // binary MP3 bytes
            ws.send(ttsResp.audioContent);
          }
          sendJSON(ws, { type: "ai_done" });
        } catch (e) {
          sendJSON(ws, { type: "error", message: "TTS error: " + (e.message || e) });
        }
      });
  }

  ws.on("message", (msg, isBinary) => {
    // JSON control messages
    if (!isBinary) {
      const s = msg.toString();

      if (s.startsWith("{")) {
        const obj = JSON.parse(s);

        if (obj.type === "start") {
          started = true;

          if (obj.scenario) scenario = obj.scenario;

          if (obj.stt?.languageCode) sttLanguageCode = obj.stt.languageCode;

          if (obj.tts?.languageCode) ttsLanguageCode = obj.tts.languageCode;
          if (obj.tts?.voiceName) ttsVoiceName = obj.tts.voiceName;

          history.length = 0;

          try { recognizeStream?.end(); } catch {}
          recognizeStream = null;

          startSTT();
          sendJSON(ws, { type: "status", message: "started" });
          return;
        }

        if (obj.type === "stop") {
          started = false;
          try { recognizeStream?.end(); } catch {}
          recognizeStream = null;
          sendJSON(ws, { type: "status", message: "stopped" });
          return;
        }
      }

      return;
    }

    // Binary = PCM16 audio chunk
    if (isBinary && started && recognizeStream) {
      recognizeStream.write(msg);
    }
  });

  ws.on("close", () => {
    try { recognizeStream?.end(); } catch {}
    recognizeStream = null;
  });
});

server.listen(PORT, () => {
  console.log(`Backend listening on port ${PORT}`);
});
