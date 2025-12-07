// server.js
require("dotenv").config();
const express = require("express");
const cors = require("cors");

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

// POST /api/client-secret
// Returns a short-lived client secret for the Realtime API
app.post("/api/client-secret", async (req, res) => {
  try {
    const scenario = req.body?.scenario || "default";

    // Different patient scenarios, if you want
    let instructions;
    if (scenario === "anxious-chest-pain") {
      instructions = `
You are a 42-year-old patient called Sam with mild chest discomfort.
You are anxious but polite and cooperative.
Answer like a real patient: describe symptoms, history, and feelings.
Do NOT give medical advice. Only talk about your own experience.
Speak in short, natural spoken sentences.
      `.trim();
    } else {
      instructions = `
You are a cooperative patient in a medical consultation.
Answer like a real patient, not a doctor.
Use short, conversational spoken sentences.
Do NOT explain medical theory, only how you feel and what you notice.
      `.trim();
    }

    // Call OpenAI to create a Realtime client secret
    const response = await fetch("https://api.openai.com/v1/realtime/client_secrets", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${process.env.OPENAI_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        expires_after: {
          anchor: "created_at",
          seconds: 600 // token valid for 10 minutes
        },
        session: {
          type: "realtime",
          model: "gpt-realtime-mini", // or "gpt-realtime" if you want the bigger one
          instructions,
          output_modalities: ["audio"],
          audio: {
            input: {
              format: {
                type: "audio/pcm",
                rate: 24000
              },
              turn_detection: {
                type: "server_vad"
              }
            },
            output: {
              format: {
                type: "audio/pcm",
                rate: 24000
              },
              voice: "alloy",
              speed: 1.0
            }
          }
        }
      }),
    });

    if (!response.ok) {
      const text = await response.text();
      console.error("Error from OpenAI:", text);
      return res.status(500).json({ error: "OpenAI client_secret failed", details: text });
    }

    const data = await response.json();
    // data.value is the ephemeral token (starts with ek_...)
    res.json({ clientSecret: data.value });
  } catch (err) {
    console.error("Server error:", err);
    res.status(500).json({ error: "Server error" });
  }
});

app.listen(PORT, () => {
  console.log(`Backend listening on port ${PORT}`);
});