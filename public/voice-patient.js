// voice-patient.js
(() => {
  const backendUrl = "https://voice-patient-backend.onrender.com";

  function $(id) { return document.getElementById(id); }

  function log(message) {
    console.log(message);
    const logEl = $("log");
    if (!logEl) return;
    logEl.value += message + "\n";
    logEl.scrollTop = logEl.scrollHeight;
  }

  function safeSetAIState(mode) {
    if (typeof window.setAIState === "function") window.setAIState(mode);
  }

  // -------------------- Case dropdown --------------------
  async function populateCaseDropdown() {
    const sel = $("caseSelect");
    if (!sel) return;

    sel.innerHTML = `<option>Loading cases…</option>`;

    try {
      const resp = await fetch(`${backendUrl}/cases`, { cache: "no-store" });
      const data = await resp.json();
      if (!data.ok) throw new Error(data.error || "Failed to load cases");

      sel.innerHTML = "";
      for (const n of data.cases) {
        const opt = document.createElement("option");
        opt.value = String(n);
        opt.textContent = `Case ${n}`;
        sel.appendChild(opt);
      }

      if (data.cases.length) sel.value = String(data.cases[data.cases.length - 1]);
      log(`[CASES] loaded ${data.cases.length} cases`);

    } catch (err) {
      sel.innerHTML = `<option>Error loading cases</option>`;
      log("[CASES] error: " + (err?.message || String(err)));
    }
  }

  // -------------------- Audio helpers --------------------
  function base64ToUint8Array(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes;
  }

  function int16ToFloat32(int16) {
    const f32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
      f32[i] = Math.max(-1, Math.min(1, int16[i] / 32768));
    }
    return f32;
  }

  // Downsample Float32 -> 16kHz PCM16
  function downsampleTo16kPCM16(float32, inRate) {
    const outRate = 16000;
    if (outRate === inRate) {
      const out = new Int16Array(float32.length);
      for (let i = 0; i < float32.length; i++) {
        out[i] = Math.max(-1, Math.min(1, float32[i])) * 32767;
      }
      return out;
    }

    const ratio = inRate / outRate;
    const outLen = Math.round(float32.length / ratio);
    const out = new Int16Array(outLen);

    let offset = 0;
    for (let i = 0; i < outLen; i++) {
      const nextOffset = Math.round((i + 1) * ratio);
      let sum = 0;
      let count = 0;
      for (let j = offset; j < nextOffset && j < float32.length; j++) {
        sum += float32[j];
        count++;
      }
      const sample = count ? sum / count : 0;
      out[i] = Math.max(-1, Math.min(1, sample)) * 32767;
      offset = nextOffset;
    }
    return out;
  }

  // -------------------- Playback --------------------
  let ws = null;
  let audioCtx = null;
  let micStream = null;
  let processor = null;

  const playbackQueue = [];
  let playing = false;
  let outputSampleRate = 24000;

  async function playPcmChunk(pcmBytesB64, sampleRate) {
    if (!audioCtx) return;

    const bytes = base64ToUint8Array(pcmBytesB64);
    const i16 = new Int16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
    const f32 = int16ToFloat32(i16);

    const buffer = audioCtx.createBuffer(1, f32.length, sampleRate);
    buffer.copyToChannel(f32, 0);

    playbackQueue.push(buffer);
    if (!playing) {
      playing = true;
      while (playbackQueue.length) {
        const buf = playbackQueue.shift();
        const src = audioCtx.createBufferSource();
        src.buffer = buf;
        src.connect(audioCtx.destination);

        safeSetAIState("talking");
        await new Promise((resolve) => {
          src.onended = resolve;
          src.start();
        });
      }
      safeSetAIState("listening");
      playing = false;
    }
  }

  // -------------------- UI helpers --------------------
  function setUiConnected(connected) {
    const startBtn = $("startBtn");
    const stopBtn = $("stopBtn");
    if (startBtn) startBtn.disabled = connected;
    if (stopBtn) stopBtn.disabled = !connected;
  }

  function setStatus(text) {
    const statusEl = $("status");
    if (statusEl) statusEl.textContent = text;
  }

  function cleanup() {
    try {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "stop_audio" }));
        ws.close();
      }
    } catch {}

    ws = null;

    if (processor) {
      try { processor.disconnect(); } catch {}
      processor = null;
    }

    if (micStream) {
      micStream.getTracks().forEach((t) => t.stop());
      micStream = null;
    }

    if (audioCtx) {
      try { audioCtx.close(); } catch {}
      audioCtx = null;
    }

    playbackQueue.length = 0;
    safeSetAIState("listening");
    setUiConnected(false);
    setStatus("Stopped.");
  }

  // -------------------- Consultation --------------------
  async function startConsultation() {
    try {
      setUiConnected(true);
      setStatus("Connecting...");
      safeSetAIState("listening");

      audioCtx = new (window.AudioContext || window.webkitAudioContext)();

      const wsUrl = backendUrl.replace("https://", "wss://").replace("http://", "ws://") + "/ws";
      ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";

      ws.onopen = async () => {
        log("[WS] connected");
        setStatus("Requesting microphone...");

        // Send init (case selection) FIRST
        const sel = $("caseSelect");
        const caseId = Number(sel?.value) || 1;
        ws.send(JSON.stringify({ type: "init", caseId }));
        log(`[INIT] sent caseId=${caseId}`);

        // Request mic with suppression
        micStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });

        const source = audioCtx.createMediaStreamSource(micStream);
        processor = audioCtx.createScriptProcessor(2048, 1, 1);

        // ---- Speech gate + DEBUG RMS meter ----
        const RMS_THRESHOLD = 0.008;  // start low; raise if noise triggers it
        const SILENCE_HOLD_MS = 350;

        let isStreamingSpeech = false;
        let silenceMs = 0;
        let meterMs = 0;

        processor.onaudioprocess = (e) => {
          if (!ws || ws.readyState !== WebSocket.OPEN) return;

          const input = e.inputBuffer.getChannelData(0);

          // RMS energy
          let sum = 0;
          for (let i = 0; i < input.length; i++) sum += input[i] * input[i];
          const rms = Math.sqrt(sum / input.length);

          const frameMs = (input.length / audioCtx.sampleRate) * 1000;

          // Debug meter once per second
          meterMs += frameMs;
          if (meterMs >= 1000) {
            meterMs = 0;
            log(`[RMS] ${rms.toFixed(4)} streaming=${isStreamingSpeech}`);
          }

          const isLoudEnough = rms >= RMS_THRESHOLD;

          if (isLoudEnough) {
            isStreamingSpeech = true;
            silenceMs = 0;

            const pcm16 = downsampleTo16kPCM16(input, audioCtx.sampleRate);
            ws.send(pcm16.buffer);
            return;
          }

          if (!isStreamingSpeech) return;

          silenceMs += frameMs;

          if (silenceMs < SILENCE_HOLD_MS) {
            const pcm16 = downsampleTo16kPCM16(input, audioCtx.sampleRate);
            ws.send(pcm16.buffer);
            return;
          }

          isStreamingSpeech = false;
          silenceMs = 0;
        };

        source.connect(processor);
        processor.connect(audioCtx.destination);

        setStatus("Connected. Start talking!");
        safeSetAIState("listening");
      };

      ws.onmessage = async (evt) => {
        let msg;
        try { msg = JSON.parse(evt.data); } catch { return; }

        if (msg.type === "ready") {
          log(`[READY] model=${msg.model}`);
          if (msg.caseTable || msg.caseId) log(`[READY] loaded ${msg.caseTable || ("Case " + msg.caseId)}`);
          if (msg.outputRate) outputSampleRate = msg.outputRate;
          return;
        }

        if (msg.type === "diag") {
          log(`[DIAG] ${msg.message}`);
          return;
        }

        if (msg.type === "transcript") log("[You] " + msg.text);
        if (msg.type === "ai_text") log("[AI] " + msg.text);

        if (msg.type === "audio") {
          await playPcmChunk(msg.data, outputSampleRate);
        }

        if (msg.type === "interrupted") {
          playbackQueue.length = 0;
          safeSetAIState("listening");
        }

        if (msg.type === "error") {
          log("[ERROR] " + msg.message);
          setStatus("Error: " + msg.message);
          cleanup();
        }

        if (msg.type === "closed") {
          log("[CLOSED] " + msg.message);
          cleanup();
        }
      };

      ws.onclose = () => {
        log("[WS] closed");
        cleanup();
      };

      ws.onerror = () => {
        log("[WS] error");
        cleanup();
      };
    } catch (err) {
      log("[ERROR] " + (err?.message || String(err)));
      setStatus("Error: " + (err?.message || String(err)));
      cleanup();
    }
  }

  function stopConsultation() {
    log("Stopping consultation.");
    cleanup();
  }

  // -------------------- Init --------------------
  window.addEventListener("DOMContentLoaded", () => {
    const startBtn = $("startBtn");
    const stopBtn = $("stopBtn");

    if (!startBtn || !stopBtn) {
      log("UI ERROR: startBtn/stopBtn not found. Check element IDs in HTML.");
      return;
    }

    startBtn.addEventListener("click", startConsultation);
    stopBtn.addEventListener("click", stopConsultation);

    populateCaseDropdown();

    setUiConnected(false);
    setStatus("Not connected");
  });
})();
