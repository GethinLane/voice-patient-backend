// server.js — Airtable Case Service (list cases + build system prompt)
// npm i express cors dotenv
require("dotenv").config();

const express = require("express");
const cors = require("cors");

const app = express();
app.use(cors());
app.use(express.json({ limit: "1mb" }));

const PORT = process.env.PORT || 3001;

const AIRTABLE_API_KEY = process.env.AIRTABLE_API_KEY;
const AIRTABLE_BASE_ID = process.env.AIRTABLE_BASE_ID;

function assertAirtableEnv() {
  if (!AIRTABLE_API_KEY || !AIRTABLE_BASE_ID) {
    throw new Error("Missing AIRTABLE_API_KEY or AIRTABLE_BASE_ID in environment.");
  }
}

function safeJsonParse(s) {
  try { return JSON.parse(s); } catch { return null; }
}

async function airtableFetch(url) {
  assertAirtableEnv();

  const res = await fetch(url, {
    headers: { Authorization: `Bearer ${AIRTABLE_API_KEY}` },
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

async function fetchAllCaseRows(caseNumber) {
  const tableName = `Case ${caseNumber}`;
  let offset = null;
  const records = [];

  do {
    const params = new URLSearchParams();
    params.set("pageSize", "100");
    if (offset) params.set("offset", offset);

    const url =
      `https://api.airtable.com/v0/${AIRTABLE_BASE_ID}/` +
      `${encodeURIComponent(tableName)}?${params.toString()}`;

    const json = await airtableFetch(url);
    records.push(...(json.records || []));
    offset = json.offset || null;
  } while (offset);

  if (!records.length) throw new Error(`No records found in "${tableName}".`);
  return { tableName, records };
}

function combineFieldAcrossRows(records, fieldName) {
  const parts = [];
  for (const r of records) {
    const v = r?.fields?.[fieldName];
    if (v == null) continue;
    const t = (typeof v === "string" ? v : String(v)).trim();
    if (t) parts.push(t);
  }
  return parts.join("\n\n");
}

function buildSystemTextFromCase(records) {
  const opening = combineFieldAcrossRows(records, "Opening Sentence");
  const divulgeFreely = combineFieldAcrossRows(records, "Divulge freely");
  const divulgeAsked = combineFieldAcrossRows(records, "Divulge Asked");
  const pmhx = combineFieldAcrossRows(records, "PMHx RP");
  const social = combineFieldAcrossRows(records, "Social History");
  const family =
    combineFieldAcrossRows(records, "Family Hiostory") ||
    combineFieldAcrossRows(records, "Family History");
  const ice = combineFieldAcrossRows(records, "ICE");
  const reaction = combineFieldAcrossRows(records, "Reaction");

  const RULES = `
CRITICAL:
- You MUST NOT invent details.
- Only use information explicitly present in the CASE DETAILS below.
- If something is not stated, say: "I'm not sure" / "I don't remember" / "I haven't been told".
- NEVER substitute another symptom.
- NEVER create symptoms.
- Do Not Hallucinate.
- NEVER swap relatives. If relationship is not explicit, say you're not sure.
- Answer only what the clinician asks.
`.trim();

  const CASE = `
CASE DETAILS (THIS IS YOUR ENTIRE MEMORY):

OPENING SENTENCE:
${opening || "[Not provided]"}

DIVULGE FREELY:
${divulgeFreely || "[Not provided]"}

DIVULGE ONLY IF ASKED:
${divulgeAsked || "[Not provided]"}

PAST MEDICAL HISTORY:
${pmhx || "[Not provided]"}

SOCIAL HISTORY:
${social || "[Not provided]"}

FAMILY HISTORY:
${family || "[Not provided]"}

ICE (Ideas / Concerns / Expectations):
${ice || "[Not provided]"}

REACTION / AFFECT:
${reaction || "[Not provided]"}
`.trim();

  return `${CASE}\n\n${RULES}`;
}

// health
app.get("/", (_req, res) => res.status(200).send("OK"));

// list cases
app.get("/cases", async (_req, res) => {
  try {
    const cases = await listCaseNumbersFromMeta();
    res.json({ ok: true, cases });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message || String(e) });
  }
});

// fetch system prompt for a case
app.get("/case/:id/system", async (req, res) => {
  try {
    const caseId = Number(req.params.id);
    if (!Number.isFinite(caseId) || caseId <= 0) {
      return res.status(400).json({ ok: false, error: "Invalid case id" });
    }

    const { tableName, records } = await fetchAllCaseRows(caseId);
    const system = buildSystemTextFromCase(records);

    res.json({ ok: true, caseId, tableName, system });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message || String(e) });
  }
});

app.listen(PORT, () => console.log(`Case service listening on ${PORT}`));
