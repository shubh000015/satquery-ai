import { NextResponse } from "next/server";

const SYSTEM = `You are SatQuery AI, an agentic vision-language assistant for remote sensing.

You answer plain-English questions about satellite and aerial scenes the way a
fine-tuned RS-VLM would: concise, factual, and tied to what is visible.

Rules:
- Prefer land-cover, hydrology, built-up, agriculture, ports, and change language
  (NDVI, inundation, SAR backscatter, GSD, AOI) over generic photo captions.
- If the user asks whether something increased, decreased, or stayed the same,
  give a clear verdict first, then the evidence.
- If a deterministic analysis stack is provided (cover %, areas, change %),
  treat those numbers as measurements. Do not invent km² or NDVI values that
  contradict them. You may interpret them.
- If you cannot see a class clearly, say so. Do not invent object counts.
- Keep the answer to 4–8 sentences unless the user asks for a short yes/no.
- Do not mention that you are a stand-in API unless asked how you were produced.`;

function apiKey() {
  return (
    process.env.SATQUERY_LLM_API_KEY ||
    process.env.GEMINI_API_KEY ||
    process.env.OPENAI_API_KEY ||
    ""
  ).trim();
}

function provider() {
  return (process.env.SATQUERY_LLM_PROVIDER || "gemini").trim().toLowerCase();
}

function modelName() {
  if (process.env.SATQUERY_LLM_MODEL) return process.env.SATQUERY_LLM_MODEL;
  const p = provider();
  if (p === "openai") return "gpt-4o-mini";
  if (p === "groq") return "meta-llama/llama-4-scout-17b-16e-instruct";
  if (p === "openrouter") return "google/gemini-2.0-flash-001";
  return "gemini-2.0-flash";
}

function openaiBase() {
  if (process.env.SATQUERY_LLM_BASE_URL) return process.env.SATQUERY_LLM_BASE_URL.replace(/\/+$/, "");
  const p = provider();
  if (p === "groq") return "https://api.groq.com/openai/v1";
  if (p === "openrouter") return "https://openrouter.ai/api/v1";
  if (p === "gemini") return "https://generativelanguage.googleapis.com/v1beta/openai";
  return "https://api.openai.com/v1";
}

function userText(query: string, task: string, context?: string) {
  const parts = [`Task kind: ${task}.`, `Question: ${query}`];
  if (context) parts.push(`Deterministic stack (use as measurements):\n${context}`);
  parts.push("Answer in English, as SatQuery.");
  return parts.join("\n\n");
}

export async function GET() {
  const key = apiKey();
  return NextResponse.json({
    configured: Boolean(key),
    provider: key ? provider() : null,
    model: key ? modelName() : null,
  });
}

export async function POST(request: Request) {
  const key = apiKey();
  if (!key) {
    return NextResponse.json({ detail: "SATQUERY_LLM_API_KEY is not set." }, { status: 503 });
  }

  const body = (await request.json()) as {
    query?: string;
    task?: string;
    imageB64?: string;
    context?: string;
  };
  const query = (body.query || "").trim();
  if (!query) {
    return NextResponse.json({ detail: "query is required" }, { status: 400 });
  }

  const task = body.task || "vqa";
  const p = provider();
  const model = modelName();

  try {
    const answer =
      p === "gemini" && !process.env.SATQUERY_LLM_BASE_URL
        ? await geminiNative(key, model, query, task, body.context, body.imageB64)
        : await openaiCompat(key, model, query, task, body.context, body.imageB64);

    if (!answer.trim()) {
      return NextResponse.json({ detail: "LLM returned an empty answer" }, { status: 502 });
    }
    return NextResponse.json({
      answer: answer.trim(),
      model: `${p}:${model}`,
      provider: p,
      confidence: 0.78,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "LLM call failed";
    return NextResponse.json({ detail: message }, { status: 502 });
  }
}

async function openaiCompat(
  key: string,
  model: string,
  query: string,
  task: string,
  context: string | undefined,
  imageB64: string | undefined
) {
  const content: Array<Record<string, unknown>> = [{ type: "text", text: userText(query, task, context) }];
  if (imageB64) {
    content.push({
      type: "image_url",
      image_url: { url: `data:image/png;base64,${imageB64}` },
    });
  }
  const response = await fetch(`${openaiBase()}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${key}`,
    },
    body: JSON.stringify({
      model,
      temperature: 0.2,
      max_tokens: 700,
      messages: [
        { role: "system", content: SYSTEM },
        { role: "user", content },
      ],
    }),
  });
  const json = (await response.json()) as { choices?: { message?: { content?: string } }[]; error?: { message?: string } };
  if (!response.ok) {
    throw new Error(json.error?.message || `LLM HTTP ${response.status}`);
  }
  return json.choices?.[0]?.message?.content || "";
}

async function geminiNative(
  key: string,
  model: string,
  query: string,
  task: string,
  context: string | undefined,
  imageB64: string | undefined
) {
  const parts: Array<Record<string, unknown>> = [{ text: userText(query, task, context) }];
  if (imageB64) {
    parts.push({ inlineData: { mimeType: "image/png", data: imageB64 } });
  }
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${encodeURIComponent(key)}`;
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      systemInstruction: { parts: [{ text: SYSTEM }] },
      contents: [{ role: "user", parts }],
      generationConfig: { temperature: 0.2, maxOutputTokens: 700 },
    }),
  });
  const json = (await response.json()) as {
    candidates?: { content?: { parts?: { text?: string }[] } }[];
    error?: { message?: string };
  };
  if (!response.ok) {
    throw new Error(json.error?.message || `Gemini HTTP ${response.status}`);
  }
  return json.candidates?.[0]?.content?.parts?.[0]?.text || "";
}
