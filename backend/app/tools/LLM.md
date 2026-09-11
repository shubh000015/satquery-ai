# Interim LLM stand-in

Used on branch `feat/external-llm` until the Qwen2.5-VL QLoRA adapter is
trained. Not a replacement for the SIH “fine-tune at least one model”
requirement — only a live answer so the demo can talk while training runs.

## Env (backend/.env)

```
SATQUERY_LLM_API_KEY=...
SATQUERY_LLM_PROVIDER=gemini
SATQUERY_LLM_MODEL=gemini-2.0-flash
```

Also accepted: `openai`, `groq`, `openrouter`, or `compatible` + `SATQUERY_LLM_BASE_URL`.

## Env (repo `.env.local` for Next.js only)

Same `SATQUERY_*` names. The Next route `/api/satquery-llm` never prefixes
`NEXT_PUBLIC_`, so the key is not shipped to the browser.
