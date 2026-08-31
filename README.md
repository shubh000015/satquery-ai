# SatQuery AI — SIH26167

Interactive vision-language assistant for multimodal remote sensing image analysis through **natural-language chat**. Smart India Hackathon 2026 · ISRO.

## What this is

Non-experts ask a satellite scene a question in English. An agentic controller classifies the query, picks specialist tools, and returns an evidence-grounded reply (text + overlays on the image).

**Defined inputs**
- Single optical / MSI / SAR — captioning, VQA, text-guided grounding
- Co-registered optical + SAR — fusion (e.g. flood through cloud)
- Bi-temporal pair — change description / change VQA

**Demo scenes (UI, mock agent — models not wired yet)**
| Scene | Proves |
| --- | --- |
| Kosi Fan, Bihar | Optical × SAR flood VQA + settlement grounding |
| Mundra, Gujarat | Single-image grounding / VQA / caption |
| Gurugram, Haryana | Bi-temporal change VQA |
| Doaba, Punjab | Captioning + land-cover VQA |

## Run

```bash
cd satquery
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Type a question on the landing chat, or open a scene and continue the thread.

### With the agent backend

The FastAPI backend in [`backend/`](backend/README.md) runs the real pipeline:
input validation, task routing, specialist tool selection, evidence generation
and an auditable execution trace.

```bash
cd backend
python -m venv .venv && .venv\Scripts\activate   # source .venv/bin/activate elsewhere
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Then add to `.env.local` in the repo root and restart `npm run dev`:

```bash
NEXT_PUBLIC_SATQUERY_API=http://localhost:8000
```

Uploaded scenes are now analysed server-side and the audit panel streams the
real trace. Leave the variable unset and the UI runs entirely on its local mock
agent — the four scripted demo missions always do.

Fine-tuned RS models are not wired yet: the specialists run a documented
heuristic baseline and every response reports that in `inferenceBackend`. See
the backend README for how to plug a checkpoint into a tool.

## Stack

Next.js 16 (App Router) · TypeScript · Tailwind v4 · Framer Motion · Three.js

FastAPI · NumPy · Pillow · rasterio (optional) for the agent backend.

RS-adapted VLM weights (BigEarthNet, VRSBench, RSVQA, CDVQA) — to be integrated.
