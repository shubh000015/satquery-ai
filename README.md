# SatQuery AI — SIH26167

Interactive vision-language assistant for multimodal remote sensing image analysis through **natural-language chat**. Smart India Hackathon 2026 · ISRO.

## What this is

Non-experts ask a satellite scene a question in English. An agentic controller classifies the query, picks specialist tools, and returns an evidence-grounded reply (text + overlays on the image).

**Defined inputs**
- Single optical / MSI / SAR — captioning, VQA, text-guided grounding
- Co-registered optical + SAR — fusion (e.g. flood through cloud)
- Bi-temporal pair — change description / change VQA

**Demo scenes (scripted mock agent — always local)**
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

`.env.development` already points the workspace at `http://localhost:8000`.
Restart `npm run dev` after the backend is up.

Uploaded scenes are analysed server-side and the audit panel streams the real
trace. The four scripted demo missions stay on the local mock.

Specialists call the Kaggle-hosted RS models when `SATQUERY_ML_ENDPOINT` is set
in `backend/.env` (the cloudflared URL from
`ml/notebooks/kaggle_serve_models.ipynb`). If that tunnel is down they fall
back to the heuristic baseline and say so in `inferenceBackend`.

## Stack

Next.js 16 (App Router) · TypeScript · Tailwind v4 · Framer Motion · Three.js

FastAPI · NumPy · Pillow · rasterio (optional) for the agent backend.

RS specialists (BigEarthNet ResNet-50, RSCoVLM-7B, Grounding DINO, SAM 2.1,
ChangeFormerV6, CROMA) are served from `ml/serve.py` and reached through
`SATQUERY_ML_ENDPOINT`. See [`ml/README.md`](ml/README.md).
