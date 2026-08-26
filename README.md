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

## Stack

Next.js 16 (App Router) · TypeScript · Tailwind v4 · Framer Motion · Three.js

Backend / RS-adapted VLM, BigEarthNet, VRSBench, RSVQA, CDVQA — not in this repo yet.
