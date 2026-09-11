# SatQuery AI — agent backend (FastAPI)

Agentic remote-sensing backend for SIH26167. Upload one scene or a pair, ask a
question in plain language, and the controller validates the inputs, routes the
query to a task, selects specialist tools, runs them, and returns a grounded
answer with visual evidence, confidence and an auditable execution trace.

```
User input ──▶ Input validator ──▶ Router agent ──┬─▶ Single-image (VQA · caption · grounding)
                     │ invalid                    ├─▶ Change analysis (bi-temporal)
                     ▼                            └─▶ Optical + SAR fusion
                  error                                        │
                                                               ▼
                                       Result integration ──▶ Response ──▶ Execution trace
```

## Status: pipeline real, weights not wired yet

The orchestration, validation, evidence generation, tracing, sessions and
reports are all real. The **specialists currently run a deterministic heuristic
baseline** (spectral indices, Otsu thresholds, connected components,
radiometrically normalised bi-temporal differencing) rather than fine-tuned
models, and every response says so in `inferenceBackend`.

That is deliberate: it makes the agentic layer demonstrable and testable now,
and swapping in RS-adapted checkpoints later is a per-tool change behind an
unchanged API. See [Wiring a fine-tuned model](#wiring-a-fine-tuned-model).

## Quick start

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate         # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

- API docs: <http://localhost:8000/docs>
- Health: <http://localhost:8000/api/health>

Point the frontend at it by adding to `.env.local` in the repo root:

```bash
NEXT_PUBLIC_SATQUERY_API=http://localhost:8000
```

With that set, uploaded scenes in the workspace are analysed by this backend and
the audit panel streams the real trace. Without it the UI keeps using its local
mock agent, so demos never depend on a running Python process. The four scripted
demo missions always use the mock.

Run the tests, or see the whole pipeline without a browser:

```bash
pytest                              # 39 tests, no network or model weights needed
python scripts/demo_pipeline.py     # synthetic GeoTIFFs through every task
python scripts/demo_pipeline.py --report
```

## Input scope (per the problem statement)

| Format | Accepted |
| --- | --- |
| GeoTIFF `.tif` / `.tiff` | Yes — the operational path; CRS and GSD give real km² |
| TIFF `.tif` / `.tiff` | Yes |
| PNG `.png` | Benchmark datasets only |
| JPEG `.jpg` / `.jpeg` | Benchmark datasets only |

PNG/JPEG uploads are flagged. Declare the dataset (`benchmark_dataset=RSVQA`) and
the flag becomes an informational note; set `SATQUERY_STRICT_FORMAT_POLICY=true`
to reject them outright instead of warning.

Cardinality is a single scene or a pair. Pair mode is inferred from the sensor
families — optical/MSI + SAR is cross-modal, same-family is bi-temporal — and
the client can override with an explicit `mode`.

## What the validator checks

Errors (the run stops): no input, more than two scenes, CRS mismatch, aspect
ratio mismatch, footprint overlap below 40%, and unreadable rasters.

Warnings (the run continues, and they are attached to the answer and subtract
from confidence): pixel dimension mismatch, partial overlap, GSD mismatch beyond
2×, missing geotransform, unverifiable co-registration, benchmark-only formats,
a pair declared cross-modal that is not, a bi-temporal pair mixing sensors, and
two scenes reporting the same acquisition date.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Status, whether rasterio is present, how many model rows are wired |
| GET | `/api/registry` | The model plan and specialist tools, with per-row status |
| POST | `/api/assets` | Upload 1–2 scenes (multipart), returns derived assets + validation + suggested queries |
| POST | `/api/assets/validate` | Dry-run compatibility check for a set of asset ids |
| GET | `/api/assets/{id}` | Derived asset record |
| GET | `/api/assets/{id}/preview` | 8-bit PNG rendering (browsers cannot display GeoTIFF) |
| GET | `/api/assets/{id}/raw` | Original uploaded bytes |
| POST | `/api/query` | Full pipeline, one JSON response |
| POST | `/api/query/stream` | Same pipeline as SSE: `trace`, `step`, `result`, `error` |
| GET | `/api/sessions` | History for the workspace sidebar |
| GET | `/api/sessions/{id}` | Assets and results for one session |
| GET | `/api/sessions/{id}/report` | Markdown report (`?queryId=` for a single run) |
| DELETE | `/api/sessions/{id}` | Delete a session and its assets |

Responses are camelCase and shaped to match `src/lib/types.ts`, so a
`QueryResult` drops straight into the existing workspace: boxes are percentages
in the UI's 0–100 viewBox, masks are SVG paths, and mask ids reuse the ones the
imagery stage already maps to layer toggles (`flood`, `channel`, `gain`, `ag`).

Example:

```bash
curl -F "files=@kosi_optical_20240715.tif" \
     -F "files=@kosi_risat_sar_20240715.tif" \
     http://localhost:8000/api/assets

curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Which settlements are affected by water-covered areas?",
       "assetIds":["<id1>","<id2>"]}'
```

## Tasks and tool chains

The first tool in a chain owns the answer; the rest contribute evidence, which
is what makes a run genuinely multi-tool rather than one model doing everything.

| Task | Chain |
| --- | --- |
| `vqa` | RS-VQA (+ RS-Grounder when the question names a class or object) |
| `caption` | RS-Captioner (+ RS-Grounder for a named class) |
| `grounding` | RS-Grounder → RS-Captioner |
| `change` | CD-Mapper → CDVQA |
| `change-vqa` | CDVQA → CD-Mapper |
| `cross-modal` | OptSAR-Fusion (+ RS-Grounder for objects/built-up) |
| `measure` | Geometry (straight off the geotransform) |

A query the inputs cannot support is **degraded rather than refused**: asking
what changed with only one scene answers about that scene, warns why, and
records `degraded: true` in the trace.

## Execution trace

Every run emits six stages — input validation, modality detection, query parse,
tool selection, visual evidence, integration — each with its detail line,
duration and machine-readable outputs. The evidence stage records, per tool, the
model key, the backend that served it, the parameters used (methods, thresholds,
target class, t0/t1) and the numbers produced. `GET /api/sessions/{id}/report`
renders all of it as markdown.

## Wiring a fine-tuned model

1. Serve the checkpoint behind an HTTP endpoint.
2. Set the matching variable (see `.env.example`): `SATQUERY_VLM_ENDPOINT`,
   `SATQUERY_GROUNDING_ENDPOINT`, `SATQUERY_CHANGE_ENDPOINT`,
   `SATQUERY_FUSION_ENDPOINT`. The registry row flips to `wired`.
3. Implement `run_endpoint(ctx, endpoint)` on that tool in `app/tools/`,
   returning a `ToolOutput`.

Until step 3, a configured endpoint falls back to the baseline and says so in
the trace and in `warnings` — it never silently pretends. Nothing else changes:
same routing, same schemas, same evidence contract, same frontend.

## Interim external LLM (while the adapter trains)

Set a provider key so VQA / caption / change-VQA / grounding / fusion answers
come from a hosted vision model. Masks, boxes and metrics still come from the
deterministic stack.

```
SATQUERY_LLM_API_KEY=your-key
SATQUERY_LLM_PROVIDER=gemini
SATQUERY_LLM_MODEL=gemini-2.0-flash
```

`GET /api/health` then reports `llmWired: true`. `POST /api/llm/ask` is the
same stand-in without an uploaded GeoTIFF (used by the Next.js workspace).
When `SATQUERY_VLM_ENDPOINT` is later set, it wins over the LLM.

## Layout

```
app/
  main.py              FastAPI app, CORS, error handlers
  core/                settings (SATQUERY_* env) and error types
  schemas/             pydantic models mirroring src/lib/types.ts
  agent/
    controller.py      the pipeline + auditable trace, sync and SSE
    validator.py       format, cardinality, modality, co-registration
    router.py          query -> task, and target-class extraction
    registry.py        the model plan; which rows have weights wired
    integrator.py      merges specialist outputs, weights confidence
    suggestions.py     representative queries per input mode
  tools/               specialist tools (see table above) + shared statistics
  services/
    raster.py          GeoTIFF/TIFF/PNG/JPEG reading; rasterio optional
    analysis.py        indices, Otsu, components, mask -> SVG, change detection
    storage.py         asset store, previews, scene cache
    sessions.py        session history, JSON-backed
    report.py          markdown reports
scripts/demo_pipeline.py
tests/                 39 tests over router, validator, analysis and the API
```

`rasterio` is optional. With it, CRS, geotransform and multi-band stacks are read
properly; without it there is a Pillow fallback that parses the GeoTIFF tags
(pixel scale, tiepoint, GeoKey directory) by hand. `/api/health` reports which
path is active.

## Notes on the baseline's honesty

- Water uses NDWI when a NIR band exists, a blue-vs-red proxy on RGB-only input,
  and low-backscatter thresholding on SAR — each response names the method used.
- Built-up is thresholded *within* the non-water, non-vegetated residual, so
  ordinary bare land is reported as other/bare instead of being forced into a
  class.
- Bi-temporal differencing normalises t1 to t0 (median/MAD) first, so a
  difference in sun angle is not reported as change.
- Object counts (ships, tanks) are returned as a *shortlist of candidates* with
  an explicit caveat, because class identity needs the fine-tuned grounder.
- Confidence comes from threshold separability, is penalised when a mask
  degenerates to nearly nothing or nearly everything, and drops further for each
  validator warning.
