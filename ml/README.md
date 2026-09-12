# SatQuery ML — specialist model server

Runs the remote-sensing specialist models on Kaggle GPUs and serves them to the
SatQuery backend over HTTP.

## What we built and what we reuse

**Read this before putting a model name on a slide.** Every checkpoint here was
trained by someone else. What we built is the orchestration: the band contract,
the grounded-evidence pipeline, the two-GPU serving layer, and the agent that
routes a question to the right models and assembles an auditable trace. That is
also what the problem statement calls the novelty of the system — "the agentic,
query-driven framework" — so it is the right thing to claim.

| Component | Model | Base | Adapted on | Trained by |
| --- | --- | --- | --- | --- |
| multi-label land cover | [BigEarthNet-19 ResNet-50 (S1+S2)](https://huggingface.co/BIFOLD-BigEarthNetv2-0/resnet50-all-v0.2.0) | ResNet-50 | BigEarthNet v2.0 (reBEN), Sentinel-1 + Sentinel-2 | BIFOLD / TU Berlin |
| VQA, captioning, phrasing | [RSCoVLM-7B](https://huggingface.co/Qingyun/RSCoVLM-7B-2512) | Qwen2.5-VL-7B-Instruct | remote-sensing multi-task recipe | VisionXLab (Li et al., 2026) |
| text-to-box grounding | [Grounding DINO base](https://huggingface.co/IDEA-Research/grounding-dino-base) | Swin-B + BERT | Objects365, OpenImages, GoldG | IDEA-Research |
| referring segmentation | [SAM 2.1 Hiera-L](https://huggingface.co/facebook/sam2.1-hiera-large) | Hiera-L | SA-V / SA-1B | Meta AI |
| bi-temporal change | [ChangeFormerV6](https://github.com/wgcban/ChangeFormer) | SegFormer-B2 | LEVIR-CD | Bandara & Patel |
| optical–SAR fusion | [CROMA-base](https://github.com/antofuller/CROMA) | ViT-B | SSL4EO (1M paired S1/S2) | Fuller et al., NeurIPS 2023 |

Two of these are remote-sensing adapted on open training data — the BigEarthNet
classifier and RSCoVLM — which is what the mandatory "at least one visual or
vision-language component must be fine-tuned or otherwise adapted" clause asks
for. Note the flip side of that clause: plain Qwen2.5-VL would *not* satisfy it,
because "a generic LLM or VLM without remote-sensing adaptation will not satisfy
the requirements". Cite the adapted models and the requirement is met.

`train_classifier.py` and `evaluate_classifier.py` are kept but unused. They
fine-tune the classifier head on the BigEarthNet dataset we built, which takes
well under an hour on a free T4. Run them if you want the adaptation to be
*yours* as well as cited — it is cheap insurance against a strict reading of that
clause, and it produces a real before/after number.

## The design rule

Specialists decide what is true; the language model only decides how it reads.

ChangeFormer emits a mask, Grounding DINO emits boxes, CROMA emits embeddings,
the classifier emits labels — none of them produce prose. Each reduces its output
to `facts.Evidence`: a verdict, some findings, and numbers measured from the model
output. That `Evidence` is the *only* thing the LLM ever sees on these paths, so
it has no image to hallucinate from, it cannot alter a measured number, and if it
contradicts the verdict it is discarded (`verbalizer.enforce_verdict`). The
confidence reported to the user is the specialist's, never the LLM's.

The exception is open-ended questions the 19 land-cover classes cannot express.
There the VLM genuinely looks at the image, and the response says so.

## Bands, not images

The models disagree about their inputs in ways that fail silently:

- BigEarthNet S1+S2 wants **12 channels in a published order**: `VV, VH, B02,
  B03, B04, B05, B06, B07, B08, B8A, B11, B12`
- CROMA wants **two tensors**: 2 SAR channels and 12 optical bands
- ChangeFormer, Grounding DINO, SAM 2 and the VLM want **3-channel RGB**

Feed a 12-band model an RGB image padded with zeros and every model still runs,
just badly. So a scene travels as a *named* `BandStack`, each model asks for the
slice it was trained on, and a model whose bands are absent raises `MissingBands`
and declines. An RGB upload cannot satisfy a 12-band request — there is a test
pinning exactly that.

Practical consequence: Cartosat-2S and RISAT uploads carry at most R/G/B/NIR and
VV, not 12 Sentinel-2 bands. The 12-band models will often decline real uploads,
and the response says which bands were missing rather than returning a confident
number from fabricated channels.

## Layout

```
satquery_ml/
  bands.py        canonical band stack, per-model selection, npz wire format
  registry.py     model specs + provenance; feeds /health and the trace
  facts.py        the Evidence contract every adapter produces
  labels.py       BigEarthNet-19 vocabulary, question -> class matching
  loader.py       lazy loading, two-GPU placement, eviction on one GPU
  verbalizer.py   grounded phrasing prompt + the verdict guard
  adapters/       one module per checkpoint
serve.py          FastAPI server, five task endpoints
notebooks/
  kaggle_serve_models.ipynb      download all weights, serve, open the tunnel
  kaggle_train_classifier.ipynb  optional: the fine-tune described above
dataset_builder/  built the BigEarthNet VQA dataset on Kaggle (our work)
tests/            110 tests, CPU only, no weights needed
```

## Running it

On Kaggle, open `notebooks/kaggle_serve_models.ipynb`, set the accelerator to
**GPU T4 x2**, turn Internet on, and run every cell. It downloads roughly 12 GB
of weights, smoke-tests each model separately so a failure names itself, starts
the server, and prints a `https://*.trycloudflare.com` URL.

Put that URL in `backend/.env` — all four settings point at the same server:

```
SATQUERY_VLM_ENDPOINT=https://<subdomain>.trycloudflare.com
SATQUERY_GROUNDING_ENDPOINT=https://<subdomain>.trycloudflare.com
SATQUERY_CHANGE_ENDPOINT=https://<subdomain>.trycloudflare.com
SATQUERY_FUSION_ENDPOINT=https://<subdomain>.trycloudflare.com
```

Locally, without GPUs, the server still starts and every task degrades to its
deterministic path:

```bash
pip install -r requirements.txt
python serve.py --no-vlm --port 8100
python -m pytest tests -q
```

## Endpoints

| Endpoint | Task | Models |
| --- | --- | --- |
| `POST /v1/vqa` | VQA and captioning (`task=caption`) | classifier → VLM |
| `POST /v1/grounding` | text-guided grounding | Grounding DINO → SAM 2 → VLM |
| `POST /v1/change` | bi-temporal change | ChangeFormer → VLM |
| `POST /v1/fusion` | optical–SAR joint analysis | classifier + CROMA → VLM |
| `POST /v1/classify` | raw classifier scores, no LLM | classifier |
| `GET /health` | placement, load state, provenance | — |
| `GET /v1/models` | full attribution table | — |

Each scene is either `imageB64` (RGB PNG) or `bands` (a `BandStack.encode()`
payload). Send `bands` when you have them.

## Degradation, by design

Weights come from four different hosts over a Kaggle connection, so any of them
can be missing on the day. Every failure is scoped to the task that needs it and
labelled in the response:

- a model that will not load is retried **once per process**, not per request
- `ChangeFormer` missing → normalised image differencing, and the note says the
  numbers are indicative only, because an unlabelled fallback is worse than none
- `SAM 2` missing → grounding returns boxes without masks
- `CROMA` missing or wrong bands → fusion answers from the per-modality
  land-cover comparison alone
- the VLM missing → deterministic template phrasing, every verdict unchanged

## What the tests do and do not prove

The 110 tests run on CPU with no weights. They cover the band contract, the
evidence contract, grounding-prompt formatting, change measurement, fusion
composition, device planning, and every endpoint including its failure paths,
using stubbed models.

They cannot prove a checkpoint downloads or that its predictions are sensible.
That is what cell 5 and 6 of the Kaggle notebook are for: each model is loaded
and run individually before the server starts.
