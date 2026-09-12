# SatQuery AI — model fine-tuning (SIH26167 task #2)

This folder adapts a vision-language model to remote sensing and plugs it into
the agent backend. It is the piece SIH26167 explicitly requires: *"adapt or
fine-tune at least one model"* on RS data. Everything else (agent, evidence,
audit trail) is already shipped in `backend/`.

## The approach, in one paragraph

We fine-tune **Qwen2.5-VL-3B-Instruct** with **QLoRA**: the base model is
loaded in 4-bit (NF4) so it fits a free 16 GB GPU, and small **LoRA adapters**
(r=16, α=32) are trained on the language decoder's attention projections
(q/k/v/o) while the **vision encoder stays frozen**. This is the configuration
recent RS-VLM papers converged on.

**Training data is BigEarthNet** (the adaptation dataset named in SIH26167):
each Sentinel-2 patch carries multi-label land-cover classes, which
`prepare_data.py bigearthnet` converts into instruction pairs — "which
land-cover classes are present?" plus balanced yes/no presence questions.
This mirrors how the official RSVQAxBEN benchmark ("RSVQA meets BigEarthNet")
was built. **RSVQA-LR is held out for evaluation only.** RSICD captions can
optionally be mixed into training for richer language. The result is a
~100–200 MB adapter file, not a new model: at inference we load the public
base model plus our adapter.

Why not full fine-tuning? A 3B model in fp16 needs ~40 GB+ with optimizer
states. QLoRA gets ~equivalent task performance on a free T4, and the small
adapter is trivial to share between teammates.

## Where to run it

| Option | Hardware | Cost | Verdict |
|---|---|---|---|
| **Kaggle Notebooks** | T4/P100 16 GB | free, **30 h/week guaranteed** | **Recommended.** Predictable quota, 9–12 h sessions, phone verification needed |
| Google Colab Free | T4 16 GB | free, ~15–30 h/week (variable) | Good backup; disconnects on idle ~90 min |
| Friend's laptop | needs **NVIDIA RTX, ≥ 8 GB VRAM** (3060/4060+) | free | Works for the 3B model. AMD/Intel/Mac GPUs will NOT work (bitsandbytes needs CUDA). On Windows prefer WSL2 |
| Rented GPU (RunPod/Lightning) | RTX 4090 / A10 | ~$0.3–0.5/h | Only if quotas run out; a full run is ≤ $3 |

Time estimates for the default run (RSVQA-LR subsampled to ~12k QA pairs,
1 epoch, 3B model): **~3–5 h on T4**, ~2–4 h on P100, ~1–2 h on a 4090.

**Recommended plan:** train on Kaggle (it cannot serve an API), download the
adapter, then **serve it from the friend's laptop** (inference only needs
~4 GB VRAM) or any machine with an NVIDIA GPU on the demo network.

## Step by step

### 0. Smoke test the loop first (30 min, do this before spending GPU hours)

```bash
pip install -r requirements.txt
python prepare_data.py smoke --out data/smoke.jsonl
python train.py --train data/smoke.jsonl --out runs/smoke --epochs 2
python evaluate.py --data data/smoke.jsonl --limit 50 --adapter runs/smoke/adapter
```

If accuracy on the synthetic tiles jumps to ~1.0, the pipeline is correct.

### 1. Get the data

- **BigEarthNet (training):** https://bigearth.net/ → `BigEarthNet-S2` archive
  + `metadata.parquet`. The full archive is ~66 GB — **you do not need all of
  it.** Extract a handful of the 115 tile folders (a few GB, tens of thousands
  of patches); the converter simply skips metadata rows whose patches are not
  extracted, and `--max-patches` caps the rest.
- **RSVQA-LR (evaluation only):** https://rsvqa.sylvainlobry.com/ →
  `Images_LR.zip` and the **test** questions/answers JSONs. (~200 MB)
- RSICD (optional extra training text): Kaggle dataset
  `thedevastator/rsicd-image-caption-dataset`.

On Kaggle, upload these as a private Dataset once and attach it to your
notebook — no re-downloading every session.

### 2. Build the JSONL splits

```bash
# TRAIN: BigEarthNet labels -> instruction pairs (2 per patch)
python prepare_data.py bigearthnet --images BigEarthNet-S2 \
    --metadata metadata.parquet --split train \
    --out data/train.jsonl --max-patches 6000

# TEST: RSVQA-LR, never seen during training
python prepare_data.py rsvqa --images Images_LR \
    --questions LR_split_test_questions.json --answers LR_split_test_answers.json \
    --out data/test.jsonl
```

6000 patches → 12k pairs keeps the first run inside one Kaggle session. Scale
up (or concatenate RSICD JSONL) once the first adapter works.

### 3. Baseline score, train, re-score

```bash
python evaluate.py --data data/test.jsonl --limit 500 --out runs/base_report.json
python train.py --train data/train.jsonl --out runs/ben-lora
python evaluate.py --data data/test.jsonl --limit 500 \
    --adapter runs/ben-lora/adapter --out runs/tuned_report.json
```

The two report files are your before/after evidence for the judges: the model
never saw RSVQA during training, so any gain on it demonstrates genuine
adaptation to remote sensing rather than memorisation of the test benchmark.

### 4. Serve the adapter and wire it into SatQuery

The Kaggle notebook `ml/kaggle_train_unsloth.ipynb` produces
`ben-lora/adapter/` — a directory with `adapter_config.json` +
`adapter_model.safetensors` + tokenizer files. Download `ben-lora-adapter.zip`
from the Kaggle output tab and extract it wherever suits you. Then on any
NVIDIA machine (Linux, WSL2, or Windows) with the adapter folder present:

```bash
python ml/serve.py --adapter path/to/ben-lora/adapter --port 8100
```

`serve.py` reads `adapter_config.json` to pick the base model automatically —
you don't need to pass `--model` for adapters trained with the Unsloth notebook
(they default to `unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit`).

Then in `backend/.env` on the machine running the SatQuery backend:

```
SATQUERY_VLM_ENDPOINT=http://<gpu-machine-ip>:8100
```

Restart the backend and check:

- `GET /api/health` → `weightsWired: 1`
- `GET /api/registry` → the VQA and captioning rows show **wired**
- Ask a question on an uploaded scene → the answer comes from the model, the
  masks/metrics still come from the deterministic stack, and
  `inferenceBackend` in the response names the endpoint.

If the endpoint is down or errors, the backend automatically falls back to the
heuristic baseline and says so in the trace — the demo cannot hard-fail.

### 4b. Test locally like a chatbot (recommended before demo day)

**One-time Windows setup (Python 3.13, CUDA 12.4):**

```powershell
# 1. Install a CUDA-enabled PyTorch (pip's default is CPU-only)
python -m pip uninstall -y torch
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision

# 2. Install the rest of the stack — the version pins in requirements.txt
#    matter (transformers 5.x + bitsandbytes 0.50 has a regression that
#    breaks the Qwen2.5-VL vision encoder; we stay on transformers 4.54.x)
python -m pip install -r ml\requirements.txt

# Sanity check
python -c "import torch; print('cuda', torch.cuda.is_available(), 'vram', torch.cuda.get_device_properties(0).total_memory/1e9 if torch.cuda.is_available() else 0, 'GB')"
```

Then run `ml/tests/test_serve_utils_smoke.py` to generate a synthetic S1
patch, or drop a real BigEarthNet patch folder somewhere on disk.

`ml/chat.py` is an interactive REPL that eats **BigEarthNet GeoTIFFs directly**
— it reads VV/VH bands and renders the exact pseudo-RGB the model saw during
training (`R=VV, G=VH, B=|VV−VH|`, 2/98 percentile stretch), so you can sanity
check quality without spinning up the frontend.

Two ways to run it:

```bash
# A) Talk to a running serve.py (best on a shared GPU machine)
python ml/serve.py --adapter path/to/ben-lora/adapter --port 8100
python ml/chat.py  --server http://127.0.0.1:8100 \
                   --image path/to/S1B_IW_GRDH_.../S1B_IW_GRDH_..._61_15   # patch folder

# B) Boot the model in-process (no HTTP; simplest on a laptop with a GPU)
python ml/chat.py --adapter path/to/ben-lora/adapter \
                  --image path/to/patch.png
```

The `--image` argument accepts:

| Input | What we do |
|---|---|
| a BigEarthNet-S1 patch folder (`*_VV.tif` + optional `*_VH.tif`) | render the training-time pseudo-RGB |
| a BigEarthNet-S2 patch folder (`*_B04.tif` + `*_B03.tif` + `*_B02.tif`) | render a true-colour composite |
| a single `.tif` / `.tiff` file | treat as VV grayscale, percentile-stretch, replicate to RGB |
| a `.png` / `.jpg` | use as-is (resized to 224x224 to match training) |

Inside the REPL:

```
you> which land-cover classes are present?
bot> Broad-leaved forest, mixed forest, transitional woodland/shrub.
     confidence 0.87 · 2410 ms
you> /image path/to/another_patch
you> is there any built-up area?
bot> No, this scene is fully natural land cover.
```

Slash commands: `/image <path>`, `/reset`, `/system <text>`, `/save log.jsonl`,
`/help`, `/quit`.

## Kaggle notebook cheat-sheet

```python
# Cell 1 — code + deps (repo is private: use a token, or upload ml/ as a Dataset)
!pip -q install peft bitsandbytes qwen-vl-utils accelerate

# Cell 2 — data (attached Kaggle Dataset appears under /kaggle/input)
!python prepare_data.py bigearthnet \
   --images /kaggle/input/bigearthnet-subset/BigEarthNet-S2 \
   --metadata /kaggle/input/bigearthnet-subset/metadata.parquet \
   --out data/train.jsonl --max-patches 6000

# Cell 3 — train (T4: ~3-5 h)
!python train.py --train data/train.jsonl --out /kaggle/working/ben-lora

# Cell 4 — zip the adapter for download
!cd /kaggle/working && zip -r ben-lora-adapter.zip ben-lora/adapter
```

Turn on **Settings → Accelerator → GPU T4 x2 (or P100)** and **Persistence →
Files** before running.

## Scaling up later

- **7B model:** `--model Qwen/Qwen2.5-VL-7B-Instruct` still fits a T4 with the
  same script (slower, ~2-3× training time). Do this only after the 3B run is
  scored.
- **Change VQA (CDVQA):** same recipe, two images per message — extend the
  collator's message builder with a second `{"type": "image"}` entry and point
  `SATQUERY_CHANGE_ENDPOINT` at a second serve instance.
- **Grounding (DIOR-RSVG / VRSBench):** train with box-coordinate answers in
  the text, then implement `run_endpoint` on `GroundingTool` the same way it
  is done for VQA/caption in `backend/app/tools/single_image.py`.
