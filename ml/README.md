# SatQuery ML — land-cover classifier + Qwen2.5 answer layer

Two stages, with a clean split of responsibility:

| Stage | What it does | Output |
|---|---|---|
| **1. Land-cover classifier** | Looks at the image. A single network handles both Sentinel-1 (SAR) and Sentinel-2 (optical) patches and predicts which of the 19 BigEarthNet classes are present. | Labels + a score per class |
| **2. Qwen2.5 answer layer** | Never sees the image. Takes stage 1's labels and the analyst's question, and writes the reply. | A sentence or two |

The reason for the split is reliability. A vision-language model asked to answer
directly will happily describe a river that is not there, and you cannot tell
from the output whether it looked or guessed. Here every claim in the answer
traces back to a number the classifier produced, the yes/no verdict is decided
by thresholded classifier scores rather than by the language model, and the
confidence we report to the UI is the classifier's own probability. If Qwen
disagrees with the verdict it is overridden (`verbalizer.py`), and if the LLM
cannot load at all the endpoint still answers from a template.

It is also the practical choice: a ResNet-50 multi-label head trains to a useful
accuracy on a free Kaggle T4 in a few hours, where end-to-end VLM fine-tuning on
the same hardware does not.

## What we trained, and what we reuse

Worth being precise about, because judges ask.

- **We train:** the land-cover classifier — all layers, on BigEarthNet S1+S2
  patches, with a randomly initialised 19-class multi-label head. Starting from
  ImageNet-pretrained ResNet weights is the standard practice for remote-sensing
  classification, since low-level ImageNet features transfer well to overhead
  imagery. `train_classifier.py` is the whole recipe and
  `evaluate_classifier.py` produces the numbers.
- **We reuse as-is:** `Qwen/Qwen2.5-7B-Instruct`, frozen, prompted, not trained.
  It is a phrasing layer, not the thing being evaluated.
- **We built:** the dataset itself — `dataset_builder/build_bigearthnet_vqa.py`
  pairs each Sentinel-2 patch with its Sentinel-1 twin via `metadata.parquet`,
  renders both to PNG (optical from B04/B03/B02, SAR false-colour from
  VV/VH/|VV-VH|), and writes the labels.

## Layout

```
ml/
  satquery_ml/
    labels.py        BigEarthNet-19 vocabulary, short names, coarse groups
    dataset.py       train.jsonl -> multi-label tensors, deterministic split
    model.py         ResNet backbone + modality-conditioned 19-class head
    metrics.py       mAP, micro/macro F1, per-class threshold tuning
    inference.py     image -> Prediction (the grounded facts)
    verbalizer.py    Prediction + question -> answer, via Qwen2.5
  train_classifier.py
  evaluate_classifier.py
  serve.py           FastAPI, speaks the backend's existing endpoint contract
  notebooks/
    kaggle_train_classifier.ipynb
    kaggle_serve_tunnel.ipynb
  dataset_builder/   builds the Kaggle dataset from raw BigEarthNet (CPU only)
```

## 1. Data

The dataset is already on Kaggle as `knayamket/bigearthnet-vqa` (S1+S2) — attach
it to the notebook and skip this step. To rebuild it from the raw archives, see
`dataset_builder/README.md`.

Stage 1 reads the `category: "multi-label"` rows of `train.jsonl`, which carry
the full label set for each patch. The presence and modality rows are QA
phrasings of the same ground truth and are ignored here.

One caveat the loader handles for you: the builder joined labels with `", "`,
and four class names contain commas themselves (`Transitional woodland, shrub`
and friends). `labels.parse_label_string` recovers the real label set by
longest-match instead of splitting on the separator.

## 2. Train

Smoke-test the loop before spending GPU hours — this should run in a couple of
minutes and the loss should fall:

```bash
pip install -r requirements.txt
python train_classifier.py --data /kaggle/input/bigearthnet-vqa \
    --out runs/smoke --limit 400 --epochs 1 --batch-size 8
```

Then the real run (T4, roughly 40 min/epoch on ~120k patches):

```bash
python train_classifier.py \
    --data /kaggle/input/bigearthnet-vqa \
    --out /kaggle/working/landcover \
    --epochs 6 --batch-size 64
```

The loop checkpoints on a wall-clock timer (`--save-every-minutes`, default 10)
and auto-resumes from `last.ckpt`, so a killed Kaggle session costs you minutes
rather than the run. `classifier.pt` is rewritten whenever validation macro F1
improves, and carries its own tuned per-class thresholds and metrics.

Use `notebooks/kaggle_train_classifier.ipynb` to do all of this on Kaggle.

## 3. Evaluate

```bash
python evaluate_classifier.py \
    --data /kaggle/input/bigearthnet-vqa \
    --checkpoint /kaggle/working/landcover/classifier.pt \
    --out /kaggle/working/landcover/eval_report.json
```

The split is reproduced by hashing image paths exactly as training did, so these
patches were never trained on. The report gives mAP, micro/macro F1, precision,
recall, per-class AP with support counts, and a separate optical-vs-SAR
breakdown — SAR is the harder modality and averaging the two hides that.

Multi-label land cover is not a single-answer task, so read mAP and macro F1
rather than exact-match accuracy, which is punishing by construction.

## 4. Serve

```bash
python serve.py --checkpoint runs/landcover/classifier.pt --port 8100
```

Endpoints:

- `GET /health` — model label, the classifier's stored metrics, whether the LLM loaded
- `POST /v1/vqa` — `{question, imageB64, task, modality?}` → `{answer, confidence, model, elapsedMs, oneWord, labels, modality}`
- `POST /v1/classify` — stage 1 only, every class score. Useful for the demo: it shows what the classifier said before the language layer touched it.

`--no-llm` starts in seconds with template phrasing, which is the right setting
while you are debugging the wiring. `oneWord` is the classifier's terse answer
(`yes`, `no`, or the top class) alongside the full sentence.

Kaggle cannot expose a port, so `notebooks/kaggle_serve_tunnel.ipynb` runs this
behind a cloudflared tunnel and prints the public URL.

## 5. Wire it into the backend

The endpoint contract is unchanged, so `backend/` needs no edits. In
`backend/.env`:

```
SATQUERY_VLM_ENDPOINT=https://<your-tunnel>.trycloudflare.com
```

Restart the backend, then check:

- `GET /api/health` → `weightsWired: 1`
- `GET /api/registry` → the VQA and captioning rows read **wired**
- Ask a question about an uploaded scene → the answer comes from this pipeline,
  while masks and area metrics still come from the deterministic stack in
  `backend/app/services/analysis.py`

If the endpoint is unreachable the backend falls back to its heuristic baseline
and says so in the trace, so a dead tunnel degrades the demo instead of breaking it.

## Notes and limits

- The modality flag matters: pass `modality` in the request when you know it.
  Otherwise `inference.detect_modality` guesses from saturation and from the
  `B ≈ |R−G|` signature of our own SAR renders, which is reliable for our
  dataset but is only a heuristic for arbitrary uploads.
- BigEarthNet is European. Applied to Indian scenes the class vocabulary still
  broadly holds, but the score calibration will drift — worth saying out loud
  rather than being caught by it.
- Object questions ("how many ships?") are out of scope for a patch-level
  classifier. The backend answers those from its own baseline and labels them as
  unverified; a detection head would be the honest way to support them.
