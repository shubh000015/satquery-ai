# Kaggle + Unsloth training (copy-paste cells)

Use this **after** you built `ben_vqa_third` on a Mac with `dataset_builder/`
(S1 + S2 PNGs + `train.jsonl`) and uploaded the **folder** to Kaggle (CLI
`--dir-mode zip` is fine — you do not need a local .zip).

## Notebook setup

1. Kaggle → **New Notebook**
2. **Settings → Accelerator → GPU T4** (or P100)
3. **Internet → On**
4. **Add Input** → your `ben_vqa_third` dataset
5. Optional: add RSVQA-LR test dataset for evaluation later

---

## Cell 1 — Install Unsloth

```python
!pip install -q unsloth
# If the above fails on a fresh image, use Unsloth's install line from their docs:
# !pip install -q --upgrade --no-cache-dir --force-reinstall --no-deps unsloth unsloth_zoo
```

---

## Cell 2 — Paths (edit the dataset name)

```python
from pathlib import Path

# Change "ben-vqa-third" to whatever Kaggle named your uploaded dataset
DATA = Path("/kaggle/input/ben-vqa-third")
TRAIN_JSONL = DATA / "train.jsonl"
# images live next to the jsonl as images_s2/ and images_s1/

assert TRAIN_JSONL.exists(), f"Missing {TRAIN_JSONL}. Attach the dataset and fix DATA."
print("train.jsonl OK:", TRAIN_JSONL)
print("s2 images:", (DATA / "images_s2").exists(), "s1 images:", (DATA / "images_s1").exists())
```

---

## Cell 3 — Load JSONL into Unsloth vision format

```python
import json
from PIL import Image

SYSTEM = (
    "You are SatQuery, an assistant for satellite and aerial imagery. "
    "Answer questions about optical and SAR remote sensing scenes "
    "concisely and factually."
)

def load_rows(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows

raw = load_rows(TRAIN_JSONL)
print("samples:", len(raw))

def to_unsloth(row):
    img_path = DATA / row["image"]  # relative path like images_s2/....png
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SYSTEM + "\n\n" + row["question"]},
                    {"type": "image", "image": str(img_path)},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": row["answer"]}],
            },
        ]
    }

# Start smaller if you want a smoke run: raw = raw[:2000]
dataset = [to_unsloth(r) for r in raw if (DATA / r["image"]).exists()]
print("usable samples:", len(dataset))
```

---

## Cell 4 — Load Qwen2.5-VL with Unsloth (4-bit)

```python
from unsloth import FastVisionModel
import torch

# 3B is safer on free T4. Switch to 7B only after a successful 3B run.
MODEL_NAME = "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit"
# MODEL_NAME = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit"

model, tokenizer = FastVisionModel.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,   # freeze vision tower (recommended)
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    random_state=13,
)
```

---

## Cell 5 — Train

```python
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_ratio=0.03,
        num_train_epochs=1,
        learning_rate=1e-4,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=13,
        output_dir="/kaggle/working/ben-lora",
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=2048,
    ),
)

trainer.train()
```

Tip: for a 30–60 min smoke test, set `max_steps=100` and comment out `num_train_epochs`.

---

## Cell 6 — Save adapter + zip for download

```python
adapter_dir = "/kaggle/working/ben-lora/adapter"
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print("saved:", adapter_dir)

!cd /kaggle/working && zip -r ben-lora-adapter.zip ben-lora/adapter
print("Download ben-lora-adapter.zip from the Notebook Output panel")
```

---

## After Kaggle (laptop with NVIDIA GPU)

```bash
cd ml
python serve.py --adapter /path/to/adapter --port 8100
```

In `backend/.env`:

```
SATQUERY_VLM_ENDPOINT=http://localhost:8100
```

---

## Notes

- Keep the notebook awake; Kaggle may kill idle sessions.
- Download the zip **before** the session ends.
- RSVQA is for **testing later**, not for this training cell.
- If Unsloth package names shift slightly, follow the latest Unsloth Qwen2.5-VL vision notebook and keep our dataset loading cells (2–3) unchanged.
