# Kaggle + Unsloth training (with auto-checkpoints)

Use this **after** you built `ben_vqa_third` on a Mac with `dataset_builder/`
and uploaded it to Kaggle.

## Why sessions die — and how we handle it

Kaggle GPU notebooks shut down after ~9–12 hours (or sooner if idle).
`train_unsloth.py` **saves progress every 10 minutes** (and every 50 steps).
Next session you **resume** from the latest `checkpoint-*` folder.

**Critical:** `/kaggle/working` is empty in a brand-new session unless you
carry checkpoints over. Do **one** of these:

| Method | How |
|---|---|
| **A. Persistence (easiest)** | Notebook → Settings → **Persistence → Files** (keeps `/kaggle/working`) |
| **B. Checkpoint Dataset** | After each session, zip `checkpoint-*`, upload as Kaggle Dataset `ben-lora-checkpoints`, attach next time |
| **C. Save Version** | Save Version mid-run, then open that version / Copy & Edit so outputs come back |

---

## Notebook setup

1. Kaggle → **New Notebook**
2. **Settings → Accelerator → GPU T4** (or P100)
3. **Internet → On**
4. **Persistence → Files** (recommended)
5. **Add Input** → `ben-vqa-third` dataset
6. Optional next sessions: also add `ben-lora-checkpoints`
7. Add this repo’s `ml/` folder (clone or upload as a Dataset) so `train_unsloth.py` is available

---

## Cell 1 — Install Unsloth

```python
!pip install -q unsloth
```

---

## Cell 2 — Train (saves every 10 minutes, auto-resumes)

Point `--data` at your uploaded dataset. If you re-uploaded previous
checkpoints as a Dataset, pass `--resume-dir`.

```python
# First session (or Persistence already has /kaggle/working/ben-lora):
!python /kaggle/input/satquery-ml/train_unsloth.py \
  --data /kaggle/input/ben-vqa-third \
  --out /kaggle/working/ben-lora \
  --save-every-minutes 10 \
  --save-steps 50

# Later session — if checkpoints were uploaded as a Dataset:
# !python /kaggle/input/satquery-ml/train_unsloth.py \
#   --data /kaggle/input/ben-vqa-third \
#   --out /kaggle/working/ben-lora \
#   --resume-dir /kaggle/input/ben-lora-checkpoints \
#   --save-every-minutes 10 \
#   --save-steps 50
```

Adjust the path to `train_unsloth.py` if you cloned the GitHub repo instead:

```python
!git clone https://github.com/shubh000015/satquery-ai.git /kaggle/working/satquery
!python /kaggle/working/satquery/ml/train_unsloth.py \
  --data /kaggle/input/ben-vqa-third \
  --out /kaggle/working/ben-lora \
  --save-every-minutes 10
```

You should see lines like:

```text
[checkpoint] timed save at step 120 (every 10 min)
RESUMING from /kaggle/working/ben-lora/checkpoint-120
```

Smoke test (short):

```python
!python .../train_unsloth.py --data /kaggle/input/ben-vqa-third \
  --out /kaggle/working/ben-lora-smoke --max-steps 30 --max-samples 200 \
  --save-every-minutes 2 --save-steps 10
```

---

## Cell 3 — After a session ends (or before it might die): backup checkpoints

Run this anytime — even mid-training in another cell if needed after interrupt:

```python
!cd /kaggle/working && zip -r ben-lora-checkpoints.zip ben-lora/checkpoint-* ben-lora/adapter 2>/dev/null || \
  cd /kaggle/working && zip -r ben-lora-checkpoints.zip ben-lora
print("Download ben-lora-checkpoints.zip from Output, then upload it as a Kaggle Dataset")
print("Next session: Add Input → that dataset → pass --resume-dir /kaggle/input/<slug>")
```

Or publish straight to a Dataset with the API (if `kaggle.json` secrets are set):

```python
# optional — only if Kaggle API credentials are configured in the notebook
# !kaggle datasets version -p /kaggle/working/ben-lora -m "checkpoint backup" --dir-mode zip
```

---

## Cell 4 — Final adapter zip (when training finishes)

```python
!cd /kaggle/working && zip -r ben-lora-adapter.zip ben-lora/adapter
print("Download ben-lora-adapter.zip — this is what serve.py needs")
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

## Manual / cell-by-cell Unsloth (same checkpoint behaviour)

If you prefer pasting Unsloth cells instead of the script, use this **Cell 5**
replacement (keeps timed saves + resume):

```python
from pathlib import Path
import time
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

OUT = Path("/kaggle/working/ben-lora")
OUT.mkdir(parents=True, exist_ok=True)

class TimedCheckpointCallback(TrainerCallback):
    def __init__(self, every_minutes=10):
        self.every_seconds = every_minutes * 60
        self._last = time.monotonic()
    def on_step_end(self, args, state, control, **kwargs):
        if time.monotonic() - self._last >= self.every_seconds:
            control.should_save = True
            self._last = time.monotonic()
            print(f"\n[checkpoint] timed save at step {state.global_step}\n", flush=True)
        return control

def latest_ckpt(root: Path):
    ckpts = [p for p in root.glob("checkpoint-*") if p.is_dir()]
    return max(ckpts, key=lambda p: int(p.name.split("-")[-1])) if ckpts else None

# ... load model + dataset as in earlier cells, then:
FastVisionModel.for_training(model)
resume = latest_ckpt(OUT)
print("RESUMING from", resume) if resume else print("fresh start")

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
        output_dir=str(OUT),
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=2048,
        save_strategy="steps",
        save_steps=50,          # backup every 50 steps
        save_total_limit=3,     # keep last 3 only (disk)
    ),
    callbacks=[TimedCheckpointCallback(10)],  # every 10 minutes
)
trainer.train(resume_from_checkpoint=str(resume) if resume else None)
```

---

## Notes

- Keep the tab awake; idle kills still happen.
- Download / persist checkpoints **before** the session vanishes.
- RSVQA is for testing later, not this training run.
- `--save-every-minutes 5` if you want denser saves (uses more disk).
