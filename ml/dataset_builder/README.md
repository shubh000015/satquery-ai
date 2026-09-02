# Build BigEarthNet S1+S2 → RSVQA-style training data (laptop only)

This folder is **separate** from model training. Use it on a laptop to turn
**both** BigEarthNet-S1 (SAR) and BigEarthNet-S2 (optical) into a zip you can
upload to Kaggle.

## Why both S1 and S2?

SIH26167 needs optical **and** SAR. BigEarthNet provides matched pairs:

| Archive | Sensor | What we make |
|---|---|---|
| **BigEarthNet-S2** | Sentinel-2 optical | `images_s2/*.png` from B04/B03/B02 |
| **BigEarthNet-S1** | Sentinel-1 SAR | `images_s1/*.png` from VV/VH false-color |
| `metadata.parquet` | links them | `patch_id` (S2) ↔ `s1_name` (S1) + land-cover labels |

Each patch becomes RSVQA-style Q&A pairs for both modalities.

## One third of ~66 GB

| Thing | Approx size |
|---|---|
| Full S2 + S1 archives | tens of GB each (together often cited ~66 GB+ class) |
| Extract ~1/3 of tile folders from **both** | ~20–25 GB raw on disk |
| After this script (224×224 PNGs + JSONL) | usually **~2–6 GB** for Kaggle |

**Best approach:** extract the **same ~1/3 of tiles** from S1 and from S2, then
run with `--fraction 1.0 --require-s1`.

## Laptop steps

### 1. Download from https://bigearth.net/
- `BigEarthNet-S2`
- `BigEarthNet-S1`
- `metadata.parquet`

Layout example:

```text
D:\ben\
  BigEarthNet-S2\
  BigEarthNet-S1\
  metadata.parquet
```

### 2. Install (CPU only, once)

```powershell
cd ml\dataset_builder
pip install -r requirements.txt
```

### 3. Build (S1 + S2, one third)

```powershell
python build_bigearthnet_vqa.py `
  --images-s2 D:\ben\BigEarthNet-S2 `
  --images-s1 D:\ben\BigEarthNet-S1 `
  --metadata D:\ben\metadata.parquet `
  --out D:\ben\ben_vqa_third `
  --fraction 0.33 `
  --require-s1
```

If you already extracted only ~1/3 of matching tiles:

```powershell
  --fraction 1.0 --require-s1
```

### 4. Output

```text
ben_vqa_third\
  train.jsonl
  images_s2\
  images_s1\
  manifest.json
```

### 5. Zip for Kaggle

```powershell
cd D:\ben\ben_vqa_third
Compress-Archive -Path train.jsonl,images_s2,images_s1,manifest.json `
  -DestinationPath D:\ben\ben_vqa_third.zip
```

Upload as a Kaggle Dataset.

## RSVQA (test only, separate)

Download RSVQA-LR **test** from https://rsvqa.sylvainlobry.com/ and upload as
a second Kaggle Dataset. Do not mix into training.

## After this → train on Kaggle

See `../kaggle_unsloth_train.md` for copy-paste Unsloth notebook cells.
