# Build BigEarthNet S1+S2 → RSVQA-style training data (Mac)

This folder is **separate** from model training. Use it on a Mac to turn
**both** BigEarthNet-S1 (SAR) and BigEarthNet-S2 (optical) into a Kaggle
dataset. You do **not** have to zip the result yourself.

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
| Full S2 + S1 archives | tens of GB each |
| Extract ~1/3 of tile folders from **both** | ~20–25 GB raw on disk |
| After this script (224×224 PNGs + JSONL) | usually **~2–6 GB** |

**Best approach:** extract the **same ~1/3 of tiles** from S1 and from S2, then
run with `--fraction 1.0 --require-s1`.

---

## Mac setup

Need **Python 3.10+**. On a fresh Mac:

```bash
# if python3 is missing:
brew install python

python3 --version
```

Use `python3` and `pip3` (not `python` / `pip`) unless you created a venv.

---

## Laptop steps

### 1. Download from https://bigearth.net/

- `BigEarthNet-S2`
- `BigEarthNet-S1`
- `metadata.parquet`

Suggested layout (home folder — change the username if needed):

```text
/Users/you/ben/
  BigEarthNet-S2/
  BigEarthNet-S1/
  metadata.parquet
```

Extract archives with Finder (double-click) or:

```bash
cd ~/ben
# example if you downloaded .tar.gz / split archives — use whatever you actually have
# tar -xf BigEarthNet-S2.tar.gz
# tar -xf BigEarthNet-S1.tar.gz
```

### 2. Install (CPU only, once)

From the repo:

```bash
cd ~/satquery/ml/dataset_builder
# or: cd "/path/to/satquery/ml/dataset_builder"

python3 -m pip install -r requirements.txt
```

### 3. Build (S1 + S2, one third)

```bash
python3 build_bigearthnet_vqa.py \
  --images-s2 ~/ben/BigEarthNet-S2 \
  --images-s1 ~/ben/BigEarthNet-S1 \
  --metadata ~/ben/metadata.parquet \
  --out ~/ben/ben_vqa_third \
  --fraction 0.33 \
  --require-s1
```

If you already extracted only ~1/3 of matching tiles:

```bash
python3 build_bigearthnet_vqa.py \
  --images-s2 ~/ben/BigEarthNet-S2 \
  --images-s1 ~/ben/BigEarthNet-S1 \
  --metadata ~/ben/metadata.parquet \
  --out ~/ben/ben_vqa_third \
  --fraction 1.0 \
  --require-s1
```

Optional smaller first run: add `--max-patches 6000`.

### 4. Output

```text
~/ben/ben_vqa_third/
  train.jsonl
  images_s2/
  images_s1/
  manifest.json
```

Image paths inside `train.jsonl` are **relative** (`images_s2/...`, `images_s1/...`),
so they work on Mac, Windows, and Kaggle the same way.

---

## Upload to Kaggle — skip making a zip

**Yes, you can upload the folder as-is.** You do not need `zip` or
`Compress-Archive`.

### Recommended: Kaggle CLI (folder in, no local zip)

1. On Kaggle: **Settings → API → Create New Token** → you get `kaggle.json`
2. On the Mac:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
python3 -m pip install kaggle
```

3. Copy the metadata file into the dataset folder and create the dataset:

```bash
cp dataset-metadata.json ~/ben/ben_vqa_third/dataset-metadata.json
# edit title / id (must be yourusername/dataset-slug, all lowercase)

kaggle datasets create -p ~/ben/ben_vqa_third --dir-mode zip
```

`--dir-mode zip` means: **the CLI packs the folder for the upload**. You never
create `ben_vqa_third.zip` yourself. After Kaggle processes it, a notebook sees
the **unzipped** tree:

```text
/kaggle/input/ben-vqa-third/train.jsonl
/kaggle/input/ben-vqa-third/images_s2/...
/kaggle/input/ben-vqa-third/images_s1/...
```

To update later:

```bash
kaggle datasets version -p ~/ben/ben_vqa_third --dir-mode zip -m "one-third S1+S2"
```

### Also fine: Kaggle website, one folder

**Datasets → New Dataset → Upload** and drop the `ben_vqa_third` folder
(or select those four items). Do **not** pick every PNG by hand.

---

## Are there technical problems if we do not zip ourselves?

| Approach | Verdict |
|---|---|
| **CLI `--dir-mode zip`** (folder on disk, CLI packs it) | **Best.** Same as a zip upload, without you compressing first. |
| **Web UI: upload the folder** | OK if the folder is a few GB and you drop the **parent folder**, not 20k+ files one by one. |
| **Web UI: select thousands of PNGs** | **Avoid.** Browsers time out; Kaggle’s uploader struggles with huge file counts. |
| **Keep a local .zip** | Works, but extra disk and an extra unzip step if you upload the zip as a **single file** (then the notebook sees `something.zip` instead of `train.jsonl`). |

Real constraints to know:

1. **File count** — one third of BigEarthNet can mean **tens of thousands of PNGs**. The website upload of individual files is the failure mode. Folder/CLI upload is fine.
2. **Size** — Kaggle datasets allow large uploads (many GB). A 2–6 GB converted pack is normal. Stay on a stable network; resume if the CLI drops.
3. **Mac extras** — Finder can add `.DS_Store`. Harmless. Do not upload the raw 20 GB GeoTIFF trees, only `ben_vqa_third`.
4. **Training** — Unsloth does **not** need a zip. It needs `train.jsonl` + image folders on `/kaggle/input/...`.
5. **Do not zip if you then upload that zip as the only file** unless you unzip it in the notebook. Folder/CLI upload avoids that extra step.

So: skip local compression. Upload the output folder with the CLI (or the website folder picker).

---

## RSVQA (test only, separate)

Download RSVQA-LR **test** from https://rsvqa.sylvainlobry.com/ and upload as
a second Kaggle Dataset. Do not mix into training.

## After this → train on Kaggle

See `../kaggle_unsloth_train.md` for copy-paste Unsloth notebook cells.
