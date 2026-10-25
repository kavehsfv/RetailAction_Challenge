# RetailAction Challenge @ WACV 2027 — Official Submission Repository (`0.7942` Public LB)

## Authors & Affiliation

- **Authors:** Kaveh Safavigerdini, Kannappan Palaniappan
- **Affiliation:** EECS Department, University of Missouri, Columbia, MO, USA
- **Competition:** RetailAction Challenge (PRAW Workshop @ WACV 2027) — Top-View Spatio-Temporal Action Localization
- **Final Public Leaderboard Score:** **`0.7942`** (Local Validation `mAP`: **`0.8541`**, Local Test `mAP`: **`0.8420`**)

For the complete scientific description of our method (model architecture, training data, progressive multi-stage training recipe, and post-processing/ensembling), see **[`METHOD_REPORT.md`](METHOD_REPORT.md)**.

---

## 1. Repository Structure

```text
RetailAction_WACV2027_Submission/
├── README.md                        # Step-by-step reproduction & training guide
├── METHOD_REPORT.md                 # Short Markdown report for the WACV 2027 Challenge Paper
├── Dockerfile                       # Containerized environment for exact reproduction
├── requirements.txt                 # Python dependencies
├── reproduce_submission.py          # One-command inference & verification entrypoint
├── train_full_pipeline.sh           # Full multi-stage training & V4 fine-tuning script
├── weights/                         # Trained V4 checkpoints (256x256 and 288x288)
│   ├── v4_control256_best.pt        # Full 256px (grid_hw=8) checkpoint (RAW + EMA + optimizer)
│   ├── v4_control256_inference.pt   # Slim 256px inference checkpoint (RAW + EMA, 323.5 MB)
│   ├── v4_high288_best.pt           # Full 288px (grid_hw=9) checkpoint (RAW + EMA + optimizer)
│   └── v4_high288_inference.pt      # Slim 288px inference checkpoint (RAW + EMA, 323.5 MB)
├── src/retailaction/                # Core model, dataset, loss, fusion, and training package
│   ├── data.py                      # Dual-view video + 32-frame 2D hand-track loader
│   ├── model.py                     # MultiViewActionDETR + HandTrackEncoder + obj/iou/dense heads
│   ├── loss.py                      # Per-layer fresh Hungarian matcher + multi-task criterion
│   ├── engine.py                    # Slot-aligned 4-pass TTA, ModelEma, and evaluation loop
│   ├── fuse.py                      # Slot-aligned multi-pass / multi-model fusion
│   ├── postprocess.py               # Spatio-temporal duplicate merging utilities
│   ├── metrics.py                   # Official 10x10 threshold mAP evaluator
│   ├── predict.py                   # CLI prediction & CSV serialization utilities
│   ├── train.py                     # Training CLI with AdamW wd-hygiene & grad checkpointing
│   └── util.py                      # Anatomical scale calibration, temporal DIoU, 3D sin-cos PE
├── scripts/                         # Provenance, validation selection, and contract test suites
│   ├── eval_v4.py                   # Full validation / local test evaluator with cache provenance
│   ├── select_v4.py                 # Pre-test validation selection & config freeze
│   ├── submit_v4.py                 # Frozen local-test confirmation & submission generator
│   ├── verify_v4_artifacts.py       # Independent CSV & manifest integrity verifier
│   ├── test_eval_contracts.py       # Unit tests for metric & ensemble contracts
│   └── test_v4_provenance.py        # Unit tests for provenance & CSV formatting
└── evidence/                        # Frozen configuration, validation/test metrics, and reference CSV
    ├── selected_config.json         # Frozen V4 ensemble configuration
    ├── validation_results.json      # Full validation split (1,277 clips) results
    ├── test_results.json            # Full local test split (2,501 clips) results
    ├── submission_manifest.json     # manifest for the 0.7942 submission
    └── reference_submission_v4_0.7942.csv
```

---

## 2. Trained Weights

Both V4 checkpoints contain the **RAW** (`"model"`) and **EMA** (`"model_ema"`) state dictionaries used by the 4-member ensemble (`c_ema`, `c_raw`, `h_ema`, `h_raw`). We provide both the full training checkpoints (`644.4 MB` each, including `AdamW` optimizer state) and lightweight inference-only checkpoints (`323.5 MB` each, with identical `"model"` and `"model_ema"` tensors):

| File | Resolution (`grid_hw`) | Size |
|---|---:|---:|
| `weights/v4_control256_best.pt` | `256×256` (`8×8`) | `644.4 MB` |
| `weights/v4_control256_inference.pt` | `256×256` (`8×8`) | `323.5 MB` |
| `weights/v4_high288_best.pt` | `288×288` (`9×9`) | `644.4 MB` |
| `weights/v4_high288_inference.pt` | `288×288` (`9×9`) | `323.5 MB` |

**Download the trained weights** (they are not stored in the repository):

1. Download the zip archive from Google Drive: <https://drive.google.com/file/d/1RXN2al-A_ycb381IqEHMPovggk4NlwM1/view?usp=sharing>
2. Unzip it so the checkpoints land in a `weights/` folder beside the cloned files:

```bash
unzip RetailAction_weights.zip -d weights/
# resulting layout:
#   weights/v4_control256_best.pt
#   weights/v4_control256_inference.pt
#   weights/v4_high288_best.pt
#   weights/v4_high288_inference.pt
```

---

## 3. Environment Setup & Dataset Layout

### 3.1 Python Environment
```bash
conda create -n retailaction python=3.11 -y
conda activate retailaction
pip install -r requirements.txt
```
*(Alternatively, build the included Docker image: `docker build -t retailaction-v4 .`)*

### 3.2 Dataset Layout
Place or symlink the official RetailAction Challenge dataset at `data/raw/`:
```text
data/raw/
├── sample_submission.csv
├── annotations/
│   ├── annotations_train.json
│   ├── annotations_validation.json
│   └── annotations_test.json
└── videos/
    ├── train/
    ├── validation/
    ├── test/
    └── challenge/
```

---

## 4. Reproducing the `0.7942` Kaggle Submission

Run the standalone entrypoint from the repository root:

```bash
python reproduce_submission.py --data-root data/raw --out submission.csv
# The reproduced submission.csv is byte-identical to combine_ideas_v4_0.7942.csv
```

What this command executes:
1. Builds synchronized dual-view challenge DataLoaders (`2,500` clips, `frames=16`, `common_size=288`) for `frame_shift=0` (`s0`, `flip`) and `frame_shift=1` (`s1`, `flip1`).
2. Sequentially runs all 4 ensemble members (`v4_control256` EMA, `v4_control256` RAW, `v4_high288` EMA, `v4_high288` RAW) across the 4 TTA passes (`s0,s1,flip,flip1`) with calibrated `obj*q` scoring (`16` passes per clip total).
3. Averages per-slot temporal bounds, 2D coordinates, and `obj*q` scores with equal `1:1:1:1` weights, then applies symmetric post-ensemble temporal boundary padding (`pad_t = 0.015`) once.
4. Writes `submission.csv` (`2,500` rows, `12,500` actions) in the exact clip ID order of `data/raw/sample_submission.csv` and runs automatic format and reference verification.

To also run using the lightweight inference-only checkpoints (`323.5 MB` each):
```bash
python reproduce_submission.py \
  --data-root data/raw \
  --ckpt-256 weights/v4_control256_inference.pt \
  --ckpt-288 weights/v4_high288_inference.pt \
  --out submission.csv   # rename/copy to combine_ideas_v4_0.7942.csv for Kaggle upload
```

To evaluate the exact same 4-member ensemble on the labeled **validation** (`1,277` clips, expected `mAP = 0.854122`) or **local test** (`2,501` clips, expected `mAP = 0.841961`) splits:
```bash
python reproduce_submission.py --data-root data/raw --split validation
python reproduce_submission.py --data-root data/raw --split test
```

---

## 5. Reproducing Training From Scratch or Final V4 Stage

All training stages are automated in `train_full_pipeline.sh`:

```bash
# Run the complete progressive curriculum from Kinetics-400 initialization through V4:
bash train_full_pipeline.sh all

# Or run only the final paired V4 256px + 288px continuations from Stage 5b (combine_v3b_256):
bash train_full_pipeline.sh final_v4
```

You can also run the CPU verification test suites at any time:
```bash
python scripts/test_eval_contracts.py
python scripts/test_v4_provenance.py
python reproduce_submission.py --verify-only --out submission.csv
```

---

## 6. Citation & Contact

- **Kaveh Safavigerdini** — EECS Department, University of Missouri
- **Kannappan Palaniappan** — EECS Department, University of Missouri
