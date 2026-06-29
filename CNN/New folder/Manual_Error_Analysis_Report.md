# Manual Error Analysis & Bias/Variance Report
### Rajuk Image-Level Classifier (ResNet18, 3 classes: negative / moderate / positive)

> Companion to `rajuk_cnn_improved_trained.ipynb`.
> Fill in the bracketed `[…]` fields with your group's actual names, numbers, and findings,
> then bring a **printed copy** to class.

---

## 0. How to generate the inputs for this report

1. Run the notebook top-to-bottom. The relevant cells are:
   - **Section 12** – fine-tune + evaluate (produces `val_preds`, `val_true`).
   - **Section 12b** – *Misclassified Validation Examples*:
     - exports **`val_misclassified.csv`** (one row per error, with blank columns to fill in),
     - saves **`val_misclassified.png`** (a labelled contact sheet of every wrong image).
2. Open `val_misclassified.csv` in Excel/Sheets. Each row already has an
   `assigned_member` column (≈20 errors per member — set `N_MEMBERS` in the notebook to
   your real group size).
3. Each member fills `reason_category`, `possible_reason`, and `suggested_fix` for their rows.

---

## 1. Current results (from the latest run — update with your numbers)

| Split | Loss | Accuracy | Notes |
|-------|------|----------|-------|
| Train (real + augmented) | ~0.01 | **~0.99** | Augmented, on-the-fly |
| Validation (real only)   | ~2.9  | **~0.54–0.67** | Real images only |
| Test (real only)         | ~2.3  | **~0.66** | Real images only |

**Key observation:** training accuracy ≈ 99 % while validation/test accuracy ≈ 55–67 %.
The ~33–45 % gap between train and val is the single most important signal in this report.

---

## 2. Manual Error Analysis

Each group member receives ~20 misclassified validation images (see `assigned_member`
column in `val_misclassified.csv`). For each image, record the **most likely reason** the
model got it wrong, using the category list below.

### 2.1 Reason categories (use these in the `reason_category` column)

| Code | Category | Description |
|------|----------|-------------|
| A | **Ambiguous / borderline** | Genuinely between two classes (e.g. moderate vs positive). |
| B | **Image quality** | Blur, low resolution, over/under-exposure, motion. |
| C | **Occlusion / framing** | Subject partly hidden, cropped, or off-centre. |
| D | **Lighting / colour** | Shadows, glare, colour cast not seen in (augmented) training data. |
| E | **Possible label error** | The ground-truth label looks wrong/inconsistent. |
| F | **Distribution mismatch** | Real image looks unlike the heavily-augmented training images. |
| G | **Class imbalance effect** | Minority-class (positive/moderate) image pushed to majority (negative). |
| H | **Other** | Anything not covered above (describe in `possible_reason`). |

### 2.2 Per-member sheet (one table per member)

**Member: [Name] — Errors assigned: [n]**

| # | val_index | image_id | True | Pred | Category (A–H) | Possible reason (1 line) |
|---|-----------|----------|------|------|----------------|--------------------------|
| 1 | […] | […] | […] | […] | […] | […] |
| 2 | […] | […] | […] | […] | […] | […] |
| … | | | | | | |

*(Copy the table for each member, or just print the filled-in `val_misclassified.csv`.)*

### 2.3 Per-member brief explanation + way forward

> Each member writes ~5–8 sentences based on their own sheet.

**Member [Name] — summary of findings:**
- Most common category in my 20 errors: **[e.g. F – distribution mismatch]** ([x]/20).
- Brief explanation: [why these images were hard; what they have in common].
- Way forward suggested by my findings: [e.g. reduce augmentation aggressiveness so real
  and training images look more alike; collect more real images of class positive; etc.].

### 2.4 Group-level error summary (aggregate all members' sheets)

| Category | Count | % of errors | Lead recommendation |
|----------|-------|-------------|---------------------|
| A Ambiguous | […] | […] | Better label guidelines |
| B Image quality | […] | […] | Quality filter / re-shoot |
| C Occlusion | […] | […] | More varied real data |
| D Lighting | […] | […] | Milder colour augmentation |
| E Label error | […] | […] | Re-label & clean dataset |
| F Distribution mismatch | […] | […] | Align train↔val distributions |
| G Class imbalance | […] | […] | More minority-class data / resampling |
| H Other | […] | […] | — |

**Dominant failure mode (fill in):** [e.g. categories E + F dominate → the model overfits
heavily-augmented training images and a few labels are noisy.]

---

## 3. Human-Level Performance (HLP)

We do **not** employ professional data labellers, so HLP is estimated two ways:

### 3.1 Group perception of task difficulty
- Distinguishing **negative** from **positive** is usually easy for a human (clear visual
  difference). The hard boundary is **moderate ↔ positive** and **moderate ↔ negative**.
- Based on our own labelling of a sample, we estimate a careful human gets roughly
  **[~88–92] %** accuracy on this 3-class task (≈8–12 % human error).

### 3.2 State-of-the-art reference
- Comparable building-condition / structural-damage / surface-crack classification papers
  report roughly **85–95 %** accuracy on curated datasets.
  *(Add the 2–3 specific papers your group found here, with citation and reported accuracy.)*
  - [Paper 1 — title, venue, year — reported accuracy]
  - [Paper 2 — title, venue, year — reported accuracy]

### 3.3 Adopted HLP for this project
> **Human-Level Performance ≈ [90] % accuracy  →  Bayes-error proxy ≈ [10] % error.**

We use this single number as the "best achievable" reference for the bias/variance analysis.

---

## 4. Bias vs Variance: what to fix first

### 4.1 Decomposition (using HLP ≈ 10 % error as the proxy for Bayes error)

| Quantity | Value | Computed as |
|----------|-------|-------------|
| Human-level error (proxy Bayes) | **~10 %** | from Section 3 |
| Training error | **~1 %** | 1 − 0.99 |
| Validation error | **~33–45 %** | 1 − 0.55…0.67 |
| **Avoidable bias** | **~ −9 % (≈ 0)** | train_err − human_err → training error is *below* human |
| **Variance** | **~32–44 %** | val_err − train_err |

### 4.2 Conclusion — variance reduction first

- **Avoidable bias is essentially zero** (training error ~1 % is already *better* than
  human-level). The model has more than enough capacity; it is **not** underfitting.
- **Variance is huge** (~32–44 % gap between train and val). This is the dominant problem.
- A large part of this "variance" is actually a **train/validation distribution mismatch**:
  training uses heavily-augmented images, while val/test are real-only. The aggressive
  augmentation (vertical flip, strong colour jitter, blur, grayscale, random-resized-crop)
  makes training images systematically different from the real validation images.

> **Decision: apply VARIANCE-REDUCTION techniques first, before any bias-reduction.**
> Bias reduction (bigger model, more layers unfrozen, less regularisation) would only make
> overfitting worse.

### 4.3 First method to try, and why

**First: reduce the train↔validation distribution mismatch + increase effective real data.**
Concretely, in priority order:

1. **Tone down augmentation** so training images resemble real ones
   (drop/limit `RandomVerticalFlip`, reduce `ColorJitter`, soften `GaussianBlur`,
   raise `RandomResizedCrop` lower-scale from 0.75 toward 0.9). *Why first:* it is free,
   directly targets the train/val mismatch, and the error sheets point to category **F**.
2. **Get more real images** (especially minority classes positive/moderate) and/or a
   "train-dev" set drawn from the *real* distribution to separate variance from mismatch.
3. **Stronger regularisation on the head** — keep dropout, keep early stopping (already on),
   optionally freeze `layer4` again so fewer parameters can memorise the augmented set.
4. **Re-check noisy labels** flagged as category **E** during manual error analysis.

*Rationale:* the dominant gap is variance/mismatch, not bias, so the cheapest high-impact
move is to make the training distribution look like the real validation distribution and to
add real data — only after the train/val gap closes do we revisit bias (e.g. unfreezing
more layers) if accuracy is still below HLP.

---

## 5. Changes already made to the notebook (this iteration)

- **Larger validation set:** real-subject split changed to `VAL_FRACTION = 0.25`,
  `TEST_FRACTION = 0.15` (was ~0.15/0.15) → roughly **doubles** the number of validation
  images so the manual error analysis has enough examples per member.
- **Fewer epochs / faster training:** `NUM_EPOCHS 60 → 25`, `PATIENCE 12 → 6`,
  `KFOLD_SPLITS 5 → 3`, learning-curve points `10 → 4` and `15 → 8` epochs per point.
- **New Section 12b:** automatically lists, exports (`val_misclassified.csv`), and plots
  (`val_misclassified.png`) every misclassified validation image, pre-assigned to members.

---

## 6. Action items

- [ ] Each member fills their rows in `val_misclassified.csv`.
- [ ] Aggregate into the Section 2.4 table.
- [ ] Finalise HLP number + add SOTA citations (Section 3).
- [ ] Implement variance-reduction step 1 (tone down augmentation) and re-measure the
      train/val gap.
- [ ] Print this report for class.
