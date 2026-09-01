# Detecting Unhealthy Salt Marsh Banks from Drone Imagery

**Final Report — EPA Salt Marsh Monitoring Project**

*[TODO: report author(s), affiliations, date, EPA program identifier, contract/grant number]*

---

## 1. Executive Summary

> **DRAFT NOTE:** Written last, once results are in. Placeholder structure:
> - What we set out to do (one sentence)
> - How we did it (one sentence: two-stage cascade on drone imagery across 9 MA marshes)
> - What we found (headline metrics: median unhealthy recall, precision, F1 across marshes)
> - What it enables operationally (field ecologists inspect a targeted subset instead of every drone tile)
> - What's next (cross-site generalization, operationalization)

---

## 2. Introduction

> **TODO:** Adapt the original project proposal wording. Key threads to develop:
>
> - **Why salt marshes matter.** Coastal protection, carbon sequestration, nursery habitat, ecosystem services.
> - **Why bank condition specifically.** Bank erosion is a leading indicator of marsh loss; unhealthy (eroding) banks predict where the marsh edge is receding and where restoration or intervention is most urgent. This is the paragraph that preempts the "who cares about unhealthy banks" objection — the audience needs to leave this section agreeing that identifying unhealthy banks is a management priority, not an academic curiosity.
> - **Why current monitoring falls short.** Field surveys are labor-intensive and don't scale to the linear extent of coastline that managers need to cover. Manual review of drone imagery is possible but time-consuming when a single flight produces thousands of tiles.
> - **What this project delivered.** A per-marsh classification pipeline that flags likely unhealthy bank tiles for targeted field inspection, tested across 9 Massachusetts coastal marshes.
>
> Aim for one to two pages. The proposal wording likely already makes the ecological/management case — pull the strongest paragraph(s) from there and adapt.

---

## 3. Study Area and Data

### 3.1 Sites

We applied the pipeline to nine salt marshes along the Massachusetts coast.

> **TODO:** List the 9 marshes with brief characterization. Suggested table:
>
| Marsh | Flight date | Notes |
|-------|----------|-------------|
| Old Town Hill | 27apr21 | low tide |
| Old Town Hill | 27apr21 | low tide |
| Old Town Hill | 27apr21 | low tide |
| Old Town Hill | 27apr21 | low tide |
| Old Town Hill | 27apr21 | low tide |
| Old Town Hill | 27apr21 | low tide |
| Old Town Hill | 27apr21 | low tide |
| Old Town Hill | 27apr21 | low tide |
| Old Town Hill | 27apr21 | low tide |


### 3.2 Imagery

*[TODO: describe sensor(s) used — RGB from onboard camera, DEM from photogrammetry or lidar, or both. Include ground sampling distance / tile resolution.]*

Elevation data from each site was rendered as RGB using the `terrain` colormap so that a standard three-channel CNN could consume it. Per-tile robust normalization (2nd/98th percentile) was applied before colormapping to maximize visual contrast of local elevation features. Tiles with no dynamic range (constant elevation, e.g., open water) or with too few valid pixels were dropped at extraction time.

### 3.3 Tiling

Each marsh raster was tiled into 299 × 299 pixel patches at *[TODO: physical tile size in meters — 10m from earlier context but confirm]* ground resolution. Tiles are indexed by their (row, column) position in the marsh grid.

### 3.4 Labeling

The source labels use a five-class scheme:

*[TODO: enumerate the 5 raw classes. Best guess based on typical marsh work: healthy bank, unhealthy (eroding) bank, plus three non-bank categories such as vegetation, water/mudflat, and something else. Confirm the actual scheme.]*

For modeling, these were collapsed into two binary problems:

- **Bank detection** (Stage 1): {bank, non-bank}, where "bank" combines the healthy and unhealthy bank classes.
- **Bank condition** (Stage 2): {healthy, unhealthy}, evaluated on tiles that are actually banks.

*[TODO: describe labeling protocol — who labeled, how disagreements were resolved, any QC pass.]*

---

## 4. Approach: Two-Stage Cascade

The core modeling task — "which tiles show unhealthy bank?" — is naturally a three-class problem (healthy bank, unhealthy bank, everything else). We split it into two sequential binary problems instead:

- **Stage 1** decides whether each tile shows a bank at all.
- **Stage 2** decides whether tiles Stage 1 identified as banks are healthy or unhealthy.

The rationale is that these are visually distinct problems. Distinguishing a bank from open water or vegetation depends on gross morphology (linear features, edge transitions, elevation gradients). Distinguishing an eroding bank from an intact one depends on finer texture cues (undercut edges, sediment slumping, vegetation loss). Training a single three-class model forces one network to learn both at once; splitting the problem lets each stage specialize on the visual features that matter for its decision.

The cost of this design is that overall cascade behavior is not directly measured by either stage's evaluation. A tile can be lost at Stage 1 (a real bank mistaken for something else) or at Stage 2 (a bank correctly identified but misclassified as healthy). We describe how we account for both loss modes in Section 6.

### Per-marsh models

We trained separate Stage 1 and Stage 2 models for each of the 9 marshes rather than a single model applied across sites. Marshes differ substantially in bank morphology, dominant vegetation, tide state at flight time, and lighting — a site-specific model captures these local characteristics without the difficulty of learning them jointly. The trade-off is that we have not demonstrated cross-site generalization; a model trained on one marsh applied cold to another is untested (see Limitations, Section 8).

---

## 5. Model Architecture and Training

Both Stage 1 and Stage 2 use the same Inception-style CNN architecture, differing only in their training data and operating threshold.

### 5.1 Architecture

The network is a modified Inception design with four inception blocks of increasing filter count (96, 192, 384, 768), interleaved with max-pooling and dropout layers. Each inception block runs four parallel branches (1×1, 3×3, 5×5 convolutions, and a max-pool + 1×1 branch) whose outputs are concatenated. Activations are ELU throughout, with L2 weight regularization (λ = 1e-4). After the final inception block, global average pooling feeds two fully-connected layers (2048 and 1024 units), each with batch normalization and 0.5 dropout. The output is a single sigmoid unit for binary classification.

Input images are 299 × 299 × 3.

### 5.2 Preprocessing

*[TODO: confirm which stages use which preprocessing. From the training code:]*

- **Stage 1** applies image sharpening before training and inference.
- **Stage 2** does not use sharpening.
- Both stages apply on-the-fly augmentation (approximately 15 augmentations per training image) and normalize pixel values to [-1, 1] as the final preprocessing step.

### 5.3 Training

- Optimizer: Adam, learning rate 5×10⁻⁴, gradient clipping at norm 1.0
- Loss: binary cross-entropy with label smoothing 0.1
- Callbacks: early stopping on validation loss (patience 15, restore best weights); ReduceLROnPlateau (factor 0.5, patience 7, min LR 1×10⁻⁶)
- Maximum 150 epochs

### 5.4 Operating thresholds

- **Stage 1**: threshold chosen per marsh to maximize F1 on the marsh's validation set.
- **Stage 2**: *[TODO: state the Stage 2 threshold policy — consistent across marshes? if so, what value, and how chosen?]*
- **Test-time augmentation (TTA)**: used at Stage 2.

---

## 6. Evaluation Framework

Because the pipeline is a cascade, per-stage metrics do not directly describe end-to-end behavior. This section explains how we compose stage-level measurements into cascade-level metrics.

### 6.1 Data splits

For each marsh, tiles were split 70/15/15 into training, validation, and test sets. The split uses stratified sampling on **spatial components** rather than individual tiles — a critical detail described in the next subsection. A fixed random seed was used across all marshes for reproducibility.

Per-marsh, per-class split counts are given in Appendix A.

### 6.2 Spatial splitting

A common failure mode in remote-sensing ML is spatial leakage: when adjacent tiles are randomly assigned to train and test, the model isn't really being tested on new terrain. It's being tested on terrain it has effectively already seen. Reported metrics under random splitting can substantially overstate real-world performance on unseen sites.

To avoid this, we split by **spatial component** rather than by individual tile. Two tiles are treated as belonging to the same component if they share a class label and are directly adjacent (up, down, left, or right — 4-connected). Components are identified via breadth-first search over the tile grid, then assigned as indivisible units to train, validation, or test splits using stratified sampling on component class labels. Every same-class contiguous region of the marsh therefore ends up wholly in one split — a model is never tested on a tile whose neighbor it trained on.

The trade-off is that stratification becomes approximate. A marsh dominated by a few very large components has fewer "units" available to balance across splits, so class ratios in train, validation, and test can drift from the target. In practice this is a small cost compared to the alternative of inflated metrics.

*[TODO: add a small figure — a grid example showing tiles, components, and split assignment. Also confirm whether components are formed on 2-class {bank, non-bank} or the raw 5-class labels; the granularity implication should be stated.]*

We verified the spatial constraint held on every marsh (no adjacent same-class tiles ended up in different splits).

Full algorithmic details are in Appendix B.

### 6.3 Cascade metrics

At each stage we report the standard binary classification metrics (precision, recall, F1) on that stage's test set. To compute end-to-end performance for the unhealthy-bank detection task, we compose these:

- **Cascade recall on unhealthy** = (Stage 1 recall on unhealthy tiles) × (Stage 2 recall on unhealthy tiles). An unhealthy tile is caught only if both stages classify it correctly.
- **Cascade precision on unhealthy** = TP / (TP + FP), where TP counts true unhealthy tiles Stage 2 called unhealthy, and FP counts everything Stage 2 called unhealthy that was not — from either healthy banks (Stage 2 misclassifications) or non-banks that Stage 1 forwarded incorrectly.

The false positives from non-banks require a separate measurement, described next.

### 6.4 Stage 2 behavior on non-banks

Stage 2 was trained on banks only, so its behavior on non-bank tiles that leak through Stage 1 must be characterized separately. We measured this on a held-out ballast set — non-bank tiles that Stage 2 was never exposed to in training, validation, or test. The measured false-positive rate on this set (typically 0.14–0.20 across marshes) is used in the cascade computation to compute end-to-end precision.

*[TODO: confirm the 0.14–0.20 range holds across all 9 marshes; report per-marsh values in the appendix. Also note per-marsh ballast set size (matters for confidence in the estimate).]*

### 6.5 Loss attribution

For each marsh we report a stage-by-stage attrition table showing how many unhealthy bank tiles are lost at each stage. This makes clear whether the cascade's misses are dominated by Stage 1 (bank detection failing) or Stage 2 (condition classification failing) — useful for understanding where future improvement effort should go. Attribution tables appear in the per-marsh appendix (Appendix C).

---

## 7. Results

*[Draft paragraph — fill in with real values once per-marsh runs complete]*

Across the 9 marshes, the cascade caught between **[min]%** and **[max]%** of unhealthy bank tiles (median **[X]%**). Precision on unhealthy-bank detection ranged from **[min]** to **[max]** (median **[Y]**). A field ecologist working from the model's outputs would examine roughly **[Z]** tiles per marsh instead of the full drone survey, without missing significant erosion sites.

| Marsh | Test tiles | Unhealthy recall | Unhealthy precision | F1 |
|-------|-----------:|-----------------:|--------------------:|---:|
| *[Marsh 1]* | | | | |
| *[Marsh 2]* | | | | |
| *[Marsh 3]* | | | | |
| *[Marsh 4]* | | | | |
| *[Marsh 5]* | | | | |
| *[Marsh 6]* | | | | |
| *[Marsh 7]* | | | | |
| *[Marsh 8]* | | | | |
| *[Marsh 9]* | | | | |
| **Median** | — | | | |

Test tiles counts all classes (healthy banks, unhealthy banks, non-banks). Per-marsh confusion matrices, thresholds, and attrition tables are in Appendix C.

---

## 8. Limitations and Future Work

### No cross-site generalization tested
We trained one Stage 1 and one Stage 2 model per marsh. We have not tested how a model trained on marsh A performs when applied cold to marsh B. Operational deployment across new sites would benefit from that evaluation — and if generalization proves poor, from either a joint multi-site model or a fine-tuning protocol for new marshes.

### Stage 1 non-bank sample sizes
For some marshes, the number of non-bank tiles in the test set is small, so the measured Stage 1 false-positive rate on non-banks (and by extension the cascade precision estimate) has wide uncertainty. Larger non-bank evaluation sets would tighten this.

### Stage 2 non-bank behavior estimate
The held-out ballast used to characterize Stage 2's behavior on non-banks is drawn from the same marsh Stage 2 was trained on. While these specific tiles were never seen by Stage 2, they share spatial context with the training set. A fully independent non-bank evaluation (e.g., drawn from a held-out region of the marsh) would give a stricter estimate.

### Temporal variation not tested
Each marsh was flown once. Bank appearance varies with tide state, season, and post-storm conditions. The pipeline's robustness to these factors is not characterized.

### Drone flight logistics
The pipeline assumes drone imagery is available. Weather, permitting, and operator availability constrain flight frequency in practice.

---

## 9. Conclusion

*[TODO: written last. Placeholder points:]*

- The two-stage cascade produces useful unhealthy-bank flags on all 9 marshes tested.
- Performance is consistent enough across sites that per-marsh training is a viable operational pattern, at least until cross-site generalization is characterized.
- The spatial-splitting protocol we adopted gives conservative, defensible performance estimates that reflect real generalization to unseen bank sections rather than memorization of adjacent pixels.
- Field ecologists can now allocate inspection effort to a targeted subset of drone tiles rather than reviewing full surveys manually.

---

## Appendix A — Per-Marsh Data Split Counts

| Marsh | Train (H / U / N) | Val (H / U / N) | Test (H / U / N) |
|-------|-------------------|-----------------|------------------|
| *[Marsh 1]* | | | |
| *[Marsh 2]* | | | |
| ... | | | |

H = healthy banks, U = unhealthy banks, N = non-banks.

---

## Appendix B — Spatial Splitting Algorithm

The spatial-component-based splitter proceeds in four steps.

**Step 1: Build connected components.** Two tiles are treated as connected if they are directly adjacent (up, down, left, or right — 4-connectivity, not 8-connectivity) and share the same class label. Using breadth-first search, we identify all maximal connected components of same-class tiles.

Example. Consider a 3 × 4 grid of tiles with class labels:

```
        col 0   col 1   col 2   col 3
row 0:    0       0       1       1
row 1:    0       1       1       1
row 2:    1       1       0       0
```

This produces three components:
- Component A (class 0): three tiles at (0,0), (0,1), (1,0)
- Component B (class 1): seven tiles at (0,2), (0,3), (1,1), (1,2), (1,3), (2,0), (2,1)
- Component C (class 0): two tiles at (2,2), (2,3)

**Step 2: Label each component.** Each component inherits its class label (all tiles within a component share the same class by construction).

**Step 3: Split components (not tiles).** Components are treated as indivisible units. Scikit-learn's `train_test_split` is called with stratification on component labels. A two-step split separates train from (val + test) first, then val from test.

**Step 4: Expand components back to tiles.** Once components are assigned to splits, they are flattened to individual tiles for training.

### Why this works

The key insight is that stratification happens at the component level, not the tile level. Since components are indivisible:

- **Spatial constraint is guaranteed** — no adjacent same-class tiles can be split because they're in the same component.
- **Stratification is approximate** — we balance class labels as well as possible given the constraint that components can't be broken.

A verification step (`verify_spatial_constraints`) explicitly checks that no same-class adjacent tiles ended up in different splits. This check passed for every marsh.

### The trade-off in practice

With a modest number of tiles, a few very large components can dominate the split. For instance, a marsh might have:

- One large healthy component of 200 tiles
- One large unhealthy component of 150 tiles
- Fifty small mixed components of 3 tiles each

That leaves only ~52 "units" to split across train/val/test, and two of them dominate. If the large healthy component goes to train and the large unhealthy component goes to test, the class balance across splits will be visibly off. This is the cost of the spatial constraint. Empirically, on our marshes, this drift is modest — see per-marsh split counts in Appendix A.

---

## Appendix C — Per-Marsh Detailed Results

*Template for each marsh — reproduce for all 9.*

### C.1 [Marsh name]

**Test set composition**

| Class | Count |
|-------|------:|
| Non-banks | *[N]* |
| Healthy banks | *[N]* |
| Unhealthy banks | *[N]* |
| **Total** | *[N]* |

**Stage 1 — bank vs. non-bank**

Operating threshold: *[X]* (F1-optimal on validation)

|                     | Predicted non-bank | Predicted bank |
|---------------------|:------------------:|:--------------:|
| **Actual non-bank** |         *[N]*      |     *[N]*      |
| **Actual bank**     |         *[N]*      |     *[N]*      |

- Precision: *[X]* · Recall: *[X]* · F1: *[X]* · Accuracy: *[X]*
- Of the tiles missed by Stage 1: *[N]* were unhealthy (lost to the cascade at this stage), *[N]* were healthy.
- Of the tiles caught by Stage 1: *[N]* were unhealthy, *[N]* were healthy.

**Stage 2 — healthy vs. unhealthy (with TTA)**

Operating threshold: *[X]*

|                      | Predicted healthy | Predicted unhealthy |
|----------------------|:-----------------:|:-------------------:|
| **Actual healthy**   |       *[N]*       |        *[N]*        |
| **Actual unhealthy** |       *[N]*       |        *[N]*        |

- Precision: *[X]* · Recall: *[X]* · F1: *[X]* · Accuracy: *[X]*
- Stage 2 FPR on non-banks (measured on held-out ballast, N=*[N]*): *[X]*

**Cascade — end-to-end unhealthy detection**

- Recall: *[X]*
- Precision: *[X]*
- F1: *[X]*

**Where unhealthy banks were lost**

| Stage    | Started | Lost | Miss rate | Passed |
|----------|--------:|-----:|----------:|-------:|
| Stage 1  |   *[N]* | *[N]* |   *[X]%*  |  *[N]* |
| Stage 2  |   *[N]* | *[N]* |   *[X]%*  |  *[N]* |
| **Total**|   *[N]* | *[N]* |   *[X]%*  |  *[N]* |

*[Brief interpretation paragraph: where the losses concentrated for this marsh, whether the pattern suggests a Stage 1 or Stage 2 problem, any notable observations.]*

---

*[Repeat C.1 template for each of the remaining 8 marshes.]*
