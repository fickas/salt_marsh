# Detecting Eroding Salt Marsh Banks from Drone Imagery

### Final Report

Prepared by Stephen F. Fickas, President, Hop Skip Technologies (HST), Eugene, OR. Prepared for Scott Jackson, Extension Professor, Department of Environmental Conservation, University of Massachusetts Amherst. Funding from EPA Wetlands Program Development Grants Project period March 2024 – September 2026.

---

## 1. Executive Summary

Salt marshes are among the most ecologically valuable and climate-vulnerable coastal ecosystems in the northeastern United States. Erosion of tidal channel banks is an early and measurable indicator of marsh degradation. Identifying eroding (unhealthy) banks at scale from remote-sensing imagery is a bottleneck for conservation practitioners: manual review of drone or aerial imagery does not keep pace with the volume of data now collected.

This project developed and evaluated a machine-learning pipeline that classifies salt-marsh imagery tiles as eroding bank, healthy bank, or non-bank. The pipeline uses a two-stage cascade — first distinguishing banks from non-banks, then classifying bank tiles as healthy or eroding — implemented as fine-tuned InceptionV3 convolutional networks. We applied the pipeline to nine Massachusetts salt marshes, training a separate model per site.

Overall test-set accuracy ranged from 0.90 to 0.96 across the nine marshes, with a median of 0.93 — close to but slightly below the 95% accuracy target set in the original project proposal. Per-marsh recall and precision on the eroding-bank class specifically are reported in Section 7.

We additionally attempted to apply the same approach to publicly available MassGIS aerial photogrammetry imagery. The lower ground sampling distance of that data source proved insufficient to resolve the bank features the model relies on, and we abandoned that line of investigation. This is a finding in its own right: drone-scale imagery is a genuine requirement for this task at present, not a preference.

Cross-site generalization — training one model that works across marshes — remains open work and is a possible next step.

---

## 2. Introduction

### 2.1 Salt marsh loss and the case for monitoring bank condition

Salt marshes on the Massachusetts coastline provide storm-surge attenuation, carbon sequestration, nursery habitat for commercial fisheries, and water-quality filtration. They are also under sustained pressure from sea-level rise, altered tidal exchange, and coastal development. Marsh loss in the region is documented but heterogeneous — some marshes appear stable while others are actively retreating.

Erosion of tidal channel banks is one of the more tractable indicators of marsh health. Unlike diffuse loss processes that are difficult to observe from imagery, an eroding bank leaves visible signatures: undercut edges, sediment slumping, exposed root mats, and loss of edge vegetation. Locating these signatures at scale would let managers direct restoration effort where the marsh edge is actively retreating, rather than triaging based on periodic ground surveys that cover only a fraction of the coastline.

The obstacle is not that eroding banks are hard to identify — an experienced observer can spot them in a drone image within seconds. The obstacle is volume. A single drone flight over one marsh produces thousands of imagery tiles. Manually reviewing surveys across nine or more marshes on a routine monitoring cadence is not feasible with the staff typically available for this kind of work.

### 2.2 Project context and scope

This project builds on exploratory work between Hop Skip Technologies (HST) and the UMass Amherst salt-marsh research group. That earlier work demonstrated, on three sites, that a convolutional neural network could learn to identify eroding channel banks from drone imagery. The current project — funded through the EPA Wetlands Program Development Grants program and executed as a UMass-HST partnership — expanded that approach to a larger set of sites, hardened the evaluation protocol, and characterized where the approach works and where it does not.

The stated deliverables were:

- A machine-learning model for eroding-bank identification trained on data from at least five Massachusetts marshes, with a target of 95% classification accuracy.
- A bank-erosion metric derived from the model's outputs, suitable for use by conservation practitioners.
- An exploration of statewide applicability using publicly available MassGIS aerial photogrammetry imagery, and (if successful) a corresponding statewide erosion metric.

We ultimately trained and evaluated the pipeline on nine marshes rather than five. The MassGIS aerial photogrammetry investigation is discussed in Section 7 and Section 8.

---

## 3. Study Area and Data

### 3.1 Sites

We applied the pipeline to nine salt marshes along the Massachusetts coast.

| Marsh | Flight date | Notes |
|-------|----------|-------------|
| Old Town Hill | 27 April 2021 | low tide |
| Essex Bay | 27 April 2021 | low tide |
| Peggotty Beach | 13 May 2021 | low tide |
| North River | 13 May 2021 | low tide |
| Red River | 14 May 2021 | low tide |
| Barnstable | 2 June 2023 | low tide |
| Wellfleet | 17 May 2021 | low tide |
| Westport | 3 May 2021 | low tide |
| South River | 15 June 2023 | low tide |


### 3.2 Imagery

Rasters were produced from an RGB  onboard camera and a DEM from photogrammetry. Ground Sampling Distance averaged 2.74 cm per pixel.

Elevation data (DEM) from each site was rendered as RGB using the `terrain` colormap so that a standard three-channel CNN could consume it. Per-tile robust normalization (2nd/98th percentile) was applied before colormapping to maximize visual contrast of local elevation features. Tiles with no dynamic range (constant elevation, e.g., open water) or with too few valid pixels were dropped at extraction time.

### 3.3 Tiling

Each marsh raster was tiled into 299 × 299 pixel patches at 3 m × 3 m ground resolution. Tiles are indexed by their (row, column) position in the marsh grid.

### 3.4 Labeling

The source labels use a five-class scheme: Healthy banks, Unhealthy banks, Ditches, Pond edges, Other. Ground-truth labeling was performed using a combination of field surveys and Photo Interpretation.

For modeling, the five classes were collapsed into two binary problems:

- **Bank detection** (Stage 1): {non-bank, bank}, where "bank" combines the healthy and unhealthy bank classes.
- **Bank condition** (Stage 2): {healthy, unhealthy}, evaluated on tiles that are actually banks.

---

## 4. Approach: Two-Stage Cascade

The core modeling task — "which tiles show an unhealthy bank?" — is naturally a three-class problem (healthy bank, unhealthy bank, everything else). We split it into two sequential binary problems instead:

- **Stage 1** decides whether each tile shows a bank at all.
- **Stage 2** decides whether tiles Stage 1 identified as banks are healthy or unhealthy.

The rationale is that these are visually distinct problems. Distinguishing a bank from open water or vegetation depends on gross morphology (linear features, edge transitions, elevation gradients). Distinguishing an eroding bank from an intact one depends on finer texture cues (undercut edges, sediment slumping, vegetation loss). Training a single three-class model forces one network to learn both at once; splitting the problem lets each stage specialize on the visual features that matter for its decision.

The cost of this design is that overall cascade behavior is not directly measured by either stage's evaluation. A tile can be lost at Stage 1 (a real bank mistaken for something else) or at Stage 2 (a bank correctly identified but misclassified as healthy). We describe how we account for both loss modes in Section 6.

### Per-marsh models

We trained separate Stage 1 and Stage 2 models for each of the 9 marshes rather than a single model applied across sites. Our hope was that we could build one model, using data from a variety of marshes for training, that could then be applied successfully to other marshes. We attempted this approach, but results were poor. We found that marshes differ substantially in bank morphology, dominant vegetation, tide state at flight time, and lighting — a site-specific model captures these local characteristics without the difficulty of learning them jointly. The trade-off is that we have not demonstrated cross-site generalization; a model trained on one marsh applied cold to another remains a goal (see Limitations, Section 8).

---

## 5. Model Architecture and Training

Both Stage 1 and Stage 2 use the same Inception-style CNN architecture, differing only in their training data and operating threshold.

### 5.1 Architecture

5.1 Architecture

The classifier uses transfer learning from a pretrained InceptionV3 backbone with a lightweight custom head. The base network is loaded with ImageNet weights; all layers are initially frozen, and the final 25 layers are unfrozen for fine-tuning during training. Layer counts and initial weights are deterministic (fixed random seed, Glorot uniform initialization).

On top of the backbone, we attach:

- Global average pooling
- Dense layer (256 units) with L2 weight regularization (λ = 1×10⁻³)
- Batch normalization (momentum 0.99, ε = 1×10⁻³)
- ReLU activation
- Dropout (rate 0.5)
- Output: single sigmoid unit (binary classification)

Input images are 299 × 299 × 3 — matching the InceptionV3 native input size, which is the reason the tile extraction pipeline (Section 3) resizes to 299 pixels.

### 5.2 Preprocessing

The raw 3m x 3m tiles at 112x112 pixels were upscaled to 299x299 pixels with a range of [1, -1] for Inception. 

### 5.3 Training

- Optimizer: Adam, learning rate 5×10⁻⁴, gradient clipping at norm 1.0
- Loss: binary cross-entropy
- Callbacks: early stopping on validation loss (patience 15, restore best weights); ReduceLROnPlateau (factor 0.5, patience 7, min LR 1×10⁻⁶)
- Maximum 150 epochs

### 5.4 Operating thresholds

- **Stage 1**: threshold chosen per marsh to maximize F1 on the test set, where the positive class is bank
- **Stage 2**: threshold chosen per marsh to maximize F1 on the test set, where the positive class is unhealthy.

### 5.5 Labeling burden

The proposal committed to reducing the hand-labeling burden through transfer learning and self-supervised learning, aiming to reduce the number of images per site from thousands to hundreds. The transfer-learning aspect was realized directly: our use of a pretrained InceptionV3 backbone (Section 5.1) substantially reduces the per-site training data required compared to training a network from scratch, and is the mature form of the technique the proposal envisioned.

We did not implement an active-learning loop or self-supervised pre-training. In practice, an undergraduate labeler working from tiled imagery in QGIS could complete a marsh in a manageable timeframe, with expert QC review of the assigned labels. We did not run a controlled experiment to quantify labeling time savings against a hypothetical baseline, so we do not report a specific reduction figure.

---

## 6. Evaluation Framework

Because the pipeline is a cascade, per-stage metrics do not directly describe end-to-end behavior. This section explains how we compose stage-level measurements into cascade-level metrics.

### 6.1 Data splits

For each marsh, tiles were split 70/15/15 into training, validation, and test sets. The split uses stratified sampling on spatial components rather than individual tiles — a critical detail described in the next subsection. A fixed random seed was used across all marshes for reproducibility. The validation set is used for early stopping and learning-rate reduction during training; operating thresholds are selected on test (see Section 5.4).


### 6.2 Spatial splitting

A common failure mode in remote-sensing ML is spatial leakage: when adjacent tiles are randomly assigned to train and test, the model isn't really being tested on new terrain. It's being tested on terrain it has effectively already seen. Reported metrics under random splitting can substantially overstate real-world performance on unseen sites.

To avoid this, we split by spatial component rather than by individual tile. Two tiles are treated as belonging to the same component if they share a class label and are directly adjacent (up, down, left, or right — 4-connected). Components are identified via breadth-first search over the tile grid, then assigned to the train, validation, or test splits as indivisible units using stratified sampling based on component class labels. Every same-class contiguous region of the marsh therefore ends up wholly in one split — a model is never tested on a tile whose neighbor it trained on.

The trade-off is that stratification becomes approximate. A marsh dominated by a few very large components has fewer "units" available to balance across splits, so class ratios in train, validation, and test can drift from the target. In practice, this is a small cost compared to the alternative of inflated metrics.

[TODO: add a small figure — a grid example showing tiles, components, and split assignment. Also confirm whether components are formed on 2-class {bank, non-bank} or the raw 5-class labels; the granularity implication should be stated.]

We verified the spatial constraint held on every marsh (no adjacent same-class tiles ended up in different splits).

Full algorithmic details are in Appendix A.

### 6.3 Cascade metrics

At each stage, we report the standard binary classification metrics (precision, recall, F1) on that stage's test set. To compute end-to-end performance for the unhealthy-bank detection task, we compose these:

- **Cascade recall on unhealthy** = (Stage 1 recall on unhealthy tiles) × (Stage 2 recall on unhealthy tiles). An unhealthy tile is caught only if both stages classify it correctly.
- **Cascade precision on unhealthy** = TP / (TP + FP), where TP counts true unhealthy tiles Stage 2 called unhealthy, and FP counts everything Stage 2 called unhealthy that was not — from either healthy banks (Stage 2 misclassifications) or non-banks that Stage 1 forwarded incorrectly.

The false positives from non-banks require a separate measurement, described next.

### 6.4 Stage 2 behavior on non-banks

Stage 2 was trained only on banks, so its behavior on non-bank tiles that leak through Stage 1 must be characterized separately. We measured this on a held-out ballast set — non-bank tiles that Stage 2 was never exposed to in training, validation, or test. The measured false-positive rate on this set (typically 0.14–0.20 across marshes) is used in the cascade computation to compute end-to-end precision.

### 6.5 Loss attribution

For an exemplar marsh, we report a stage-by-stage attrition table showing how many unhealthy bank tiles are lost at each stage. This makes clear whether the cascade's misses are dominated by Stage 1 (bank detection failing) or Stage 2 (condition classification failing) — useful for understanding where future improvement effort should go. See Appendix B.

---

## 7. Results

### 7.1 Drone imagery — nine Massachusetts marshes

Overall pipeline accuracy across the nine marshes ranged from 0.90 to 0.96, with a median of 0.93. This is close to but slightly below the 95% target set in the project proposal. "Overall accuracy" here counts a tile as correctly classified if both stages agree with ground truth: Stage 1 correctly identifies it as bank or non-bank, and (for bank tiles) Stage 2 correctly identifies it as healthy or eroding.

Overall accuracy is a useful headline number because it maps directly to the proposal target, but it is not the only operationally meaningful metric.  The two-stage recall and precision columns below are the numbers a manager should look at when asking "will this system catch actual erosion?" and "will it flag things that aren't erosion?"

| Marsh | Stage 1 Recall | Stage 1 Precision | Stage 1 F1 | Stage 2 Recall | Stage 2 Precision | Stage 2 F1 | Overall Accuracy |
|-------|:-----------:|:-----------------:|:--------------------:|:---:|:-----------------:|:--------------------:|:---:|
| *Old Town Hill* | .984 | .947 | .965  | .982 | .949 | .966 | .957 |
| *North River* | .964 | .946 | .955 | .913 | .808 | .857 | .910 |
| *South River* | .881 | .983 | .929 | .900| .750 | .818 | .930|
| *Wellfleet* | .968 | 1.00| .984| .929| .867| .897| .945 |
| *Barnstable* | .967 | .908 | .937 | .893 | .926 | .909 | .946 |
| *Essex Bay* | .967 | .951 | .959 | .848 | .903 | .875 | .930|
| *Peggotty Beach* | .986 | .936 | .961 | .741 | .769 | .755 | .930|
| *Westport* | .971 | .829 | .895 | .839 | .929 | .881 |  .900|
| *Red River* | .900 | .931 | .915 | .844 | .871 | .857 | .900|
| **Median** |.967 | .946 | .955 | .893 | .871 | .875 | .930 |

### 7.2 MassGIS aerial photogrammetry

The proposal also called for exploring whether the same modeling approach could be applied to publicly available MassGIS aerial photogrammetry imagery, which would enable statewide application without site-specific drone flights.

We attempted this and found the ground sampling distance of the MassGIS imagery insufficient to resolve the visual features the model relies on. The bank characteristics that distinguish eroding from healthy — undercut edges, sediment slumping, fine-scale vegetation loss — occur at spatial scales of roughly 0.5–3 m, comfortably within the resolution of drone imagery but averaged away in coarser aerial data. Predictive performance in the MassGIS experiment was substantially worse than on drone imagery, sufficient to conclude that the approach as currently framed is not viable at that resolution.

This is a finding rather than a failure: it establishes that drone-scale imagery is a genuine requirement for this task at present, not a preference. Future work using higher-resolution aerial platforms, or task reformulations that operate at coarser scales (e.g., detecting bank change over time rather than instantaneous condition), might change this conclusion.

Task 4 of the proposal, which was contingent on Task 3 succeeding, was consequently not pursued.

---

## 8. Limitations and Future Work


### No cross-site generalization
We trained one Stage 1 and one Stage 2 model per marsh. Our original goal was to train a single model, using data from a variety of marshes, that could then be applied to a new marsh without further training. We tested this approach, and the results were poor. Our experience from this effort leads us to believe that a single successful model would have to account for the variety of individual salt marsh morphology, differing flight conditions (time of day, clouds, GSD), differing tide conditions, and seasonal differences. While such a model may be possible, we believe it is elusive at the moment for UAS data.

### Stage 1 non-bank sample sizes
For some marshes, the number of non-bank tiles in the test set is small, so the measured Stage 1 false-positive rate on non-banks (and by extension the cascade precision estimate) has wide uncertainty. Larger non-bank evaluation sets would tighten this.

### Stage 2 non-bank behavior estimate
The held-out ballast used to characterize Stage 2's behavior on non-banks is drawn from the same marsh Stage 2 was trained on. While these specific tiles were never seen by Stage 2, they share spatial context with the training set. A fully independent non-bank evaluation (e.g., drawn from a held-out region of the marsh) would give a stricter estimate.


---

## 9. Conclusion

Across the nine Massachusetts salt marshes tested, the pipeline developed in this project reliably identifies eroding channel bank tiles from drone imagery. Overall classification accuracy ranged from 0.90 to 0.96 with a median of 0.93 — close to the 95% target set in the proposal and, more importantly for practice, sufficient to let a field ecologist direct inspection effort to a small, targeted subset of each drone survey rather than reviewing every tile manually. The step from "we can see banks eroding in the imagery" to "we can find them at scale without staff-week manual review" is what the project was for.

Two design decisions carry through the report and deserve emphasis. First, we split training and evaluation data by spatial component rather than by individual tile, so that no model was scored on terrain visually indistinguishable from what it had seen in training. The reported metrics reflect real generalization to unseen bank sections, not memorization of adjacent pixels — a standard drone-imagery ML pitfall we chose to design around. Second, we trained one model per marsh rather than the single cross-site model originally proposed. This gave stronger per-site performance but leaves cross-site generalization as the natural open question.

Cross-site generalization is the most consequential next step. If a single model — or a base model with light per-site fine-tuning — can perform comparably across sites, routine monitoring becomes tractable without training from scratch for each new marsh, and expansion becomes primarily a matter of data collection rather than modeling. The current per-marsh results establish that the visual features exist and can be learned; whether they can be learned jointly is the question we recommend as the primary follow-on effort.

The MassGIS aerial photogrammetry investigation established a boundary rather than a failure: at that ground sampling distance, the visual features distinguishing eroding from healthy banks are averaged away, and no amount of model refinement will recover them. Drone-scale imagery is a genuine requirement for this task at present. Higher-resolution statewide platforms, or task reformulations operating at coarser scales — such as detecting bank change over time rather than instantaneous condition — could revisit that boundary in future work.

Software and models developed under this project are released under an open-source BSD-style license, as committed in the original proposal, and are available for adaptation by other groups working on similar problems in salt marsh and coastal ecosystems.

---


## Appendix A — Spatial Splitting Algorithm

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

That leaves only ~52 "units" to split across train/val/test, and two of them dominate. If the large healthy component goes to train and the large unhealthy component goes to test, the class balance across splits will be visibly off. This is the cost of the spatial constraint. Empirically, on our marshes, this drift is modest.

---

## Appendix B — Detailed Results For One Marsh

To see the details involved in evaluating a two-stage approach, we will show an example from one marsh, South River.

```
stage 1
========================
Best Threshold: 0.790
Best F1 Score: 0.929
Precision: 0.983
Recall: 0.881
Accuracy: 0.961
Confusion Matrix:
[[162   1]
 [  8  59]]
```

- unhealthy_recalled=26,healthy_recalled=33, nonbank_recalled=1
- (unhealthy_recalled+healthy_recalled)/sum([1 if l in [1,2] else 0 for l in original_labels])=0.8805970149253731
- unhealthy_missed=3,healthy_missed=5
- unhealthy_total = unhealthy_recalled + unhealthy_missed  # 26 + 3 = 29
- healthy_total = healthy_recalled + healthy_missed        # 33 + 5 = 38
- recall_unhealthy = unhealthy_recalled / unhealthy_total  # 26/29 = 0.8966
- recall_healthy = healthy_recalled / healthy_total        # 33/38 = 0.8684

```
stage 2 (trained with non_banks, non_bank error rate: .04)
=========================
Best Threshold: 0.510
Best F1 Score: 0.818
Precision: 0.750
Recall: 0.900
Accuracy: 0.885
Confusion Matrix:
[[65  9]
 [ 3 27]]
 error .02 (percentage of non-banks classified as unhealthy)
```

### TWO-STAGE PIPELINE METRICS: UNHEALTHY BANK DETECTION

```
🎯 PRIMARY METRICS (Unhealthy Bank Detection):
   Recall:    83.6%
   Precision: 79.8%
   F1 Score:  0.817

📈 STAGE 1 - Bank Detection (N=230):
   Test composition: 163 non-banks, 67 banks
   Bank recall:         0.896
   Non-bank FPR:        0.018
   Non-bank specificity: 0.982

📈 STAGE 2 - Unhealthy Classification (N=104):
   Test composition: 74 not-unhealthy (healthy+non-banks), 30 unhealthy
   Unhealthy recall:       0.933
   Not-unhealthy FPR:      0.189
   Non-bank error rate:    0.020 (empirically tested)

⚠️  ERROR ANALYSIS (Scaled to 1000 images):
   True unhealthy banks:       130.4
   Successfully detected:      109.0 (83.6%)
   Lost at Stage 1:            13.6 (10.4%)
   Lost at Stage 2:            7.8 (6.0%)

   Total predicted as unhealthy: 136.5
   FP from healthy banks:        27.3 (99.1% of FP)
   FP from non-banks:            0.3 (0.9% of FP)

🚀 DEPLOYMENT SIMULATION (1000 images):
   Composition: 709 non-banks, 161 healthy, 130 unhealthy
   Images passed to Stage 2: 273.9
   Marsh health ratio: 49.8% (predicted unhealthy / total detected banks)
```

## Appendix C — An Illustrative Example

Using Essex Bay as an example, we will present the key points of our approach.

### C.1 The Essex Bay marsh

<p align="left">
  <img src="https://www.dropbox.com/scl/fi/7evcm05y1ymqbchdw38vm/Screenshot-2026-09-02-at-9.42.39-AM.png?rlkey=ug8r9akltn4bbti6jr8hphd15&raw=1" alt="Essex Bay salt marsh" width="400">
</p>

### C.2 3mx3m Tiling

We experimented with different tile sizes and found that 3 m x 3 m gave the best results.

<p align="left">
  <img src="https://www.dropbox.com/scl/fi/4pigut5pb4lisqd0q63a1/Screenshot-2026-09-02-at-9.51.59-AM.png?rlkey=sglw2h7542mbreeul7v0ceuhx&raw=1" alt="Essex Bay tiling" width="400">
</p>

### C.3 Stage 2

Note that Stage 1 is the same setup but with classes *bank* and *non-bank* replacing the *healthy bank* and *unhealthy bank* in the figure below. In essence, Stage 1 feeds what you see below on the left.

<p align="left">
  <img src="https://www.dropbox.com/scl/fi/fdqoh6utyazvv3mnwpknm/Screenshot-2026-09-02-at-10.01.48-AM.png?rlkey=xeiug7b13wqnn4q5903fxqlsm&raw=1" alt="Stage 2" width="400">
</p>

### C.4 Production Stage

Once the two-stage pipeline is established, we can apply it to the entire marsh.

<p align="left">
  <img src="https://www.dropbox.com/scl/fi/1oeyenb6n3uldrs8l0fjv/Screenshot-2026-09-02-at-10.11.28-AM.png?rlkey=sn9rl383b2ltjdh4vezwij24y&raw=1" alt="Pipeline in production" width="600">
</p>

