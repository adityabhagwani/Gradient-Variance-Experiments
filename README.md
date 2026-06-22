# Gradient Variance Experiments

**Do overfit neurons exhibit systematically noisier gradients than generalizing neurons — and can this be detected without ever touching the validation set?**

This repository investigates whether memorization in deep neural networks leaves a detectable signature in the per-neuron gradient dynamics of the first hidden layer, independent of any explicit train/validation comparison. It is the second strand of a two-part research program; the first strand ([Curve Optimizer](https://github.com/adityabhagwani/Curve-Optimizer)) approached the same memorization phenomenon from the frequency domain of a model's *output*. This work approaches it from the *gradients themselves*.

---

## Hypothesis

When a network overfits, it doesn't merely achieve low training loss — it fits idiosyncratic, sample-specific noise that has no consistent underlying structure to anchor on. The central claim tested here:

> **Neurons in an overfit network will show (1) lower consistency in gradient *direction* across consecutive training epochs, and (2) higher volatility in gradient *magnitude* across consecutive training epochs, relative to the same neurons in an architecturally identical but regularized network.**

The intuition: a neuron learning genuine structure receives gradient signal that points in a stable, slowly-evolving direction epoch over epoch (it's converging toward a consistent feature). A neuron absorbing memorized noise receives gradient signal that is comparatively erratic — what improves the fit to one idiosyncratic training point this epoch may be irrelevant or counterproductive the next, since there's no shared structure across samples to consistently push toward.

This is an output-agnostic test: it never looks at train/val accuracy gaps directly. It looks only at the *trajectory* of gradients flowing into each neuron.

---

## Methodology

### Paired-model design

For each dataset, two architecturally identical MLPs are trained from the **same weight initialization** (same seed):

- **Model A ("overfit"):** no dropout, no weight decay, trained blind to validation signal for the full epoch budget.
- **Model B ("regularized"):** identical architecture, but with dropout + L2 weight decay, trained for the *same number of epochs* (not early-stopped), so the epoch axes align for direct per-epoch comparison.

Because both models share an initialization, neuron index $i$ in Model A and neuron index $i$ in Model B are literally the same neuron at $t=0$ — any divergence in their gradient behavior over training is attributable to the regularization difference, not to a different random draw of weights.

### Per-neuron gradient logging

At every training epoch, the full gradient vector flowing into each first-layer (`fc1`) neuron's incoming weight vector is logged, along with its L2 norm. This produces, per neuron, two time series across training: a sequence of gradient vectors (for direction analysis) and a sequence of gradient norms (for magnitude analysis).

### Dead neuron filtering

Neurons whose post-burn-in mean gradient norm falls below a threshold are excluded as "dead" (gradient signal too small to support a meaningful direction/magnitude comparison). The threshold is either fixed per dataset or auto-selected per run as the smallest value in a dataset-specific range that yields 5–15 dead neurons — keeping the dead-neuron rate small and stable without hand-tuning per seed.

### Two metrics, computed per valid neuron

**Direction disagreement** — for each neuron, compute the cosine similarity between its gradient vector at epoch $t$ and epoch $t+1$, for all $t$ after a burn-in window. A neuron is flagged if Model A's *mean* cosine similarity across this window is lower than Model B's — i.e., the overfit model's gradient direction is less stable for that neuron.

**Magnitude volatility** — for each neuron, compute the log-ratio of consecutive gradient norms, $\log(\|g_{t+1}\| / \|g_t\|)$, across the same post-burn-in window. A neuron is flagged if Model A's *standard deviation* of this log-ratio sequence is higher than Model B's — i.e., the overfit model's gradient magnitude swings more erratically.

The hypothesis predicts that a majority of valid neurons should be flagged on both metrics whenever Model A has genuinely memorized noise that Model B has resisted via regularization.

### Noise injection

Every dataset includes an explicit noise-injection step so the overfit model has concrete idiosyncratic signal to memorize that cannot be explained by the true input–output relationship:
- **Classification tasks:** 10% of training labels are flipped (to a random wrong class for multi-class).
- **Regression tasks:** a fraction of training targets receive a large random offset.

### Multi-seed validation

Every dataset is run across **5 seeds (42, 7, 13, 99, 2024)** to separate a robust, seed-general effect from a lucky single run.

---

## Datasets

| Dataset | Task | Features | Loss | Train / Val size | Noise injection |
|---|---|---|---|---|---|
| `make_moons` *(baseline, prior work)* | Binary classification | 2 (synthetic, geometric) | BCEWithLogitsLoss | — | Label flip 10% |
| Breast Cancer (`load_breast_cancer`) | Binary classification | 30 (real, mixed scale) | BCEWithLogitsLoss | 40 / 150 | Label flip 10% |
| Digits (`load_digits`) | 10-class classification | 64 (8×8 pixel images) | CrossEntropyLoss | 100 / 400 | Label flip 10% (to random wrong class) |
| Diabetes (`load_diabetes`) | Regression | 10 (real) | MSELoss | 40 / 150 | Target offset injection |
| Synthetic High-Dim (`make_classification`) | Binary classification | 40 (5 informative, 10 redundant, rest noise) | BCEWithLogitsLoss | 60 / 300 | Label flip 10% |

Each dataset notebook is fully self-contained: data loading, model definitions, paired training loops, gradient logging, dead-neuron filtering, metric computation, and per-seed result export.

---

## Results

### Summary across all seeds (mean ± SD of flagged-neuron count)

| Dataset | Dead neurons (range) | Direction disagreement | Magnitude volatility |
|---|---|---|---|
| Breast Cancer | 7–28* | 67.2 ± 13.0 / ~115 valid (≈58%) | 74.6 ± 22.4 / ~115 valid (≈65%) |
| Digits | 5–6 | 93.2 ± 6.1 / 123 valid (≈76%) | 112.2 ± 6.9 / 123 valid (≈91%) |
| Diabetes | 5 | 122.4 ± 0.9 / 123 valid (≈99.5%) | 119.8 ± 2.9 / 123 valid (≈97%) |
| Synthetic High-Dim | 5 | 21.2 ± 5.9 / 123 valid (≈17%) | 27.0 ± 6.9 / 123 valid (≈22%) |

\* Seed 13 produced 28 dead neurons on Breast Cancer (vs. 7–20 for other seeds) — a one-time approved exception, included in the analysis.

### Per-dataset detail

**Breast Cancer** — moderate, seed-variable support. Direction disagreement ranges from 47.2% to 70.2% of valid neurons across seeds; magnitude volatility from 36.1% to 78.3%. Higher seed-to-seed variance than other datasets, plausibly reflecting genuine instability in small (40-sample) stratified splits on real tabular data.

**Digits** — clean, low-variance support. Dead-neuron count is remarkably stable (5–6 across all 5 seeds). Direction disagreement holds in 69.9–79.7% of valid neurons and magnitude volatility in 82.1–96.7%, with the tightest seed-to-seed spread of any dataset. The strongest classification-setting replication, suggesting the effect may be more robust in multi-class/high-feature-count regimes.

**Diabetes** — the strongest result in the entire sweep. Direction disagreement holds in 98.4–100% of valid neurons across every seed; magnitude volatility in 94.3–100%. Dead-neuron count is perfectly stable (5/128 in all 5 seeds). This is also the only regression task in the sweep, and it produces the cleanest signal — possibly because R² divergence between the overfit and regularized model is unusually dramatic here (val R² goes negative for the overfit model while train R² approaches 1.0), giving the gradient dynamics a starker contrast to track.

**Synthetic High-Dimensional — a negative result that supports the hypothesis.** Direction disagreement holds in only 10.6–24.4% of valid neurons, and magnitude volatility in 14.6–29.3% — well below the other four datasets. Investigation revealed the root cause: **the "overfit" model never developed a meaningful generalization gap on this dataset.** Both models converge to similarly poor validation accuracy (0.54–0.75 across seeds), and for seeds 13 and 99 the *regularized* model actually outperforms the "overfit" one on validation. Since the hypothesis is conditioned on one model genuinely memorizing while the other generalizes, the absence of that precondition here predicts — and produces — the absence of the gradient-variance signal. This null result functions as an internal negative control: the effect disappears exactly when its precondition is absent, rather than appearing as random noise.

---

## Repository Structure
Gradient Variance Exps/
├── Neuron_Gradient_Variance_Experiments.ipynb # Original make_moons baseline
└── Exps for Different Datasets/
├── Breast Cancer Dataset/
│ ├── Neuron_Gradient_Variance_BreastCancer.ipynb
│ ├── Seed 42 Result Files/ ... Seed 2024 Result Files/
│ └── breast_cancer_experiment_results.xlsx
├── Digits Dataset/
│ ├── Neuron_Gradient_Variance_Digits.ipynb
│ ├── Seed 42 Result Files/ ... Seed 2024 Result Files/
│ └── digits_experiment_results.xlsx
├── Diabetes Dataset/
│ ├── Neuron_Gradient_Variance_Diabetes.ipynb
│ ├── Seed 42 Results/ ... Seed 2024 Results/
│ └── diabetes_experiment_results.xlsx
└── Synthetic High Dimensional Dataset/
├── Neuron_Gradient_Variance_SyntheticHighDim.ipynb
├── Seed 42 Results/ ... Seed 2024 Results/
└── synthetichighdim_experiment_results.xlsx


Each `Seed X Result(s)` folder contains the per-neuron gradient plots (direction cosine similarity over epochs, magnitude log-ratio over epochs) and a JSON dump of the raw per-seed statistics. Each `.xlsx` file aggregates Dataset / Seed / dead neurons / valid neurons / direction-disagreement count / magnitude-volatility count across all 5 seeds, with mean and standard deviation computed via live Excel formulas.

---

## Interpreting the Results as a Whole

Four real-world-adjacent or synthetic datasets, two task types (classification and regression), two loss functions (BCE/CrossEntropy and MSE), and dataset dimensionalities ranging from 2 to 64 features all converge on the same qualitative pattern: **whenever a clear overfit/generalization gap exists, the overfit model's first-layer neurons show measurably less stable gradient directions and more volatile gradient magnitudes than the same neurons in the regularized model** — and this gap disappears in the one dataset where the models failed to diverge in generalization behavior. The effect strength varies (from ~20% to ~99.5% of valid neurons flagged), but its *direction* is consistent across every dataset where the precondition holds, and its *absence* is consistent with the one dataset where the precondition fails.

This is consistent with — but does not prove — a causal story in which memorization is mechanistically expressed at the level of individual neuron gradient trajectories, not just at the level of aggregate loss curves.

---

## Caveats and Open Questions

- **Effect size is not uniform.** The breast cancer result is noticeably noisier across seeds than digits or diabetes; whether this reflects genuine sensitivity to small real-world dataset structure or instability from very small (40-sample) training splits is untested.
- **Burn-in window choice matters.** Metrics are computed post-epoch-200 (or post-epoch-50 for diabetes, which converges faster). The sensitivity of the reported counts to this cutoff has not been systematically swept.
- **Dead-neuron threshold selection is heuristic.** The auto-selection logic targets a 5–15 dead-neuron count by construction; this could mask threshold sensitivity in the underlying gradient norm distribution.
- **Only the first hidden layer is probed.** Whether the effect holds, strengthens, or weakens in deeper layers is unexplored.
- **No causal manipulation.** All results are observational comparisons between two trained models; no experiment here directly intervenes on gradient direction/magnitude to test whether suppressing the effect changes generalization.

---

## Relationship to the Curve Optimizer Project

This work is the gradient-level counterpart to the [Curve Optimizer](https://github.com/adityabhagwani/Curve-Optimizer) project, which targets the same underlying phenomenon — memorization in overfit models — from the frequency domain of a model's *output* curve rather than its internal gradients. Curve Optimizer shows that memorization deposits high-frequency noise in a model's output signal that can be filtered out post-hoc; this repository asks whether the same memorization process is visible even earlier, in the gradient signal that produced the weights in the first place. Together they sketch two independent, non-overlapping observational handles on the same underlying question: *what does memorization look like, mechanistically, inside a trained network?*

---

## Dependencies
torch
numpy
scikit-learn
matplotlib
openpyxl
