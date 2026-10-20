# cangu / tutorial + demo

This tutorial runs the bundled demo in [example/](/example/demo.rs), uses it to show how a triad is scored and how the fit works, and then covers preparing your own data, choosing settings, reading the results, and designing stimuli.

## 1. What cangu does

Each row of your dataset is a triad: a target word, two candidates $b$ and $c$, and a label saying which candidate sounds closer to the target. cangu learns the ALINE weights (feature saliences and edit costs) that best predict these labels.

For each triad, ALINE (computed by pho) gives each candidate a similarity $S$ to the target, and the Bradley-Terry model turns the difference into a probability:

$$
d = S(\text{target}, b) - S(\text{target}, c),
\qquad
P(y = b) = \sigma(k d) = \frac{1}{1 + e^{-k d}}
$$

$k$ sets the sensitivity: with a higher $k$, a small similarity difference already produces a confident choice. cangu fits the weights listed in `free_weights`, together with $k$, by minimizing

$$
\frac{1}{n}\sum_{i=1}^{n} -\ln P(y_i)
\;+\; \lambda \sum_j \bigl(\theta_j - \theta_{\text{default},j}\bigr)^2
\;+\; \text{clamp penalty},
$$

where $\theta_j = \ln w_j$ is the logarithm of each fitted weight. The first term is the log-loss of the labels. The $\lambda$ term pulls weights toward their default ALINE values unless the triads justify moving them, and cross-validation picks $\lambda$ from the values in `lambda_grid`.

The optimizer is CMA-ES started at the default weights, followed by BIPOP restarts. [Bradley-Terry](bradley-terry.md) and [CMA-ES](cma-es.md) describe both parts in more detail.

## 2. Run the demo

cangu is a library, so there is no `cangu` command. You run a fit with a small Rust program that builds an ALINE config for your language, sets the fit settings, and calls `Fitter`. The demo in `example/` is such a program:

- `example/demo.rs` is the script. It sets the data path and fit settings, fits `voice` and `nasal` (plus $k$), and writes the results.
- `example/aline.rs` builds a tiny ALINE config in code: 9 consonants (`p b t d k g m n s`), 3 vowels (`a i u`), and Kondrak's standard costs, saliences, and feature values. It uses the MangoCats variant, so stress (`ˈ`) counts.
- `example/dataset.csv` has 150 triads over 25 made-up target words.

```bash
cargo run --release --example demo
```

Always use `--release`, because the optimizer is much slower in a debug build. The run takes about 20 seconds.

### The demo data

Each row is one triad, where each candidate differs from the target in one way (voicing, place, nasality, a vowel, or stress position):

```csv
target,b,c,label
baˈbu,daˈbu,paˈbu,b
```

Here the target is `baˈbu`, the candidates are $b$ = `daˈbu` and $c$ = `paˈbu`, and the label $y = b$ says that `daˈbu` sounds closer to `baˈbu` than `paˈbu` does.

No listeners were involved. Each label was made in four steps:

1. Pick "true" weights: ALINE's defaults, except that voicing counts four times as much ($w_\text{voice} = 20$ instead of 5).
2. Score the triad with those weights to get $d$.
3. Turn $d$ into $P(y = b)$ with $k = 40$.
4. Draw the label at random with that probability.

So most labels follow the true weights, but some do not, as with real judgments. The fit only sees the CSV. Because the true weights are known, you can check whether it finds them.

### The demo output

The demo writes two TOML files to `out/demo/`:

- `aline.toml` is the fitted ALINE config, which pho can load. It holds the final weights, including `voice ≈ 20.69` and `nasal ≈ 21.60`.
- `report.toml` is the training report: the settings, the fitted $k$ and $\lambda$, the starting value of each fitted weight, held-out scores, flags, and the results of every fold. Section 8 describes it.

`voice` moved from 5 to about 20, the effect hidden in the labels, and `nasal`, which already had the right value, stayed near 20. In the report's `[performance]` section, held-out accuracy is 69.3% for the fitted weights (`model`) against 60.7% for plain ALINE with only $k$ fitted (`baseline`), and the log-loss falls from 0.685 to 0.507. Accuracy cannot reach 100% because the labels are noisy draws. With only 150 noisy labels, $k$ is the hardest value to pin down: `[final_fit] k` comes out near 25, below the true 40.

## 3. How a triad is scored

Take the row above. `daˈbu` changes the place of the first consonant, and `paˈbu` changes its voicing:

| weights | $S(\text{baˈbu}, \text{daˈbu})$ | $S(\text{baˈbu}, \text{paˈbu})$ | $d = S_b - S_c$ | $P(y = b)$ at $k = 40$ |
| --- | --- | --- | --- | --- |
| default ($w_\text{voice} = 5$) | 0.950 | 0.958 | −0.008 | 0.42 |
| true ($w_\text{voice} = 20$) | 0.950 | 0.833 | +0.117 | 0.99 |

With the default weights a voicing change (`b` to `p`) is cheap, so ALINE puts `paˈbu` slightly closer and predicts $c$. With the true weights voicing costs four times as much, so `daˈbu` is clearly closer and the model gives $b$ a probability of 0.99. The dataset says $b$.

## 4. How the fit works

1. For any candidate weights $w$, cangu scores every triad as above and computes the log-loss $-\frac{1}{n}\sum_i \ln P(y_i)$, which measures how far the predicted probabilities are from the labels. Better weights give a lower loss.
2. CMA-ES, a derivative-free optimizer, tries batches of weight settings, moves toward the ones with lower loss, and repeats until the budget (`max_evals`) runs out. It starts from the defaults and searches over $\theta = \ln w$, so every weight it proposes is positive.
3. The penalty $\lambda \lVert \theta - \theta_\text{default} \rVert^2$ pulls each weight back toward its default unless the data justifies moving it. Cross-validation picks $\lambda$.
4. Evaluation uses grouped, nested cross-validation. Folds are grouped by target word, so every held-out score comes from target words the fit never saw.

## 5. Prepare your data

A dataset is a CSV file with one row per triad. Four columns are required, in any order:

| column | meaning |
| --- | --- |
| `target` | Target word in IPA, including stress marks (such as `ˈkabajo`). Folds are grouped by this string, so a target must be spelled the same way in every row. |
| `b`, `c` | Candidates $b$ and $c$ in IPA. |
| `label` | The candidate judged closer to the target, `b` or `c`. |

cangu ignores any other column, so you can keep spellings, notes, or contrast annotations such as `contrast_b` and `contrast_c` in the file. Outputs identify triads by their CSV line number.

Sample rows from `tests/fixtures/clean.csv`:

```csv
target,b,c,contrast_b,contrast_c,label
ˈkabajo,ˈgabajo,kaˈbajo,voicing,stress,b
ˈkabajo,ˈgabajo,ˈkamajo,voicing,nasal,c
```

cangu fits on every row, so each row should hold a final, resolved label. If several participants judged the same triads, combine their answers first (for example by majority vote) and drop failed attention checks.

`Fitter::load_dataset` validates the file before fitting and reports each problem with its line number. It checks that:

- every IPA string tokenizes against the sound inventory of the ALINE config passed to `Fitter::new`,
- no triad appears twice (same target and candidates, in either order),
- `label` is `b` or `c` and no required field is empty.

## 6. Choose the settings

Your program sets every field of `ModelConfig`, `OptimizationConfig`, and `EvaluationConfig`; there are no defaults. Edit them in your copy of `example/demo.rs`, or keep one program per experiment.

1. Set the data path (`data` in `run()`). Paths are relative to the directory you run from.
2. Set `free_weights` in `ModelConfig` to the weights you want to fit. Every other weight keeps its value from your ALINE config, and $k$ is always fitted. Only list weights your triads can actually pin down (see section 9). `substitute` cannot be fitted, because the other weights are measured as ratios to it.
3. Set the budget with `max_evals` in `OptimizationConfig`. For 600 triads on 4 CPU cores, 6000 evaluations take about 1.5 hours.
4. Pass the seed and whether to compute a learning curve to `Fitter::fit`.

Each outer fold picks $\lambda$ by inner cross-validation, which takes $F \times L$ fits for $F$ = `n_folds` and $L$ values in `lambda_grid`, and then runs one outer fit. A final fit on all data repeats the selection. In total that is

$$
(F + 1)(F L + 1)
$$

fits, or 156 for $F = 5$ and $L = 5$. A `max_evals` between 100 and 300 gives a quick check before a full run.

## 7. Fit and save the results

`fitter.fit(&dataset, seed, learning_curve)` returns a `FitResult`, and progress prints as the folds finish. Your program saves two TOML files wherever it chooses:

- `result.write_aline(path)` writes the fitted ALINE config as a pho TOML file. Load it with `pho::utils::io::import` to score words with the fitted weights. The weights are stored exactly as fitted, for example `place = 22.209833`.
- `result.write_report(path)` writes the training report, described in the next section.

The two files do not repeat each other. The fitted weights are only in the ALINE file; $k$, the held-out scores, and everything else about the training run are only in the report. $k$ is not part of an ALINE config, because it only converts similarity differences into choice probabilities.

The same data, settings, and seed always give identical files.

## 8. Read the report

The report is a TOML file with these parts.

### Flags

`flags` lists conditions worth checking:

- `RESEARCHER CHOICE`: the ALINE config set the stress salience to 0, so `stress_prior` became the starting value for `stress`. Report this value with your results.
- `weight moved more than Nx from default`: a large change in a weight. It can mean a strong signal in the data or a weight the data does not constrain.
- `parameters at the theta_default +/- bound`: the optimizer hit the limit set by `bound`. The best value may lie beyond it, or the data may not identify the weight.
- `baseline k at its bound`: in that fold, plain ALINE similarities barely separated the candidates.
- `objective evaluations hit NonPositiveSelfScore`: some candidate weights made a word's similarity to itself non-positive, so pho could not compute a similarity. Those evaluations got a fixed high loss.

### Settings

`[settings]` records the dataset path, the pho revision, the seed, and the number of triads, targets, and distinct target-candidate pairs. `[settings.model]`, `[settings.optimization]`, and `[settings.evaluation]` hold the settings exactly as your program gave them.

### Final fit

`[final_fit]` has the fitted $k$, the $\lambda$ chosen by cross-validation on the whole dataset, and `lambda_log_loss`, the cross-validated log-loss for each value of `lambda_grid` in the same order. `[final_fit.start_weights]` gives the value each fitted weight started from: its value in your ALINE config, or `stress_prior` for a stress salience of 0. Compare these with the fitted values in the ALINE file to see how far each weight moved. ALINE's similarity depends only on the ratios between weights, with `substitute` held fixed.

### Performance

`[performance.model]` and `[performance.baseline]` hold the held-out results of the outer cross-validation, pooled over all folds: `accuracy`, `log_loss`, and `ties` (triads where both candidates were equally similar). `model` is the fitted weights; `baseline` is default ALINE with $k$ fitted. A useful fit has a lower held-out log-loss than the baseline.

### Learning curve

`[[learning_curve]]` appears when you pass `learning_curve = true`. Each entry gives the held-out `accuracy` and `log_loss` when training on a `fraction` of the targets. If the scores are still improving at 100%, more data would probably improve the estimates.

### Similarity distribution

`[similarity_distribution]` summarizes the similarities between targets and candidates under the fitted weights: minimum, quartiles, maximum, and the shares at or below 0 and at or above 0.999. If they pile up at 0 or 1, the weights may be pushing similarities to the extremes.

### Folds

Each `[[folds]]` entry describes one outer fold: its size, the $\lambda$ it chose (with `lambda_log_loss`), the $k$ fitted with the weights and for the baseline, the optimizer's evaluation counts, the fitted weights in that fold (`[folds.weights]`), and its held-out scores (`[folds.performance]`). If a weight varies a lot between folds, the data does not pin it down well.

`[[folds.predictions]]` has one entry per held-out triad: its CSV `line`, the probability of label $b$ under the fold's fitted weights (`p_b`), and under the baseline (`baseline_p_b`). Join them with your CSV on the line number to find the triads the model gets wrong.

## 9. Design triads that identify the weights

Add `"place"` to `free_weights` in `example/demo.rs` and run the demo again. `voice` is still recovered, but `place` ends far below its true value of 40 (about 8.5 in one run). The data is not missing place contrasts; about a third of the triads have one. The problem is what those changes are compared against.

ALINE charges a feature's salience times the distance between the two values. With the true weights, the changes in the demo data cost:

| change | cost |
| --- | --- |
| place, `p`→`t` / `t`→`k` / `k`→`p` | $40 \times 0.15 = 6$, $\ 40 \times 0.25 = 10$, $\ 40 \times 0.4 = 16$ |
| voicing, such as `p`→`b` | $20 \times 1 = 20$ |
| nasality, such as `b`→`m` | $20 \times 1 = 20$ |

Every triad with a place change pits it against a voicing or nasal change, and the place change is always the cheaper one, so the place candidate is always closer. Lowering `place` to 10 would make place changes cost 1.5 to 4, which is still cheaper, so every prediction stays the same. The labels show that place matters less than voicing and nasality, but not by how much. Any small enough value of `place` fits them equally well, and the fit settles on one of them.

A weight is only learnable if some triads could go either way depending on it, which happens when the two candidates' changes cost about the same. For `place` that would mean pairing a place change with a change of similar cost, such as stress ($w_\text{stress} = 10$), or pairing two place changes of different sizes, such as `p`→`t` against `p`→`k`. More triads of the lopsided kind do not help.

Keep this in mind when designing stimuli and choosing `free_weights`. In the report, a weight the data does not pin down often ends at its `bound` (listed under `flags`) or varies a lot between folds (`[folds.weights]`).

## 10. Troubleshooting

| error | resolution |
| --- | --- |
| `free_weights: unknown weight 'X'` | Use the weight names listed in the `ModelConfig` docs (`free_weights`). |
| `... does not tokenize` | Correct the IPA transcription, or add the missing sound to your ALINE config. |
| `only N targets; need at least F for F-fold grouped CV` | Add target words or lower `n_folds`. |
| `default for fitted weight 'X' is 0` | Give the weight a non-zero default in your ALINE config, or remove it from `free_weights`. The fit works on logarithms, and $\ln 0$ is undefined. |
