# CMA-ES

cangu fits ALINE weights to triads, where each label says which of two candidate words sounds closer to a target. [Bradley-Terry in cangu](bradley-terry.md) explains how a set of weights is scored.

The optimizer that minimizes this log-loss cannot rely on derivatives. cangu uses the Covariance Matrix Adaptation Evolution Strategy (CMA-ES), which needs only the values of the objective.

## Why not gradient descent?

Gradient descent computes the derivative of the loss with respect to each parameter and steps downhill. That does not work here, for three reasons:

- ALINE finds the best alignment by dynamic programming. A small change in the weights can make the best alignment jump to a different path, which puts kinks in the loss surface.
- Some weight combinations make a word's self-alignment score non-positive, and pho then cannot compute a normalized similarity.
- pho computes no derivatives. To the optimizer, it maps a weight vector to a loss and nothing more.

More details: [What a gradient is, and why it breaks here](#what-a-gradient-is-and-why-it-breaks-here).

## How CMA-ES works

CMA-ES (Hansen and Ostermeier, 2001) keeps a multivariate normal search distribution over the parameters, described by a mean $m$ (the current best guess), a step size $\sigma$ (how widely it searches), and a covariance matrix $C$ (the shape and orientation of the search).

Each generation, it:

1. draws a population of $p$ candidates,

   $$
   x_i = m + \sigma \, z_i, \qquad z_i \sim \mathcal{N}(0, C), \qquad i = 1, \dots, p,
   $$

2. evaluates the objective at each candidate,
3. ranks the candidates and keeps the better half,
4. moves $m$ to a weighted average of those candidates, giving more weight to better ranks,
5. stretches $C$ along the directions of successful steps, which lets it learn correlations between parameters,
6. grows $\sigma$ when consecutive steps point the same way and shrinks it when they cancel out.

It repeats until the evaluation budget runs out or the tolerances are met.

More details: [The search as a cloud of guesses](#the-search-as-a-cloud-of-guesses).

### Why it fits ALINE

CMA-ES only uses the ranking of candidates, so it is unaffected by any monotonic transformation of the objective. Flat plateaus and sharp kinks do not stop it, and a weight combination that cannot be scored can simply get a large fixed loss.

ALINE weights trade off against each other: raising `place` may call for lowering `manner`. The covariance matrix captures such dependencies, so the search moves along them instead of along one axis at a time.

Its internal settings (population size, recombination weights, learning rates) follow standard formulas based on the number of parameters $D$. The default population size is

$$
p = 4 + \lfloor 3 \ln D \rfloor,
$$

so fitting 13 weights plus $k$ ($D = 14$) gives $p = 4 + \lfloor 3 \ln 14 \rfloor = 11$ candidates per generation.

### Restarts: BIPOP

A single run can get stuck in a local minimum. BIPOP-CMA-ES (Hansen, 2009) restarts the optimizer within the evaluation budget and alternates between two regimes: large populations, doubled at each restart, which smooth over local roughness and find broad basins, and small populations with small random step sizes, which search nearby cheaply. cangu keeps the lowest loss found across all restarts.

More details: [Local minima and why restarts help](#local-minima-and-why-restarts-help).

## How cangu sets up the search

### Searching in log space

cangu optimizes the logarithms of the parameters:

$$
\theta = (\ln w_1, \dots, \ln w_n, \ln k),
$$

where $w_1, \dots, w_n$ are the weights in `free_weights` and $k$ is the Bradley-Terry sensitivity.

This keeps every parameter positive. For a negative cost such as `skip`, cangu works with the logarithm of its absolute value and restores the sign when it builds the ALINE config.

A step in $\theta$ multiplies a weight instead of adding to it: a step of $+0.1$ scales a weight by $e^{0.1} \approx 1.105$, whether it started at 2 or at 50. The default ALINE weights differ by orders of magnitude, and CMA-ES starts with the same step size in every direction, so searching on a log scale puts all weights on an equal footing.

The search starts at $\theta_{\text{default}}$, the logarithms of the weights in the ALINE config passed to `Fitter::new`, together with $\ln k_{\text{init}}$. `substitute` stays fixed and sets the overall scale, as [Bradley-Terry in cangu](bradley-terry.md) explains.

More details: [Why logarithms make the search fair](#why-logarithms-make-the-search-fair).

### The objective

On a training set, cangu minimizes

$$
J(\theta) = \frac{1}{N} \sum_{t=1}^{N} \ell_t(\theta)
+ \lambda \sum_{i=1}^{n} \bigl(\theta_i - \theta_{\text{default},i}\bigr)^2
+ c \sum_{i=1}^{n+1} \bigl(\theta_i - \operatorname{clamp}(\theta_i)\bigr)^2,
$$

where $\ell_t$ is the Bradley-Terry loss of triad $t$, $N$ is the number of training triads, $c$ is `clamp_penalty`, and $\operatorname{clamp}$ limits each parameter to $[\theta_{\text{default},i} - b,\ \theta_{\text{default},i} + b]$ with $b$ = `bound`.

The first term is the mean log-loss on the training triads. The $\lambda$ term pulls the weights toward their ALINE defaults; it covers the $n$ weights but not $k$. With little data the weights stay near their defaults, and with plenty of data the log-loss dominates.

The last term keeps the parameters within $\theta_{\text{default}} \pm b$ in log space. With $b = 5$, a weight can grow or shrink by a factor of up to $e^5 \approx 148$. A candidate outside the bounds is evaluated at the clamped values and charged a quadratic penalty, which grows the further it strays.

If pho cannot score a candidate because of an invalid self-alignment, the objective returns a fixed penalty of $10^6$. Those candidates rank last, and the report counts how often it happened.

More details: [What each term of the objective does](#what-each-term-of-the-objective-does).

### The two phases

`fit_theta` in `src/eval/fit.rs` runs in two phases.

1. Plain CMA-ES starts at $\theta_{\text{default}}$ with step size $\sigma_0$ = `sigma0`. A value of 0.5 in log space moves weights by roughly $e^{0.5} - 1 \approx 65\%$ at first. This phase gets two thirds of `max_evals` and stops early when the objective improves by less than `tol_fun` or the steps fall below `tol_x`.
2. BIPOP restarts use the remaining third of the budget, starting from points drawn uniformly within the $\theta_{\text{default}} \pm b$ box to explore regions far from the start. If fewer than `min_bipop_evals` evaluations remain, cangu skips this phase.

cangu keeps whichever phase found the lower loss and says which in the progress output (`best from default_start` or `best from bipop`). The first phase makes use of the hand-tuned defaults, and the second guards against a poor local minimum near them.

All random numbers derive from the seed passed to `Fitter::fit`, so the same seed gives the same fit.

### Choosing λ by cross-validation

cangu picks $\lambda$ from `lambda_grid` with nested cross-validation:

```
for each outer fold:
    hold out the test triads
    for each lambda in lambda_grid:
        inner cross-validation on the training triads -> held-out log-loss
    pick the lambda with the lowest inner log-loss
    fit on all training triads with that lambda
    score the held-out test triads
```

Folds are grouped by target word: every triad with the same target goes into the same fold, so the test triads always have targets the fit never saw. Targets are shuffled with the run's seed and dealt to the folds in turn.

After cross-validation, a final fit on the whole dataset with the chosen $\lambda$ gives the final weights, which `FitResult::write_aline` saves as a pho TOML config.

Every fold and every grid value needs its own optimization run. With $F$ folds and $L$ values of $\lambda$, a full `Fitter::fit` performs

$$
(F + 1)(F L + 1)
$$

CMA-ES fits, which is $(5 + 1)(25 + 1) = 156$ for $F = 5$ and $L = 5$.

More details: [Cross-validation and counting the fits](#cross-validation-and-counting-the-fits).

## Checking on simulated data

The labels in the demo dataset (`example/dataset.csv`) were generated from known weights: ALINE's defaults, except $w_\text{voice} = 20$ instead of 5. Fitting `voice` and `nasal` with a small budget of 300 evaluations per fit gives:

```
  weight      default     fitted       true
  voice          5.00      20.69      20.00
  nasal         20.00      21.60      20.00
```

Both weights come within 10% of their true values.

`tests/recovery.rs` runs a larger end-to-end check. It simulates labels with raised `back`, `round`, and `stress` weights and fits them with the full budget. The fit must beat baseline ALINE on held-out triads by at least 0.05 in log-loss and 5 percentage points in accuracy, and recover `stress` and `back + round` to within 30%.

## Limitations

- A weight only moves if the dataset has triads whose outcome depends on it. A weight the data does not constrain may drift until it hits `bound`, and the report flags parameters that end up at their bounds.
- CMA-ES stops when the budget runs out or the step size falls below the tolerance. If the fitted weights still change between runs or seeds, increase `max_evals`.
- The Rust `cmaes` crate evaluates the candidates of a generation one after another. Each evaluation scores its triads in parallel across CPU cores, but candidates are not evaluated concurrently.

## Further reading

- N. Hansen, "The CMA Evolution Strategy: A Tutorial", <https://arxiv.org/abs/1604.00772>.
- N. Hansen, "Benchmarking a BI-population CMA-ES on the BBOB-2009 noisy testbed", <https://doi.org/10.1145/1570256.1570334>.
- [Bradley-Terry in cangu](bradley-terry.md)

## More details

These sections go through the math above more slowly, with worked numbers.

### What a gradient is, and why it breaks here

Picture the loss as a landscape. Each possible setting of the weights is a location, and the loss is the height of the ground there. Fitting means finding the lowest point.

The derivative of the loss with respect to one weight says how steeply the ground rises or falls when you move a little in that weight's direction. The gradient collects these slopes for all the weights, so it points uphill. Gradient descent takes a small step the opposite way, recomputes the slopes, and repeats, like walking downhill in fog by feeling the slope under your feet.

That only works if the ground is smooth. ALINE picks the single best alignment of two words. Change a weight slightly and, at some point, a different alignment suddenly becomes the best one, and the similarity jumps or bends sharply. The landscape gets creases and small cliffs, where the slope under your feet tells you little about where the valley is.

Some weight settings cannot be scored at all (see the self-alignment point above), which is like holes in the ground. And pho has no way of reporting slopes in the first place. An optimizer that only needs the height at the points it visits avoids all three problems.

Back to [Why not gradient descent?](#why-not-gradient-descent).

### The search as a cloud of guesses

CMA-ES never computes slopes. It keeps a cloud of guesses around its current best estimate and lets the results reshape the cloud.

The cloud is a multivariate normal distribution, the multi-dimensional version of the bell curve. Its three parts each have a plain meaning:

- the mean $m$ is the center of the cloud, the current best estimate of the weights;
- the step size $\sigma$ is how spread out the cloud is overall;
- the covariance matrix $C$ is the cloud's shape. With two weights, the cloud is an ellipse that can be round, long and thin, or tilted.

In each generation, CMA-ES scatters $p$ guesses through the cloud, scores each one, and keeps the better half. The center moves toward the good guesses, with the best ones pulling hardest.

The shape is where the method gets its name. Suppose the good guesses keep landing along a diagonal, where `place` goes up while `manner` goes down. CMA-ES stretches $C$ along that diagonal, so the next generation searches more along that direction and less across it. Over the generations, the cloud learns which combinations of weights tend to improve the fit together.

The step size works the same way. If successive moves of the center keep pointing the same way, the cloud is too timid and $\sigma$ grows. If they keep reversing direction, the center is bouncing around a minimum and $\sigma$ shrinks, so the search settles in.

The population size formula works out as follows for 14 parameters: $\ln 14 \approx 2.64$, three times that is about 7.92, rounding down gives 7, and adding 4 gives 11 guesses per generation. More parameters call for slightly larger generations, but the number grows slowly.

Back to [How CMA-ES works](#how-cma-es-works).

### Local minima and why restarts help

A local minimum is a dip in the landscape that is lower than everything around it but not the lowest point overall. Once the cloud has shrunk inside such a dip, every nearby guess is worse, so the search stays there.

Restarting means beginning a new search somewhere else and keeping the best result of all the searches. BIPOP alternates two kinds of restart. Large populations spread many guesses widely; they are slow per generation, but they see the overall shape of the landscape and are less easily caught by small dips. Small populations with small, randomly chosen step sizes are cheap and search a neighborhood closely. Alternating between them covers both rough landscapes and ones with a narrow, well-hidden minimum.

Back to [Restarts: BIPOP](#restarts-bipop).

### Why logarithms make the search fair

The default Filipino weights include `back = 2` and `manner = 50`. Suppose the optimizer added 1 to every weight. For `back` that is a 50% change, for `manner` only 2%. A fixed step would be huge for small weights and negligible for large ones.

On a logarithmic scale, equal steps mean equal ratios. Adding 0.1 to $\ln w$ multiplies $w$ by $e^{0.1} \approx 1.105$, a 10.5% increase, whatever $w$ was:

| weight | $\ln w$ | after $+0.1$ | new weight |
| --- | --- | --- | --- |
| 2 | 0.693 | 0.793 | 2.21 |
| 50 | 3.912 | 4.012 | 55.26 |

Both weights grow by the same proportion. Since CMA-ES starts with one step size for every direction, this puts all weights on an equal footing.

Logarithms also keep weights positive for free: whatever value $\ln w$ takes, $w = e^{\ln w}$ is positive. `skip` is negative in ALINE, so cangu works with $\ln 10$ for `skip = -10` and puts the minus sign back when it builds the config.

Back to [Searching in log space](#searching-in-log-space).

### What each term of the objective does

The first term is the average log-loss from [Bradley-Terry in cangu](bradley-terry.md): how badly the weights predict the training labels.

The $\lambda$ term acts like a rubber band tying each weight to its default. Moving a weight costs a little; moving it far costs a lot, because the distance is squared. Doubling a weight changes $\theta$ by $\ln 2 \approx 0.693$, so the penalty for that one weight is

$$
\lambda \times 0.693^2 \approx 0.48 \lambda.
$$

With $\lambda = 0.01$ that is about 0.005, small next to a typical log-loss of around 0.5, so the weight moves if the data supports it even a little. With $\lambda = 1$ it is about 0.48, and the weight only moves if that improves the fit a great deal. Cross-validation chooses between such settings by checking which one predicts unseen triads best.

The clamp term is a fence. Inside $\theta_\text{default} \pm b$ it costs nothing. Outside, the candidate is scored as if it stood at the fence, plus a cost that grows with the square of how far past the fence it went, which pushes the search back inside. With $b = 5$ the fence allows a weight to become up to $e^5 \approx 148$ times larger or smaller than its default.

A candidate that pho cannot score at all gets a loss of $10^6$, far worse than any real loss. CMA-ES only compares candidates by rank, so such a candidate comes last and the search moves away from it.

Back to [The objective](#the-objective).

### Cross-validation and counting the fits

A model always fits the data it was trained on better than new data, so scores on the training triads are too optimistic. Cross-validation measures how well the weights predict triads the fit did not see. It splits the data into $F$ parts called folds, trains on all but one, tests on the one left out, and repeats until every fold has been the test set once.

Grouping by target word matters because triads with the same target are related. If some triads for `ˈkabajo` were in training and others in testing, the test would partly measure memory of that word rather than generalization to new words.

"Nested" means there is a second cross-validation inside each training set, used only to choose $\lambda$. The outer test fold stays untouched until the very end, so choosing $\lambda$ cannot leak information from it.

The fit count adds up as follows. For each of the $F$ outer folds:

- choosing $\lambda$ takes one fit per inner fold and per $\lambda$ value: $F \times L$ fits,
- then one more fit trains on the whole outer training set with the chosen $\lambda$.

That is $F L + 1$ fits per outer fold, or $F (F L + 1)$ for all of them. The final fit on the whole dataset chooses $\lambda$ the same way, adding another $F L + 1$. Together:

$$
F (F L + 1) + (F L + 1) = (F + 1)(F L + 1).
$$

With $F = 5$ and $L = 5$: $5 \times 26 + 26 = 156$.

Back to [Choosing λ by cross-validation](#choosing-λ-by-cross-validation).
