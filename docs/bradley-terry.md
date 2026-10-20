# Bradley-Terry

ALINE compares how two words sound. It aligns them segment by segment and adds up the similarity of the aligned sounds, using feature saliences (such as place of articulation or voicing) and edit costs (such as deleting or expanding a sound). The default ALINE weights come from linguistic judgment and were never calibrated against Tagalog listeners.

cangu tunes these weights on triad data, where listeners answer "which of two candidate words sounds closer to the target?". To score a set of candidate weights against those answers, it needs a probabilistic choice model, and it uses the Bradley-Terry model. [CMA-ES in cangu](cma-es.md) covers the optimizer that searches for the best-scoring weights.

## The setup

Each row in the dataset records one triad:

| target | $b$ | $c$ | label |
|---|---|---|---|
| ˈkabajo | ˈgabajo (voicing) | kaˈbajo (stress) | b |

ALINE (through the pho library) gives each candidate a similarity $S$ to the target. Under the Filipino default weights, with `stress_prior = 10`:

$$
S(\text{ˈkabajo}, \text{ˈgabajo}) = 0.972,
\qquad
S(\text{ˈkabajo}, \text{kaˈbajo}) = 0.889
$$

$b$ scores higher, which matches the listener's answer. Other weights could reverse the ranking; a lower stress salience, for instance. The fit looks for weights that agree with listeners consistently and by a clear margin.

Accuracy alone is a poor objective, because it is a step function: it stays flat under small weight changes until some triad flips. The optimizer needs a smooth objective.

## Paired comparisons

Bradley and Terry (1952) proposed a model for choosing between two items. Each item has a positive strength $\pi$, and item $i$ is chosen over item $j$ with probability

$$
P(i \text{ beats } j) = \frac{\pi_i}{\pi_i + \pi_j}.
$$

Writing the strengths as $\pi = e^{k s}$ turns this into the logistic function of the score difference:

$$
P(i \text{ beats } j)
= \frac{e^{k s_i}}{e^{k s_i} + e^{k s_j}}
= \frac{1}{1 + e^{-k (s_i - s_j)}}.
$$

The probability depends only on $s_i - s_j$, so adding the same constant to every score changes nothing.

More details: [Where this formula comes from](#where-this-formula-comes-from).

## Applied to triads

In a triad the two items are the candidates $b$ and $c$, and each candidate's score is its ALINE similarity to the target:

$$
d = S(\text{target}, b) - S(\text{target}, c)
$$

The probability that a listener picks $b$ is

$$
P(y = b) = \sigma(k d) = \frac{1}{1 + e^{-k d}}.
$$

$k$ controls how deterministic the choices are. As $k \to 0$, choices approach coin flips whatever the similarities. A large $k$ turns even a small similarity margin into a confident choice.

cangu fits $k$ together with the ALINE weights. A lower fitted $k$ means the responses in the data are noisier or less consistent.

For the example triad, $d = 0.972 - 0.889 = 0.083$, and at $k = 40$

$$
P(y = b) = \frac{1}{1 + e^{-40 \cdot 0.083}} = \frac{1}{1 + e^{-3.33}} \approx 0.966.
$$

At $k = 10$ the same margin gives only 0.70.

More details: [Reading the S-shaped curve and the role of k](#reading-the-s-shaped-curve-and-the-role-of-k).

## The loss

The loss for one triad is the negative log-likelihood of the observed answer:

$$
\ell =
\begin{cases}
-\ln P(y = b) & \text{if the answer was } b, \\
-\ln \bigl(1 - P(y = b)\bigr) & \text{if the answer was } c.
\end{cases}
$$

In the example, an observed $b$ costs $-\ln 0.966 \approx 0.035$ and an observed $c$ costs $-\ln 0.034 \approx 3.37$. Confident correct predictions cost little, and confident mistakes cost a lot.

The fit minimizes the mean log-loss over the training triads. Unlike accuracy, log-loss changes continuously as the weights move the probabilities, and it separates a narrow margin from a decisive one. It is also a proper scoring rule: the true probabilities minimize it, so it does not reward overconfidence.

Since $-\ln \sigma(x) = \ln(1 + e^{-x})$, the loss is the softplus function $\operatorname{softplus}(x) = \ln(1 + e^{x})$ evaluated at $-k d$ when the answer is $b$ and at $k d$ when it is $c$. `src/model/loss.rs` implements `softplus` so that it does not overflow when $\lvert k d \rvert$ is large:

```rust
pub fn bt_loss(d: f64, k: f64, label_b: bool) -> f64 {
    if label_b { softplus(-k * d) } else { softplus(k * d) }
}
```

More details: [Why the loss uses a logarithm](#why-the-loss-uses-a-logarithm).

## How cangu computes similarities

### Fixed scale

pho normalizes an alignment score by the larger of the two self-alignment scores:

$$
S(x, y) = \frac{\operatorname{score}(x, y)}{\max\bigl(\operatorname{score}(x, x), \operatorname{score}(y, y)\bigr)}
$$

Multiplying every weight by the same constant scales the numerator and the denominator equally, so the similarity does not change. The data therefore cannot tell you the absolute size of the weights. cangu holds `substitute` fixed and fits the other weights, so their values are in effect ratios to `substitute`.

The overall sensitivity comes from $k$, which scales similarity differences separately from the size of the weights.

More details: [Why only ratios between weights matter](#why-only-ratios-between-weights-matter).

### Unclamped similarity

cangu uses pho's unclamped similarity, which can be negative. The standard similarity is clamped at 0, so two very dissimilar candidates would both score 0, giving $d = 0$ and a 50/50 prediction no matter which one is further away.

### Caching similarity pairs

Triads often share target-candidate pairs. For each set of candidate weights, cangu builds the ALINE config once, computes the similarity of every distinct pair the triads need, and looks up $d$ from that table. The pairs are computed in parallel and stored by index, so the results are the same on every run.

## The baseline

Evaluation compares the fitted weights with a baseline: the default ALINE weights with only $k$ fitted.

For fixed similarities, log-loss is strictly convex in $k$, so `fit_k` in `src/eval/fit.rs` finds the best $k$ directly with a golden-section search over $\ln k$ within the model's bounds.

Fitting $k$ for the baseline as well means that a reported improvement comes from the feature weights and not from a better choice of $k$.

More details: [Finding the best k quickly](#finding-the-best-k-quickly).

## Reading the numbers

Reports give these Bradley-Terry metrics on held-out triads, for both the fitted weights and the baseline:

- Log-loss (`log_loss`) is the mean negative log-likelihood; lower is better. A 50/50 guess scores $\ln 2 \approx 0.693$.
- Accuracy (`accuracy`) is the share of triads where the sign of $d$ matches the listener's choice. A tie ($d = 0$) counts as half correct.
- $k$ is the fitted sensitivity in each evaluation fold.

Fitting the demo dataset `example/dataset.csv`, generated with $k = 40$, gives these held-out scores in the report's `[performance]` section:

| | accuracy | log-loss |
| --- | --- | --- |
| fitted weights (`model`) | 69.3% | 0.5070 |
| default ALINE, $k$ fitted (`baseline`) | 60.7% | 0.6848 |

The fitted weights lower the held-out log-loss from 0.6848 to 0.5070.

## Limits

- All responses share one set of weights and one $k$, so the model describes an average listener and not individual differences.
- Each triad adds to the loss independently, with no grouping of repeated or similar items.
- The probabilities depend only on the difference $d$, so the model captures which candidate is closer and not how far apart the words sound in absolute terms.

## Further reading

- R. A. Bradley and M. E. Terry, "Rank analysis of incomplete block designs: I. The method of paired comparisons", <https://doi.org/10.2307/2334029>.
- G. Kondrak, "A new algorithm for the alignment of phonetic sequences", <https://aclanthology.org/A00-2038.pdf>.
- [CMA-ES in cangu](cma-es.md) covers how the weights are optimized.

## More details

These sections go through the math above more slowly, with worked numbers.

### Where this formula comes from

Think of two players with fixed strengths. If player $i$ has strength 3 and player $j$ has strength 1, the model says $i$ wins three games out of four:

$$
P(i \text{ beats } j) = \frac{3}{3 + 1} = 0.75.
$$

The stronger player usually wins, but not always, and the bigger the gap in strength, the more reliably they win. Two equal players each win half the time.

For words, we do not have strengths, only similarity scores, and a score can be any number. A strength has to be positive, and the exponential function $e^{x}$ turns any number into a positive one, so we set $\pi = e^{k s}$. The constant $k$ controls how much a difference in score matters.

To get from the first form to the second, divide the top and the bottom of the fraction by $e^{k s_i}$:

$$
\frac{e^{k s_i}}{e^{k s_i} + e^{k s_j}}
= \frac{1}{1 + e^{k s_j} / e^{k s_i}}
= \frac{1}{1 + e^{-k (s_i - s_j)}}.
$$

The last step uses $e^{a} / e^{b} = e^{a - b}$. Only the difference $s_i - s_j$ is left, which is why adding the same constant to both scores changes nothing: the constant cancels in the difference.

Back to [Paired comparisons](#paired-comparisons).

### Reading the S-shaped curve and the role of k

The function $\sigma(x) = \frac{1}{1 + e^{-x}}$ is called the logistic or sigmoid function. It takes any number and returns a value between 0 and 1, so it can serve as a probability:

- at $x = 0$ it gives exactly 0.5, a coin flip,
- for large positive $x$ it approaches 1,
- for large negative $x$ it approaches 0.

Here $x = k d$. If both candidates are equally similar to the target ($d = 0$), the model gives each a 50% chance. The more similar $b$ is compared with $c$, the closer $P(y = b)$ gets to 1.

$k$ stretches or squeezes the curve. The table shows $P(y = b)$ for a few similarity differences:

| $d$ | $k = 10$ | $k = 40$ |
| --- | --- | --- |
| 0.01 | 0.525 | 0.599 |
| 0.05 | 0.622 | 0.881 |
| 0.083 | 0.696 | 0.965 |
| 0.2 | 0.881 | 0.9997 |

With $k = 10$, even a fairly clear difference ($d = 0.083$) only gives about 70%. With $k = 40$, the same difference gives about 97%. A dataset where listeners agree with ALINE very consistently leads to a large fitted $k$; a dataset with many surprising answers leads to a small one.

Back to [Applied to triads](#applied-to-triads).

### Why the loss uses a logarithm

The model gives a probability to every observed answer. A good set of weights should make the answers that listeners actually gave look likely. If the triads are independent, the probability of seeing all the answers together is the product of the individual probabilities, for example

$$
0.9 \times 0.8 \times 0.95 \times \dots
$$

This number is called the likelihood. With hundreds of triads, the product becomes tiny and awkward to work with, so we take its logarithm, which turns the product into a sum:

$$
\ln(0.9 \times 0.8 \times 0.95) = \ln 0.9 + \ln 0.8 + \ln 0.95.
$$

Each $\ln P$ is negative (the logarithm of a number below 1 is negative), so we flip the sign to get a positive loss, and divide by the number of triads to get an average. Lower is better.

Some single-triad losses $-\ln P$, where $P$ is the probability the model gave to the answer the listener chose:

| $P$ given to the observed answer | loss |
| --- | --- |
| 0.9 (confident and right) | 0.105 |
| 0.5 (no opinion) | 0.693 |
| 0.1 (confident and wrong) | 2.303 |

A confident mistake costs about 20 times as much as a confident correct answer, and as $P$ approaches 0 the loss grows without limit.

"Proper scoring rule" means the loss rewards honest probabilities. Suppose that, for some triad, 70% of listeners would pick $b$. If the model predicts a probability $q$ for $b$, its average loss over many listeners is $-(0.7 \ln q + 0.3 \ln(1 - q))$:

| predicted $q$ | average loss |
| --- | --- |
| 0.5 | 0.693 |
| 0.7 | 0.611 |
| 0.9 | 0.765 |
| 0.99 | 1.389 |

The lowest loss comes from predicting exactly 0.7. Being more confident than the data justifies costs more, not less.

The softplus rewrite is about computer arithmetic. $e^{x}$ for a large $x$, say $e^{1000}$, is too big to store in a 64-bit floating-point number. `softplus` uses the identity $\ln(1 + e^{x}) = x + \ln(1 + e^{-x})$ for positive $x$, so it never computes $e^{x}$ for a large $x$.

Back to [The loss](#the-loss).

### Why only ratios between weights matter

An ALINE score is a sum of terms, each of them a weight times something that does not depend on the weights (a feature distance, or a count of skipped segments). Doubling every weight therefore doubles every score.

Suppose that for two words $x$ and $y$

$$
\operatorname{score}(x, y) = 30, \quad \operatorname{score}(x, x) = 40, \quad \operatorname{score}(y, y) = 36,
\qquad
S(x, y) = \frac{30}{\max(40, 36)} = 0.75.
$$

With every weight doubled, the scores become 60, 80, and 72, and $S(x, y) = 60 / 80 = 0.75$ again. Only the proportions between the weights change the similarities.

So if the fit were free to move every weight, it could slide all of them up or down together without changing anything, and the optimizer would wander. Fixing `substitute` removes that freedom. A fitted `place` of 70 then means "place counts twice as much as `substitute`", and that is the kind of statement the data can support.

Back to [Fixed scale](#fixed-scale).

### Finding the best k quickly

When the similarities are fixed, the loss depends on $k$ alone, and as a function of $k$ it has a single lowest point with no other dips. That is what "strictly convex" guarantees.

Golden-section search finds that lowest point the way you might search for the deepest spot of a valley by probing. It starts with an interval of possible values, evaluates the loss at two points inside it, and throws away the part of the interval on the side of the worse point. Each round keeps about 61.8% of the interval (the golden ratio, which gives the method its name), so after 20 rounds the interval is less than a ten-thousandth of its original width. It needs no derivatives, only loss values.

The search runs over $\ln k$ instead of $k$, so that "twice as large" counts as the same step whether $k$ is 5 or 50.

Back to [The baseline](#the-baseline).
