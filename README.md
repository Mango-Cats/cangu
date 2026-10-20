# cangu

**Purpose**: cangu fits ALINE phonetic similarity parameters to model auditory confusability and near-homophone judgments from triad perceptual data.

**Why the name?** The name derives from Vietnamese cá ngừ (tuna), reflecting its role in tuning ALINE parameters for phonetic similarity. Also see  and [bowl](https://github.com/Mango-Cats/bowl).

## Abstract

- **Similarity computation**: ALINE global dynamic programming alignment implemented in [pho](https://github.com/Mango-Cats/pho), using the MangoCats (stress-aware) variant.
- **Choice probability**: Bradley-Terry logistic model predicting candidate preference based on similarity differences scaled by sensitivity $k$.
- **Optimization parameters**: Log-transformed free weights and log $k$ bounded within an L2 shrinkage penalty toward ALINE defaults.
- **Search algorithm**: Covariance Matrix Adaptation Evolution Strategy (CMA-ES) combined with BIPOP restarts.
- **Validation**: Grouped cross-validation clustered by target word, tuning regularization parameter lambda on inner folds.

## Docs

For a tutorial and short write ups on the mathematical foundations, see:

- [Tutorial](docs/tutorial.md): Step-by-step guide from build to fitted weights.
- [Bradley-Terry choice model](docs/bradley-terry.md): Triad scoring formulation and loss derivation.
- [CMA-ES optimization](docs/cma-es.md): Derivative-free search strategy and restart schedules.
