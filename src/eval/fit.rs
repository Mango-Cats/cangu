//! Optimization with CMA-ES, and grouped, nested cross-validation.
//!
//! Fits the ALINE weights and the choice sensitivity k with CMA-ES and BIPOP
//! restarts. Cross-validation folds are grouped by target word, so every score
//! comes from targets the fit did not see.

use crate::config::EvaluationConfig;
use crate::eval::metrics::Scores;
use crate::model::loss::{
    judgment_d, mean_loss, prob_b, similarity_table, FitData, Objective,
};
use crate::model::params::ParamSpace;
use cmaes::restart::{RestartOptions, RestartStrategy};
use cmaes::{CMAESOptions, DVector};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::Serialize;
use std::collections::BTreeSet;

/// CMA-ES budget and tolerances.
#[derive(Debug, Clone, Serialize)]
pub struct OptimOptions {
    /// Objective evaluations per fit.
    pub max_evals: usize,
    /// Initial step size of the phase that starts from the defaults.
    pub sigma0: f64,
    /// Stop early when the objective improves by less than this.
    pub tol_fun: f64,
    /// Stop early when the step size falls below this.
    pub tol_x: f64,
    /// Skip the BIPOP restarts when fewer evaluations than this remain.
    pub min_bipop_evals: usize,
}

/// Derives a seed from `seed` and `tags` with SplitMix64, so each fold and phase
/// gets its own reproducible random stream.
pub fn derive_seed(seed: u64, tags: &[u64]) -> u64 {
    fn mix(mut z: u64) -> u64 {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    tags.iter().fold(mix(seed), |acc, &t| mix(acc ^ mix(t)))
}

/// The result of one optimization run (both phases).
#[derive(Debug, Clone, Serialize)]
pub struct FitOutcome {
    /// The λ used.
    pub lambda: f64,
    /// The best parameter vector found.
    pub theta: Vec<f64>,
    /// The objective value at `theta`.
    pub objective: f64,
    /// Objective evaluations used, across both phases.
    pub evals: usize,
    /// Evaluations that hit a non-positive self-alignment score.
    pub invalid_evals: usize,
    /// Number of BIPOP restarts run.
    pub bipop_runs: usize,
    /// The phase that found `theta`: "default_start" or "bipop".
    pub best_from: String,
    /// Why the phase that starts from the defaults stopped.
    pub default_start_termination: Vec<String>,
}

/// Minimizes the training objective: CMA-ES from the defaults, then BIPOP restarts.
pub fn fit_theta(
    space: &ParamSpace,
    data: &FitData,
    train: &[usize],
    lambda: f64,
    opts: &OptimOptions,
    seed: u64,
) -> FitOutcome {
    let objective = Objective::new(space, data, train, lambda);
    let dim = space.dim();
    let center = DVector::from_vec(space.theta_default.clone());

    let first_budget = (opts.max_evals * 2 / 3).max(1);
    let mut state = CMAESOptions::new(space.theta_default.clone(), opts.sigma0)
        .seed(derive_seed(seed, &[0]))
        .tol_fun(opts.tol_fun)
        .tol_x(opts.tol_x)
        .max_function_evals(first_budget)
        .build(|x: &DVector<f64>| objective.value(x.as_slice()))
        .expect("valid CMA-ES options");
    let first = state.run();
    let first_evals = state.function_evals();
    let mut best = first
        .overall_best
        .map(|b| (b.point.as_slice().to_vec(), b.value))
        .unwrap_or_else(|| {
            let t = space.theta_default.clone();
            let v = objective.value(&t);
            (t, v)
        });
    let mut best_from = "default_start".to_string();
    let default_start_termination = first.reasons.iter().map(|r| format!("{r:?}")).collect();

    let remaining = opts.max_evals.saturating_sub(first_evals);
    let mut bipop_runs = 0;
    if remaining >= opts.min_bipop_evals {
        let restarter = RestartOptions::new(
            dim,
            -space.bound..=space.bound,
            RestartStrategy::BIPOP(Default::default()),
        )
        .seed(derive_seed(seed, &[1]))
        .max_function_evals(remaining)
        .max_function_evals_per_run(remaining)
        .build()
        .expect("valid restart options");
        let results = restarter.run_with_reuse(|z: &DVector<f64>| {
            let theta = z + &center;
            objective.value(theta.as_slice())
        });
        bipop_runs = results.runs;
        if let Some(b) = results.best
            && b.value < best.1
        {
            best = ((b.point + &center).as_slice().to_vec(), b.value);
            best_from = "bipop".to_string();
        }
    }

    FitOutcome {
        lambda,
        theta: best.0,
        objective: best.1,
        evals: objective.evals(),
        invalid_evals: objective.invalid_evals(),
        bipop_runs,
        best_from,
        default_start_termination,
    }
}

/// The best k for fixed similarities, found by golden-section search over ln k.
pub fn fit_k(space: &ParamSpace, data: &FitData, table: &[f64], train: &[usize]) -> f64 {
    let f = |log_k: f64| mean_loss(data, table, train, log_k.exp());
    let (lo, hi) = space.k_bounds();
    let (mut a, mut b) = (lo.ln(), hi.ln());
    let g = (5f64.sqrt() - 1.0) / 2.0;
    let mut c = b - g * (b - a);
    let mut d = a + g * (b - a);
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..200 {
        if fc <= fd {
            b = d;
            d = c;
            fd = fc;
            c = b - g * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + g * (b - a);
            fd = f(d);
        }
    }
    ((a + b) / 2.0).exp()
}

/// Partitions judgments into folds grouped by target word: targets are shuffled and
/// dealt round-robin, so every judgment for a target lands in the same fold.
pub fn grouped_folds(
    data: &FitData,
    subset: &[usize],
    n_folds: usize,
    seed: u64,
) -> Result<Vec<Vec<usize>>, String> {
    let mut groups: Vec<usize> = subset
        .iter()
        .map(|&j| data.judgments[j].group)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if groups.len() < n_folds {
        return Err(format!(
            "only {} targets; need at least {n_folds} for {n_folds}-fold grouped CV",
            groups.len()
        ));
    }
    let mut rng = StdRng::seed_from_u64(seed);
    groups.shuffle(&mut rng);
    let mut fold_of_group = vec![usize::MAX; data.groups.len()];
    for (i, &g) in groups.iter().enumerate() {
        fold_of_group[g] = i % n_folds;
    }
    let mut folds = vec![Vec::new(); n_folds];
    for &j in subset {
        folds[fold_of_group[data.judgments[j].group]].push(j);
    }
    Ok(folds)
}

/// The elements of `all` that are not in `exclude`.
fn without(all: &[usize], exclude: &[usize]) -> Vec<usize> {
    let ex: BTreeSet<usize> = exclude.iter().copied().collect();
    all.iter().copied().filter(|j| !ex.contains(j)).collect()
}

/// The result of choosing λ by inner cross-validation.
#[derive(Debug, Clone, Serialize)]
pub struct LambdaSelection {
    /// The λ with the lowest held-out log-loss.
    pub lambda: f64,
    /// Each λ tried, with its mean held-out log-loss.
    pub scores: Vec<(f64, f64)>,
}

/// Picks λ from `eval_cfg.lambda_grid` by grouped cross-validation on `train`.
pub fn select_lambda(
    space: &ParamSpace,
    data: &FitData,
    train: &[usize],
    opts: &OptimOptions,
    eval_cfg: &EvaluationConfig,
    seed: u64,
    label: &str,
) -> Result<LambdaSelection, String> {
    let folds = grouped_folds(data, train, eval_cfg.n_folds, derive_seed(seed, &[0]))
        .map_err(|e| format!("{label}: inner CV: {e}"))?;
    let mut scores = Vec::new();
    for (li, &lambda) in eval_cfg.lambda_grid.iter().enumerate() {
        let mut total = 0.0;
        let mut n = 0usize;
        for (fi, test) in folds.iter().enumerate() {
            let inner_train = without(train, test);
            let fit = fit_theta(
                space,
                data,
                &inner_train,
                lambda,
                opts,
                derive_seed(seed, &[1, li as u64, fi as u64]),
            );
            let preds = predict_theta(space, data, &fit.theta, test)?;
            total += preds.iter().map(|p| p.loss).sum::<f64>();
            n += preds.len();
        }
        let mean = total / n as f64;
        eprintln!("  {label}: lambda = {lambda}: inner held-out log-loss {mean:.4}");
        scores.push((lambda, mean));
    }
    let lambda = scores
        .iter()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|s| s.0)
        .expect("lambda grid is not empty");
    Ok(LambdaSelection { lambda, scores })
}

/// The model's prediction for one triad.
#[derive(Debug, Clone, Serialize)]
pub struct Prediction {
    /// Line number of the triad in the CSV file.
    pub line: usize,
    /// Similarity difference d = S(target, B) - S(target, C).
    pub d: f64,
    /// Predicted probability that B is chosen.
    pub p_b: f64,
    /// Whether the label is B.
    pub label_b: bool,
    /// Bradley-Terry negative log-likelihood of the label.
    pub loss: f64,
}

/// Predictions for `judgments` from a similarity table and k.
pub fn predict(data: &FitData, table: &[f64], judgments: &[usize], k: f64) -> Vec<Prediction> {
    judgments
        .iter()
        .map(|&j| {
            let jd = &data.judgments[j];
            let d = judgment_d(table, jd);
            Prediction {
                line: jd.line,
                d,
                p_b: prob_b(d, k),
                label_b: jd.label_b,
                loss: crate::model::loss::bt_loss(d, k, jd.label_b),
            }
        })
        .collect()
}

/// Predictions for `judgments` under the parameter vector `theta`.
pub fn predict_theta(
    space: &ParamSpace,
    data: &FitData,
    theta: &[f64],
    judgments: &[usize],
) -> Result<Vec<Prediction>, String> {
    let dec = space.decode(theta);
    let aline = space.build(&dec.weights)?;
    let table = similarity_table(&aline, &data.pairs, &data.pairs_of(judgments))?;
    Ok(predict(data, &table, judgments, dec.k))
}

/// Similarities of every pair under the default weights.
pub fn baseline_table(space: &ParamSpace, data: &FitData) -> Result<Vec<f64>, String> {
    let aline = space.build(&space.defaults)?;
    let all: Vec<usize> = (0..data.pairs.len()).collect();
    similarity_table(&aline, &data.pairs, &all)
}

/// One point on the learning curve.
#[derive(Debug, Clone, Serialize)]
pub struct LearningPoint {
    /// Fraction of the training targets used.
    pub fraction: f64,
    /// Number of training targets used.
    pub n_train_targets: usize,
    /// Held-out scores.
    pub scores: Scores,
}

/// Results for one outer fold.
#[derive(Debug, Clone, Serialize)]
pub struct FoldResult {
    /// Fold index, starting at 0.
    pub fold: usize,
    /// Number of training triads.
    pub n_train: usize,
    /// Number of held-out triads.
    pub n_test: usize,
    /// Number of held-out target words.
    pub n_test_targets: usize,
    /// How λ was chosen for this fold.
    pub lambda: LambdaSelection,
    /// The fit on this fold's training triads.
    pub fit: FitOutcome,
    /// k fitted for the default weights (the baseline).
    pub baseline_k: f64,
    /// Held-out predictions of the fitted weights.
    pub model: Vec<Prediction>,
    /// Held-out predictions of the baseline.
    pub baseline: Vec<Prediction>,
    /// The learning curve, if requested.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub learning_curve: Vec<LearningPoint>,
}

/// Results of the outer cross-validation.
#[derive(Debug, Clone, Serialize)]
pub struct CvResult {
    /// One result per fold.
    pub folds: Vec<FoldResult>,
}

/// Runs the outer grouped cross-validation, choosing λ by inner cross-validation in each fold.
pub fn cross_validate(
    space: &ParamSpace,
    data: &FitData,
    opts: &OptimOptions,
    eval_cfg: &EvaluationConfig,
    seed: u64,
    learning_curve: bool,
) -> Result<CvResult, String> {
    let all: Vec<usize> = (0..data.judgments.len()).collect();
    let folds = grouped_folds(data, &all, eval_cfg.n_folds, derive_seed(seed, &[1]))
        .map_err(|e| format!("outer CV: {e}"))?;
    let base_table = baseline_table(space, data)?;
    let mut results = Vec::new();
    for (f, test) in folds.iter().enumerate() {
        let train = without(&all, test);
        let label = format!("outer fold {}/{}", f + 1, eval_cfg.n_folds);
        eprintln!("{label}: {} train / {} test judgments", train.len(), test.len());
        let lambda = select_lambda(
            space,
            data,
            &train,
            opts,
            eval_cfg,
            derive_seed(seed, &[2, f as u64]),
            &label,
        )?;
        let fit = fit_theta(
            space,
            data,
            &train,
            lambda.lambda,
            opts,
            derive_seed(seed, &[3, f as u64]),
        );
        eprintln!(
            "{label}: lambda = {}, objective {:.4}, {} evals ({} invalid), best from {}",
            lambda.lambda, fit.objective, fit.evals, fit.invalid_evals, fit.best_from
        );
        let model = predict_theta(space, data, &fit.theta, test)?;
        let baseline_k = fit_k(space, data, &base_table, &train);
        let baseline = predict(data, &base_table, test, baseline_k);

        let mut curve = Vec::new();
        if learning_curve {
            let groups: Vec<usize> = train
                .iter()
                .map(|&j| data.judgments[j].group)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            for (fi, &fraction) in eval_cfg.learning_fractions.iter().enumerate() {
                let n = ((groups.len() as f64 * fraction).ceil() as usize).clamp(1, groups.len());
                let mut order = groups.clone();
                order.shuffle(&mut StdRng::seed_from_u64(derive_seed(
                    seed,
                    &[4, f as u64, fi as u64],
                )));
                let keep: BTreeSet<usize> = order[..n].iter().copied().collect();
                let sub: Vec<usize> = train
                    .iter()
                    .copied()
                    .filter(|&j| keep.contains(&data.judgments[j].group))
                    .collect();
                let sub_fit = if n == groups.len() {
                    fit.clone()
                } else {
                    fit_theta(
                        space,
                        data,
                        &sub,
                        lambda.lambda,
                        opts,
                        derive_seed(seed, &[5, f as u64, fi as u64]),
                    )
                };
                let preds = predict_theta(space, data, &sub_fit.theta, test)?;
                eprintln!("{label}: learning curve {:.0}% of targets done", fraction * 100.0);
                curve.push(LearningPoint {
                    fraction,
                    n_train_targets: n,
                    scores: Scores::from_predictions(&preds),
                });
            }
        }

        results.push(FoldResult {
            fold: f,
            n_train: train.len(),
            n_test: test.len(),
            n_test_targets: test
                .iter()
                .map(|&j| data.judgments[j].group)
                .collect::<BTreeSet<_>>()
                .len(),
            lambda,
            fit,
            baseline_k,
            model,
            baseline,
            learning_curve: curve,
        });
    }
    Ok(CvResult { folds: results })
}

/// The final fit on all triads, with its in-sample predictions.
#[derive(Debug, Clone, Serialize)]
pub struct FinalFit {
    /// How λ was chosen by grouped cross-validation.
    pub lambda: LambdaSelection,
    /// The fit on the whole dataset.
    pub fit: FitOutcome,
    /// The fitted k.
    pub k: f64,
    /// In-sample predictions for every triad.
    pub train: Vec<Prediction>,
}

/// Chooses λ by cross-validation on all triads, then fits on all of them.
pub fn final_fit(
    space: &ParamSpace,
    data: &FitData,
    opts: &OptimOptions,
    eval_cfg: &EvaluationConfig,
    seed: u64,
) -> Result<FinalFit, String> {
    let all: Vec<usize> = (0..data.judgments.len()).collect();
    let label = "final fit";
    eprintln!("{label}: {} judgments", all.len());
    let lambda = select_lambda(space, data, &all, opts, eval_cfg, derive_seed(seed, &[6]), label)?;
    let fit = fit_theta(space, data, &all, lambda.lambda, opts, derive_seed(seed, &[7]));
    eprintln!(
        "{label}: lambda = {}, objective {:.4}, {} evals ({} invalid), best from {}",
        lambda.lambda, fit.objective, fit.evals, fit.invalid_evals, fit.best_from
    );
    let train = predict_theta(space, data, &fit.theta, &all)?;
    let k = space.decode(&fit.theta).k;
    Ok(FinalFit {
        lambda,
        fit,
        k,
        train,
    })
}
