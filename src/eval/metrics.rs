//! Scores, sanity checks, and the evaluation report.
//!
//! Computes accuracy, log-loss, the similarity distribution, and flags worth checking.

use crate::config::{EvaluationConfig, ModelConfig};
use crate::data::Dataset;
use crate::eval::fit::{CvResult, FinalFit, OptimOptions, Prediction};
use crate::model::loss::similarity_table;
use crate::model::params::{ParamSpace, Weights};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

/// Accuracy and log-loss over a set of predictions.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Scores {
    /// Number of triads. Not written to the report, which records the counts once.
    #[serde(skip)]
    pub n: usize,
    /// Share of triads where the sign of d matches the label (d = 0 counts as 0.5).
    pub accuracy: f64,
    /// Mean Bradley-Terry negative log-likelihood.
    pub log_loss: f64,
    /// Number of triads where both candidates had the same similarity.
    pub ties: usize,
}

impl Scores {
    /// Scores for `preds`.
    pub fn from_predictions(preds: &[Prediction]) -> Scores {
        let refs: Vec<&Prediction> = preds.iter().collect();
        Scores::from_refs(&refs)
    }

    /// Scores for borrowed predictions.
    fn from_refs(preds: &[&Prediction]) -> Scores {
        if preds.is_empty() {
            return Scores::default();
        }
        let mut correct = 0.0;
        let mut ties = 0;
        let mut loss = 0.0;
        for p in preds {
            if p.d == 0.0 {
                correct += 0.5;
                ties += 1;
            } else if (p.d > 0.0) == p.label_b {
                correct += 1.0;
            }
            loss += p.loss;
        }
        Scores {
            n: preds.len(),
            accuracy: correct / preds.len() as f64,
            log_loss: loss / preds.len() as f64,
            ties,
        }
    }
}

/// Distribution of target-candidate similarities.
#[derive(Debug, Clone, Serialize)]
pub struct Distribution {
    /// Number of distinct pairs. Not written to the report, which has it under `settings`.
    #[serde(skip)]
    pub n_pairs: usize,
    /// Pairs whose target and candidate are the same string.
    pub n_identical_strings: usize,
    /// Lowest similarity.
    pub min: f64,
    /// 25th percentile.
    pub q25: f64,
    /// Median.
    pub median: f64,
    /// 75th percentile.
    pub q75: f64,
    /// Highest similarity.
    pub max: f64,
    /// Share of similarities at or below 0.
    pub share_le_0: f64,
    /// Share of similarities at or above 0.999.
    pub share_ge_0_999: f64,
}

impl Distribution {
    /// The distribution of `values`, which need not be sorted.
    pub fn of(values: &[f64], n_identical_strings: usize) -> Distribution {
        let mut v = values.to_vec();
        v.sort_by(f64::total_cmp);
        let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
        let n = v.len() as f64;
        Distribution {
            n_pairs: v.len(),
            n_identical_strings,
            min: v[0],
            q25: q(0.25),
            median: q(0.5),
            q75: q(0.75),
            max: v[v.len() - 1],
            share_le_0: v.iter().filter(|&&x| x <= 0.0).count() as f64 / n,
            share_ge_0_999: v.iter().filter(|&&x| x >= 0.999).count() as f64 / n,
        }
    }
}

/// Held-out scores of the fitted weights and of the baseline (default weights with k fitted).
#[derive(Debug, Clone, Serialize)]
pub struct Performance {
    /// Scores of the fitted weights.
    pub model: Scores,
    /// Scores of the baseline.
    pub baseline: Scores,
}

/// The predictions for one held-out triad.
#[derive(Debug, Clone, Serialize)]
pub struct TriadPrediction {
    /// Line number of the triad in the CSV file.
    pub line: usize,
    /// Probability of label B under the fold's fitted weights.
    pub p_b: f64,
    /// Probability of label B under the baseline.
    pub baseline_p_b: f64,
}

/// Results for one outer fold.
#[derive(Debug, Clone, Serialize)]
pub struct FoldReport {
    /// Fold index, starting at 0.
    pub fold: usize,
    /// Number of training triads.
    pub n_train: usize,
    /// Number of held-out triads.
    pub n_test: usize,
    /// Number of held-out target words.
    pub n_test_targets: usize,
    /// The λ chosen by inner cross-validation.
    pub lambda: f64,
    /// Inner cross-validation log-loss for each value in `settings.evaluation.lambda_grid`.
    pub lambda_log_loss: Vec<f64>,
    /// k fitted together with the weights.
    pub k: f64,
    /// k fitted for the baseline.
    pub baseline_k: f64,
    /// Objective evaluations used.
    pub evals: usize,
    /// Evaluations that hit a non-positive self-alignment score.
    pub invalid_evals: usize,
    /// Number of BIPOP restarts run.
    pub bipop_runs: usize,
    /// The phase that found the fold's weights: "default_start" or "bipop".
    pub best_from: String,
    /// Why the phase that starts from the defaults stopped.
    pub default_start_termination: Vec<String>,
    /// The fitted weights in this fold, by name.
    pub weights: BTreeMap<String, f64>,
    /// Held-out scores.
    pub performance: Performance,
    /// One entry per held-out triad.
    pub predictions: Vec<TriadPrediction>,
}

/// One learning-curve point, pooled over the outer folds.
#[derive(Debug, Clone, Serialize)]
pub struct LearningCurvePoint {
    /// Fraction of the training targets used.
    pub fraction: f64,
    /// Held-out scores.
    #[serde(flatten)]
    pub scores: Scores,
}

/// The run's settings and dataset.
#[derive(Debug, Clone, Serialize)]
pub struct Settings {
    /// Path of the dataset.
    pub data: String,
    /// pho revision used.
    pub pho_rev: &'static str,
    /// Seed of the run.
    pub seed: u64,
    /// Number of triads.
    pub n_triads: usize,
    /// Number of target words.
    pub n_targets: usize,
    /// Number of distinct target-candidate pairs.
    pub n_pairs: usize,
    /// Model settings as given to [`crate::Fitter::new`].
    pub model: ModelConfig,
    /// Optimizer settings.
    pub optimization: OptimOptions,
    /// Evaluation settings.
    pub evaluation: EvaluationConfig,
}

/// The final fit on the whole dataset. The fitted weights themselves are in the ALINE
/// TOML written by [`crate::FitResult::write_aline`].
#[derive(Debug, Clone, Serialize)]
pub struct FinalFitReport {
    /// The fitted k.
    pub k: f64,
    /// The λ chosen by cross-validation on the whole dataset.
    pub lambda: f64,
    /// Cross-validation log-loss for each value in `settings.evaluation.lambda_grid`.
    pub lambda_log_loss: Vec<f64>,
    /// Objective evaluations used.
    pub evals: usize,
    /// Evaluations that hit a non-positive self-alignment score.
    pub invalid_evals: usize,
    /// The phase that found the final weights: "default_start" or "bipop".
    pub best_from: String,
    /// The values the fitted weights started from: the ALINE config's values, or
    /// `stress_prior` for a stress salience of 0.
    pub start_weights: BTreeMap<String, f64>,
}

/// The training report. Together with the fitted ALINE TOML it records the whole run;
/// nothing in one file is repeated in the other.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    /// Conditions worth checking, as messages.
    pub flags: Vec<String>,
    /// The run's settings and dataset.
    pub settings: Settings,
    /// The final fit.
    pub final_fit: FinalFitReport,
    /// Held-out scores, pooled over the outer folds.
    pub performance: Performance,
    /// The learning curve, if requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub learning_curve: Option<Vec<LearningCurvePoint>>,
    /// Distribution of similarities under the final fitted weights.
    pub similarity_distribution: Distribution,
    /// Results per outer fold.
    pub folds: Vec<FoldReport>,
}

/// The pho revision pinned in Cargo.toml. Update it whenever pho is bumped.
pub const PHO_REV: &str = "945a983edcd4456ad2c55c26f65a00d9aac36446";

/// Everything needed to build a report.
pub struct ReportInputs<'a> {
    /// The dataset.
    pub dataset: &'a Dataset,
    /// The dataset prepared for fitting.
    pub data: &'a crate::model::loss::FitData,
    /// The parameter space.
    pub space: &'a ParamSpace,
    /// Seed of the run.
    pub seed: u64,
    /// Model settings.
    pub model: &'a ModelConfig,
    /// Optimizer settings.
    pub opts: &'a OptimOptions,
    /// Evaluation settings.
    pub eval_cfg: &'a EvaluationConfig,
    /// Outer cross-validation results.
    pub cv: CvResult,
    /// The final fit on the whole dataset.
    pub final_fit: FinalFit,
}

/// The fitted weights in `weights`, by name.
fn free_weights(space: &ParamSpace, weights: &Weights) -> BTreeMap<String, f64> {
    space.free.iter().map(|n| (n.clone(), weights.get(n))).collect()
}

/// Builds the report from the cross-validation and the final fit.
pub fn build_report(inp: ReportInputs<'_>) -> Result<Report, String> {
    let space = inp.space;
    let move_flag_factor = inp.eval_cfg.move_flag_factor;
    let dec = space.decode(&inp.final_fit.fit.theta);
    let fitted = dec.weights.clone();
    let log_loss_only = |scores: &[(f64, f64)]| scores.iter().map(|s| s.1).collect::<Vec<f64>>();

    let all_model: Vec<Prediction> = inp.cv.folds.iter().flat_map(|f| f.model.clone()).collect();
    let all_base: Vec<Prediction> = inp
        .cv
        .folds
        .iter()
        .flat_map(|f| f.baseline.clone())
        .collect();
    let performance = Performance {
        model: Scores::from_predictions(&all_model),
        baseline: Scores::from_predictions(&all_base),
    };
    let folds = inp
        .cv
        .folds
        .iter()
        .map(|f| {
            let dec = space.decode(&f.fit.theta);
            FoldReport {
                fold: f.fold,
                n_train: f.n_train,
                n_test: f.n_test,
                n_test_targets: f.n_test_targets,
                lambda: f.lambda.lambda,
                lambda_log_loss: log_loss_only(&f.lambda.scores),
                k: dec.k,
                baseline_k: f.baseline_k,
                evals: f.fit.evals,
                invalid_evals: f.fit.invalid_evals,
                bipop_runs: f.fit.bipop_runs,
                best_from: f.fit.best_from.clone(),
                default_start_termination: f.fit.default_start_termination.clone(),
                weights: free_weights(space, &dec.weights),
                performance: Performance {
                    model: Scores::from_predictions(&f.model),
                    baseline: Scores::from_predictions(&f.baseline),
                },
                predictions: f
                    .model
                    .iter()
                    .zip(&f.baseline)
                    .map(|(m, b)| TriadPrediction {
                        line: m.line,
                        p_b: m.p_b,
                        baseline_p_b: b.p_b,
                    })
                    .collect(),
            }
        })
        .collect();

    let learning_curve = inp
        .cv
        .folds
        .first()
        .filter(|f| !f.learning_curve.is_empty())
        .map(|first| {
            (0..first.learning_curve.len())
                .map(|i| {
                    let pts: Vec<&crate::eval::fit::LearningPoint> =
                        inp.cv.folds.iter().map(|f| &f.learning_curve[i]).collect();
                    let n: usize = pts.iter().map(|p| p.scores.n).sum();
                    let w = |g: fn(&Scores) -> f64| {
                        pts.iter()
                            .map(|p| g(&p.scores) * p.scores.n as f64)
                            .sum::<f64>()
                            / n as f64
                    };
                    LearningCurvePoint {
                        fraction: pts[0].fraction,
                        scores: Scores {
                            n,
                            accuracy: w(|s| s.accuracy),
                            log_loss: w(|s| s.log_loss),
                            ties: pts.iter().map(|p| p.scores.ties).sum(),
                        },
                    }
                })
                .collect()
        });

    let mut pairs: Vec<(String, String)> = Vec::new();
    let mut seen = BTreeSet::new();
    for r in &inp.dataset.rows {
        for c in [&r.b, &r.c] {
            let p = (r.target.clone(), c.clone());
            if seen.insert(p.clone()) {
                pairs.push(p);
            }
        }
    }
    let aline = space.build(&fitted)?;
    let idx: Vec<usize> = (0..pairs.len()).collect();
    let sims = similarity_table(&aline, &pairs, &idx)?;
    let identical = pairs.iter().filter(|(t, c)| t == c).count();
    let similarity_distribution = Distribution::of(&sims, identical);

    let mut flags = Vec::new();
    if let Some(p) = space.stress_prior_used {
        flags.push(format!(
            "RESEARCHER CHOICE: ALINE config has no stress salience; stress default (theta_default and lambda-penalty centre) set to stress_prior = {p}"
        ));
    }
    for name in &space.free {
        let (default, value) = (space.defaults.get(name), fitted.get(name));
        let factor = value / default;
        if factor > move_flag_factor || factor < 1.0 / move_flag_factor {
            flags.push(format!(
                "weight moved more than {move_flag_factor}x from default: {name} ({factor:.3}x default)"
            ));
        }
    }
    let at_bound: Vec<&str> = space
        .param_names()
        .into_iter()
        .enumerate()
        .filter(|(i, _)| {
            (dec.theta_clamped[*i] - space.theta_default[*i]).abs() >= space.bound - 1e-6
        })
        .map(|(_, n)| n)
        .collect();
    if !at_bound.is_empty() {
        flags.push(format!(
            "parameters at the theta_default +/- {} bound: {}",
            space.bound,
            at_bound.join(", ")
        ));
    }
    let (k_lo, k_hi) = space.k_bounds();
    let k_at_bound: Vec<String> = inp
        .cv
        .folds
        .iter()
        .filter(|f| f.baseline_k < k_lo * 1.01 || f.baseline_k > k_hi * 0.99)
        .map(|f| format!("{}", f.fold + 1))
        .collect();
    if !k_at_bound.is_empty() {
        flags.push(format!(
            "baseline k at its bound [{k_lo:.3}, {k_hi:.1}] in outer fold(s) {}",
            k_at_bound.join(", ")
        ));
    }
    let invalid: usize = inp
        .cv
        .folds
        .iter()
        .map(|f| f.fit.invalid_evals)
        .sum::<usize>()
        + inp.final_fit.fit.invalid_evals;
    if invalid > 0 {
        flags.push(format!(
            "{invalid} objective evaluations hit NonPositiveSelfScore and were given value {}",
            crate::model::loss::INVALID_OBJECTIVE
        ));
    }

    Ok(Report {
        flags,
        settings: Settings {
            data: inp.dataset.path.display().to_string(),
            pho_rev: PHO_REV,
            seed: inp.seed,
            n_triads: inp.data.judgments.len(),
            n_targets: inp.data.groups.len(),
            n_pairs: inp.data.pairs.len(),
            model: inp.model.clone(),
            optimization: inp.opts.clone(),
            evaluation: inp.eval_cfg.clone(),
        },
        final_fit: FinalFitReport {
            k: inp.final_fit.k,
            lambda: inp.final_fit.lambda.lambda,
            lambda_log_loss: log_loss_only(&inp.final_fit.lambda.scores),
            evals: inp.final_fit.fit.evals,
            invalid_evals: inp.final_fit.fit.invalid_evals,
            best_from: inp.final_fit.fit.best_from.clone(),
            start_weights: free_weights(space, &space.defaults),
        },
        performance,
        learning_curve,
        similarity_distribution,
        folds,
    })
}

/// The ALINE weights encoded in the parameter vector `theta`.
pub fn weights_of(space: &ParamSpace, theta: &[f64]) -> Weights {
    space.decode(theta).weights
}
