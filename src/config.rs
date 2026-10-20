//! Fit settings for the ALINE parameter space, the CMA-ES optimizer, and evaluation.
//!
//! Every field has to be set; there are no defaults.

use serde::Serialize;

use crate::eval::fit::OptimOptions;

/// Which ALINE weights to fit and how far they may move.
#[derive(Debug, Clone, Serialize)]
pub struct ModelConfig {
    /// Starting stress salience, used when the ALINE config sets stress to 0.
    pub stress_prior: f64,
    /// Names of the weights to fit, in parameter-vector order.
    pub free_weights: Vec<String>,
    /// Starting value of the choice sensitivity k.
    pub k_init: f64,
    /// How far each weight may move from its default, in log space.
    pub bound: f64,
    /// Weight of the quadratic penalty for moving past `bound`.
    pub clamp_penalty: f64,
}

/// CMA-ES settings.
#[derive(Debug, Clone, Serialize)]
pub struct OptimizationConfig {
    /// Objective evaluations per fit.
    pub max_evals: usize,
    /// Initial CMA-ES step size around the default weights, in log space.
    pub sigma0: f64,
    /// Stop early when the objective improves by less than this.
    pub tol_fun: f64,
    /// Stop early when the step size falls below this.
    pub tol_x: f64,
    /// Skip the BIPOP restart phase when fewer evaluations than this remain.
    pub min_bipop_evals: usize,
}

impl From<&OptimizationConfig> for OptimOptions {
    fn from(opt: &OptimizationConfig) -> Self {
        OptimOptions {
            max_evals: opt.max_evals,
            sigma0: opt.sigma0,
            tol_fun: opt.tol_fun,
            tol_x: opt.tol_x,
            min_bipop_evals: opt.min_bipop_evals,
        }
    }
}

/// Cross-validation and report settings.
#[derive(Debug, Clone, Serialize)]
pub struct EvaluationConfig {
    /// Number of cross-validation folds, grouped by target word.
    pub n_folds: usize,
    /// Regularization strengths compared by inner cross-validation.
    pub lambda_grid: Vec<f64>,
    /// Fractions of the training targets used for the learning curve.
    pub learning_fractions: Vec<f64>,
    /// Flag fitted weights that move by more than this factor from their default.
    pub move_flag_factor: f64,
}
