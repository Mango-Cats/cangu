//! A [`Fitter`] takes an ALINE config (the language's sound inventory and default
//! weights) and three settings structs.
//!
//! ```no_run
//! use cangu::{EvaluationConfig, Fitter, ModelConfig, OptimizationConfig};
//! # fn my_language_aline() -> pho::algorithms::Aline { unimplemented!() }
//!
//! let fitter = Fitter::new(
//!     my_language_aline(),
//!     ModelConfig {
//!         stress_prior: 10.0,
//!         free_weights: vec!["place".into(), "manner".into(), "stress".into()],
//!         k_init: 20.0,
//!         bound: 5.0,
//!         clamp_penalty: 1.0,
//!     },
//!     OptimizationConfig {
//!         max_evals: 300,
//!         sigma0: 0.5,
//!         tol_fun: 1e-6,
//!         tol_x: 1e-4,
//!         min_bipop_evals: 50,
//!     },
//!     EvaluationConfig {
//!         n_folds: 3,
//!         lambda_grid: vec![0.0, 0.01, 0.1],
//!         learning_fractions: vec![0.5, 1.0],
//!         move_flag_factor: 5.0,
//!     },
//! )?;
//! let dataset = fitter.load_dataset("data.csv".as_ref())?;
//! let result = fitter.fit(&dataset, 1, false)?;
//! println!("stress = {}, k = {}", result.weights.stress, result.k);
//! # Ok::<(), String>(())
//! ```

use crate::config::{EvaluationConfig, ModelConfig, OptimizationConfig};
use crate::data::Dataset;
use crate::eval::fit::{OptimOptions, Prediction, cross_validate, final_fit, predict_theta};
use crate::eval::metrics::{Report, ReportInputs, build_report};
use crate::model::loss::FitData;
use crate::model::params::{ParamFile, ParamSpace, Weights};
use pho::algorithms::Aline;
use std::path::Path;

/// Fits ALINE weights to triad datasets with fixed model, optimizer, and evaluation settings.
pub struct Fitter {
    space: ParamSpace,
    model: ModelConfig,
    opts: OptimOptions,
    evaluation: EvaluationConfig,
}

impl Fitter {
    /// Builds a fitter for one language's ALINE config.
    ///
    /// `aline` provides the sound inventory and the default weights the fit starts from.
    pub fn new(
        aline: Aline,
        model: ModelConfig,
        optimization: OptimizationConfig,
        evaluation: EvaluationConfig,
    ) -> Result<Fitter, String> {
        Ok(Fitter {
            space: ParamSpace::new(aline, &model)?,
            model,
            opts: OptimOptions::from(&optimization),
            evaluation,
        })
    }

    /// The base ALINE config, the default weights, and the bounds.
    pub fn space(&self) -> &ParamSpace {
        &self.space
    }

    /// Loads a dataset and validates it against this fitter's sound inventory.
    pub fn load_dataset(&self, path: &Path) -> Result<Dataset, String> {
        Dataset::load(path, &self.space.base).map_err(|e| e.to_string())
    }

    /// Runs nested grouped cross-validation, then the final fit on all of `dataset`.
    ///
    /// The same dataset, settings, and `seed` always give the same result. With
    /// `learning_curve`, each outer fold is also refit on fractions of its training
    /// targets (`evaluation.learning_fractions`).
    pub fn fit(
        &self,
        dataset: &Dataset,
        seed: u64,
        learning_curve: bool,
    ) -> Result<FitResult, String> {
        let data = FitData::from_rows(&dataset.rows);
        let cv = cross_validate(
            &self.space,
            &data,
            &self.opts,
            &self.evaluation,
            seed,
            learning_curve,
        )?;
        let final_fit = final_fit(&self.space, &data, &self.opts, &self.evaluation, seed)?;
        let theta = final_fit.fit.theta.clone();
        let decoded = self.space.decode(&theta);
        let report = build_report(ReportInputs {
            dataset,
            data: &data,
            space: &self.space,
            seed,
            model: &self.model,
            opts: &self.opts,
            eval_cfg: &self.evaluation,
            cv,
            final_fit,
        })?;
        Ok(FitResult {
            space: self.space.clone(),
            theta,
            weights: decoded.weights,
            k: decoded.k,
            report,
        })
    }
}

/// The result of [`Fitter::fit`]. This has the fitted weights, k, and the evaluation report.
pub struct FitResult {
    space: ParamSpace,
    theta: Vec<f64>,
    /// Every ALINE weight after fitting. Weights not in `free_weights` keep their defaults.
    pub weights: Weights,
    /// Fitted choice sensitivity.
    pub k: f64,
    /// The training report that [`FitResult::write_report`] writes.
    pub report: Report,
}

impl FitResult {
    /// The fitted model's probability of label B for every triad in `dataset`. Each
    /// prediction's `loss` is computed against the dataset's `label` column.
    pub fn predict(&self, dataset: &Dataset) -> Result<Vec<Prediction>, String> {
        let data = FitData::from_rows(&dataset.rows);
        let all: Vec<usize> = (0..data.judgments.len()).collect();
        predict_theta(&self.space, &data, &self.theta, &all)
    }

    /// The fitted weights and k in the parameter-file format.
    pub fn params(&self) -> ParamFile {
        ParamSpace::to_file(&self.weights, Some(self.k))
    }

    /// Writes the training report to `path` as TOML: the settings, k and λ, held-out
    /// scores, flags, and the results of every fold. The fitted weights are not repeated
    /// here; [`FitResult::write_aline`] writes them.
    pub fn write_report(&self, path: &Path) -> Result<(), String> {
        let text = toml::to_string(&self.report).map_err(|e| e.to_string())?;
        std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// The fitted ALINE config: the base config with the fitted saliences and costs.
    /// The fit scored exactly this config.
    pub fn aline(&self) -> Result<Aline, String> {
        self.space.build(&self.weights)
    }

    /// Writes the fitted ALINE config to `path` as a pho TOML file, which
    /// `pho::utils::io::import` loads back. `path` must end in `.toml`.
    pub fn write_aline(&self, path: &Path) -> Result<(), String> {
        let name = path
            .to_str()
            .ok_or_else(|| format!("{}: path is not UTF-8", path.display()))?;
        pho::utils::io::export(name, &self.aline()?, false)
            .map_err(|e| format!("{}: {e}", path.display()))
    }
}
