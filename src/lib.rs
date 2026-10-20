pub mod config;
pub mod data;
pub mod eval;
pub mod fitter;
pub mod model;

pub use config::{EvaluationConfig, ModelConfig, OptimizationConfig};
pub use fitter::{FitResult, Fitter};
