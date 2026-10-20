//! Optimization, grouped nested cross-validation, and the evaluation report.
//!
//! Submodules:
//! - [`fit`] runs CMA-ES with BIPOP restarts and the cross-validation.
//! - [`metrics`] computes accuracy and log-loss, runs the sanity checks, and renders the report.

pub mod fit;
pub mod metrics;
