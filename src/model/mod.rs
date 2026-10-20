//! Mapping between optimizer parameters and ALINE weights, and the Bradley-Terry loss.
//!
//! Submodules:
//! - [`params`] converts between the unconstrained parameter vector and ALINE weights.
//! - [`loss`] has the Bradley-Terry triad loss, the training objective, and the similarity table.

pub mod loss;
pub mod params;
