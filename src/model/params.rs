//! Conversion between the optimizer's parameter vector and ALINE weights.
//!
//! ALINE needs non-negative saliences and costs, with a non-positive skip cost. The
//! optimizer works on the logarithms of the fitted weights and of the sensitivity k,
//! which can take any real value.

use crate::config::ModelConfig;
use pho::algorithms::aline::config::{Costs, Salience};
use pho::algorithms::Aline;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Every ALINE feature salience and edit cost.
///
/// Following ALINE, the `skip` cost is negative.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Weights {
    /// Salience of syllabicity.
    pub syllabic: f64,
    /// Salience of place of articulation.
    pub place: f64,
    /// Salience of manner of articulation.
    pub manner: f64,
    /// Salience of voicing.
    pub voice: f64,
    /// Salience of nasality.
    pub nasal: f64,
    /// Salience of retroflexion.
    pub retroflex: f64,
    /// Salience of laterality.
    pub lateral: f64,
    /// Salience of aspiration.
    pub aspirated: f64,
    /// Salience of vowel length.
    pub long: f64,
    /// Salience of vowel height.
    pub high: f64,
    /// Salience of vowel backness.
    pub back: f64,
    /// Salience of rounding.
    pub round: f64,
    /// Salience of phonation type.
    pub phonation: f64,
    /// Salience of airstream mechanism.
    pub airstream: f64,
    /// Salience of secondary articulation.
    pub secondary: f64,
    /// Salience of stress (used by the MangoCats variant).
    pub stress: f64,
    /// Cost of skipping a segment (insertion or deletion); negative.
    pub skip: f64,
    /// Substitution cost. It stays fixed and sets the scale for the other weights.
    pub substitute: f64,
    /// Cost of an expansion or compression (one segment aligned with two).
    pub expand_compress: f64,
    /// Extra cost of aligning a vowel with a consonant.
    pub vowel_consonant: f64,
}

/// Names of all ALINE weights, in the order of the [`Weights`] fields.
pub const ALL_WEIGHTS: [&str; 20] = [
    "syllabic",
    "place",
    "manner",
    "voice",
    "nasal",
    "retroflex",
    "lateral",
    "aspirated",
    "long",
    "high",
    "back",
    "round",
    "phonation",
    "airstream",
    "secondary",
    "stress",
    "skip",
    "substitute",
    "expand_compress",
    "vowel_consonant",
];

impl Weights {
    /// The weight called `name`. Panics on an unknown name.
    pub fn get(&self, name: &str) -> f64 {
        *self.field(name)
    }

    /// Sets the weight called `name`. Panics on an unknown name.
    pub fn set(&mut self, name: &str, value: f64) {
        *self.field_mut(name) = value;
    }

    /// The field holding the weight called `name`.
    fn field(&self, name: &str) -> &f64 {
        match name {
            "syllabic" => &self.syllabic,
            "place" => &self.place,
            "manner" => &self.manner,
            "voice" => &self.voice,
            "nasal" => &self.nasal,
            "retroflex" => &self.retroflex,
            "lateral" => &self.lateral,
            "aspirated" => &self.aspirated,
            "long" => &self.long,
            "high" => &self.high,
            "back" => &self.back,
            "round" => &self.round,
            "phonation" => &self.phonation,
            "airstream" => &self.airstream,
            "secondary" => &self.secondary,
            "stress" => &self.stress,
            "skip" => &self.skip,
            "substitute" => &self.substitute,
            "expand_compress" => &self.expand_compress,
            "vowel_consonant" => &self.vowel_consonant,
            _ => panic!("unknown weight '{name}'"),
        }
    }

    /// The field holding the weight called `name`, mutably.
    fn field_mut(&mut self, name: &str) -> &mut f64 {
        match name {
            "syllabic" => &mut self.syllabic,
            "place" => &mut self.place,
            "manner" => &mut self.manner,
            "voice" => &mut self.voice,
            "nasal" => &mut self.nasal,
            "retroflex" => &mut self.retroflex,
            "lateral" => &mut self.lateral,
            "aspirated" => &mut self.aspirated,
            "long" => &mut self.long,
            "high" => &mut self.high,
            "back" => &mut self.back,
            "round" => &mut self.round,
            "phonation" => &mut self.phonation,
            "airstream" => &mut self.airstream,
            "secondary" => &mut self.secondary,
            "stress" => &mut self.stress,
            "skip" => &mut self.skip,
            "substitute" => &mut self.substitute,
            "expand_compress" => &mut self.expand_compress,
            "vowel_consonant" => &mut self.vowel_consonant,
            _ => panic!("unknown weight '{name}'"),
        }
    }

    /// The saliences and costs of an ALINE config.
    pub fn from_config(aline: &Aline) -> Weights {
        let s = &aline.salience;
        let c = &aline.costs;
        Weights {
            syllabic: s.syllabic() as f64,
            place: s.place() as f64,
            manner: s.manner() as f64,
            voice: s.voice() as f64,
            nasal: s.nasal() as f64,
            retroflex: s.retroflex() as f64,
            lateral: s.lateral() as f64,
            aspirated: s.aspirated() as f64,
            long: s.long() as f64,
            high: s.high() as f64,
            back: s.back() as f64,
            round: s.round() as f64,
            phonation: s.phonation() as f64,
            airstream: s.airstream() as f64,
            secondary: s.secondary() as f64,
            stress: s.stress() as f64,
            skip: c.skip() as f64,
            substitute: c.substitute() as f64,
            expand_compress: c.expand_compress() as f64,
            vowel_consonant: c.vowel_consonant() as f64,
        }
    }
}

/// Weights and an optional k, as stored in a parameter file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamFile {
    /// Weight values by name.
    pub weights: serde_json::Map<String, serde_json::Value>,
    /// Choice sensitivity k, if present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub k: Option<f64>,
}

/// A parameter vector converted back to weights.
#[derive(Debug, Clone)]
pub struct Decoded {
    /// The ALINE weights.
    pub weights: Weights,
    /// The choice sensitivity k.
    pub k: f64,
    /// The parameter vector after clamping to the bounds.
    pub theta_clamped: Vec<f64>,
    /// Sum of squared distances by which the parameters exceeded their bounds.
    pub clamp_excess_sq: f64,
}

/// The parameters being fitted, their defaults and bounds, and the base ALINE config.
#[derive(Debug, Clone)]
pub struct ParamSpace {
    /// The base ALINE config; fitted weights are written into a copy of it.
    pub base: Aline,
    /// The default weights the fit starts from.
    pub defaults: Weights,
    /// The default parameter vector, in log space.
    pub theta_default: Vec<f64>,
    /// The `stress_prior` used, if the base config had a stress salience of 0.
    pub stress_prior_used: Option<f64>,
    /// Names of the fitted weights, in parameter-vector order.
    pub free: Vec<String>,
    /// How far each parameter may move from its default, in log space.
    pub bound: f64,
    /// Weight of the quadratic penalty for moving past `bound`.
    pub clamp_penalty: f64,
    /// Starting value of k.
    pub k_init: f64,
}

impl ParamSpace {
    /// Sets up the parameter space around the weights of the ALINE config `base`.
    pub fn new(base: Aline, model: &ModelConfig) -> Result<ParamSpace, String> {
        let free = validate_free_weights(&model.free_weights)?;
        for (name, v) in [
            ("bound", model.bound),
            ("k_init", model.k_init),
        ] {
            if !(v > 0.0 && v.is_finite()) {
                return Err(format!("{name} is {v}; it must be a positive number"));
            }
        }
        if !(model.clamp_penalty >= 0.0) {
            return Err(format!(
                "clamp_penalty is {}; it must be >= 0",
                model.clamp_penalty
            ));
        }
        let stress_prior = model.stress_prior;
        let mut defaults = Weights::from_config(&base);
        let mut stress_prior_used = None;
        if defaults.stress == 0.0 {
            if stress_prior.is_nan() || stress_prior <= 0.0 {
                return Err(format!(
                    "config has stress salience 0 and stress_prior is {stress_prior}; it must be > 0"
                ));
            }
            defaults.stress = stress_prior;
            stress_prior_used = Some(stress_prior);
        }
        let mut theta_default = Vec::with_capacity(free.len() + 1);
        for name in &free {
            let v = defaults.get(name).abs();
            if v <= 0.0 {
                return Err(format!(
                    "default for fitted weight '{name}' is 0, so it has no log; set it in the config"
                ));
            }
            theta_default.push(v.ln());
        }
        theta_default.push(model.k_init.ln());
        Ok(ParamSpace {
            base,
            defaults,
            theta_default,
            stress_prior_used,
            free,
            bound: model.bound,
            clamp_penalty: model.clamp_penalty,
            k_init: model.k_init,
        })
    }

    /// Number of fitted weights.
    pub fn n_weights(&self) -> usize {
        self.free.len()
    }

    /// Index of ln k in the parameter vector.
    pub fn k_index(&self) -> usize {
        self.free.len()
    }

    /// Length of the parameter vector (the fitted weights plus ln k).
    pub fn dim(&self) -> usize {
        self.free.len() + 1
    }

    /// The range k is searched in: k_init * e^(+/-bound).
    pub fn k_bounds(&self) -> (f64, f64) {
        (
            (self.k_init.ln() - self.bound).exp(),
            (self.k_init.ln() + self.bound).exp(),
        )
    }

    /// Names of all parameters, in vector order.
    pub fn param_names(&self) -> Vec<&str> {
        self.free
            .iter()
            .map(String::as_str)
            .chain(std::iter::once("k"))
            .collect()
    }

    /// Converts a log-space parameter vector to weights and k, clamping it to the bounds.
    pub fn decode(&self, theta: &[f64]) -> Decoded {
        assert_eq!(theta.len(), self.dim(), "theta has the wrong length");
        let mut weights = self.defaults.clone();
        let mut theta_clamped = Vec::with_capacity(theta.len());
        let mut clamp_excess_sq = 0.0;
        for (i, &t) in theta.iter().enumerate() {
            let lo = self.theta_default[i] - self.bound;
            let hi = self.theta_default[i] + self.bound;
            let c = if t.is_nan() {
                self.theta_default[i]
            } else {
                t.clamp(lo, hi)
            };
            clamp_excess_sq += if t.is_nan() {
                self.bound * self.bound
            } else {
                (t - c).powi(2)
            };
            theta_clamped.push(c);
        }
        for (i, name) in self.free.iter().enumerate() {
            let magnitude = theta_clamped[i].exp();
            weights.set(
                name,
                if name == "skip" {
                    -magnitude
                } else {
                    magnitude
                },
            );
        }
        Decoded {
            weights,
            k: theta_clamped[self.k_index()].exp(),
            theta_clamped,
            clamp_excess_sq,
        }
    }

    /// Converts weights and k to a log-space parameter vector.
    pub fn encode(&self, weights: &Weights, k: f64) -> Vec<f64> {
        let mut theta: Vec<f64> = self
            .free
            .iter()
            .map(|name| weights.get(name).abs().ln())
            .collect();
        theta.push(k.ln());
        theta
    }

    /// Builds the base ALINE config with `w` as its saliences and costs.
    pub fn build(&self, w: &Weights) -> Result<Aline, String> {
        let f = |v: f64| v as f32;
        let mut aline = self.base.clone();
        aline.set_salience(
            Salience::try_new(
                f(w.syllabic),
                f(w.place),
                f(w.manner),
                f(w.voice),
                f(w.nasal),
                f(w.retroflex),
                f(w.lateral),
                f(w.aspirated),
                f(w.long),
                f(w.high),
                f(w.back),
                f(w.round),
                f(w.phonation),
                f(w.airstream),
                f(w.secondary),
                f(w.stress),
            )
            .map_err(|e| e.to_string())?,
        );
        aline.set_costs(
            Costs::try_new(
                f(w.skip),
                f(w.substitute),
                f(w.expand_compress),
                f(w.vowel_consonant),
            )
            .map_err(|e| e.to_string())?,
        );
        Ok(aline)
    }

    /// Weights from a parameter file; weights the file omits keep their defaults.
    pub fn weights_from_file(&self, file: &ParamFile) -> Result<Weights, String> {
        let mut w = self.defaults.clone();
        for (name, value) in &file.weights {
            if !ALL_WEIGHTS.contains(&name.as_str()) {
                return Err(format!("unknown weight '{name}'"));
            }
            let v = value
                .as_f64()
                .ok_or_else(|| format!("weight '{name}' is not a number"))?;
            w.set(name, v);
        }
        Ok(w)
    }

    /// A parameter file holding `weights` and `k`.
    pub fn to_file(weights: &Weights, k: Option<f64>) -> ParamFile {
        let mut map = serde_json::Map::new();
        for name in ALL_WEIGHTS {
            map.insert(name.to_string(), serde_json::json!(weights.get(name)));
        }
        ParamFile { weights: map, k }
    }
}

/// Checks that every name in `free_weights` is a known ALINE weight other than
/// `substitute`, and that none is listed twice.
fn validate_free_weights(names: &[String]) -> Result<Vec<String>, String> {
    let mut seen = BTreeSet::new();
    for name in names {
        if name == "substitute" {
            return Err(
                "free_weights: 'substitute' cannot be fitted; it is the fixed reference the other weights are reported against"
                    .to_string(),
            );
        }
        if name == "k" {
            return Err("free_weights: 'k' is always fitted; leave it out of the list".to_string());
        }
        if !ALL_WEIGHTS.contains(&name.as_str()) {
            return Err(format!(
                "free_weights: unknown weight '{name}' (known: {})",
                ALL_WEIGHTS.join(", ")
            ));
        }
        if !seen.insert(name.as_str()) {
            return Err(format!("free_weights: '{name}' is listed twice"));
        }
    }
    Ok(names.to_vec())
}
