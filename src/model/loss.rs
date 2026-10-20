//! Bradley-Terry triad loss and the training objective.
//!
//! The probability of choosing a candidate depends on how much more similar it is
//! to the target than the other candidate, scaled by the sensitivity k.

use crate::data::{Label, Row};
use crate::model::params::ParamSpace;
use pho::algorithms::Aline;
use rayon::prelude::*;
use std::cell::Cell;
use std::collections::HashMap;

/// Objective value for weights that give a word a non-positive self-alignment score.
pub const INVALID_OBJECTIVE: f64 = 1.0e6;

/// Softplus, `ln(1 + e^x)`, computed without overflow for large `|x|`.
pub fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// Bradley-Terry negative log-likelihood of one triad's label.
pub fn bt_loss(d: f64, k: f64, label_b: bool) -> f64 {
    if label_b {
        softplus(-k * d)
    } else {
        softplus(k * d)
    }
}

/// Probability that candidate B is chosen over candidate C.
pub fn prob_b(d: f64, k: f64) -> f64 {
    1.0 / (1.0 + (-k * d).exp())
}

/// One triad, with indices into the shared pair and target tables.
#[derive(Debug, Clone)]
pub struct Judgment {
    /// Line number in the CSV file, which identifies the triad.
    pub line: usize,
    /// Index of the target in [`FitData::groups`].
    pub group: usize,
    /// Index of the (target, B) pair in [`FitData::pairs`].
    pub pair_b: usize,
    /// Index of the (target, C) pair in [`FitData::pairs`].
    pub pair_c: usize,
    /// Whether the label is B.
    pub label_b: bool,
}

/// A dataset prepared for fitting, with each (target, candidate) pair stored once.
#[derive(Debug, Clone)]
pub struct FitData {
    /// The distinct (target, candidate) pairs.
    pub pairs: Vec<(String, String)>,
    /// The triads.
    pub judgments: Vec<Judgment>,
    /// The distinct targets, which are the cross-validation groups.
    pub groups: Vec<String>,
}

impl FitData {
    /// Builds the fitting data from rows, storing each pair and target once.
    pub fn from_rows<'a>(rows: impl IntoIterator<Item = &'a Row>) -> FitData {
        let mut pairs = Vec::new();
        let mut pair_index: HashMap<(String, String), usize> = HashMap::new();
        let mut groups = Vec::new();
        let mut group_index: HashMap<String, usize> = HashMap::new();
        let mut judgments = Vec::new();

        fn intern<K: Clone + std::hash::Hash + Eq>(
            key: K,
            list: &mut Vec<K>,
            index: &mut HashMap<K, usize>,
        ) -> usize {
            *index.entry(key.clone()).or_insert_with(|| {
                list.push(key);
                list.len() - 1
            })
        }

        for row in rows {
            let pair_b = intern(
                (row.target.clone(), row.b.clone()),
                &mut pairs,
                &mut pair_index,
            );
            let pair_c = intern(
                (row.target.clone(), row.c.clone()),
                &mut pairs,
                &mut pair_index,
            );
            let group = intern(row.target.clone(), &mut groups, &mut group_index);
            judgments.push(Judgment {
                line: row.line,
                group,
                pair_b,
                pair_c,
                label_b: row.label == Label::B,
            });
        }
        FitData {
            pairs,
            judgments,
            groups,
        }
    }

    /// The sorted, distinct pair indices that `judgments` use.
    pub fn pairs_of(&self, judgments: &[usize]) -> Vec<usize> {
        let mut p: Vec<usize> = judgments
            .iter()
            .flat_map(|&j| [self.judgments[j].pair_b, self.judgments[j].pair_c])
            .collect();
        p.sort_unstable();
        p.dedup();
        p
    }
}

/// ALINE similarities for the pairs in `which`, computed in parallel. Other entries are NaN.
pub fn similarity_table(
    aline: &Aline,
    pairs: &[(String, String)],
    which: &[usize],
) -> Result<Vec<f64>, String> {
    let scored: Vec<Result<f64, String>> = which
        .par_iter()
        .map(|&p| {
            let (t, c) = &pairs[p];
            aline
                .similarity_unclamped(t, c)
                .map(f64::from)
                .map_err(|e| format!("sim('{t}', '{c}'): {e}"))
        })
        .collect();
    let mut table = vec![f64::NAN; pairs.len()];
    for (&p, s) in which.iter().zip(scored) {
        table[p] = s?;
    }
    Ok(table)
}

/// The similarity difference d = S(target, B) - S(target, C).
pub fn judgment_d(table: &[f64], j: &Judgment) -> f64 {
    table[j.pair_b] - table[j.pair_c]
}

/// Mean Bradley-Terry log-loss over `judgments`.
pub fn mean_loss(data: &FitData, table: &[f64], judgments: &[usize], k: f64) -> f64 {
    let total: f64 = judgments
        .iter()
        .map(|&j| {
            let jd = &data.judgments[j];
            bt_loss(judgment_d(table, jd), k, jd.label_b)
        })
        .sum();
    total / judgments.len() as f64
}

/// The training objective: mean log-loss plus the λ and bound penalties. It also
/// counts evaluations.
pub struct Objective<'a> {
    space: &'a ParamSpace,
    data: &'a FitData,
    train: Vec<usize>,
    needed_pairs: Vec<usize>,
    lambda: f64,
    invalid_evals: Cell<usize>,
    evals: Cell<usize>,
}

impl<'a> Objective<'a> {
    /// An objective over the triads in `train` with regularization strength `lambda`.
    pub fn new(space: &'a ParamSpace, data: &'a FitData, train: &[usize], lambda: f64) -> Self {
        assert!(!train.is_empty(), "training set is empty");
        Objective {
            space,
            data,
            train: train.to_vec(),
            needed_pairs: data.pairs_of(train),
            lambda,
            invalid_evals: Cell::new(0),
            evals: Cell::new(0),
        }
    }

    /// The objective at the parameter vector `theta`.
    pub fn value(&self, theta: &[f64]) -> f64 {
        self.evals.set(self.evals.get() + 1);
        let dec = self.space.decode(theta);
        let penalty = self.space.clamp_penalty * dec.clamp_excess_sq;
        let table = match self
            .space
            .build(&dec.weights)
            .and_then(|aline| similarity_table(&aline, &self.data.pairs, &self.needed_pairs))
        {
            Ok(t) => t,
            Err(_) => {
                self.invalid_evals.set(self.invalid_evals.get() + 1);
                return INVALID_OBJECTIVE + penalty;
            }
        };
        let reg: f64 = (0..self.space.n_weights())
            .map(|i| (dec.theta_clamped[i] - self.space.theta_default[i]).powi(2))
            .sum();
        mean_loss(
            self.data,
            &table,
            &self.train,
            dec.theta_clamped[self.space.k_index()].exp(),
        ) + self.lambda * reg
            + penalty
    }

    /// How many evaluations hit a non-positive self-alignment score.
    pub fn invalid_evals(&self) -> usize {
        self.invalid_evals.get()
    }

    /// How many evaluations have run.
    pub fn evals(&self) -> usize {
        self.evals.get()
    }
}
