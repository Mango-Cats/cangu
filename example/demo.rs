//! A cangu demo on synthetic data.
//!
//! ```bash
//! cargo run --release --example demo
//! ```

mod aline;

use cangu::{EvaluationConfig, Fitter, ModelConfig, OptimizationConfig};
use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let data = "example/dataset.csv";
    // Outputs: the fitted ALINE config and the training report.
    let aline_out = "out/demo/aline.toml";
    let report_out = "out/demo/report.toml";

    let fitter = Fitter::new(
        // Start from ALINE's default saliences (voice = 5, stress = 10).
        aline::demo_aline(aline::salience(5.0, 10.0)),
        ModelConfig {
            // Not used here, because the demo config already has a non-zero stress salience.
            stress_prior: 10.0,
            // Fit these two weights and k; every other weight keeps its default.
            free_weights: ["voice", "nasal"].map(String::from).to_vec(),
            k_init: 20.0,
            bound: 5.0,
            clamp_penalty: 1.0,
        },
        OptimizationConfig {
            max_evals: 300,
            sigma0: 0.5,
            tol_fun: 1e-6,
            tol_x: 1e-4,
            min_bipop_evals: 50,
        },
        EvaluationConfig {
            n_folds: 5,
            lambda_grid: vec![0.0, 0.01, 0.1],
            learning_fractions: vec![0.5, 1.0],
            move_flag_factor: 5.0,
        },
    )?;

    let dataset = fitter.load_dataset(Path::new(data))?;
    eprint!("{}", dataset.summary());
    let result = fitter.fit(&dataset, 1, false)?;
    std::fs::create_dir_all("out/demo").map_err(|e| format!("out/demo: {e}"))?;
    result.write_aline(Path::new(aline_out))?;
    result.write_report(Path::new(report_out))?;
    eprintln!("wrote {aline_out} and {report_out}");

    Ok(())
}
