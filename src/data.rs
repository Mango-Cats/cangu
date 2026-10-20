//! Loading and validating triad datasets.
//!
//! A dataset is a CSV file with one triad per row. Loading checks the columns,
//! labels, duplicate triads, and that every word tokenizes, and reports each error
//! with its line number.

use pho::algorithms::Aline;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

/// Required CSV columns. Any other column is ignored.
pub const COLUMNS: [&str; 4] = ["target", "b", "c", "label"];

/// The candidate judged closer to the target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Label {
    /// Candidate B.
    B,
    /// Candidate C.
    C,
}

impl Label {
    /// The label as written in the CSV: "b" or "c".
    pub fn as_str(self) -> &'static str {
        match self {
            Label::B => "b",
            Label::C => "c",
        }
    }
}

/// One validated triad.
#[derive(Debug, Clone)]
pub struct Row {
    /// Line number in the CSV file (the header is line 1).
    pub line: usize,
    /// Target word in IPA. Cross-validation folds are grouped by this string.
    pub target: String,
    /// Candidate B in IPA.
    pub b: String,
    /// Candidate C in IPA.
    pub c: String,
    /// The candidate listeners judged closer.
    pub label: Label,
}

/// A validated dataset.
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Path of the CSV file.
    pub path: PathBuf,
    /// The triads, in file order.
    pub rows: Vec<Row>,
}

/// Every validation error found while loading a file.
#[derive(Debug)]
pub struct LoadError {
    /// Path of the CSV file.
    pub path: PathBuf,
    /// Error messages, each starting with its line number.
    pub errors: Vec<String>,
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{}: {} validation error(s):",
            self.path.display(),
            self.errors.len()
        )?;
        for e in &self.errors {
            writeln!(f, "  {e}")?;
        }
        Ok(())
    }
}

impl std::error::Error for LoadError {}

/// A CSV row as read, before validation.
#[derive(Debug, Deserialize)]
struct RawRow {
    target: String,
    b: String,
    c: String,
    label: String,
}

impl Dataset {
    /// Reads `path` and validates it, using `aline` to check that every word tokenizes.
    pub fn load(path: &Path, aline: &Aline) -> Result<Dataset, LoadError> {
        let fail = |errors: Vec<String>| LoadError {
            path: path.to_path_buf(),
            errors,
        };
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(path)
            .map_err(|e| fail(vec![format!("cannot open: {e}")]))?;

        let headers = reader
            .headers()
            .map_err(|e| fail(vec![format!("line 1: cannot read header: {e}")]))?
            .clone();
        let missing: Vec<&str> = COLUMNS
            .iter()
            .copied()
            .filter(|c| !headers.iter().any(|h| h == *c))
            .collect();
        if !missing.is_empty() {
            return Err(fail(vec![format!(
                "line 1: missing column(s): {}",
                missing.join(", ")
            )]));
        }

        let mut errors = Vec::new();
        let mut rows = Vec::new();
        for result in reader.records() {
            let record = match result {
                Ok(record) => record,
                Err(e) => {
                    let line = e.position().map_or(0, |p| p.line() as usize);
                    errors.push(format!("line {line}: unreadable row: {e}"));
                    continue;
                }
            };
            let line = record.position().map_or(0, |p| p.line() as usize);
            let raw: RawRow = match record.deserialize(Some(&headers)) {
                Ok(raw) => raw,
                Err(e) => {
                    errors.push(format!("line {line}: unreadable row: {e}"));
                    continue;
                }
            };
            match parse_row(raw, line) {
                Ok(row) => rows.push(row),
                Err(mut row_errors) => errors.append(&mut row_errors),
            }
        }

        if errors.is_empty() {
            validate_rows(&rows, aline, &mut errors);
        }
        if !errors.is_empty() {
            return Err(fail(errors));
        }
        Ok(Dataset {
            path: path.to_path_buf(),
            rows,
        })
    }

    /// A one-line description: the number of triads and the file.
    pub fn summary(&self) -> String {
        format!(
            "Loaded {} triads from {}\n",
            self.rows.len(),
            self.path.display()
        )
    }
}

/// Checks one row's fields and converts it to a [`Row`].
fn parse_row(raw: RawRow, line: usize) -> Result<Row, Vec<String>> {
    let mut errors = Vec::new();
    let fields = [
        ("target", &raw.target),
        ("b", &raw.b),
        ("c", &raw.c),
    ];
    for (name, value) in fields {
        if value.trim().is_empty() {
            errors.push(format!("line {line}: column '{name}' is empty"));
        } else if value.contains('\n') || value.contains('\r') {
            errors.push(format!(
                "line {line}: column '{name}' contains a line break"
            ));
        }
    }
    let label = match raw.label.as_str() {
        "b" => Some(Label::B),
        "c" => Some(Label::C),
        other => {
            errors.push(format!(
                "line {line}: label must be 'b' or 'c', found '{other}'"
            ));
            None
        }
    };
    if !errors.is_empty() {
        return Err(errors);
    }
    Ok(Row {
        line,
        target: raw.target,
        b: raw.b,
        c: raw.c,
        label: label.unwrap(),
    })
}

/// Checks that every word tokenizes and that no triad repeats (in either candidate order).
fn validate_rows(rows: &[Row], aline: &Aline, errors: &mut Vec<String>) {
    let mut token_errors: HashMap<&str, Option<String>> = HashMap::new();
    for row in rows {
        for (column, ipa) in [
            ("target", &row.target),
            ("b", &row.b),
            ("c", &row.c),
        ] {
            let result = token_errors
                .entry(ipa.as_str())
                .or_insert_with(|| aline.tokenize(ipa).err().map(|e| e.to_string()));
            if let Some(e) = result {
                errors.push(format!(
                    "line {}: {column} '{ipa}' does not tokenize: {e}",
                    row.line
                ));
            }
        }
    }

    let mut seen: HashMap<(&str, &str, &str), &Row> = HashMap::new();
    for row in rows {
        let (b, c) = if row.b <= row.c {
            (&row.b, &row.c)
        } else {
            (&row.c, &row.b)
        };
        if let Some(first) = seen.get(&(row.target.as_str(), b.as_str(), c.as_str())) {
            errors.push(format!(
                "line {}: duplicate triad '{}' / '{}' / '{}', first seen on line {}",
                row.line, row.target, row.b, row.c, first.line
            ));
        } else {
            seen.insert((&row.target, b, c), row);
        }
    }
}
