use rust_umap::{InitMethod, Metric, SparseCsrMatrix, UmapParams};
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct SparseInputArgs {
    pub indptr_path: String,
    pub indices_path: String,
    pub data_path: String,
    pub n_cols: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct UmapParamOverrides {
    pub learning_rate: Option<f32>,
    pub min_dist: Option<f32>,
    pub spread: Option<f32>,
    pub local_connectivity: Option<f32>,
    pub set_op_mix_ratio: Option<f32>,
    pub repulsion_strength: Option<f32>,
    pub negative_sample_rate: Option<usize>,
}

impl UmapParamOverrides {
    pub fn apply_to(self, params: &mut UmapParams) {
        if let Some(value) = self.learning_rate {
            params.learning_rate = value;
        }
        if let Some(value) = self.min_dist {
            params.min_dist = value;
        }
        if let Some(value) = self.spread {
            params.spread = value;
        }
        if let Some(value) = self.local_connectivity {
            params.local_connectivity = value;
        }
        if let Some(value) = self.set_op_mix_ratio {
            params.set_op_mix_ratio = value;
        }
        if let Some(value) = self.repulsion_strength {
            params.repulsion_strength = value;
        }
        if let Some(value) = self.negative_sample_rate {
            params.negative_sample_rate = value;
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CliOptionalArgs {
    pub metric: Metric,
    pub knn_metric: Option<Metric>,
    pub sparse_input: Option<SparseInputArgs>,
    pub overrides: UmapParamOverrides,
}

fn parse_flag_value<T>(args: &[String], index: usize, flag: &str) -> Result<T, Box<dyn Error>>
where
    T: std::str::FromStr,
    T::Err: Error + 'static,
{
    if index + 1 >= args.len() {
        return Err(format!("{flag} requires a value").into());
    }
    args[index + 1].parse::<T>().map_err(|e| Box::new(e) as Box<dyn Error>)
}

pub fn parse_bool(s: &str) -> Result<bool, Box<dyn Error>> {
    match s.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "y" => Ok(true),
        "0" | "false" | "no" | "n" => Ok(false),
        _ => Err(format!("cannot parse bool from '{s}'").into()),
    }
}

pub fn parse_init(s: &str) -> Result<InitMethod, Box<dyn Error>> {
    match s.to_ascii_lowercase().as_str() {
        "random" => Ok(InitMethod::Random),
        "spectral" => Ok(InitMethod::Spectral),
        _ => Err(format!("unsupported init '{s}', expected random|spectral").into()),
    }
}

pub fn parse_metric(s: &str) -> Result<Metric, Box<dyn Error>> {
    match s.to_ascii_lowercase().as_str() {
        "euclidean" => Ok(Metric::Euclidean),
        "manhattan" | "l1" => Ok(Metric::Manhattan),
        "cosine" => Ok(Metric::Cosine),
        _ => Err(format!("unsupported metric '{s}', expected euclidean|manhattan|cosine").into()),
    }
}

pub fn extract_optional_args(args: &mut Vec<String>) -> Result<CliOptionalArgs, Box<dyn Error>> {
    let mut metric = Metric::Euclidean;
    let mut knn_metric: Option<Metric> = None;
    let mut csr_indptr: Option<String> = None;
    let mut csr_indices: Option<String> = None;
    let mut csr_data: Option<String> = None;
    let mut csr_n_cols: Option<usize> = None;
    let mut overrides = UmapParamOverrides::default();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--metric" {
            if i + 1 >= args.len() {
                return Err("--metric requires a value".into());
            }
            metric = parse_metric(&args[i + 1])?;
            args.drain(i..=i + 1);
        } else if args[i] == "--knn-metric" {
            if i + 1 >= args.len() {
                return Err("--knn-metric requires a value".into());
            }
            knn_metric = Some(parse_metric(&args[i + 1])?);
            args.drain(i..=i + 1);
        } else if args[i] == "--csr-indptr" {
            if i + 1 >= args.len() {
                return Err("--csr-indptr requires a file path".into());
            }
            csr_indptr = Some(args[i + 1].clone());
            args.drain(i..=i + 1);
        } else if args[i] == "--csr-indices" {
            if i + 1 >= args.len() {
                return Err("--csr-indices requires a file path".into());
            }
            csr_indices = Some(args[i + 1].clone());
            args.drain(i..=i + 1);
        } else if args[i] == "--csr-data" {
            if i + 1 >= args.len() {
                return Err("--csr-data requires a file path".into());
            }
            csr_data = Some(args[i + 1].clone());
            args.drain(i..=i + 1);
        } else if args[i] == "--csr-n-cols" {
            if i + 1 >= args.len() {
                return Err("--csr-n-cols requires an integer value".into());
            }
            csr_n_cols = Some(args[i + 1].parse::<usize>()?);
            args.drain(i..=i + 1);
        } else if args[i] == "--learning-rate" {
            overrides.learning_rate = Some(parse_flag_value(args, i, "--learning-rate")?);
            args.drain(i..=i + 1);
        } else if args[i] == "--min-dist" {
            overrides.min_dist = Some(parse_flag_value(args, i, "--min-dist")?);
            args.drain(i..=i + 1);
        } else if args[i] == "--spread" {
            overrides.spread = Some(parse_flag_value(args, i, "--spread")?);
            args.drain(i..=i + 1);
        } else if args[i] == "--local-connectivity" {
            overrides.local_connectivity =
                Some(parse_flag_value(args, i, "--local-connectivity")?);
            args.drain(i..=i + 1);
        } else if args[i] == "--set-op-mix-ratio" {
            overrides.set_op_mix_ratio = Some(parse_flag_value(args, i, "--set-op-mix-ratio")?);
            args.drain(i..=i + 1);
        } else if args[i] == "--repulsion-strength" {
            overrides.repulsion_strength =
                Some(parse_flag_value(args, i, "--repulsion-strength")?);
            args.drain(i..=i + 1);
        } else if args[i] == "--negative-sample-rate" {
            overrides.negative_sample_rate =
                Some(parse_flag_value(args, i, "--negative-sample-rate")?);
            args.drain(i..=i + 1);
        } else {
            i += 1;
        }
    }

    let has_any_csr =
        csr_indptr.is_some() || csr_indices.is_some() || csr_data.is_some() || csr_n_cols.is_some();
    let sparse = if has_any_csr {
        let indptr_path = csr_indptr.ok_or("--csr-indptr is required when using CSR input")?;
        let indices_path = csr_indices.ok_or("--csr-indices is required when using CSR input")?;
        let data_path = csr_data.ok_or("--csr-data is required when using CSR input")?;
        let n_cols = csr_n_cols.ok_or("--csr-n-cols is required when using CSR input")?;
        Some(SparseInputArgs {
            indptr_path,
            indices_path,
            data_path,
            n_cols,
        })
    } else {
        None
    };

    Ok(CliOptionalArgs {
        metric,
        knn_metric,
        sparse_input: sparse,
        overrides,
    })
}

pub fn read_csv(path: &Path) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut data = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let row = trimmed
            .split(',')
            .map(|x| x.trim().parse::<f32>())
            .collect::<Result<Vec<f32>, _>>()?;
        data.push(row);
    }
    Ok(data)
}

pub fn read_csv_usize(path: &Path) -> Result<Vec<Vec<usize>>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut data = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let row = trimmed
            .split(',')
            .map(|x| x.trim().parse::<usize>())
            .collect::<Result<Vec<usize>, _>>()?;
        data.push(row);
    }
    Ok(data)
}

fn read_csv_flat_usize(path: &Path) -> Result<Vec<usize>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut values = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        for token in trimmed.split(',') {
            let token = token.trim();
            if token.is_empty() {
                continue;
            }
            values.push(token.parse::<usize>()?);
        }
    }

    Ok(values)
}

fn read_csv_flat_f32(path: &Path) -> Result<Vec<f32>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut values = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        for token in trimmed.split(',') {
            let token = token.trim();
            if token.is_empty() {
                continue;
            }
            values.push(token.parse::<f32>()?);
        }
    }

    Ok(values)
}

pub fn read_sparse_csr(spec: &SparseInputArgs) -> Result<SparseCsrMatrix, Box<dyn Error>> {
    let indptr = read_csv_flat_usize(Path::new(&spec.indptr_path))?;
    let indices = read_csv_flat_usize(Path::new(&spec.indices_path))?;
    let values = read_csv_flat_f32(Path::new(&spec.data_path))?;
    if indptr.is_empty() {
        return Err("csr indptr cannot be empty".into());
    }
    let n_rows = indptr.len() - 1;
    SparseCsrMatrix::new(n_rows, spec.n_cols, indptr, indices, values)
        .map_err(|e| -> Box<dyn Error> { Box::new(e) })
}

pub fn write_csv(path: &Path, arr: &[Vec<f32>]) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    for row in arr {
        for (j, val) in row.iter().enumerate() {
            if j > 0 {
                writer.write_all(b",")?;
            }
            write!(writer, "{val:.8}")?;
        }
        writer.write_all(b"\n")?;
    }

    Ok(())
}

// Keep this file buildable as an inert auxiliary bin target when Cargo scans src/bin.
#[allow(dead_code)]
fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_optional_args_parses_overrides_and_sparse_input() {
        let mut args = vec![
            "fit_csv".to_string(),
            "in.csv".to_string(),
            "out.csv".to_string(),
            "--metric".to_string(),
            "cosine".to_string(),
            "--knn-metric".to_string(),
            "manhattan".to_string(),
            "--learning-rate".to_string(),
            "0.25".to_string(),
            "--min-dist".to_string(),
            "0.05".to_string(),
            "--spread".to_string(),
            "1.7".to_string(),
            "--local-connectivity".to_string(),
            "2.5".to_string(),
            "--set-op-mix-ratio".to_string(),
            "0.6".to_string(),
            "--repulsion-strength".to_string(),
            "1.9".to_string(),
            "--negative-sample-rate".to_string(),
            "11".to_string(),
            "--csr-indptr".to_string(),
            "indptr.csv".to_string(),
            "--csr-indices".to_string(),
            "indices.csv".to_string(),
            "--csr-data".to_string(),
            "data.csv".to_string(),
            "--csr-n-cols".to_string(),
            "128".to_string(),
            "fit".to_string(),
        ];

        let parsed = extract_optional_args(&mut args).expect("flags should parse");
        assert_eq!(parsed.metric, Metric::Cosine);
        assert_eq!(parsed.knn_metric, Some(Metric::Manhattan));
        assert_eq!(
            parsed.overrides,
            UmapParamOverrides {
                learning_rate: Some(0.25),
                min_dist: Some(0.05),
                spread: Some(1.7),
                local_connectivity: Some(2.5),
                set_op_mix_ratio: Some(0.6),
                repulsion_strength: Some(1.9),
                negative_sample_rate: Some(11),
            }
        );

        let sparse = parsed.sparse_input.expect("csr flags should produce sparse input");
        assert_eq!(sparse.indptr_path, "indptr.csv");
        assert_eq!(sparse.indices_path, "indices.csv");
        assert_eq!(sparse.data_path, "data.csv");
        assert_eq!(sparse.n_cols, 128);

        assert_eq!(
            args,
            vec![
                "fit_csv".to_string(),
                "in.csv".to_string(),
                "out.csv".to_string(),
                "fit".to_string(),
            ]
        );
    }

    #[test]
    fn extract_optional_args_rejects_missing_override_value() {
        let mut args = vec!["fit_csv".to_string(), "--learning-rate".to_string()];
        let err = extract_optional_args(&mut args).expect_err("missing value should fail");
        assert!(err.to_string().contains("--learning-rate requires a value"));
    }

    #[test]
    fn extract_optional_args_requires_complete_csr_tuple() {
        let mut args = vec![
            "fit_csv".to_string(),
            "--csr-indptr".to_string(),
            "indptr.csv".to_string(),
        ];
        let err = extract_optional_args(&mut args).expect_err("partial csr input should fail");
        assert!(err.to_string().contains("--csr-indices is required"));
    }
}
