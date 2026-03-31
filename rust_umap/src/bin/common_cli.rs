use rust_umap::{InitMethod, Metric, SparseCsrMatrix};
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct SparseInputArgs {
    pub indptr_path: String,
    pub indices_path: String,
    pub data_path: String,
    pub n_cols: usize,
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

pub fn extract_optional_args(
    args: &mut Vec<String>,
) -> Result<(Metric, Option<Metric>, Option<SparseInputArgs>), Box<dyn Error>> {
    let mut metric = Metric::Euclidean;
    let mut knn_metric: Option<Metric> = None;
    let mut csr_indptr: Option<String> = None;
    let mut csr_indices: Option<String> = None;
    let mut csr_data: Option<String> = None;
    let mut csr_n_cols: Option<usize> = None;
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

    Ok((metric, knn_metric, sparse))
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
