use rust_umap::{InitMethod, Metric, SparseCsrMatrix, UmapModel, UmapParams};
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

fn parse_bool(s: &str) -> Result<bool, Box<dyn Error>> {
    match s.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "y" => Ok(true),
        "0" | "false" | "no" | "n" => Ok(false),
        _ => Err(format!("cannot parse bool from '{s}'").into()),
    }
}

fn parse_init(s: &str) -> Result<InitMethod, Box<dyn Error>> {
    match s.to_ascii_lowercase().as_str() {
        "random" => Ok(InitMethod::Random),
        "spectral" => Ok(InitMethod::Spectral),
        _ => Err(format!("unsupported init '{s}', expected random|spectral").into()),
    }
}

fn parse_metric(s: &str) -> Result<Metric, Box<dyn Error>> {
    match s.to_ascii_lowercase().as_str() {
        "euclidean" => Ok(Metric::Euclidean),
        "manhattan" | "l1" => Ok(Metric::Manhattan),
        "cosine" => Ok(Metric::Cosine),
        _ => Err(format!("unsupported metric '{s}', expected euclidean|manhattan|cosine").into()),
    }
}

#[derive(Debug, Clone)]
struct SparseInputArgs {
    indptr_path: String,
    indices_path: String,
    data_path: String,
    n_cols: usize,
}

fn extract_optional_args(
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

fn metric_name(metric: Metric) -> &'static str {
    match metric {
        Metric::Euclidean => "euclidean",
        Metric::Manhattan => "manhattan",
        Metric::Cosine => "cosine",
    }
}

fn read_csv(path: &Path) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
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

fn read_csv_usize(path: &Path) -> Result<Vec<Vec<usize>>, Box<dyn Error>> {
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

fn read_sparse_csr(spec: &SparseInputArgs) -> Result<SparseCsrMatrix, Box<dyn Error>> {
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

fn write_csv(path: &Path, arr: &[Vec<f32>]) -> Result<(), Box<dyn Error>> {
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

fn usage() {
    eprintln!(
        "Usage:\n  bench_fit_csv <input.csv> <output.csv> <n_neighbors> <n_components> <n_epochs> <seed> \\\n          <init:random|spectral> <use_approx:bool> <approx_candidates> <approx_iters> <approx_threshold> \\\n          <warmup> <repeats> [knn_idx.csv] [knn_dist.csv] [--metric euclidean|manhattan|cosine] [--knn-metric euclidean|manhattan|cosine]\n\
          Optional CSR input (fit mode): --csr-indptr <path> --csr-indices <path> --csr-data <path> --csr-n-cols <n>"
    );
}

fn mean_std(vals: &[f64]) -> (f64, f64) {
    if vals.is_empty() {
        return (0.0, 0.0);
    }
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let var = vals
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f64>()
        / vals.len() as f64;
    (mean, var.sqrt())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args: Vec<String> = env::args().collect();
    let (metric, knn_metric_opt, sparse_input) = extract_optional_args(&mut args)?;
    if args.len() < 14 {
        usage();
        return Err("insufficient arguments".into());
    }

    let input_path = Path::new(&args[1]);
    let output_path = Path::new(&args[2]);

    let n_neighbors = args[3].parse::<usize>()?;
    let n_components = args[4].parse::<usize>()?;
    let n_epochs = args[5].parse::<usize>()?;
    let seed = args[6].parse::<u64>()?;
    let init = parse_init(&args[7])?;
    let use_approximate_knn = parse_bool(&args[8])?;
    let approx_knn_candidates = args[9].parse::<usize>()?;
    let approx_knn_iters = args[10].parse::<usize>()?;
    let approx_knn_threshold = args[11].parse::<usize>()?;
    let warmup = args[12].parse::<usize>()?;
    let repeats = args[13].parse::<usize>()?;

    let precomputed = if args.len() >= 16 {
        Some((
            read_csv_usize(Path::new(&args[14]))?,
            read_csv(Path::new(&args[15]))?,
        ))
    } else {
        None
    };
    if precomputed.is_none() && knn_metric_opt.is_some() {
        return Err("--knn-metric requires precomputed kNN inputs".into());
    }
    if sparse_input.is_some() && precomputed.is_some() {
        return Err("CSR input cannot be combined with precomputed kNN files".into());
    }
    let knn_metric = knn_metric_opt.unwrap_or(metric);

    let dense_data = if sparse_input.is_some() {
        None
    } else {
        Some(read_csv(input_path)?)
    };
    let sparse_data = if let Some(spec) = sparse_input.as_ref() {
        Some(read_sparse_csr(spec)?)
    } else {
        None
    };

    let params = UmapParams {
        n_neighbors,
        n_components,
        n_epochs: Some(n_epochs),
        metric,
        learning_rate: 1.0,
        min_dist: 0.1,
        spread: 1.0,
        local_connectivity: 1.0,
        set_op_mix_ratio: 1.0,
        repulsion_strength: 1.0,
        negative_sample_rate: 5,
        random_seed: seed,
        init,
        use_approximate_knn,
        approx_knn_candidates,
        approx_knn_iters,
        approx_knn_threshold,
    };

    let total = warmup + repeats;
    let mut times = Vec::with_capacity(repeats);
    let mut last_embedding: Option<Vec<Vec<f32>>> = None;

    for i in 0..total {
        let mut model = UmapModel::new(params.clone());
        let t0 = Instant::now();
        let embedding = if let Some((ref idx, ref dist)) = precomputed {
            let data = dense_data
                .as_ref()
                .ok_or("precomputed kNN requires dense input data")?;
            model.fit_transform_with_knn_metric(data, idx, dist, knn_metric)?
        } else if let Some(data) = sparse_data.as_ref() {
            model.fit_transform_sparse_csr(data.clone())?
        } else {
            let data = dense_data
                .as_ref()
                .ok_or("dense input data is unavailable")?;
            model.fit_transform(&data)?
        };
        let dt = t0.elapsed().as_secs_f64();
        if i >= warmup {
            times.push(dt);
        }
        last_embedding = Some(embedding);
    }

    if let Some(embedding) = last_embedding.as_ref() {
        write_csv(output_path, embedding)?;
    }

    let (mean, std) = mean_std(&times);

    print!(
        "{{\"mode\":\"fit\",\
\"input_format\":\"{}\",\
\"metric\":\"{}\",\
\"knn_metric\":\"{}\",\
\"precomputed_knn\":{},\
\"n_neighbors\":{},\
\"n_components\":{},\
\"n_epochs\":{},\
\"seed\":{},\
\"warmup\":{},\
\"repeats\":{},\
\"fit_times_sec\":[",
        if sparse_data.is_some() {
            "csr"
        } else {
            "dense"
        },
        metric_name(metric),
        metric_name(knn_metric),
        precomputed.is_some(),
        n_neighbors,
        n_components,
        n_epochs,
        seed,
        warmup,
        repeats
    );
    for (i, t) in times.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        print!("{:.9}", t);
    }
    println!(
        "],\"fit_mean_sec\":{:.9},\"fit_std_sec\":{:.9}}}",
        mean, std
    );

    Ok(())
}
