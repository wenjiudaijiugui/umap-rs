use rust_umap::{InitMethod, Metric, UmapModel, UmapParams};
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

fn extract_metric_arg(args: &mut Vec<String>) -> Result<Metric, Box<dyn Error>> {
    let mut metric = Metric::Euclidean;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--metric" {
            if i + 1 >= args.len() {
                return Err("--metric requires a value".into());
            }
            metric = parse_metric(&args[i + 1])?;
            args.drain(i..=i + 1);
        } else {
            i += 1;
        }
    }
    Ok(metric)
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
        "Usage:\n  bench_fit_csv <input.csv> <output.csv> <n_neighbors> <n_components> <n_epochs> <seed> \\\n          <init:random|spectral> <use_approx:bool> <approx_candidates> <approx_iters> <approx_threshold> \\\n          <warmup> <repeats> [knn_idx.csv] [knn_dist.csv] [--metric euclidean|manhattan|cosine]"
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
    let metric = extract_metric_arg(&mut args)?;
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

    let data = read_csv(input_path)?;

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
            model.fit_transform_with_knn(&data, idx, dist)?
        } else {
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

    print!("{{\"fit_times_sec\":[");
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
