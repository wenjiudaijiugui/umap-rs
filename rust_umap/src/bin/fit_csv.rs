use rust_umap::{InitMethod, Metric, UmapModel, UmapParams};
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

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

fn extract_metric_args(args: &mut Vec<String>) -> Result<(Metric, Option<Metric>), Box<dyn Error>> {
    let mut metric = Metric::Euclidean;
    let mut knn_metric: Option<Metric> = None;
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
        } else {
            i += 1;
        }
    }
    Ok((metric, knn_metric))
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
        "Usage:\n  fit_csv <input.csv> <output.csv> <n_neighbors> <n_components> <n_epochs> <seed> \
          <init:random|spectral> <use_approx:bool> <approx_candidates> <approx_iters> \
          <approx_threshold> [mode:fit|fit_precomputed|transform|inverse] [extra args] \
          [--metric euclidean|manhattan|cosine] [--knn-metric euclidean|manhattan|cosine]\n\
          mode=fit_precomputed extra args: <knn_idx.csv> <knn_dist.csv>\n\
          mode=transform|inverse extra args: <ref_input.csv>"
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args: Vec<String> = env::args().collect();
    let (metric, knn_metric_opt) = extract_metric_args(&mut args)?;
    if args.len() < 12 {
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

    let mode = if args.len() >= 13 {
        args[12].as_str()
    } else {
        "fit"
    };
    if mode != "fit_precomputed" && knn_metric_opt.is_some() {
        return Err("--knn-metric is only valid in fit_precomputed mode".into());
    }

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

    let mut model = UmapModel::new(params);

    match mode {
        "fit" => {
            let x = read_csv(input_path)?;
            let emb = model.fit_transform(&x)?;
            write_csv(output_path, &emb)?;
        }
        "fit_precomputed" => {
            if args.len() < 15 {
                return Err(
                    "fit_precomputed mode requires knn_idx.csv and knn_dist.csv as 14th and 15th args".into(),
                );
            }
            let x = read_csv(input_path)?;
            let knn_idx = read_csv_usize(Path::new(&args[13]))?;
            let knn_dist = read_csv(Path::new(&args[14]))?;
            let knn_metric = knn_metric_opt.unwrap_or(metric);
            let emb = model.fit_transform_with_knn_metric(&x, &knn_idx, &knn_dist, knn_metric)?;
            write_csv(output_path, &emb)?;
        }
        "transform" => {
            if args.len() < 14 {
                return Err("transform mode requires ref_input.csv as 14th arg".into());
            }
            let train_x = read_csv(Path::new(&args[13]))?;
            model.fit(&train_x)?;
            let query = read_csv(input_path)?;
            let out = model.transform(&query)?;
            write_csv(output_path, &out)?;
        }
        "inverse" => {
            if args.len() < 14 {
                return Err("inverse mode requires ref_input.csv as 14th arg".into());
            }
            let train_x = read_csv(Path::new(&args[13]))?;
            model.fit(&train_x)?;
            let query = read_csv(input_path)?;
            let out = model.inverse_transform(&query)?;
            write_csv(output_path, &out)?;
        }
        _ => return Err(format!("unsupported mode '{mode}'").into()),
    }

    Ok(())
}
