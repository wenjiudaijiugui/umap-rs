use rust_umap::{UmapModel, UmapParams};
use std::env;
use std::error::Error;
use std::path::Path;

mod common_cli;
use common_cli::{
    extract_optional_args, parse_bool, parse_init, read_csv, read_csv_usize, read_sparse_csr, write_csv,
};

fn usage() {
    eprintln!(
        "Usage:\n  fit_csv <input.csv> <output.csv> <n_neighbors> <n_components> <n_epochs> <seed> \
          <init:random|spectral> <use_approx:bool> <approx_candidates> <approx_iters> \
          <approx_threshold> [mode:fit|fit_precomputed|transform|inverse] [extra args] \
          [--metric euclidean|manhattan|cosine] [--knn-metric euclidean|manhattan|cosine] \n\
          [--learning-rate <f32>] [--min-dist <f32>] [--spread <f32>] \n\
          [--local-connectivity <f32>] [--set-op-mix-ratio <f32>] \n\
          [--repulsion-strength <f32>] [--negative-sample-rate <usize>]\n\
          Optional CSR input (fit mode only): --csr-indptr <path> --csr-indices <path> --csr-data <path> --csr-n-cols <n>\n\
          mode=fit_precomputed extra args: <knn_idx.csv> <knn_dist.csv>\n\
          mode=transform|inverse extra args: <ref_input.csv>"
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args: Vec<String> = env::args().collect();
    let optional = extract_optional_args(&mut args)?;
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
    if optional.sparse_input.is_some() && mode != "fit" {
        return Err("CSR input is currently supported in fit mode only".into());
    }
    if mode != "fit_precomputed" && optional.knn_metric.is_some() {
        return Err("--knn-metric is only valid in fit_precomputed mode".into());
    }

    let mut params = UmapParams {
        n_neighbors,
        n_components,
        n_epochs: Some(n_epochs),
        metric: optional.metric,
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
    optional.overrides.apply_to(&mut params);

    let mut model = UmapModel::new(params);

    match mode {
        "fit" => {
            let emb = if let Some(spec) = optional.sparse_input.as_ref() {
                let x_csr = read_sparse_csr(spec)?;
                model.fit_transform_sparse_csr(x_csr)?
            } else {
                let x = read_csv(input_path)?;
                model.fit_transform(&x)?
            };
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
            let knn_metric = optional.knn_metric.unwrap_or(optional.metric);
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
