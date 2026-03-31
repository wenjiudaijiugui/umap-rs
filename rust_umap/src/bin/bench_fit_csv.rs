use rust_umap::{Metric, UmapModel, UmapParams};
use std::env;
use std::error::Error;
use std::fs;
use std::path::Path;
use std::time::Instant;

mod common_cli;
use common_cli::{
    extract_optional_args, parse_bool, parse_init, read_csv, read_csv_usize, read_sparse_csr,
    write_csv,
};

fn metric_name(metric: Metric) -> &'static str {
    match metric {
        Metric::Euclidean => "euclidean",
        Metric::Manhattan => "manhattan",
        Metric::Cosine => "cosine",
    }
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

fn read_proc_status_kb(field: &str) -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix(field) {
            let value = rest.split_whitespace().next()?;
            return value.parse::<u64>().ok();
        }
    }
    None
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
    let vmhwm_before = read_proc_status_kb("VmHWM:");
    let vmrss_before = read_proc_status_kb("VmRSS:");

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
    let vmhwm_after = read_proc_status_kb("VmHWM:");
    let vmrss_after = read_proc_status_kb("VmRSS:");

    if let Some(embedding) = last_embedding.as_ref() {
        write_csv(output_path, embedding)?;
    }

    let (mean, std) = mean_std(&times);
    let algo_mem_proxy_available = vmhwm_before.is_some()
        && vmhwm_after.is_some()
        && vmrss_before.is_some()
        && vmrss_after.is_some();
    let vmhwm_before_mb = vmhwm_before.unwrap_or(0) as f64 / 1024.0;
    let vmhwm_after_mb = vmhwm_after.unwrap_or(0) as f64 / 1024.0;
    let vmrss_before_mb = vmrss_before.unwrap_or(0) as f64 / 1024.0;
    let vmrss_after_mb = vmrss_after.unwrap_or(0) as f64 / 1024.0;
    let algo_peak_delta_mb = if let (Some(before), Some(after)) = (vmhwm_before, vmhwm_after) {
        after.saturating_sub(before) as f64 / 1024.0
    } else {
        0.0
    };

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
        "],\"fit_mean_sec\":{:.9},\"fit_std_sec\":{:.9},\
\"algorithm_phase_memory_proxy_available\":{},\
\"algorithm_phase_peak_rss_delta_mb\":{:.9},\
\"algorithm_phase_vmrss_before_mb\":{:.9},\
\"algorithm_phase_vmrss_after_mb\":{:.9},\
\"algorithm_phase_vmhwm_before_mb\":{:.9},\
\"algorithm_phase_vmhwm_after_mb\":{:.9}}}",
        mean,
        std,
        algo_mem_proxy_available,
        algo_peak_delta_mb,
        vmrss_before_mb,
        vmrss_after_mb,
        vmhwm_before_mb,
        vmhwm_after_mb
    );

    Ok(())
}
