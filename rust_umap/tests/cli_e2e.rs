use rust_umap::{InitMethod, Metric, UmapModel, UmapParams};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

fn synthetic_data(n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        let cluster_shift = if i < n / 2 { 0.0 } else { 5.0 };
        let row = (0..dim)
            .map(|d| {
                let t = (i as f32 + 1.3 * d as f32) / n as f32;
                cluster_shift + (10.0 * t).sin() * 0.2 + (7.0 * t).cos() * 0.1
            })
            .collect::<Vec<f32>>();
        data.push(row);
    }
    data
}

struct TestDir {
    path: PathBuf,
}

impl TestDir {
    fn new() -> Self {
        let unique = format!(
            "umap-cli-e2e-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time before unix epoch")
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        fs::create_dir_all(&path).expect("failed to create temp test directory");
        Self { path }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn fit_csv_bin() -> &'static str {
    env!("CARGO_BIN_EXE_fit_csv")
}

fn write_f32_csv(path: &Path, rows: &[Vec<f32>]) {
    let mut body = String::new();
    for row in rows {
        for (idx, value) in row.iter().enumerate() {
            if idx > 0 {
                body.push(',');
            }
            body.push_str(&format!("{value:.8}"));
        }
        body.push('\n');
    }
    fs::write(path, body).expect("failed to write f32 csv");
}

fn write_usize_csv(path: &Path, rows: &[Vec<usize>]) {
    let mut body = String::new();
    for row in rows {
        for (idx, value) in row.iter().enumerate() {
            if idx > 0 {
                body.push(',');
            }
            body.push_str(&value.to_string());
        }
        body.push('\n');
    }
    fs::write(path, body).expect("failed to write usize csv");
}

fn read_f32_csv(path: &Path) -> Vec<Vec<f32>> {
    let content = fs::read_to_string(path).expect("failed to read csv output");
    content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            line.split(',')
                .map(|value| value.trim().parse::<f32>().expect("invalid float in csv"))
                .collect::<Vec<f32>>()
        })
        .collect()
}

fn exact_knn(data: &[Vec<f32>], k: usize) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    let mut indices = Vec::with_capacity(data.len());
    let mut dists = Vec::with_capacity(data.len());

    for row in data {
        let mut neighbors = data
            .iter()
            .enumerate()
            .map(|(idx, other)| (idx, euclidean_distance(row, other)))
            .collect::<Vec<_>>();
        neighbors.sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1).then_with(|| lhs.0.cmp(&rhs.0)));

        indices.push(neighbors.iter().take(k).map(|(idx, _)| *idx).collect());
        dists.push(neighbors.iter().take(k).map(|(_, dist)| *dist).collect());
    }

    (indices, dists)
}

fn euclidean_distance(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(a, b)| {
            let diff = a - b;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

fn run_fit_csv(args: &[String]) -> Output {
    Command::new(fit_csv_bin())
        .args(args)
        .output()
        .expect("failed to spawn fit_csv")
}

fn success_output(args: &[String]) -> Output {
    let output = run_fit_csv(args);
    assert!(
        output.status.success(),
        "command failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn base_args(input: &Path, output: &Path) -> Vec<String> {
    vec![
        input.display().to_string(),
        output.display().to_string(),
        "6".to_string(),
        "2".to_string(),
        "40".to_string(),
        "2026".to_string(),
        "random".to_string(),
        "false".to_string(),
        "16".to_string(),
        "4".to_string(),
        "999999".to_string(),
    ]
}

fn assert_all_finite(points: &[Vec<f32>]) {
    assert!(
        points
            .iter()
            .flat_map(|row| row.iter())
            .all(|value| value.is_finite()),
        "expected all values to be finite"
    );
}

fn assert_close(lhs: &[Vec<f32>], rhs: &[Vec<f32>], tol: f32) {
    assert_eq!(lhs.len(), rhs.len(), "row count mismatch");
    for (row_idx, (lhs_row, rhs_row)) in lhs.iter().zip(rhs.iter()).enumerate() {
        assert_eq!(
            lhs_row.len(),
            rhs_row.len(),
            "column count mismatch at row {row_idx}"
        );
        for (col_idx, (&lhs_value, &rhs_value)) in lhs_row.iter().zip(rhs_row.iter()).enumerate() {
            let delta = (lhs_value - rhs_value).abs();
            assert!(
                delta <= tol,
                "value mismatch at row {row_idx}, col {col_idx}: lhs={lhs_value}, rhs={rhs_value}, delta={delta}, tol={tol}"
            );
        }
    }
}

#[test]
fn cli_fit_writes_embedding_csv() {
    let dir = TestDir::new();
    let input = dir.path("train.csv");
    let output = dir.path("embedding.csv");
    let data = synthetic_data(24, 5);
    write_f32_csv(&input, &data);

    let args = base_args(&input, &output);
    success_output(&args);

    let embedding = read_f32_csv(&output);
    assert_eq!(embedding.len(), data.len());
    assert_eq!(embedding[0].len(), 2);
    assert_all_finite(&embedding);
}

#[test]
fn cli_fit_precomputed_matches_library_output_for_same_knn() {
    let dir = TestDir::new();
    let input = dir.path("train.csv");
    let precomputed_output = dir.path("precomputed_embedding.csv");
    let knn_idx = dir.path("knn_idx.csv");
    let knn_dist = dir.path("knn_dist.csv");
    let data = synthetic_data(24, 5);
    write_f32_csv(&input, &data);

    let rounded_data = read_f32_csv(&input);
    let (indices, dists) = exact_knn(&rounded_data, 6);
    write_usize_csv(&knn_idx, &indices);
    write_f32_csv(&knn_dist, &dists);

    let mut precomputed_args = base_args(&input, &precomputed_output);
    precomputed_args.push("fit_precomputed".to_string());
    precomputed_args.push(knn_idx.display().to_string());
    precomputed_args.push(knn_dist.display().to_string());
    success_output(&precomputed_args);

    let mut model = UmapModel::new(UmapParams {
        n_neighbors: 6,
        n_components: 2,
        n_epochs: Some(40),
        metric: Metric::Euclidean,
        random_seed: 2026,
        init: InitMethod::Random,
        use_approximate_knn: false,
        approx_knn_candidates: 16,
        approx_knn_iters: 4,
        approx_knn_threshold: 999999,
        ..UmapParams::default()
    });
    let expected_embedding = model
        .fit_transform_with_knn_metric(&rounded_data, &indices, &dists, Metric::Euclidean)
        .expect("library precomputed fit should succeed");
    let precomputed_embedding = read_f32_csv(&precomputed_output);
    assert_close(&expected_embedding, &precomputed_embedding, 1e-5);
}

#[test]
fn cli_transform_outputs_query_embedding() {
    let dir = TestDir::new();
    let train = dir.path("train.csv");
    let query = dir.path("query.csv");
    let output = dir.path("transform.csv");
    let train_data = synthetic_data(24, 5);
    let query_data = train_data.iter().take(5).cloned().collect::<Vec<_>>();
    write_f32_csv(&train, &train_data);
    write_f32_csv(&query, &query_data);

    let mut args = base_args(&query, &output);
    args.push("transform".to_string());
    args.push(train.display().to_string());
    success_output(&args);

    let transformed = read_f32_csv(&output);
    assert_eq!(transformed.len(), query_data.len());
    assert_eq!(transformed[0].len(), 2);
    assert_all_finite(&transformed);
}

#[test]
fn cli_inverse_reconstructs_feature_dimensions() {
    let dir = TestDir::new();
    let train = dir.path("train.csv");
    let embedding_csv = dir.path("embedding.csv");
    let inverse_query = dir.path("inverse_query.csv");
    let inverse_output = dir.path("inverse.csv");
    let train_data = synthetic_data(24, 5);
    write_f32_csv(&train, &train_data);

    let fit_args = base_args(&train, &embedding_csv);
    success_output(&fit_args);
    let embedding = read_f32_csv(&embedding_csv);
    let query_embedding = embedding.iter().skip(18).cloned().collect::<Vec<_>>();
    write_f32_csv(&inverse_query, &query_embedding);

    let mut inverse_args = base_args(&inverse_query, &inverse_output);
    inverse_args.push("inverse".to_string());
    inverse_args.push(train.display().to_string());
    success_output(&inverse_args);

    let reconstructed = read_f32_csv(&inverse_output);
    assert_eq!(reconstructed.len(), query_embedding.len());
    assert_eq!(reconstructed[0].len(), train_data[0].len());
    assert_all_finite(&reconstructed);
}

#[test]
fn cli_fit_precomputed_rejects_metric_mismatch() {
    let dir = TestDir::new();
    let input = dir.path("train.csv");
    let output = dir.path("should_not_exist.csv");
    let knn_idx = dir.path("knn_idx.csv");
    let knn_dist = dir.path("knn_dist.csv");
    let data = synthetic_data(24, 5);
    write_f32_csv(&input, &data);

    let (indices, dists) = exact_knn(&data, 6);
    write_usize_csv(&knn_idx, &indices);
    write_f32_csv(&knn_dist, &dists);

    let mut args = base_args(&input, &output);
    args.push("fit_precomputed".to_string());
    args.push(knn_idx.display().to_string());
    args.push(knn_dist.display().to_string());
    args.push("--knn-metric".to_string());
    args.push("cosine".to_string());

    let output = run_fit_csv(&args);
    assert!(
        !output.status.success(),
        "expected command failure, got success\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("precomputed knn metric"),
        "unexpected stderr: {stderr}"
    );
}
