use nalgebra::{DMatrix, SymmetricEigen};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

const SMOOTH_K_TOLERANCE: f32 = 1e-5;
const MIN_K_DIST_SCALE: f32 = 1e-3;
const DEFAULT_BANDWIDTH: f32 = 1.0;
const INIT_MAX_COORD: f32 = 10.0;
const INIT_NOISE: f32 = 1e-4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitMethod {
    Random,
    Spectral,
}

impl Default for InitMethod {
    fn default() -> Self {
        Self::Spectral
    }
}

#[derive(Debug, Clone)]
pub struct UmapParams {
    pub n_neighbors: usize,
    pub n_components: usize,
    pub n_epochs: Option<usize>,
    pub learning_rate: f32,
    pub min_dist: f32,
    pub spread: f32,
    pub local_connectivity: f32,
    pub set_op_mix_ratio: f32,
    pub repulsion_strength: f32,
    pub negative_sample_rate: usize,
    pub random_seed: u64,
    pub init: InitMethod,
    pub use_approximate_knn: bool,
    pub approx_knn_candidates: usize,
    pub approx_knn_iters: usize,
    pub approx_knn_threshold: usize,
}

impl Default for UmapParams {
    fn default() -> Self {
        Self {
            n_neighbors: 15,
            n_components: 2,
            n_epochs: None,
            learning_rate: 1.0,
            min_dist: 0.1,
            spread: 1.0,
            local_connectivity: 1.0,
            set_op_mix_ratio: 1.0,
            repulsion_strength: 1.0,
            negative_sample_rate: 5,
            random_seed: 42,
            init: InitMethod::default(),
            use_approximate_knn: true,
            approx_knn_candidates: 30,
            approx_knn_iters: 10,
            approx_knn_threshold: 4096,
        }
    }
}

#[derive(Debug)]
pub enum UmapError {
    EmptyData,
    NeedAtLeastTwoSamples,
    InconsistentDimensions {
        row: usize,
        expected: usize,
        got: usize,
    },
    FeatureMismatch {
        expected: usize,
        got: usize,
    },
    EmbeddingDimensionMismatch {
        expected: usize,
        got: usize,
    },
    NotFitted,
    InvalidParameter(String),
}

impl Display for UmapError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            UmapError::EmptyData => write!(f, "input data is empty"),
            UmapError::NeedAtLeastTwoSamples => write!(f, "need at least two samples"),
            UmapError::InconsistentDimensions { row, expected, got } => {
                write!(
                    f,
                    "inconsistent row dimension at row {row}: expected {expected}, got {got}"
                )
            }
            UmapError::FeatureMismatch { expected, got } => {
                write!(f, "feature mismatch: expected {expected}, got {got}")
            }
            UmapError::EmbeddingDimensionMismatch { expected, got } => {
                write!(
                    f,
                    "embedding dimension mismatch: expected {expected}, got {got}"
                )
            }
            UmapError::NotFitted => write!(f, "model is not fitted yet"),
            UmapError::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
        }
    }
}

impl Error for UmapError {}

#[derive(Debug, Clone, Copy)]
struct Edge {
    head: usize,
    tail: usize,
    weight: f32,
}

#[derive(Debug, Clone)]
pub struct UmapModel {
    params: UmapParams,
    a: f32,
    b: f32,
    embedding: Option<Vec<Vec<f32>>>,
    training_data: Option<Vec<Vec<f32>>>,
    n_features: Option<usize>,
    fit_sigmas: Option<Vec<f32>>,
    fit_rhos: Option<Vec<f32>>,
}

impl UmapModel {
    pub fn new(params: UmapParams) -> Self {
        Self {
            params,
            a: 0.0,
            b: 0.0,
            embedding: None,
            training_data: None,
            n_features: None,
            fit_sigmas: None,
            fit_rhos: None,
        }
    }

    pub fn params(&self) -> &UmapParams {
        &self.params
    }

    pub fn ab_params(&self) -> Option<(f32, f32)> {
        if self.a > 0.0 && self.b > 0.0 {
            Some((self.a, self.b))
        } else {
            None
        }
    }

    pub fn embedding(&self) -> Option<&[Vec<f32>]> {
        self.embedding.as_deref()
    }

    pub fn fit(&mut self, data: &[Vec<f32>]) -> Result<(), UmapError> {
        self.fit_transform(data).map(|_| ())
    }

    pub fn fit_transform(&mut self, data: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, UmapError> {
        let (n_samples, n_features) = validate_data(data)?;
        validate_params(&self.params, n_samples, n_features)?;

        let (knn_indices, knn_dists) = if self.params.use_approximate_knn
            && n_samples > self.params.approx_knn_threshold
        {
            approximate_nearest_neighbors(
                data,
                self.params.n_neighbors,
                self.params.approx_knn_candidates,
                self.params.approx_knn_iters,
                self.params.random_seed ^ 0xC0FE_FEED_1234_ABCD,
            )
        } else {
            exact_nearest_neighbors(data, self.params.n_neighbors)
        };

        self.fit_transform_with_knn_internal(data, &knn_indices, &knn_dists)
    }

    pub fn fit_transform_with_knn(
        &mut self,
        data: &[Vec<f32>],
        knn_indices: &[Vec<usize>],
        knn_dists: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, UmapError> {
        self.fit_transform_with_knn_internal(data, knn_indices, knn_dists)
    }

    fn fit_transform_with_knn_internal(
        &mut self,
        data: &[Vec<f32>],
        knn_indices: &[Vec<usize>],
        knn_dists: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, UmapError> {
        let (n_samples, n_features) = validate_data(data)?;
        validate_params(&self.params, n_samples, n_features)?;

        let n_epochs = self
            .params
            .n_epochs
            .unwrap_or_else(|| if n_samples <= 10_000 { 500 } else { 200 });

        let (a, b) = find_ab_params(self.params.spread, self.params.min_dist);
        self.a = a;
        self.b = b;

        if knn_indices.len() != n_samples || knn_dists.len() != n_samples {
            return Err(UmapError::InvalidParameter(
                "precomputed knn row count must match number of samples".to_string(),
            ));
        }

        let mut knn_indices_trimmed = Vec::with_capacity(n_samples);
        let mut knn_dists_trimmed = Vec::with_capacity(n_samples);
        for row in 0..n_samples {
            let idx_row = &knn_indices[row];
            let dist_row = &knn_dists[row];
            if idx_row.len() < self.params.n_neighbors || dist_row.len() < self.params.n_neighbors {
                return Err(UmapError::InvalidParameter(
                    "precomputed knn columns must be >= n_neighbors".to_string(),
                ));
            }
            if idx_row.len() != dist_row.len() {
                return Err(UmapError::InvalidParameter(
                    "precomputed knn index/dist row lengths must match".to_string(),
                ));
            }

            let idx_trim = idx_row[..self.params.n_neighbors].to_vec();
            for &idx in idx_trim.iter() {
                if idx >= n_samples {
                    return Err(UmapError::InvalidParameter(
                        "precomputed knn index out of range".to_string(),
                    ));
                }
            }
            let dist_trim = dist_row[..self.params.n_neighbors].to_vec();
            knn_indices_trimmed.push(idx_trim);
            knn_dists_trimmed.push(dist_trim);
        }

        let (sigmas, rhos) = smooth_knn_dist(
            &knn_dists_trimmed,
            self.params.n_neighbors as f32,
            self.params.local_connectivity,
            DEFAULT_BANDWIDTH,
        );

        let directed = compute_membership_strengths(
            &knn_indices_trimmed,
            &knn_dists_trimmed,
            &sigmas,
            &rhos,
        );
        let mut edges = symmetrize_fuzzy_graph(&directed, self.params.set_op_mix_ratio);
        prune_edges(&mut edges, n_epochs);

        let mut embedding = initialize_embedding(
            data,
            self.params.n_components,
            &edges,
            self.params.init,
            self.params.random_seed,
        );

        optimize_layout_training(
            &mut embedding,
            &edges,
            n_epochs,
            a,
            b,
            self.params.learning_rate,
            self.params.negative_sample_rate,
            self.params.repulsion_strength,
            self.params.random_seed ^ 0x9E37_79B9_7F4A_7C15,
        );

        self.embedding = Some(embedding.clone());
        self.training_data = Some(data.to_vec());
        self.n_features = Some(n_features);
        self.fit_sigmas = Some(sigmas);
        self.fit_rhos = Some(rhos);

        Ok(embedding)
    }

    pub fn transform(&self, query: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, UmapError> {
        let train_data = self.training_data.as_ref().ok_or(UmapError::NotFitted)?;
        let train_embedding = self.embedding.as_ref().ok_or(UmapError::NotFitted)?;
        let expected_features = self.n_features.ok_or(UmapError::NotFitted)?;

        if query.is_empty() {
            return Ok(Vec::new());
        }

        for (idx, row) in query.iter().enumerate() {
            if row.len() != expected_features {
                if idx == 0 {
                    return Err(UmapError::FeatureMismatch {
                        expected: expected_features,
                        got: row.len(),
                    });
                }
                return Err(UmapError::InconsistentDimensions {
                    row: idx,
                    expected: expected_features,
                    got: row.len(),
                });
            }
        }

        let n_neighbors = self.params.n_neighbors.min(train_data.len());
        let n_epochs = match self.params.n_epochs {
            None => {
                if query.len() <= 10_000 {
                    100
                } else {
                    30
                }
            }
            Some(e) => (e / 3).max(1),
        };

        let (indices, dists) =
            exact_nearest_neighbors_to_reference(query, train_data, n_neighbors);

        let adjusted_local_connectivity = (self.params.local_connectivity - 1.0).max(0.0);
        let (sigmas, rhos) = smooth_knn_dist(
            &dists,
            n_neighbors as f32,
            adjusted_local_connectivity,
            DEFAULT_BANDWIDTH,
        );

        let mut edges = compute_membership_strengths_bipartite(&indices, &dists, &sigmas, &rhos);
        prune_edges(&mut edges, n_epochs);

        let mut embedding = init_graph_transform(
            query.len(),
            self.params.n_components,
            &edges,
            train_embedding,
        );

        optimize_layout_transform(
            &mut embedding,
            train_embedding,
            &edges,
            n_epochs,
            self.a,
            self.b,
            self.params.learning_rate / 4.0,
            self.params.negative_sample_rate,
            self.params.repulsion_strength,
            self.params.random_seed ^ 0xA24B_AED4_0B53_C117,
        );

        Ok(embedding)
    }

    pub fn inverse_transform(
        &self,
        embedded_query: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, UmapError> {
        let train_data = self.training_data.as_ref().ok_or(UmapError::NotFitted)?;
        let train_embedding = self.embedding.as_ref().ok_or(UmapError::NotFitted)?;
        let fit_sigmas = self.fit_sigmas.as_ref().ok_or(UmapError::NotFitted)?;
        let fit_rhos = self.fit_rhos.as_ref().ok_or(UmapError::NotFitted)?;
        let expected_features = self.n_features.ok_or(UmapError::NotFitted)?;

        if embedded_query.is_empty() {
            return Ok(Vec::new());
        }

        for (idx, row) in embedded_query.iter().enumerate() {
            if row.len() != self.params.n_components {
                if idx == 0 {
                    return Err(UmapError::EmbeddingDimensionMismatch {
                        expected: self.params.n_components,
                        got: row.len(),
                    });
                }
                return Err(UmapError::InconsistentDimensions {
                    row: idx,
                    expected: self.params.n_components,
                    got: row.len(),
                });
            }
        }

        let n_train = train_data.len();
        let n_neighbors = expected_features.min(n_train).max(1);
        let n_epochs = match self.params.n_epochs {
            None => {
                if embedded_query.len() <= 10_000 {
                    100
                } else {
                    30
                }
            }
            Some(e) => (e / 3).max(1),
        };

        let (indices, dists) =
            exact_nearest_neighbors_to_reference(embedded_query, train_embedding, n_neighbors);

        let mut graph_edges = Vec::with_capacity(embedded_query.len() * n_neighbors);
        for i in 0..embedded_query.len() {
            for j in 0..n_neighbors {
                let tail = indices[i][j];
                let dist = dists[i][j];
                let weight = 1.0 / (1.0 + self.a * dist.powf(2.0 * self.b));
                if weight > 0.0 && weight.is_finite() {
                    graph_edges.push(Edge {
                        head: i,
                        tail,
                        weight,
                    });
                }
            }
        }

        let mut inv_points =
            init_graph_transform(embedded_query.len(), expected_features, &graph_edges, train_data);

        optimize_layout_inverse(
            &mut inv_points,
            train_data,
            &graph_edges,
            fit_sigmas,
            fit_rhos,
            n_epochs,
            self.params.learning_rate / 4.0,
            self.params.negative_sample_rate,
            self.params.repulsion_strength,
            self.params.random_seed ^ 0xD1B5_4A32_D192_ED03,
        );

        Ok(inv_points)
    }
}

pub fn fit_transform(data: &[Vec<f32>], params: UmapParams) -> Result<Vec<Vec<f32>>, UmapError> {
    let mut model = UmapModel::new(params);
    model.fit_transform(data)
}

fn validate_data(data: &[Vec<f32>]) -> Result<(usize, usize), UmapError> {
    if data.is_empty() {
        return Err(UmapError::EmptyData);
    }
    if data.len() < 2 {
        return Err(UmapError::NeedAtLeastTwoSamples);
    }

    let n_features = data[0].len();
    if n_features == 0 {
        return Err(UmapError::InvalidParameter(
            "input rows must have at least one feature".to_string(),
        ));
    }

    for (idx, row) in data.iter().enumerate().skip(1) {
        if row.len() != n_features {
            return Err(UmapError::InconsistentDimensions {
                row: idx,
                expected: n_features,
                got: row.len(),
            });
        }
    }

    Ok((data.len(), n_features))
}

fn validate_params(
    params: &UmapParams,
    n_samples: usize,
    _n_features: usize,
) -> Result<(), UmapError> {
    if params.n_neighbors < 2 {
        return Err(UmapError::InvalidParameter(
            "n_neighbors must be >= 2".to_string(),
        ));
    }
    if params.n_neighbors >= n_samples {
        return Err(UmapError::InvalidParameter(format!(
            "n_neighbors ({}) must be smaller than number of samples ({n_samples})",
            params.n_neighbors
        )));
    }
    if params.n_components == 0 {
        return Err(UmapError::InvalidParameter(
            "n_components must be >= 1".to_string(),
        ));
    }
    if params.learning_rate <= 0.0 {
        return Err(UmapError::InvalidParameter(
            "learning_rate must be > 0".to_string(),
        ));
    }
    if params.min_dist < 0.0 {
        return Err(UmapError::InvalidParameter(
            "min_dist must be >= 0".to_string(),
        ));
    }
    if params.spread <= 0.0 {
        return Err(UmapError::InvalidParameter("spread must be > 0".to_string()));
    }
    if params.min_dist > params.spread {
        return Err(UmapError::InvalidParameter(
            "min_dist must be <= spread".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&params.set_op_mix_ratio) {
        return Err(UmapError::InvalidParameter(
            "set_op_mix_ratio must be in [0, 1]".to_string(),
        ));
    }
    if params.repulsion_strength < 0.0 {
        return Err(UmapError::InvalidParameter(
            "repulsion_strength must be >= 0".to_string(),
        ));
    }
    if params.negative_sample_rate == 0 {
        return Err(UmapError::InvalidParameter(
            "negative_sample_rate must be >= 1".to_string(),
        ));
    }
    if params.local_connectivity < 0.0 {
        return Err(UmapError::InvalidParameter(
            "local_connectivity must be >= 0".to_string(),
        ));
    }
    if params.approx_knn_candidates == 0 {
        return Err(UmapError::InvalidParameter(
            "approx_knn_candidates must be >= 1".to_string(),
        ));
    }
    if params.approx_knn_iters == 0 {
        return Err(UmapError::InvalidParameter(
            "approx_knn_iters must be >= 1".to_string(),
        ));
    }
    Ok(())
}

fn euclidean_distance(x: &[f32], y: &[f32]) -> f32 {
    squared_distance(x, y).sqrt()
}

fn squared_distance(x: &[f32], y: &[f32]) -> f32 {
    x.iter()
        .zip(y.iter())
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum()
}

fn exact_nearest_neighbors(
    data: &[Vec<f32>],
    n_neighbors: usize,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    let n_samples = data.len();
    let mut indices = vec![vec![0_usize; n_neighbors]; n_samples];
    let mut dists = vec![vec![0.0_f32; n_neighbors]; n_samples];

    for i in 0..n_samples {
        let mut row: Vec<(usize, f32)> = (0..n_samples)
            .map(|j| (j, euclidean_distance(&data[i], &data[j])))
            .collect();

        row.sort_by(|a, b| a.1.total_cmp(&b.1));

        for (k, (j, d)) in row.into_iter().take(n_neighbors).enumerate() {
            indices[i][k] = j;
            dists[i][k] = d;
        }
    }

    (indices, dists)
}

fn dedup_sorted_neighbors(
    mut pairs: Vec<(usize, f32)>,
    k: usize,
    all_points: &[Vec<f32>],
    row_point: &[f32],
) -> Vec<(usize, f32)> {
    pairs.sort_by(|a, b| {
        a.1.total_cmp(&b.1)
            .then_with(|| a.0.cmp(&b.0))
    });

    let mut out = Vec::with_capacity(k);
    let mut seen = HashSet::with_capacity(k * 2);
    for (idx, dist) in pairs {
        if seen.insert(idx) {
            out.push((idx, dist));
            if out.len() == k {
                break;
            }
        }
    }

    if out.len() < k {
        for (idx, candidate) in all_points.iter().enumerate() {
            if seen.insert(idx) {
                out.push((idx, euclidean_distance(row_point, candidate)));
                if out.len() == k {
                    break;
                }
            }
        }
        out.sort_by(|a, b| {
            a.1.total_cmp(&b.1)
                .then_with(|| a.0.cmp(&b.0))
        });
        out.truncate(k);
    }

    out
}

fn approximate_nearest_neighbors(
    data: &[Vec<f32>],
    n_neighbors: usize,
    candidate_pool: usize,
    n_iters: usize,
    seed: u64,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    let n_samples = data.len();
    let pool = candidate_pool.max(n_neighbors).min(n_samples - 1);
    let mut rng = SmallRng::seed_from_u64(seed);

    let mut neighbors: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let mut sampled = HashSet::with_capacity(pool + 1);
        sampled.insert(i);
        while sampled.len() < pool + 1 {
            sampled.insert(rng.gen_range(0..n_samples));
        }

        let candidates = sampled
            .into_iter()
            .map(|j| (j, euclidean_distance(&data[i], &data[j])))
            .collect::<Vec<(usize, f32)>>();

        neighbors.push(dedup_sorted_neighbors(
            candidates,
            n_neighbors,
            data,
            &data[i],
        ));
    }

    let max_candidates = candidate_pool.max(n_neighbors) * 4 + 1;

    for _ in 0..n_iters {
        let mut changed = false;
        let mut next_neighbors = neighbors.clone();

        for i in 0..n_samples {
            let mut candidate_set = HashSet::with_capacity(max_candidates);
            for (idx, _) in &neighbors[i] {
                candidate_set.insert(*idx);
                for (idx2, _) in &neighbors[*idx] {
                    candidate_set.insert(*idx2);
                }
            }
            candidate_set.insert(i);

            let mut candidate_vec = candidate_set.into_iter().collect::<Vec<usize>>();
            if candidate_vec.len() > max_candidates {
                for s in 0..max_candidates {
                    let r = rng.gen_range(s..candidate_vec.len());
                    candidate_vec.swap(s, r);
                }
                candidate_vec.truncate(max_candidates);
            }

            let candidate_pairs = candidate_vec
                .into_iter()
                .map(|j| (j, euclidean_distance(&data[i], &data[j])))
                .collect::<Vec<(usize, f32)>>();

            let updated =
                dedup_sorted_neighbors(candidate_pairs, n_neighbors, data, &data[i]);

            if updated != neighbors[i] {
                changed = true;
                next_neighbors[i] = updated;
            }
        }

        neighbors = next_neighbors;
        if !changed {
            break;
        }
    }

    let mut indices = vec![vec![0_usize; n_neighbors]; n_samples];
    let mut dists = vec![vec![0.0_f32; n_neighbors]; n_samples];
    for i in 0..n_samples {
        for j in 0..n_neighbors {
            indices[i][j] = neighbors[i][j].0;
            dists[i][j] = neighbors[i][j].1;
        }
    }

    (indices, dists)
}

fn exact_nearest_neighbors_to_reference(
    query: &[Vec<f32>],
    reference: &[Vec<f32>],
    n_neighbors: usize,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    let mut indices = vec![vec![0_usize; n_neighbors]; query.len()];
    let mut dists = vec![vec![0.0_f32; n_neighbors]; query.len()];

    for (i, x) in query.iter().enumerate() {
        let mut row: Vec<(usize, f32)> = reference
            .iter()
            .enumerate()
            .map(|(j, y)| (j, euclidean_distance(x, y)))
            .collect();

        row.sort_by(|a, b| a.1.total_cmp(&b.1));

        for (k, (j, d)) in row.into_iter().take(n_neighbors).enumerate() {
            indices[i][k] = j;
            dists[i][k] = d;
        }
    }

    (indices, dists)
}

fn smooth_knn_dist(
    distances: &[Vec<f32>],
    k: f32,
    local_connectivity: f32,
    bandwidth: f32,
) -> (Vec<f32>, Vec<f32>) {
    let target = k.log2() * bandwidth;
    let n_samples = distances.len();

    let mean_distances = {
        let mut total = 0.0_f32;
        let mut count = 0_usize;
        for row in distances {
            for &d in row {
                total += d;
                count += 1;
            }
        }
        if count > 0 { total / count as f32 } else { 0.0 }
    };

    let mut sigmas = vec![0.0_f32; n_samples];
    let mut rhos = vec![0.0_f32; n_samples];

    for i in 0..n_samples {
        let ith_distances = &distances[i];

        let non_zero_dists: Vec<f32> = ith_distances
            .iter()
            .copied()
            .filter(|d| *d > 0.0)
            .collect();

        if non_zero_dists.len() as f32 >= local_connectivity {
            let index = local_connectivity.floor() as usize;
            let interpolation = local_connectivity - index as f32;

            if index > 0 {
                rhos[i] = non_zero_dists[index - 1];
                if interpolation > SMOOTH_K_TOLERANCE && index < non_zero_dists.len() {
                    rhos[i] += interpolation * (non_zero_dists[index] - non_zero_dists[index - 1]);
                }
            } else if let Some(first) = non_zero_dists.first() {
                rhos[i] = interpolation * *first;
            }
        } else if let Some(max_dist) = non_zero_dists.iter().copied().reduce(f32::max) {
            rhos[i] = max_dist;
        }

        let mut lo = 0.0_f32;
        let mut hi = f32::INFINITY;
        let mut mid = 1.0_f32;

        for _ in 0..64 {
            let mut psum = 0.0_f32;

            for &dist in ith_distances.iter().skip(1) {
                let d = dist - rhos[i];
                if d > 0.0 {
                    psum += (-(d / mid)).exp();
                } else {
                    psum += 1.0;
                }
            }

            if (psum - target).abs() < SMOOTH_K_TOLERANCE {
                break;
            }

            if psum > target {
                hi = mid;
                mid = (lo + hi) / 2.0;
            } else {
                lo = mid;
                if hi.is_infinite() {
                    mid *= 2.0;
                } else {
                    mid = (lo + hi) / 2.0;
                }
            }
        }

        let mean_ith = ith_distances.iter().sum::<f32>() / ith_distances.len() as f32;
        let min_scale = if rhos[i] > 0.0 {
            MIN_K_DIST_SCALE * mean_ith
        } else {
            MIN_K_DIST_SCALE * mean_distances
        };

        sigmas[i] = mid.max(min_scale);
    }

    (sigmas, rhos)
}

fn compute_membership_strengths(
    knn_indices: &[Vec<usize>],
    knn_dists: &[Vec<f32>],
    sigmas: &[f32],
    rhos: &[f32],
) -> HashMap<(usize, usize), f32> {
    let n_samples = knn_indices.len();
    let n_neighbors = knn_indices[0].len();

    let mut directed = HashMap::with_capacity(n_samples * n_neighbors);

    for i in 0..n_samples {
        for j in 0..n_neighbors {
            let neighbor = knn_indices[i][j];
            let dist = knn_dists[i][j];

            let val = if neighbor == i {
                0.0
            } else if dist - rhos[i] <= 0.0 || sigmas[i] == 0.0 {
                1.0
            } else {
                (-(dist - rhos[i]) / sigmas[i]).exp()
            };

            if val > 0.0 {
                directed.insert((i, neighbor), val);
            }
        }
    }

    directed
}

fn compute_membership_strengths_bipartite(
    indices: &[Vec<usize>],
    dists: &[Vec<f32>],
    sigmas: &[f32],
    rhos: &[f32],
) -> Vec<Edge> {
    let n_samples = indices.len();
    let n_neighbors = indices[0].len();

    let mut edges = Vec::with_capacity(n_samples * n_neighbors);

    for i in 0..n_samples {
        for j in 0..n_neighbors {
            let tail = indices[i][j];
            let dist = dists[i][j];

            let val = if dist - rhos[i] <= 0.0 || sigmas[i] == 0.0 {
                1.0
            } else {
                (-(dist - rhos[i]) / sigmas[i]).exp()
            };

            if val > 0.0 {
                edges.push(Edge {
                    head: i,
                    tail,
                    weight: val,
                });
            }
        }
    }

    edges
}

fn symmetrize_fuzzy_graph(
    directed: &HashMap<(usize, usize), f32>,
    set_op_mix_ratio: f32,
) -> Vec<Edge> {
    let mut all_pairs: HashSet<(usize, usize)> = HashSet::with_capacity(directed.len() * 2);

    for &(i, j) in directed.keys() {
        all_pairs.insert((i, j));
        all_pairs.insert((j, i));
    }

    let mut edges = Vec::with_capacity(all_pairs.len());

    for (i, j) in all_pairs {
        if i == j {
            continue;
        }

        let w_ij = *directed.get(&(i, j)).unwrap_or(&0.0);
        let w_ji = *directed.get(&(j, i)).unwrap_or(&0.0);
        let prod = w_ij * w_ji;
        let sym = set_op_mix_ratio * (w_ij + w_ji - prod) + (1.0 - set_op_mix_ratio) * prod;

        if sym > 0.0 {
            edges.push(Edge {
                head: i,
                tail: j,
                weight: sym,
            });
        }
    }

    edges.sort_by_key(|e| (e.head, e.tail));
    edges
}

fn prune_edges(edges: &mut Vec<Edge>, n_epochs: usize) {
    if edges.is_empty() {
        return;
    }

    let max_weight = edges
        .iter()
        .map(|e| e.weight)
        .fold(f32::NEG_INFINITY, f32::max);

    let denom = if n_epochs > 10 {
        n_epochs as f32
    } else {
        500.0
    };

    let threshold = max_weight / denom;
    edges.retain(|e| e.weight >= threshold && e.weight > 0.0);
}

fn random_init(
    n_samples: usize,
    n_components: usize,
    seed: u64,
    low: f32,
    high: f32,
) -> Vec<Vec<f32>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n_samples)
        .map(|_| {
            (0..n_components)
                .map(|_| rng.gen_range(low..high))
                .collect::<Vec<f32>>()
        })
        .collect()
}

fn initialize_embedding(
    data: &[Vec<f32>],
    n_components: usize,
    edges: &[Edge],
    init: InitMethod,
    seed: u64,
) -> Vec<Vec<f32>> {
    let n_samples = data.len();
    match init {
        InitMethod::Random => random_init(n_samples, n_components, seed, -10.0, 10.0),
        InitMethod::Spectral => spectral_init(data, n_components, edges, seed)
            .unwrap_or_else(|| random_init(n_samples, n_components, seed, -10.0, 10.0)),
    }
}

fn spectral_init(
    data: &[Vec<f32>],
    n_components: usize,
    edges: &[Edge],
    seed: u64,
) -> Option<Vec<Vec<f32>>> {
    let n_samples = data.len();
    if n_samples <= n_components + 1 || edges.is_empty() {
        return None;
    }

    let components = connected_components_from_edges(n_samples, edges);
    if components.len() > 1 {
        return multi_component_spectral_init(data, n_components, edges, &components, seed);
    }

    spectral_init_connected(n_samples, n_components, edges, seed)
}

fn spectral_embedding_from_affinity_raw(
    affinity: &[Vec<f64>],
    embedding_dim: usize,
    drop_first: bool,
) -> Option<Vec<Vec<f32>>> {
    let n_samples = affinity.len();
    if n_samples == 0 {
        return None;
    }
    if affinity.iter().any(|row| row.len() != n_samples) {
        return None;
    }

    let mut degrees = vec![0.0_f64; n_samples];
    for i in 0..n_samples {
        degrees[i] = affinity[i].iter().sum();
    }
    if degrees.iter().all(|degree| *degree <= 0.0) {
        return None;
    }

    let mut laplacian = DMatrix::<f64>::identity(n_samples, n_samples);
    for i in 0..n_samples {
        if degrees[i] <= 0.0 {
            continue;
        }
        for j in 0..n_samples {
            if degrees[j] <= 0.0 {
                continue;
            }
            let weight = affinity[i][j];
            if weight > 0.0 {
                laplacian[(i, j)] -= weight / (degrees[i].sqrt() * degrees[j].sqrt());
            }
        }
    }

    let eig = SymmetricEigen::new(laplacian);
    let mut order: Vec<usize> = (0..n_samples).collect();
    order.sort_by(|&i, &j| {
        eig.eigenvalues[i]
            .partial_cmp(&eig.eigenvalues[j])
            .unwrap_or(Ordering::Equal)
    });

    let offset = usize::from(drop_first);
    if order.len() <= embedding_dim + offset {
        return None;
    }

    let mut coords = vec![vec![0.0_f32; embedding_dim]; n_samples];
    for out_col in 0..embedding_dim {
        let eig_col = order[out_col + offset];
        for row in 0..n_samples {
            coords[row][out_col] = eig.eigenvectors[(row, eig_col)] as f32;
        }
    }

    if coords
        .iter()
        .flat_map(|row| row.iter())
        .all(|value| value.is_finite())
    {
        Some(coords)
    } else {
        None
    }
}

fn spectral_init_connected(
    n_samples: usize,
    n_components: usize,
    edges: &[Edge],
    seed: u64,
) -> Option<Vec<Vec<f32>>> {
    if n_samples <= n_components + 1 || edges.is_empty() {
        return None;
    }

    let mut w = DMatrix::<f64>::zeros(n_samples, n_samples);
    for edge in edges {
        if edge.head == edge.tail {
            continue;
        }
        let i = edge.head;
        let j = edge.tail;
        let val = edge.weight.max(0.0) as f64;
        if val > w[(i, j)] {
            w[(i, j)] = val;
        }
    }

    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let sym = w[(i, j)].max(w[(j, i)]);
            w[(i, j)] = sym;
            w[(j, i)] = sym;
        }
    }

    let mut degrees = vec![0.0_f64; n_samples];
    for i in 0..n_samples {
        let mut sum = 0.0_f64;
        for j in 0..n_samples {
            sum += w[(i, j)];
        }
        degrees[i] = sum;
    }
    if degrees.iter().all(|degree| *degree <= 0.0) {
        return None;
    }

    let mut laplacian = DMatrix::<f64>::identity(n_samples, n_samples);
    for i in 0..n_samples {
        if degrees[i] <= 0.0 {
            continue;
        }
        for j in 0..n_samples {
            if degrees[j] <= 0.0 {
                continue;
            }
            let wij = w[(i, j)];
            if wij > 0.0 {
                laplacian[(i, j)] -= wij / (degrees[i].sqrt() * degrees[j].sqrt());
            }
        }
    }

    let eig = SymmetricEigen::new(laplacian);
    let mut order: Vec<usize> = (0..n_samples).collect();
    order.sort_by(|&i, &j| {
        eig.eigenvalues[i]
            .partial_cmp(&eig.eigenvalues[j])
            .unwrap_or(Ordering::Equal)
    });

    if order.len() <= n_components {
        return None;
    }

    let mut coords = vec![vec![0.0_f32; n_components]; n_samples];
    for out_col in 0..n_components {
        let eig_col = order[out_col + 1];
        for row in 0..n_samples {
            coords[row][out_col] = eig.eigenvectors[(row, eig_col)] as f32;
        }
    }

    noisy_scale_coords(&mut coords, seed ^ 0x5DEECE66D, INIT_MAX_COORD, INIT_NOISE);

    if coords
        .iter()
        .flat_map(|r| r.iter())
        .all(|v| v.is_finite())
    {
        Some(coords)
    } else {
        None
    }
}

fn connected_components_from_edges(n_samples: usize, edges: &[Edge]) -> Vec<Vec<usize>> {
    let mut parent = (0..n_samples).collect::<Vec<usize>>();
    let mut rank = vec![0_u8; n_samples];

    fn find(parent: &mut [usize], x: usize) -> usize {
        let mut root = x;
        while parent[root] != root {
            root = parent[root];
        }
        let mut node = x;
        while parent[node] != root {
            let next = parent[node];
            parent[node] = root;
            node = next;
        }
        root
    }

    fn union(parent: &mut [usize], rank: &mut [u8], a: usize, b: usize) {
        let root_a = find(parent, a);
        let root_b = find(parent, b);
        if root_a == root_b {
            return;
        }
        if rank[root_a] < rank[root_b] {
            parent[root_a] = root_b;
        } else if rank[root_a] > rank[root_b] {
            parent[root_b] = root_a;
        } else {
            parent[root_b] = root_a;
            rank[root_a] += 1;
        }
    }

    for edge in edges {
        if edge.head != edge.tail {
            union(&mut parent, &mut rank, edge.head, edge.tail);
        }
    }

    let mut groups = HashMap::<usize, Vec<usize>>::new();
    for node in 0..n_samples {
        let root = find(&mut parent, node);
        groups.entry(root).or_default().push(node);
    }

    let mut components = groups.into_values().collect::<Vec<Vec<usize>>>();
    for component in components.iter_mut() {
        component.sort_unstable();
    }
    components.sort_by_key(|component| component[0]);
    components
}

fn remap_component_edges(
    component: &[usize],
    component_labels: &[usize],
    component_id: usize,
    edges: &[Edge],
) -> Vec<Edge> {
    let mut mapping = vec![usize::MAX; component_labels.len()];
    for (local_idx, &global_idx) in component.iter().enumerate() {
        mapping[global_idx] = local_idx;
    }

    let mut out = Vec::new();
    for edge in edges {
        if component_labels[edge.head] == component_id && component_labels[edge.tail] == component_id {
            out.push(Edge {
                head: mapping[edge.head],
                tail: mapping[edge.tail],
                weight: edge.weight,
            });
        }
    }
    out
}

fn component_centroids(data: &[Vec<f32>], components: &[Vec<usize>]) -> Vec<Vec<f64>> {
    let n_features = data[0].len();
    let mut centroids = Vec::with_capacity(components.len());

    for component in components {
        let mut centroid = vec![0.0_f64; n_features];
        for &idx in component {
            for (feature_idx, &value) in data[idx].iter().enumerate() {
                centroid[feature_idx] += value as f64;
            }
        }
        let scale = 1.0 / component.len() as f64;
        for value in centroid.iter_mut() {
            *value *= scale;
        }
        centroids.push(centroid);
    }

    centroids
}

fn squared_distance_f64(x: &[f64], y: &[f64]) -> f64 {
    x.iter()
        .zip(y.iter())
        .map(|(a, b)| {
            let d = a - b;
            d * d
        })
        .sum()
}

fn meta_component_layout(
    data: &[Vec<f32>],
    embedding_dim: usize,
    components: &[Vec<usize>],
    seed: u64,
) -> Option<Vec<Vec<f32>>> {
    let n_graph_components = components.len();
    if n_graph_components == 0 {
        return None;
    }
    if n_graph_components == 1 {
        return Some(vec![vec![0.0; embedding_dim]]);
    }

    if n_graph_components <= 2 * embedding_dim {
        let k = n_graph_components.div_ceil(2);
        let mut layout = vec![vec![0.0_f32; embedding_dim]; n_graph_components];
        for i in 0..n_graph_components {
            let axis = i % k;
            layout[i][axis] = if i < k { 1.0 } else { -1.0 };
        }
        return Some(layout);
    }

    let centroids = component_centroids(data, components);
    let mut affinity = vec![vec![0.0_f64; n_graph_components]; n_graph_components];
    for i in 0..n_graph_components {
        affinity[i][i] = 1.0;
        for j in (i + 1)..n_graph_components {
            let dist2 = squared_distance_f64(&centroids[i], &centroids[j]);
            let weight = (-dist2).exp();
            affinity[i][j] = weight;
            affinity[j][i] = weight;
        }
    }

    spectral_embedding_from_affinity(&affinity, embedding_dim, true, seed ^ 0xC85E_7D6A_DA31_8F21)
}

fn spectral_embedding_from_affinity(
    affinity: &[Vec<f64>],
    embedding_dim: usize,
    drop_first: bool,
    seed: u64,
) -> Option<Vec<Vec<f32>>> {
    let mut coords = spectral_embedding_from_affinity_raw(affinity, embedding_dim, drop_first)?;
    noisy_scale_coords(&mut coords, seed, INIT_MAX_COORD, INIT_NOISE);
    if coords
        .iter()
        .flat_map(|row| row.iter())
        .all(|value| value.is_finite())
    {
        Some(coords)
    } else {
        None
    }
}

fn spectral_init_connected_raw(
    n_samples: usize,
    n_components: usize,
    edges: &[Edge],
) -> Option<Vec<Vec<f32>>> {
    if n_samples <= n_components + 1 || edges.is_empty() {
        return None;
    }

    let mut affinity = vec![vec![0.0_f64; n_samples]; n_samples];
    for edge in edges {
        if edge.head == edge.tail {
            continue;
        }
        let i = edge.head;
        let j = edge.tail;
        let weight = edge.weight.max(0.0) as f64;
        if weight > affinity[i][j] {
            affinity[i][j] = weight;
        }
    }
    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let sym = affinity[i][j].max(affinity[j][i]);
            affinity[i][j] = sym;
            affinity[j][i] = sym;
        }
    }

    spectral_embedding_from_affinity_raw(&affinity, n_components, true)
}

fn add_noise(coords: &mut [Vec<f32>], seed: u64, noise: f32) {
    if coords.is_empty() || coords[0].is_empty() {
        return;
    }

    let normal = Normal::new(0.0_f64, noise as f64).expect("normal distribution should be valid");
    let mut rng = SmallRng::seed_from_u64(seed);
    for row in coords.iter_mut() {
        for value in row.iter_mut() {
            *value += normal.sample(&mut rng) as f32;
        }
    }
}

fn multi_component_spectral_init(
    data: &[Vec<f32>],
    embedding_dim: usize,
    edges: &[Edge],
    components: &[Vec<usize>],
    seed: u64,
) -> Option<Vec<Vec<f32>>> {
    let n_samples = data.len();
    let mut component_labels = vec![usize::MAX; n_samples];
    for (component_id, component) in components.iter().enumerate() {
        for &idx in component {
            component_labels[idx] = component_id;
        }
    }

    let component_layout =
        meta_component_layout(data, embedding_dim, components, seed ^ 0xA13F_52A9_2D4C_B801)?;
    let mut result = vec![vec![0.0_f32; embedding_dim]; n_samples];

    for (component_id, component) in components.iter().enumerate() {
        let anchor = &component_layout[component_id];
        let mut data_range = f32::INFINITY;
        for (other_id, other_anchor) in component_layout.iter().enumerate() {
            if other_id == component_id {
                continue;
            }
            let dist = euclidean_distance(anchor, other_anchor);
            if dist > 0.0 {
                data_range = data_range.min(dist / 2.0);
            }
        }
        if !data_range.is_finite() || data_range <= 0.0 {
            data_range = 1.0;
        }

        let local_coords = if component.len() < 2 * embedding_dim || component.len() <= embedding_dim + 1 {
            random_init(
                component.len(),
                embedding_dim,
                seed ^ ((component_id as u64 + 1) * 0x9E37_79B9),
                -data_range,
                data_range,
            )
        } else {
            let component_edges =
                remap_component_edges(component, &component_labels, component_id, edges);
            let mut coords =
                spectral_init_connected_raw(component.len(), embedding_dim, &component_edges)?;
            let max_abs = coords
                .iter()
                .flat_map(|row| row.iter())
                .map(|value| value.abs())
                .fold(0.0_f32, f32::max);
            let expansion = if max_abs > 0.0 {
                data_range / max_abs
            } else {
                1.0
            };
            for row in coords.iter_mut() {
                for value in row.iter_mut() {
                    *value *= expansion;
                }
            }
            add_noise(
                &mut coords,
                seed ^ ((component_id as u64 + 1) * 0x94D0_49BB),
                INIT_NOISE,
            );
            coords
        };

        for (local_idx, &global_idx) in component.iter().enumerate() {
            for dim in 0..embedding_dim {
                result[global_idx][dim] = local_coords[local_idx][dim] + anchor[dim];
            }
        }
    }

    if result
        .iter()
        .flat_map(|row| row.iter())
        .all(|value| value.is_finite())
    {
        Some(result)
    } else {
        None
    }
}

fn noisy_scale_coords(coords: &mut [Vec<f32>], seed: u64, max_coord: f32, noise: f32) {
    if coords.is_empty() || coords[0].is_empty() {
        return;
    }

    let max_abs = coords
        .iter()
        .flat_map(|r| r.iter())
        .map(|v| v.abs())
        .fold(0.0_f32, f32::max);

    let expansion = if max_abs > 0.0 {
        max_coord / max_abs
    } else {
        1.0
    };

    let normal = Normal::new(0.0_f64, noise as f64).expect("normal distribution should be valid");
    let mut rng = SmallRng::seed_from_u64(seed);

    for row in coords.iter_mut() {
        for value in row.iter_mut() {
            *value = *value * expansion + normal.sample(&mut rng) as f32;
        }
    }
}

fn init_graph_transform(
    n_new_samples: usize,
    dim: usize,
    graph_edges: &[Edge],
    base_embedding: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let mut rows: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_new_samples];
    for edge in graph_edges {
        rows[edge.head].push((edge.tail, edge.weight));
    }

    let mut result = vec![vec![0.0_f32; dim]; n_new_samples];

    for i in 0..n_new_samples {
        if rows[i].is_empty() {
            result[i].fill(f32::NAN);
            continue;
        }

        if let Some((col_idx, _)) = rows[i]
            .iter()
            .copied()
            .find(|(_, weight)| (*weight - 1.0).abs() < f32::EPSILON)
        {
            result[i].clone_from_slice(&base_embedding[col_idx]);
            continue;
        }

        let row_sum: f32 = rows[i].iter().map(|(_, w)| *w).sum();
        if row_sum <= 0.0 {
            result[i].fill(f32::NAN);
            continue;
        }

        for (col_idx, weight) in rows[i].iter().copied() {
            let w = weight / row_sum;
            for d in 0..dim {
                result[i][d] += w * base_embedding[col_idx][d];
            }
        }
    }

    result
}

fn make_epochs_per_sample(weights: &[f32], n_epochs: usize) -> Vec<f32> {
    if weights.is_empty() {
        return Vec::new();
    }

    let max_weight = weights
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
        .max(1e-12);

    weights
        .iter()
        .copied()
        .map(|w| {
            let n_samples = n_epochs as f32 * (w / max_weight);
            if n_samples > 0.0 {
                n_epochs as f32 / n_samples
            } else {
                f32::INFINITY
            }
        })
        .collect()
}

fn clip(value: f32) -> f32 {
    value.clamp(-4.0, 4.0)
}

fn two_rows_mut<T>(rows: &mut [Vec<T>], i: usize, j: usize) -> (&mut Vec<T>, &mut Vec<T>) {
    assert!(i != j, "indices must be different");
    if i < j {
        let (left, right) = rows.split_at_mut(j);
        (&mut left[i], &mut right[0])
    } else {
        let (left, right) = rows.split_at_mut(i);
        (&mut right[0], &mut left[j])
    }
}

fn is_finite_row(row: &[f32]) -> bool {
    row.iter().all(|v| v.is_finite())
}

fn euclidean_distance_with_grad(x: &[f32], y: &[f32]) -> (f32, Vec<f32>) {
    let mut sq = 0.0_f32;
    for (a, b) in x.iter().zip(y.iter()) {
        let d = *a - *b;
        sq += d * d;
    }
    let dist = sq.sqrt();
    let denom = 1e-6 + dist;
    let grad = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| (*a - *b) / denom)
        .collect::<Vec<f32>>();
    (dist, grad)
}

fn optimize_layout_training(
    embedding: &mut Vec<Vec<f32>>,
    edges: &[Edge],
    n_epochs: usize,
    a: f32,
    b: f32,
    initial_alpha: f32,
    negative_sample_rate: usize,
    repulsion_strength: f32,
    seed: u64,
) {
    if edges.is_empty() || embedding.is_empty() || n_epochs == 0 {
        return;
    }

    let dim = embedding[0].len();
    let n_vertices = embedding.len();

    let weights: Vec<f32> = edges.iter().map(|e| e.weight).collect();
    let epochs_per_sample = make_epochs_per_sample(&weights, n_epochs);
    let mut epoch_of_next_sample = epochs_per_sample.clone();

    let epochs_per_negative_sample: Vec<f32> = epochs_per_sample
        .iter()
        .map(|eps| *eps / negative_sample_rate as f32)
        .collect();
    let mut epoch_of_next_negative_sample = epochs_per_negative_sample.clone();

    let mut rng = SmallRng::seed_from_u64(seed);

    for epoch in 0..n_epochs {
        let alpha = initial_alpha * (1.0 - epoch as f32 / n_epochs as f32);

        for (edge_idx, edge) in edges.iter().enumerate() {
            if epoch_of_next_sample[edge_idx] > epoch as f32 {
                continue;
            }

            let head = edge.head;
            let tail = edge.tail;

            if head != tail {
                let dist_squared = squared_distance(&embedding[head], &embedding[tail]);
                let grad_coeff = if dist_squared > 0.0 {
                    let dist_pow_b = dist_squared.powf(b);
                    -2.0 * a * b * dist_squared.powf(b - 1.0) / (a * dist_pow_b + 1.0)
                } else {
                    0.0
                };

                let (current, other) = two_rows_mut(embedding.as_mut_slice(), head, tail);
                for d in 0..dim {
                    let grad = clip(grad_coeff * (current[d] - other[d]));
                    current[d] += grad * alpha;
                    other[d] -= grad * alpha;
                }
            }

            epoch_of_next_sample[edge_idx] += epochs_per_sample[edge_idx];

            let eps_neg = epochs_per_negative_sample[edge_idx];
            if !eps_neg.is_finite() || eps_neg <= 0.0 {
                continue;
            }

            let n_neg_samples =
                ((epoch as f32 - epoch_of_next_negative_sample[edge_idx]) / eps_neg)
                    .floor()
                    .max(0.0) as usize;

            for _ in 0..n_neg_samples {
                let neg_idx = rng.gen_range(0..n_vertices);
                if neg_idx == head {
                    continue;
                }

                let dist_squared = squared_distance(&embedding[head], &embedding[neg_idx]);

                let grad_coeff = if dist_squared > 0.0 {
                    let dist_pow_b = dist_squared.powf(b);
                    2.0 * repulsion_strength * b
                        / ((0.001 + dist_squared) * (a * dist_pow_b + 1.0))
                } else {
                    0.0
                };

                if grad_coeff > 0.0 {
                    let other = embedding[neg_idx].clone();
                    let current = &mut embedding[head];
                    for d in 0..dim {
                        let grad = clip(grad_coeff * (current[d] - other[d]));
                        current[d] += grad * alpha;
                    }
                }
            }

            epoch_of_next_negative_sample[edge_idx] += n_neg_samples as f32 * eps_neg;
        }
    }
}

fn optimize_layout_transform(
    embedding: &mut Vec<Vec<f32>>,
    base_embedding: &[Vec<f32>],
    edges: &[Edge],
    n_epochs: usize,
    a: f32,
    b: f32,
    initial_alpha: f32,
    negative_sample_rate: usize,
    repulsion_strength: f32,
    seed: u64,
) {
    if edges.is_empty() || embedding.is_empty() || n_epochs == 0 {
        return;
    }

    let dim = embedding[0].len();
    let n_vertices = base_embedding.len();

    let weights: Vec<f32> = edges.iter().map(|e| e.weight).collect();
    let epochs_per_sample = make_epochs_per_sample(&weights, n_epochs);
    let mut epoch_of_next_sample = epochs_per_sample.clone();

    let epochs_per_negative_sample: Vec<f32> = epochs_per_sample
        .iter()
        .map(|eps| *eps / negative_sample_rate as f32)
        .collect();
    let mut epoch_of_next_negative_sample = epochs_per_negative_sample.clone();

    let mut rng = SmallRng::seed_from_u64(seed);

    for epoch in 0..n_epochs {
        let alpha = initial_alpha * (1.0 - epoch as f32 / n_epochs as f32);

        for (edge_idx, edge) in edges.iter().enumerate() {
            if epoch_of_next_sample[edge_idx] > epoch as f32 {
                continue;
            }

            let head = edge.head;
            let tail = edge.tail;

            if !is_finite_row(&embedding[head]) {
                continue;
            }

            let dist_squared = squared_distance(&embedding[head], &base_embedding[tail]);
            let grad_coeff = if dist_squared > 0.0 {
                let dist_pow_b = dist_squared.powf(b);
                -2.0 * a * b * dist_squared.powf(b - 1.0) / (a * dist_pow_b + 1.0)
            } else {
                0.0
            };

            {
                let current = &mut embedding[head];
                let other = &base_embedding[tail];
                for d in 0..dim {
                    let grad = clip(grad_coeff * (current[d] - other[d]));
                    current[d] += grad * alpha;
                }
            }

            epoch_of_next_sample[edge_idx] += epochs_per_sample[edge_idx];

            let eps_neg = epochs_per_negative_sample[edge_idx];
            if !eps_neg.is_finite() || eps_neg <= 0.0 {
                continue;
            }

            let n_neg_samples =
                ((epoch as f32 - epoch_of_next_negative_sample[edge_idx]) / eps_neg)
                    .floor()
                    .max(0.0) as usize;

            for _ in 0..n_neg_samples {
                let neg_idx = rng.gen_range(0..n_vertices);

                let dist_squared = squared_distance(&embedding[head], &base_embedding[neg_idx]);
                let grad_coeff = if dist_squared > 0.0 {
                    let dist_pow_b = dist_squared.powf(b);
                    2.0 * repulsion_strength * b
                        / ((0.001 + dist_squared) * (a * dist_pow_b + 1.0))
                } else {
                    0.0
                };

                if grad_coeff > 0.0 {
                    let current = &mut embedding[head];
                    let other = &base_embedding[neg_idx];
                    for d in 0..dim {
                        let grad = clip(grad_coeff * (current[d] - other[d]));
                        current[d] += grad * alpha;
                    }
                }
            }

            epoch_of_next_negative_sample[edge_idx] += n_neg_samples as f32 * eps_neg;
        }
    }
}

fn optimize_layout_inverse(
    head_embedding: &mut Vec<Vec<f32>>,
    tail_embedding: &[Vec<f32>],
    edges: &[Edge],
    sigmas: &[f32],
    rhos: &[f32],
    n_epochs: usize,
    initial_alpha: f32,
    negative_sample_rate: usize,
    repulsion_strength: f32,
    seed: u64,
) {
    if edges.is_empty() || head_embedding.is_empty() || n_epochs == 0 {
        return;
    }

    let dim = head_embedding[0].len();
    let n_vertices = tail_embedding.len();

    let weights: Vec<f32> = edges.iter().map(|e| e.weight).collect();
    let epochs_per_sample = make_epochs_per_sample(&weights, n_epochs);
    let mut epoch_of_next_sample = epochs_per_sample.clone();

    let epochs_per_negative_sample: Vec<f32> = epochs_per_sample
        .iter()
        .map(|eps| *eps / negative_sample_rate as f32)
        .collect();
    let mut epoch_of_next_negative_sample = epochs_per_negative_sample.clone();

    let mut rng = SmallRng::seed_from_u64(seed);

    for epoch in 0..n_epochs {
        let alpha = initial_alpha * (1.0 - epoch as f32 / n_epochs as f32);

        for (edge_idx, edge) in edges.iter().enumerate() {
            if epoch_of_next_sample[edge_idx] > epoch as f32 {
                continue;
            }

            let head = edge.head;
            let tail = edge.tail;
            if !is_finite_row(&head_embedding[head]) {
                continue;
            }

            let (_, grad_dist_output) =
                euclidean_distance_with_grad(&head_embedding[head], &tail_embedding[tail]);

            let wl = edge.weight;
            let grad_coeff = -(1.0 / (wl * sigmas[tail] + 1e-6));

            {
                let current = &mut head_embedding[head];
                for d in 0..dim {
                    let grad_d = clip(grad_coeff * grad_dist_output[d]);
                    current[d] += grad_d * alpha;
                }
            }

            epoch_of_next_sample[edge_idx] += epochs_per_sample[edge_idx];

            let eps_neg = epochs_per_negative_sample[edge_idx];
            if !eps_neg.is_finite() || eps_neg <= 0.0 {
                continue;
            }

            let n_neg_samples =
                ((epoch as f32 - epoch_of_next_negative_sample[edge_idx]) / eps_neg)
                    .floor()
                    .max(0.0) as usize;

            for _ in 0..n_neg_samples {
                let neg_tail = rng.gen_range(0..n_vertices);
                let (dist_neg, grad_neg) =
                    euclidean_distance_with_grad(&head_embedding[head], &tail_embedding[neg_tail]);

                let wh =
                    (-(dist_neg - rhos[neg_tail]).max(1e-6) / (sigmas[neg_tail] + 1e-6)).exp();
                let grad_coeff = -repulsion_strength
                    * ((0.0 - wh) / ((1.0 - wh) * sigmas[neg_tail] + 1e-6));

                {
                    let current = &mut head_embedding[head];
                    for d in 0..dim {
                        let grad_d = clip(grad_coeff * grad_neg[d]);
                        current[d] += grad_d * alpha;
                    }
                }
            }

            epoch_of_next_negative_sample[edge_idx] += n_neg_samples as f32 * eps_neg;
        }
    }
}

fn curve_loss(a: f32, b: f32, xv: &[f32], yv: &[f32]) -> f32 {
    let mut loss = 0.0_f32;

    for (&x, &y) in xv.iter().zip(yv.iter()) {
        let model = 1.0 / (1.0 + a * x.powf(2.0 * b));
        let e = model - y;
        loss += e * e;
    }

    loss / xv.len() as f32
}

pub fn find_ab_params(spread: f32, min_dist: f32) -> (f32, f32) {
    let n = 300;
    let xmax = spread * 3.0;

    let mut xv = Vec::with_capacity(n);
    let mut yv = Vec::with_capacity(n);

    for i in 0..n {
        let x = xmax * i as f32 / (n as f32 - 1.0);
        xv.push(x);
        let y = if x < min_dist {
            1.0
        } else {
            (-(x - min_dist) / spread).exp()
        };
        yv.push(y);
    }

    let mut best_a = 1.0_f32;
    let mut best_b = 1.0_f32;
    let mut best_loss = f32::INFINITY;

    for bi in 0..=70 {
        let b = 0.3 + 2.7 * bi as f32 / 70.0;
        for ai in 0..=120 {
            let log_a = -3.0 + 6.0 * ai as f32 / 120.0;
            let a = 10_f32.powf(log_a);
            let loss = curve_loss(a, b, &xv, &yv);
            if loss < best_loss {
                best_loss = loss;
                best_a = a;
                best_b = b;
            }
        }
    }

    let mut step_a = (best_a * 0.35).max(0.05);
    let mut step_b = 0.2_f32;

    for _ in 0..80 {
        let candidates = [
            (best_a + step_a, best_b),
            ((best_a - step_a).max(1e-6), best_b),
            (best_a, best_b + step_b),
            (best_a, (best_b - step_b).max(1e-6)),
            (best_a + step_a, best_b + step_b),
            (best_a + step_a, (best_b - step_b).max(1e-6)),
            ((best_a - step_a).max(1e-6), best_b + step_b),
            ((best_a - step_a).max(1e-6), (best_b - step_b).max(1e-6)),
        ];

        let mut improved = false;
        for (a, b) in candidates {
            let loss = curve_loss(a, b, &xv, &yv);
            if loss < best_loss {
                best_loss = loss;
                best_a = a;
                best_b = b;
                improved = true;
            }
        }

        if !improved {
            step_a *= 0.7;
            step_b *= 0.7;
            if step_a < 1e-5 && step_b < 1e-5 {
                break;
            }
        }
    }

    (best_a, best_b)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn disconnected_component_data(
        n_components: usize,
        points_per_component: usize,
    ) -> (Vec<Vec<f32>>, Vec<usize>) {
        let mut data = Vec::with_capacity(n_components * points_per_component);
        let mut labels = Vec::with_capacity(n_components * points_per_component);

        for component in 0..n_components {
            let shift = component as f32 * 50.0;
            for point_idx in 0..points_per_component {
                let t =
                    2.0 * std::f32::consts::PI * point_idx as f32 / points_per_component as f32;
                data.push(vec![
                    shift + t.cos(),
                    t.sin(),
                    (2.0 * t).cos(),
                    (2.0 * t).sin(),
                    -1.0 + 2.0 * point_idx as f32 / points_per_component as f32,
                    component as f32 * 0.1,
                    0.0,
                    0.0,
                ]);
                labels.push(component);
            }
        }

        (data, labels)
    }

    fn component_centroid(embedding: &[Vec<f32>], labels: &[usize], component: usize) -> Vec<f32> {
        let dim = embedding[0].len();
        let mut centroid = vec![0.0_f32; dim];
        let mut count = 0.0_f32;
        for (row, &label) in embedding.iter().zip(labels.iter()) {
            if label == component {
                count += 1.0;
                for (dim_idx, &value) in row.iter().enumerate() {
                    centroid[dim_idx] += value;
                }
            }
        }
        for value in centroid.iter_mut() {
            *value /= count;
        }
        centroid
    }

    fn component_std_norm(embedding: &[Vec<f32>], labels: &[usize], component: usize) -> f32 {
        let centroid = component_centroid(embedding, labels, component);
        let dim = embedding[0].len();
        let mut variances = vec![0.0_f32; dim];
        let mut count = 0.0_f32;

        for (row, &label) in embedding.iter().zip(labels.iter()) {
            if label == component {
                count += 1.0;
                for dim_idx in 0..dim {
                    let diff = row[dim_idx] - centroid[dim_idx];
                    variances[dim_idx] += diff * diff;
                }
            }
        }

        variances
            .iter()
            .map(|sum_sq| (sum_sq / count).sqrt())
            .map(|std| std * std)
            .sum::<f32>()
            .sqrt()
    }

    #[test]
    fn fit_transform_returns_expected_shape_and_finite_values() {
        let data = synthetic_data(80, 8);
        let params = UmapParams {
            n_neighbors: 12,
            n_components: 2,
            n_epochs: Some(60),
            init: InitMethod::Random,
            random_seed: 7,
            ..UmapParams::default()
        };

        let mut model = UmapModel::new(params);
        let embedding = model.fit_transform(&data).expect("fit_transform should succeed");

        assert_eq!(embedding.len(), 80);
        assert_eq!(embedding[0].len(), 2);

        for row in &embedding {
            for &v in row {
                assert!(v.is_finite());
            }
        }
    }

    #[test]
    fn deterministic_with_same_seed() {
        let data = synthetic_data(60, 6);
        let params = UmapParams {
            n_neighbors: 10,
            n_components: 2,
            n_epochs: Some(40),
            init: InitMethod::Random,
            random_seed: 1234,
            ..UmapParams::default()
        };

        let mut model_1 = UmapModel::new(params.clone());
        let mut model_2 = UmapModel::new(params);

        let emb_1 = model_1.fit_transform(&data).expect("model 1 fit failed");
        let emb_2 = model_2.fit_transform(&data).expect("model 2 fit failed");

        assert_eq!(emb_1, emb_2);
    }

    #[test]
    fn spectral_init_and_transform_work() {
        let data = synthetic_data(120, 5);
        let query = data.iter().take(15).cloned().collect::<Vec<_>>();

        let params = UmapParams {
            n_neighbors: 15,
            n_components: 2,
            n_epochs: Some(80),
            init: InitMethod::Spectral,
            random_seed: 9,
            ..UmapParams::default()
        };

        let mut model = UmapModel::new(params);
        let embedding = model.fit_transform(&data).expect("fit should succeed");
        assert_eq!(embedding.len(), 120);

        let transformed = model.transform(&query).expect("transform should succeed");
        assert_eq!(transformed.len(), 15);
        assert_eq!(transformed[0].len(), 2);
        assert!(transformed
            .iter()
            .flat_map(|r| r.iter())
            .all(|v| v.is_finite()));
    }

    #[test]
    fn transform_requires_fit() {
        let model = UmapModel::new(UmapParams::default());
        let query = synthetic_data(4, 3);
        let err = model.transform(&query).expect_err("must fail before fit");
        assert!(matches!(err, UmapError::NotFitted));
    }

    #[test]
    fn inverse_transform_workflow() {
        let data = synthetic_data(140, 6);
        let params = UmapParams {
            n_neighbors: 15,
            n_components: 2,
            n_epochs: Some(90),
            init: InitMethod::Spectral,
            random_seed: 11,
            ..UmapParams::default()
        };

        let mut model = UmapModel::new(params);
        let embedding = model.fit_transform(&data).expect("fit should succeed");

        let emb_query = embedding.iter().skip(130).cloned().collect::<Vec<_>>();
        let reconstructed = model
            .inverse_transform(&emb_query)
            .expect("inverse_transform should succeed");

        assert_eq!(reconstructed.len(), emb_query.len());
        assert_eq!(reconstructed[0].len(), 6);
        assert!(reconstructed
            .iter()
            .flat_map(|r| r.iter())
            .all(|v| v.is_finite()));
    }

    #[test]
    fn inverse_transform_requires_fit() {
        let model = UmapModel::new(UmapParams::default());
        let emb_query = vec![vec![0.1_f32, 0.2_f32]];
        let err = model
            .inverse_transform(&emb_query)
            .expect_err("must fail before fit");
        assert!(matches!(err, UmapError::NotFitted));
    }

    #[test]
    fn approximate_knn_path_runs() {
        let data = synthetic_data(90, 7);
        let params = UmapParams {
            n_neighbors: 12,
            n_components: 2,
            n_epochs: Some(50),
            init: InitMethod::Random,
            random_seed: 99,
            use_approximate_knn: true,
            approx_knn_candidates: 24,
            approx_knn_iters: 6,
            approx_knn_threshold: 0,
            ..UmapParams::default()
        };

        let mut model = UmapModel::new(params);
        let embedding = model.fit_transform(&data).expect("fit should succeed on ANN path");
        assert_eq!(embedding.len(), 90);
        assert_eq!(embedding[0].len(), 2);
        assert!(embedding
            .iter()
            .flat_map(|r| r.iter())
            .all(|v| v.is_finite()));
    }

    #[test]
    fn spectral_init_handles_disconnected_components() {
        let (data, labels) = disconnected_component_data(4, 40);
        let params = UmapParams {
            n_neighbors: 10,
            n_components: 2,
            n_epochs: Some(0),
            init: InitMethod::Spectral,
            random_seed: 42,
            use_approximate_knn: false,
            ..UmapParams::default()
        };

        let mut model = UmapModel::new(params);
        let embedding = model.fit_transform(&data).expect("fit should succeed");

        let component_std_mins = (0..4)
            .map(|component| component_std_norm(&embedding, &labels, component))
            .collect::<Vec<f32>>();
        let component_centroids = (0..4)
            .map(|component| component_centroid(&embedding, &labels, component))
            .collect::<Vec<Vec<f32>>>();

        let mut min_centroid_distance = f32::INFINITY;
        for i in 0..component_centroids.len() {
            for j in (i + 1)..component_centroids.len() {
                min_centroid_distance = min_centroid_distance.min(euclidean_distance(
                    &component_centroids[i],
                    &component_centroids[j],
                ));
            }
        }

        let min_component_std = component_std_mins
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);

        assert!(
            min_component_std > 0.01,
            "expected each disconnected component to keep internal spread, got {min_component_std}"
        );
        assert!(
            min_centroid_distance > 0.5,
            "expected disconnected component centroids to remain separated, got {min_centroid_distance}"
        );
    }
}
