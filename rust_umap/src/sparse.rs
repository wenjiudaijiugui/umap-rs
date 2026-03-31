use crate::UmapError;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Debug, Clone)]
pub struct SparseCsrMatrix {
    n_rows: usize,
    n_cols: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f32>,
    squared_norms: Vec<f32>,
}

impl SparseCsrMatrix {
    pub fn new(
        n_rows: usize,
        n_cols: usize,
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<f32>,
    ) -> Result<Self, UmapError> {
        if n_cols == 0 {
            return Err(UmapError::InvalidParameter(
                "csr n_cols must be >= 1".to_string(),
            ));
        }
        if indptr.len() != n_rows + 1 {
            return Err(UmapError::InvalidParameter(format!(
                "csr indptr length must be n_rows + 1 (got {}, expected {})",
                indptr.len(),
                n_rows + 1
            )));
        }
        if indptr.first().copied().unwrap_or_default() != 0 {
            return Err(UmapError::InvalidParameter(
                "csr indptr must start from 0".to_string(),
            ));
        }
        if indices.len() != data.len() {
            return Err(UmapError::InvalidParameter(
                "csr indices/data lengths must match".to_string(),
            ));
        }
        if indptr[n_rows] != indices.len() {
            return Err(UmapError::InvalidParameter(format!(
                "csr indptr last value ({}) must equal nnz ({})",
                indptr[n_rows],
                indices.len()
            )));
        }

        for row in 0..n_rows {
            if indptr[row] > indptr[row + 1] {
                return Err(UmapError::InvalidParameter(format!(
                    "csr indptr must be non-decreasing, got indptr[{row}]={} > indptr[{}]={}",
                    indptr[row],
                    row + 1,
                    indptr[row + 1]
                )));
            }
            let start = indptr[row];
            let end = indptr[row + 1];
            let row_indices = &indices[start..end];
            for w in row_indices.windows(2) {
                if w[0] >= w[1] {
                    return Err(UmapError::InvalidParameter(format!(
                        "csr row {row} indices must be strictly increasing"
                    )));
                }
            }
            for &col in row_indices {
                if col >= n_cols {
                    return Err(UmapError::InvalidParameter(format!(
                        "csr row {row} has column index {col} out of bounds for n_cols={n_cols}"
                    )));
                }
            }
        }

        if data.iter().any(|v| !v.is_finite()) {
            return Err(UmapError::InvalidParameter(
                "csr data must be finite".to_string(),
            ));
        }

        let mut squared_norms = vec![0.0_f32; n_rows];
        for (row, norm_slot) in squared_norms.iter_mut().enumerate() {
            let (_, vals) = row_slices(&indptr, &indices, &data, row);
            *norm_slot = vals.iter().map(|v| v * v).sum();
        }

        Ok(Self {
            n_rows,
            n_cols,
            indptr,
            indices,
            data,
            squared_norms,
        })
    }

    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    #[inline]
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    #[inline]
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn squared_norm(&self, row: usize) -> f32 {
        self.squared_norms[row]
    }

    #[inline]
    pub fn row(&self, row: usize) -> (&[usize], &[f32]) {
        row_slices(&self.indptr, &self.indices, &self.data, row)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct NeighborCandidate {
    idx: usize,
    dist: f32,
}

impl Eq for NeighborCandidate {}

impl Ord for NeighborCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then_with(|| self.idx.cmp(&other.idx))
    }
}

impl PartialOrd for NeighborCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[inline]
fn is_better(a: NeighborCandidate, b: NeighborCandidate) -> bool {
    match a.dist.total_cmp(&b.dist) {
        Ordering::Less => true,
        Ordering::Equal => a.idx < b.idx,
        Ordering::Greater => false,
    }
}

#[inline]
fn push_top_k(heap: &mut BinaryHeap<NeighborCandidate>, cand: NeighborCandidate, k: usize) {
    if heap.len() < k {
        heap.push(cand);
        return;
    }
    if let Some(&worst) = heap.peek() {
        if is_better(cand, worst) {
            heap.pop();
            heap.push(cand);
        }
    }
}

#[inline]
fn heap_into_sorted_rows(heap: BinaryHeap<NeighborCandidate>) -> (Vec<usize>, Vec<f32>) {
    let mut row = heap.into_vec();
    row.sort_by(|a, b| a.dist.total_cmp(&b.dist).then_with(|| a.idx.cmp(&b.idx)));
    let mut idx = Vec::with_capacity(row.len());
    let mut dist = Vec::with_capacity(row.len());
    for item in row {
        idx.push(item.idx);
        dist.push(item.dist);
    }
    (idx, dist)
}

#[inline]
fn row_slices<'a>(
    indptr: &[usize],
    indices: &'a [usize],
    data: &'a [f32],
    row: usize,
) -> (&'a [usize], &'a [f32]) {
    let start = indptr[row];
    let end = indptr[row + 1];
    (&indices[start..end], &data[start..end])
}

#[inline]
fn sparse_dot(lhs_idx: &[usize], lhs_vals: &[f32], rhs_idx: &[usize], rhs_vals: &[f32]) -> f32 {
    let mut i = 0;
    let mut j = 0;
    let mut dot = 0.0_f32;

    while i < lhs_idx.len() && j < rhs_idx.len() {
        match lhs_idx[i].cmp(&rhs_idx[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                dot += lhs_vals[i] * rhs_vals[j];
                i += 1;
                j += 1;
            }
        }
    }

    dot
}

#[inline]
fn euclidean_distance_from_dot(norm_lhs: f32, norm_rhs: f32, dot: f32) -> f32 {
    (norm_lhs + norm_rhs - 2.0 * dot).max(0.0).sqrt()
}

#[inline]
fn sparse_row_euclidean_distance(matrix: &SparseCsrMatrix, lhs: usize, rhs: usize) -> f32 {
    let (lhs_idx, lhs_vals) = matrix.row(lhs);
    let (rhs_idx, rhs_vals) = matrix.row(rhs);
    let dot = sparse_dot(lhs_idx, lhs_vals, rhs_idx, rhs_vals);
    euclidean_distance_from_dot(matrix.squared_norm(lhs), matrix.squared_norm(rhs), dot)
}

pub(crate) fn exact_nearest_neighbors_euclidean(
    data: &SparseCsrMatrix,
    n_neighbors: usize,
) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
    let n_samples = data.n_rows();
    let mut heaps =
        vec![BinaryHeap::<NeighborCandidate>::with_capacity(n_neighbors + 1); n_samples];

    for (i, heap) in heaps.iter_mut().enumerate() {
        heap.push(NeighborCandidate { idx: i, dist: 0.0 });
    }

    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let dist = sparse_row_euclidean_distance(data, i, j);
            let cand_ij = NeighborCandidate { idx: j, dist };
            let cand_ji = NeighborCandidate { idx: i, dist };
            push_top_k(&mut heaps[i], cand_ij, n_neighbors);
            push_top_k(&mut heaps[j], cand_ji, n_neighbors);
        }
    }

    let mut indices = Vec::with_capacity(n_samples);
    let mut dists = Vec::with_capacity(n_samples);
    for heap in heaps {
        let (idx_row, dist_row) = heap_into_sorted_rows(heap);
        indices.push(idx_row);
        dists.push(dist_row);
    }
    (indices, dists)
}

pub(crate) fn exact_nearest_neighbors_dense_query_euclidean(
    query: &[Vec<f32>],
    reference: &SparseCsrMatrix,
    n_neighbors: usize,
) -> Result<(Vec<Vec<usize>>, Vec<Vec<f32>>), UmapError> {
    let n_features = reference.n_cols();
    for (row, q) in query.iter().enumerate() {
        if q.len() != n_features {
            if row == 0 {
                return Err(UmapError::FeatureMismatch {
                    expected: n_features,
                    got: q.len(),
                });
            }
            return Err(UmapError::InconsistentDimensions {
                row,
                expected: n_features,
                got: q.len(),
            });
        }
    }

    let query_sq_norms = query
        .iter()
        .map(|row| row.iter().map(|v| v * v).sum::<f32>())
        .collect::<Vec<f32>>();

    let mut indices = Vec::with_capacity(query.len());
    let mut dists = Vec::with_capacity(query.len());
    for (q_row_idx, q_row) in query.iter().enumerate() {
        let mut heap = BinaryHeap::<NeighborCandidate>::with_capacity(n_neighbors + 1);
        for ref_idx in 0..reference.n_rows() {
            let (ref_cols, ref_vals) = reference.row(ref_idx);
            let mut dot = 0.0_f32;
            for (&col, &val) in ref_cols.iter().zip(ref_vals.iter()) {
                dot += q_row[col] * val;
            }
            let dist = euclidean_distance_from_dot(
                query_sq_norms[q_row_idx],
                reference.squared_norm(ref_idx),
                dot,
            );
            push_top_k(
                &mut heap,
                NeighborCandidate { idx: ref_idx, dist },
                n_neighbors,
            );
        }

        let (idx_row, dist_row) = heap_into_sorted_rows(heap);
        indices.push(idx_row);
        dists.push(dist_row);
    }

    Ok((indices, dists))
}
