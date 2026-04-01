use numpy::ndarray::{Array2, ArrayView1};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray2};
use numpy::PyUntypedArrayMethods;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rust_umap::{InitMethod, Metric, SparseCsrMatrix, UmapError, UmapModel, UmapParams};

fn parse_metric(metric: &str) -> PyResult<Metric> {
    match metric.to_ascii_lowercase().as_str() {
        "euclidean" => Ok(Metric::Euclidean),
        "manhattan" | "l1" => Ok(Metric::Manhattan),
        "cosine" => Ok(Metric::Cosine),
        _ => Err(PyValueError::new_err(format!(
            "unsupported metric '{metric}', expected euclidean|manhattan|cosine"
        ))),
    }
}

fn parse_init(init: &str) -> PyResult<InitMethod> {
    match init.to_ascii_lowercase().as_str() {
        "random" => Ok(InitMethod::Random),
        "spectral" => Ok(InitMethod::Spectral),
        _ => Err(PyValueError::new_err(format!(
            "unsupported init '{init}', expected random|spectral"
        ))),
    }
}

fn map_umap_error(err: UmapError) -> PyErr {
    match err {
        UmapError::NotFitted => PyRuntimeError::new_err(err.to_string()),
        UmapError::EmptyData
        | UmapError::NeedAtLeastTwoSamples
        | UmapError::InconsistentDimensions { .. }
        | UmapError::FeatureMismatch { .. }
        | UmapError::EmbeddingDimensionMismatch { .. }
        | UmapError::InvalidParameter(_) => PyValueError::new_err(err.to_string()),
    }
}

fn ensure_nonzero_columns(name: &str, n_cols: usize) -> PyResult<()> {
    if n_cols == 0 {
        return Err(PyValueError::new_err(format!(
            "{name} must have at least one column"
        )));
    }
    Ok(())
}

fn array2_f32_to_rows(data: &PyReadonlyArray2<'_, f32>, name: &str) -> PyResult<Vec<Vec<f32>>> {
    let view = data.as_array();
    let (n_rows, n_cols) = view.dim();
    ensure_nonzero_columns(name, n_cols)?;

    if let Ok(slice) = data.as_slice() {
        let mut rows = Vec::with_capacity(n_rows);
        for row in slice.chunks_exact(n_cols) {
            rows.push(row.to_vec());
        }
        return Ok(rows);
    }

    let mut rows = Vec::with_capacity(n_rows);
    for row in view.outer_iter() {
        rows.push(row.to_vec());
    }
    Ok(rows)
}

fn array2_i64_to_usize_rows(
    data: &PyReadonlyArray2<'_, i64>,
    name: &str,
) -> PyResult<Vec<Vec<usize>>> {
    let view = data.as_array();
    let (n_rows, n_cols) = view.dim();
    ensure_nonzero_columns(name, n_cols)?;

    if let Ok(slice) = data.as_slice() {
        let mut rows = Vec::with_capacity(n_rows);
        for row in slice.chunks_exact(n_cols) {
            let mut out_row = Vec::with_capacity(n_cols);
            for &idx in row {
                if idx < 0 {
                    return Err(PyValueError::new_err(
                        "knn indices must be non-negative integers",
                    ));
                }
                out_row.push(idx as usize);
            }
            rows.push(out_row);
        }
        return Ok(rows);
    }

    let mut rows = Vec::with_capacity(n_rows);
    for row in view.outer_iter() {
        let mut out_row = Vec::with_capacity(n_cols);
        for &idx in row {
            if idx < 0 {
                return Err(PyValueError::new_err(
                    "knn indices must be non-negative integers",
                ));
            }
            out_row.push(idx as usize);
        }
        rows.push(out_row);
    }
    Ok(rows)
}

fn array1_i64_to_usize_vec(data: &PyReadonlyArray1<'_, i64>, name: &str) -> PyResult<Vec<usize>> {
    let slice = data
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be contiguous")))?;
    let mut out = Vec::with_capacity(slice.len());
    for &value in slice {
        if value < 0 {
            return Err(PyValueError::new_err(format!(
                "{name} must contain non-negative integers"
            )));
        }
        out.push(value as usize);
    }
    Ok(out)
}

fn array1_f32_to_vec(data: &PyReadonlyArray1<'_, f32>, name: &str) -> PyResult<Vec<f32>> {
    let slice = data
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be contiguous")))?;
    Ok(slice.to_vec())
}

fn sparse_csr_from_py(
    indptr: &PyReadonlyArray1<'_, i64>,
    indices: &PyReadonlyArray1<'_, i64>,
    data: &PyReadonlyArray1<'_, f32>,
    n_cols: usize,
) -> PyResult<SparseCsrMatrix> {
    let indptr = array1_i64_to_usize_vec(indptr, "indptr")?;
    let indices = array1_i64_to_usize_vec(indices, "indices")?;
    let values = array1_f32_to_vec(data, "data")?;
    if indptr.is_empty() {
        return Err(PyValueError::new_err("indptr cannot be empty"));
    }
    let n_rows = indptr.len() - 1;
    SparseCsrMatrix::new(n_rows, n_cols, indptr, indices, values).map_err(map_umap_error)
}

fn rows_to_numpy<'py>(
    py: Python<'py>,
    rows: Vec<Vec<f32>>,
    empty_cols: usize,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let n_rows = rows.len();
    let n_cols = if n_rows == 0 {
        empty_cols
    } else {
        rows[0].len()
    };

    let mut flat = Vec::with_capacity(n_rows.saturating_mul(n_cols));
    for row in rows {
        if row.len() != n_cols {
            return Err(PyRuntimeError::new_err(
                "inconsistent output row width from rust_umap core",
            ));
        }
        flat.extend_from_slice(&row);
    }

    let arr = Array2::from_shape_vec((n_rows, n_cols), flat)
        .map_err(|err| PyRuntimeError::new_err(format!("failed to build output array: {err}")))?;
    Ok(PyArray2::from_owned_array_bound(py, arr))
}

fn copy_rows_into_out(
    rows: &[Vec<f32>],
    empty_cols: usize,
    mut out: PyReadwriteArray2<'_, f32>,
) -> PyResult<()> {
    let n_rows = rows.len();
    let n_cols = if n_rows == 0 {
        empty_cols
    } else {
        rows[0].len()
    };

    let mut out_view = out.as_array_mut();
    if out_view.nrows() != n_rows || out_view.ncols() != n_cols {
        return Err(PyValueError::new_err(format!(
            "output buffer shape mismatch: expected ({n_rows}, {n_cols}), got ({}, {})",
            out_view.nrows(),
            out_view.ncols()
        )));
    }

    if let Some(out_slice) = out_view.as_slice_memory_order_mut() {
        for (i, row) in rows.iter().enumerate() {
            if row.len() != n_cols {
                return Err(PyRuntimeError::new_err(
                    "inconsistent output row width from rust_umap core",
                ));
            }
            let start = i * n_cols;
            let end = start + n_cols;
            out_slice[start..end].copy_from_slice(row);
        }
        return Ok(());
    }

    for (i, row) in rows.iter().enumerate() {
        if row.len() != n_cols {
            return Err(PyRuntimeError::new_err(
                "inconsistent output row width from rust_umap core",
            ));
        }
        out_view.row_mut(i).assign(&ArrayView1::from(row.as_slice()));
    }

    Ok(())
}

#[pyclass(name = "UmapCore", module = "rust_umap_py._rust_umap_py")]
struct PyUmapCore {
    inner: UmapModel,
}

#[pymethods]
impl PyUmapCore {
    #[new]
    #[pyo3(signature = (
        n_neighbors = 15,
        n_components = 2,
        n_epochs = None,
        metric = "euclidean",
        learning_rate = 1.0,
        min_dist = 0.1,
        spread = 1.0,
        local_connectivity = 1.0,
        set_op_mix_ratio = 1.0,
        repulsion_strength = 1.0,
        negative_sample_rate = 5,
        random_seed = 42,
        init = "spectral",
        use_approximate_knn = true,
        approx_knn_candidates = 30,
        approx_knn_iters = 10,
        approx_knn_threshold = 4096,
    ))]
    fn new(
        n_neighbors: usize,
        n_components: usize,
        n_epochs: Option<usize>,
        metric: &str,
        learning_rate: f32,
        min_dist: f32,
        spread: f32,
        local_connectivity: f32,
        set_op_mix_ratio: f32,
        repulsion_strength: f32,
        negative_sample_rate: usize,
        random_seed: u64,
        init: &str,
        use_approximate_knn: bool,
        approx_knn_candidates: usize,
        approx_knn_iters: usize,
        approx_knn_threshold: usize,
    ) -> PyResult<Self> {
        let metric = parse_metric(metric)?;
        let init = parse_init(init)?;

        let params = UmapParams {
            n_neighbors,
            n_components,
            n_epochs,
            metric,
            learning_rate,
            min_dist,
            spread,
            local_connectivity,
            set_op_mix_ratio,
            repulsion_strength,
            negative_sample_rate,
            random_seed,
            init,
            use_approximate_knn,
            approx_knn_candidates,
            approx_knn_iters,
            approx_knn_threshold,
        };

        Ok(Self {
            inner: UmapModel::new(params),
        })
    }

    fn fit(&mut self, py: Python<'_>, data: PyReadonlyArray2<'_, f32>) -> PyResult<()> {
        let rows = array2_f32_to_rows(&data, "data")?;
        py.allow_threads(|| self.inner.fit(&rows)).map_err(map_umap_error)
    }

    fn fit_sparse_csr(
        &mut self,
        py: Python<'_>,
        indptr: PyReadonlyArray1<'_, i64>,
        indices: PyReadonlyArray1<'_, i64>,
        data: PyReadonlyArray1<'_, f32>,
        n_cols: usize,
    ) -> PyResult<()> {
        let csr = sparse_csr_from_py(&indptr, &indices, &data, n_cols)?;
        py.allow_threads(|| self.inner.fit_sparse_csr(csr))
            .map_err(map_umap_error)
    }

    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let rows = array2_f32_to_rows(&data, "data")?;
        let embedding = py
            .allow_threads(|| self.inner.fit_transform(&rows))
            .map_err(map_umap_error)?;
        rows_to_numpy(py, embedding, self.inner.params().n_components)
    }

    fn fit_transform_into<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f32>,
        out: PyReadwriteArray2<'py, f32>,
    ) -> PyResult<()> {
        let rows = array2_f32_to_rows(&data, "data")?;
        let embedding = py
            .allow_threads(|| self.inner.fit_transform(&rows))
            .map_err(map_umap_error)?;
        copy_rows_into_out(&embedding, self.inner.params().n_components, out)
    }

    fn fit_transform_sparse_csr<'py>(
        &mut self,
        py: Python<'py>,
        indptr: PyReadonlyArray1<'py, i64>,
        indices: PyReadonlyArray1<'py, i64>,
        data: PyReadonlyArray1<'py, f32>,
        n_cols: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let csr = sparse_csr_from_py(&indptr, &indices, &data, n_cols)?;
        let embedding = py
            .allow_threads(|| self.inner.fit_transform_sparse_csr(csr))
            .map_err(map_umap_error)?;
        rows_to_numpy(py, embedding, self.inner.params().n_components)
    }

    fn fit_transform_sparse_csr_into<'py>(
        &mut self,
        py: Python<'py>,
        indptr: PyReadonlyArray1<'py, i64>,
        indices: PyReadonlyArray1<'py, i64>,
        data: PyReadonlyArray1<'py, f32>,
        n_cols: usize,
        out: PyReadwriteArray2<'py, f32>,
    ) -> PyResult<()> {
        let csr = sparse_csr_from_py(&indptr, &indices, &data, n_cols)?;
        let embedding = py
            .allow_threads(|| self.inner.fit_transform_sparse_csr(csr))
            .map_err(map_umap_error)?;
        copy_rows_into_out(&embedding, self.inner.params().n_components, out)
    }

    #[pyo3(signature = (data, knn_indices, knn_dists, knn_metric = "euclidean"))]
    fn fit_transform_with_knn<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f32>,
        knn_indices: PyReadonlyArray2<'py, i64>,
        knn_dists: PyReadonlyArray2<'py, f32>,
        knn_metric: &str,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        if knn_indices.shape() != knn_dists.shape() {
            return Err(PyValueError::new_err(
                "knn_indices and knn_dists must have identical shapes",
            ));
        }
        if knn_indices.shape()[1] < self.inner.params().n_neighbors {
            return Err(PyValueError::new_err(format!(
                "knn columns must be >= n_neighbors ({})",
                self.inner.params().n_neighbors
            )));
        }

        let rows = array2_f32_to_rows(&data, "data")?;
        if knn_indices.shape()[0] != rows.len() {
            return Err(PyValueError::new_err(
                "knn row count must match data row count",
            ));
        }

        let knn_idx_rows = array2_i64_to_usize_rows(&knn_indices, "knn_indices")?;
        let knn_dist_rows = array2_f32_to_rows(&knn_dists, "knn_dists")?;
        let knn_metric = parse_metric(knn_metric)?;

        let embedding = py
            .allow_threads(|| {
                self.inner.fit_transform_with_knn_metric(
                    &rows,
                    &knn_idx_rows,
                    &knn_dist_rows,
                    knn_metric,
                )
            })
            .map_err(map_umap_error)?;

        rows_to_numpy(py, embedding, self.inner.params().n_components)
    }

    #[pyo3(signature = (data, knn_indices, knn_dists, out, knn_metric = "euclidean"))]
    fn fit_transform_with_knn_into<'py>(
        &mut self,
        py: Python<'py>,
        data: PyReadonlyArray2<'py, f32>,
        knn_indices: PyReadonlyArray2<'py, i64>,
        knn_dists: PyReadonlyArray2<'py, f32>,
        out: PyReadwriteArray2<'py, f32>,
        knn_metric: &str,
    ) -> PyResult<()> {
        if knn_indices.shape() != knn_dists.shape() {
            return Err(PyValueError::new_err(
                "knn_indices and knn_dists must have identical shapes",
            ));
        }
        if knn_indices.shape()[1] < self.inner.params().n_neighbors {
            return Err(PyValueError::new_err(format!(
                "knn columns must be >= n_neighbors ({})",
                self.inner.params().n_neighbors
            )));
        }

        let rows = array2_f32_to_rows(&data, "data")?;
        if knn_indices.shape()[0] != rows.len() {
            return Err(PyValueError::new_err(
                "knn row count must match data row count",
            ));
        }

        let knn_idx_rows = array2_i64_to_usize_rows(&knn_indices, "knn_indices")?;
        let knn_dist_rows = array2_f32_to_rows(&knn_dists, "knn_dists")?;
        let knn_metric = parse_metric(knn_metric)?;

        let embedding = py
            .allow_threads(|| {
                self.inner.fit_transform_with_knn_metric(
                    &rows,
                    &knn_idx_rows,
                    &knn_dist_rows,
                    knn_metric,
                )
            })
            .map_err(map_umap_error)?;

        copy_rows_into_out(&embedding, self.inner.params().n_components, out)
    }

    fn transform<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let rows = array2_f32_to_rows(&query, "query")?;
        let out = py
            .allow_threads(|| self.inner.transform(&rows))
            .map_err(map_umap_error)?;
        rows_to_numpy(py, out, self.inner.params().n_components)
    }

    fn transform_into<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray2<'py, f32>,
        out: PyReadwriteArray2<'py, f32>,
    ) -> PyResult<()> {
        let rows = array2_f32_to_rows(&query, "query")?;
        let transformed = py
            .allow_threads(|| self.inner.transform(&rows))
            .map_err(map_umap_error)?;
        copy_rows_into_out(&transformed, self.inner.params().n_components, out)
    }

    fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        embedded_query: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let rows = array2_f32_to_rows(&embedded_query, "embedded_query")?;
        let out = py
            .allow_threads(|| self.inner.inverse_transform(&rows))
            .map_err(map_umap_error)?;
        rows_to_numpy(py, out, 0)
    }

    fn inverse_transform_into<'py>(
        &self,
        py: Python<'py>,
        embedded_query: PyReadonlyArray2<'py, f32>,
        out: PyReadwriteArray2<'py, f32>,
    ) -> PyResult<()> {
        let rows = array2_f32_to_rows(&embedded_query, "embedded_query")?;
        let reconstructed = py
            .allow_threads(|| self.inner.inverse_transform(&rows))
            .map_err(map_umap_error)?;
        copy_rows_into_out(&reconstructed, 0, out)
    }

    #[getter]
    fn n_components(&self) -> usize {
        self.inner.params().n_components
    }

    #[getter]
    fn n_neighbors(&self) -> usize {
        self.inner.params().n_neighbors
    }
}

#[pymodule]
fn _rust_umap_py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyUmapCore>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
