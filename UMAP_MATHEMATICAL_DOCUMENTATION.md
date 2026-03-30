# UMAP Mathematical Implementation Documentation

## Overview

UMAP (Uniform Manifold Approximation and Projection) is a dimension reduction technique that operates on the following mathematical framework:

1. **High-dimensional representation**: Construct a fuzzy simplicial set from high-dimensional data
2. **Low-dimensional representation**: Construct a fuzzy simplicial set in low-dimensional space
3. **Optimization**: Minimize the cross-entropy between the two fuzzy sets

This document details every mathematical operation in the standard implementation.

---

## Stage 1: Nearest Neighbor Search

**Function**: `nearest_neighbors()` in `umap/umap_.py:255`

### Input
- `X`: Data matrix of shape `(n_samples, n_features)`
- `n_neighbors`: Number of nearest neighbors k
- `metric`: Distance metric d(x,y)

### Output
- `knn_indices`: Shape `(n_samples, n_neighbors)`, indices of k-nearest neighbors
- `knn_dists`: Shape `(n_samples, n_neighbors)`, distances to k-nearest neighbors

### Algorithm

For small datasets (n < 4096), exact computation:
```
dmat = pairwise_distances(X, metric)  # O(n²)
knn_indices[i] = argsort(dmat[i])[:k]
knn_dists[i] = dmat[i][knn_indices[i]]
```

For large datasets, approximate via NN-Descent:
```
n_trees = min(64, 5 + int(round(sqrt(n_samples) / 20)))
n_iters = max(5, int(round(log2(n_samples))))

Use pynndescent.NNDescent with:
- n_trees: Random projection trees for initialization
- n_iters: Iterations for neighbor refinement
- max_candidates=60: Candidate neighbors per iteration
```

---

## Stage 2: Fuzzy Simplicial Set Construction

### 2.1 Smooth KNN Distance

**Function**: `smooth_knn_dist()` in `umap/umap_.py:142`

**Purpose**: Convert discrete k-NN distances to continuous values σ and ρ for each point.

**Mathematical Model**

For each point i, we find σ_i such that the sum of membership strengths equals log₂(k):

```
Σⱼ exp(-max(0, d(i,j) - ρ_i) / σ_i) = log₂(k)
```

Where:
- `d(i,j)`: Distance from point i to neighbor j
- `ρ_i`: Distance to local_connectivity-th neighbor (local connectivity parameter)
- `σ_i`: Bandwidth parameter

**Algorithm** (Binary Search):

```python
target = log₂(k) * bandwidth  # bandwidth=1.0
for each point i:
    # 1. Compute ρ_i (local connectivity adjustment)
    if non_zero_dists.shape >= local_connectivity:
        idx = floor(local_connectivity)
        interp = local_connectivity - idx
        ρ_i = non_zero_dists[idx-1] + interp * (non_zero_dists[idx] - non_zero_dists[idx-1])
    
    # 2. Binary search for σ_i
    lo = 0.0, hi = ∞, mid = 1.0
    for n in range(n_iter=64):
        psum = 0.0
        for j in range(1, n_neighbors):
            d = distances[i,j] - ρ_i
            if d > 0:
                psum += exp(-d / mid)
            else:
                psum += 1.0
        
        if |psum - target| < tolerance:
            break
        
        if psum > target:
            hi = mid
        else:
            lo = mid
        mid = (lo + hi) / 2.0
    
    σ_i = mid
    
    # 3. Apply minimum scale constraint
    if ρ_i > 0:
        result[i] = max(MIN_K_DIST_SCALE * mean_distances[i], σ_i)
    else:
        result[i] = max(MIN_K_DIST_SCALE * global_mean, σ_i)
```

**Key Constants**:
- `SMOOTH_K_TOLERANCE = 1e-5`
- `MIN_K_DIST_SCALE = 1e-2`

**Return**: `sigmas` (σ), `rhos` (ρ)

---

### 2.2 Compute Membership Strengths

**Function**: `compute_membership_strengths()` in `umap/umap_.py:350`

**Purpose**: Construct the 1-skeleton of the local fuzzy simplicial set as a sparse matrix.

**Mathematical Definition**

For each point i and each neighbor j:
```
μ_ij = exp(-max(0, d(i,j) - ρ_i) / σ_i)
```

Special cases:
- If `j == i` (self): `μ_ij = 0.0` (no self-loops)
- If `d(i,j) - ρ_i <= 0` or `σ_i == 0`: `μ_ij = 1.0` (maximum strength)

**Algorithm**:

```python
for i in range(n_samples):
    for j in range(n_neighbors):
        if knn_indices[i,j] == -1:
            continue
        if knn_indices[i,j] == i:
            val = 0.0
        elif knn_dists[i,j] - ρ_i <= 0 or σ_i == 0:
            val = 1.0
        else:
            val = exp(-(knn_dists[i,j] - ρ_i) / σ_i)
        
        rows[i*n_neighbors + j] = i
        cols[i*n_neighbors + j] = knn_indices[i,j]
        vals[i*n_neighbors + j] = val
```

**Return**: `(rows, cols, vals, dists)` for COO sparse matrix format

---

### 2.3 Fuzzy Simplicial Set Union

**Function**: `fuzzy_simplicial_set()` in `umap/umap_.py:441`

**Purpose**: Combine local fuzzy simplicial sets into a global fuzzy set using fuzzy set operations.

**Mathematical Operations**

Let `G` be the directed graph where `G_ij = μ_ij` (membership from i to j).

**Step 1**: Compute symmetric membership using product t-norm:

```
P = G · G^T  (element-wise product)
P_ij = μ_ij · μ_ji
```

**Step 2**: Fuzzy union and intersection:

```
Union(G, G^T) = G + G^T - P
Intersection(G, G^T) = P
```

**Step 3**: Final combination (set_op_mix_ratio parameter):

```
result = α · Union(G, G^T) + (1-α) · Intersection(G, G^T)
       = α(G + G^T - P) + (1-α)P
```

Where `α = set_op_mix_ratio` (default: 1.0 for pure union)

**Code**:

```python
rows, cols, vals, dists = compute_membership_strengths(
    knn_indices, knn_dists, sigmas, rhos, return_dists
)
result = coo_matrix((vals, (rows, cols)), shape=(n, n))

if apply_set_operations:
    transpose = result.T
    prod_matrix = result.multiply(transpose)  # P
    result = set_op_mix_ratio * (result + transpose - prod_matrix) + \
             (1.0 - set_op_mix_ratio) * prod_matrix

result.eliminate_zeros()
```

**Return**: Sparse matrix representing the fuzzy simplicial set `graph`, `sigmas`, `rhos`, `graph_dists`

---

## Stage 3: Spectral Initialization

**Function**: `spectral_layout()` → `_spectral_layout()` in `umap/spectral.py:262`

**Purpose**: Initialize low-dimensional embedding using spectral embedding of the graph.

**Mathematical Foundation**

Compute the normalized graph Laplacian:

```
L = I - D^(-1/2) · W · D^(-1/2)
```

Where:
- `W`: Weighted adjacency matrix (our fuzzy simplicial set)
- `D`: Degree matrix, `D_ii = Σⱼ W_ij`
- `I`: Identity matrix

**Eigenvalue Problem**:

Find eigenvectors corresponding to the smallest non-zero eigenvalues:

```
L · v = λ · v
```

The first eigenvector (λ=0) is constant, so we use eigenvectors 2 to (dim+1).

**Algorithm**:

```python
# 1. Check for connected components
n_components, labels = connected_components(graph)
if n_components > 1:
    return multi_component_layout(...)  # Handle separately

# 2. Compute normalized Laplacian
sqrt_deg = sqrt(graph.sum(axis=0))
D = spdiags(1/sqrt_deg, 0, n, n)
I = identity(n)
L = I - D @ graph @ D

# 3. Compute eigenvectors
k = dim + 1
if method == "eigsh":
    eigenvalues, eigenvectors = eigsh(L, k, which="SM")
elif method == "lobpcg":
    X = random initialization or TruncatedSVD(L)
    X[:, 0] = sqrt_deg / norm(sqrt_deg)  # Exact first eigenvector
    eigenvalues, eigenvectors = lobpcg(L, X, largest=False)

# 4. Return eigenvectors 2:(dim+1)
order = argsort(eigenvalues)[1:k]
return eigenvectors[:, order]
```

**Multi-component handling**: If the graph has multiple connected components, use metric-based layout to position components relative to each other.

---

## Stage 4: Layout Optimization

### 4.1 Parameter Discovery: a and b

**Function**: `find_ab_params()` in `umap/umap_.py:1392`

**Purpose**: Fit parameters for the differentiable curve in low-dimensional space.

**Mathematical Model**

In high-dimensional space, membership strength follows:
```
μ_hd = exp(-d / σ)
```

In low-dimensional space, we use a differentiable approximation:
```
μ_ld = 1 / (1 + a · d^(2b))
```

We find `a` and `b` to match a piecewise function:
```
f(d) = { 1.0                    if d < min_dist
       { exp(-(d - min_dist) / spread)  if d ≥ min_dist
```

**Curve Fitting**:

```python
def curve(x, a, b):
    return 1.0 / (1.0 + a * x**(2*b))

xv = linspace(0, spread * 3, 300)
yv = zeros_like(xv)
yv[xv < min_dist] = 1.0
yv[xv >= min_dist] = exp(-(xv[xv >= min_dist] - min_dist) / spread)

params, covar = curve_fit(curve, xv, yv)
return params[0], params[1]  # a, b
```

**Default values**: `spread=1.0`, `min_dist=0.1`

---

### 4.2 Epochs Per Sample

**Function**: `make_epochs_per_sample()` in `umap/umap_.py:905`

**Purpose**: Determine how often to sample each edge during optimization.

**Mathematical Model**

Edges with higher weights should be sampled more frequently:

```
epochs_per_sample[i] = n_epochs / (n_epochs * weights[i] / max(weights))
                    = max(weights) / weights[i]
```

**Algorithm**:

```python
result = -1.0 * ones(n_edges)
n_samples = n_epochs * (weights / weights.max())
result[n_samples > 0] = n_epochs / n_samples[n_samples > 0]
return result
```

**Interpretation**: If an edge has weight `w_max`, it's sampled every epoch. An edge with weight `0.5 * w_max` is sampled every 2 epochs.

---

### 4.3 Optimization Objective

**Function**: `simplicial_set_embedding()` → `optimize_layout_euclidean()` in `umap/layouts.py`

**Mathematical Objective**: Minimize cross-entropy between high and low-dimensional fuzzy simplicial sets.

```
C = Σᵢⱼ [P_ij log(P_ij / Q_ij) + (1-P_ij) log((1-P_ij) / (1-Q_ij))]
```

Where:
- `P_ij`: Membership strength in high-dimensional space (from fuzzy simplicial set)
- `Q_ij`: Membership strength in low-dimensional space

**Low-dimensional membership**:

For Euclidean output metric:
```
Q_ij = 1 / (1 + a · ||y_i - y_j||^(2b))
```

**Gradient**:

For positive samples (edges in the graph):
```
∂C/∂y_i = -2ab · ||y_i - y_j||^(2b-1) / (1 + a·||y_i - y_j||^(2b)) · (y_i - y_j)
```

For negative samples (sampled randomly):
```
∂C/∂y_i = 2γb / (0.001 + ||y_i - y_j||²) / (1 + a·||y_i - y_j||^(2b)) · (y_i - y_j)
```

Where `γ` is the repulsion strength (default: 1.0).

---

### 4.4 Optimization Algorithm

**Function**: `_optimize_layout_euclidean_single_epoch()` in `umap/layouts.py:62`

**Algorithm** (Stochastic Gradient Descent):

```python
# Preprocessing
embedding = spectral_embedding  # or random/PCA initialization
embedding = normalize_to_range(embedding, 0, 10)

# Edge pruning
graph.data[graph.data < max(graph.data) / n_epochs] = 0.0
graph.eliminate_zeros()

epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)
head = graph.row  # Source vertices
tail = graph.col  # Target vertices

# Main optimization loop
for epoch in range(n_epochs):
    alpha = initial_alpha * (1.0 - epoch / n_epochs)  # Learning rate decay
    
    for each edge (i, j) in graph:
        if edge_should_be_sampled(edge, epoch):
            # POSITIVE SAMPLE
            current = embedding[i]
            other = embedding[j]
            dist_squared = ||current - other||²
            
            # Gradient for positive sample
            if dist_squared > 0:
                grad_coeff = -2ab · dist_squared^(b-1) / (1 + a·dist_squared^b)
            else:
                grad_coeff = 0.0
            
            for d in range(dim):
                grad_d = clip(grad_coeff * (current[d] - other[d]))
                current[d] += grad_d * alpha
                other[d] += -grad_d * alpha  # Symmetric update
            
            # NEGATIVE SAMPLES
            n_neg_samples = int((epoch - epoch_of_next_negative_sample[i]) / 
                               epochs_per_negative_sample[i])
            
            for p in range(n_neg_samples):
                k = random_int() % n_vertices  # Random negative sample
                other = embedding[k]
                dist_squared = ||current - other||²
                
                if dist_squared > 0:
                    grad_coeff = 2γb / ((0.001 + dist_squared) * 
                                        (1 + a·dist_squared^b))
                else:
                    grad_coeff = 0.0
                
                for d in range(dim):
                    if grad_coeff > 0:
                        grad_d = clip(grad_coeff * (current[d] - other[d]))
                        current[d] += grad_d * alpha
            
            epoch_of_next_sample[i] += epochs_per_sample[i]
            epoch_of_next_negative_sample[i] += n_neg_samples * epochs_per_negative_sample[i]
```

**Key Implementation Details**:

1. **Gradient Clipping**: `clip()` function prevents numerical instability
2. **Symmetric Updates**: Both endpoints of positive edges are updated
3. **Negative Sampling**: For each positive edge, sample `negative_sample_rate` negative examples
4. **Learning Rate Schedule**: `alpha = initial_alpha * (1 - epoch/n_epochs)`

**Constants**:
- `negative_sample_rate = 5` (default)
- `initial_alpha = 1.0` (default)
- `gamma = 1.0` (default, repulsion strength)

---

## Stage 5: Transform (New Data)

**Function**: `transform()` in `umap/umap_.py:2949`

**Purpose**: Embed new data points into existing embedding.

**Algorithm**:

1. **Find nearest neighbors in original data**:
   - Use pre-built search index from training
   - Get indices and distances to k-nearest training points

2. **Compute membership strengths**:
   ```python
   sigmas, rhos = smooth_knn_dist(dists, k, local_connectivity-1)
   rows, cols, vals, dists = compute_membership_strengths(
       indices, dists, sigmas, rhos, bipartite=True
   )
   graph = coo_matrix((vals, (rows, cols)), shape=(n_new, n_train))
   ```

3. **Initialize new embedding**:
   ```python
   embedding = init_graph_transform(graph, existing_embedding)
   # Weighted average of neighbors' embeddings
   ```

4. **Optimize**:
   - Keep training embeddings fixed (`move_other=False`)
   - Run optimization for `n_epochs // 3` epochs
   - Use same gradient formulas but only update new points

---

## Complete Algorithm Flow

```
Input: X (n_samples, n_features), n_neighbors, min_dist, spread

1. NEAREST NEIGHBORS
   knn_indices, knn_dists = nearest_neighbors(X, n_neighbors, metric)

2. FUZZY SIMPLICIAL SET (HIGH-DIM)
   sigmas, rhos = smooth_knn_dist(knn_dists, n_neighbors)
   graph = fuzzy_simplicial_set(knn_indices, knn_dists, sigmas, rhos)

3. SPECTRAL INITIALIZATION
   embedding = spectral_layout(graph, n_components)

4. PARAMETER FITTING
   a, b = find_ab_params(spread, min_dist)

5. OPTIMIZATION
   epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)
   embedding = optimize_layout_euclidean(
       embedding, graph, epochs_per_sample, a, b, n_epochs
   )

Output: embedding (n_samples, n_components)
```

---

## Mathematical Summary

### High-Dimensional Fuzzy Simplicial Set

For each point i with neighbors j ∈ N(i):
```
ρ_i = d(i, π_{local_connectivity}(i))  # Distance to local_connectivity-th neighbor
σ_i = binary_search_solution such that Σⱼ exp(-max(0, d(i,j)-ρ_i)/σ_i) = log₂(k)
μ_ij = exp(-max(0, d(i,j) - ρ_i) / σ_i)
```

Global fuzzy set (symmetrized):
```
P_ij = max(μ_ij, μ_ji) - μ_ij · μ_ji  # Fuzzy union with product t-norm
```

### Low-Dimensional Fuzzy Simplicial Set

```
Q_ij = 1 / (1 + a · ||y_i - y_j||^(2b))
```

### Cross-Entropy Loss

```
L = Σᵢⱼ [P_ij log(P_ij/Q_ij) + (1-P_ij) log((1-P_ij)/(1-Q_ij))]
```

### Gradients (Euclidean)

Positive edge (i,j):
```
∇ᵢL = -2ab · d²^(b-1) / (1 + a·d²^b) · (y_i - y_j)
where d² = ||y_i - y_j||²
```

Negative sample (i,k):
```
∇ᵢL = 2γb / (0.001 + d²) / (1 + a·d²^b) · (y_i - y_k)
```

---

## Key Equations Reference

| Symbol | Meaning |
|--------|---------|
| X | Input data (n × d) |
| Y | Output embedding (n × n_components) |
| k | n_neighbors |
| d(x,y) | Distance metric in input space |
| ||y_i - y_j|| | Euclidean distance in output space |
| ρ_i | Local connectivity offset for point i |
| σ_i | Bandwidth for point i |
| μ_ij | Local membership strength (high-dim) |
| P_ij | Global membership strength (high-dim, symmetrized) |
| Q_ij | Membership strength (low-dim) |
| a, b | Parameters for low-dim membership function |
| γ | Repulsion strength (gamma) |
| α | Learning rate |
| min_dist | Minimum distance in output space |
| spread | Scale of output distribution |

---

## File Structure Reference

- `umap/umap_.py`: Main UMAP class and core functions
  - `smooth_knn_dist()`: σ, ρ computation
  - `compute_membership_strengths()`: μ_ij
  - `fuzzy_simplicial_set()`: Fuzzy set construction
  - `simplicial_set_embedding()`: Main embedding function
  - `UMAP.fit()`: Main entry point
  - `UMAP.transform()`: New data transformation
  
- `umap/spectral.py`: Spectral initialization
  - `spectral_layout()`: Eigendecomposition of graph Laplacian
  
- `umap/layouts.py`: Optimization routines
  - `optimize_layout_euclidean()`: SGD optimization loop
  - `_optimize_layout_euclidean_single_epoch()`: Single epoch computation
  
- `umap/distances.py`: Distance metrics and gradients
- `umap/utils.py`: Utility functions (random sampling, submatrix operations)
- `umap/sparse.py`: Sparse matrix distance computations

---

## Implementation Notes

1. **Numerical Stability**: 
   - Gradient clipping prevents explosion
   - Small epsilon (0.001) prevents division by zero
   - Minimum scale constraint prevents σ from being too small

2. **Performance Optimizations**:
   - Numba JIT compilation for tight loops
   - Sparse matrix operations for efficiency
   - Approximate nearest neighbors for large datasets
   - Edge pruning based on weight thresholds

3. **Memory Management**:
   - Low memory mode for NN-descent
   - Sparse matrices throughout
   - In-place updates where possible

4. **Reproducibility**:
   - Fixed random seeds for initialization
   - Parallel execution disabled when seed is set
   - Deterministic operations (no race conditions)

---

This documentation covers every mathematical operation in the UMAP implementation with exact formulas, algorithms, and code references. Use this as the complete specification for reimplementation.
