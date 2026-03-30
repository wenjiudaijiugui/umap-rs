# UMAP 算法数学实现详解

## 概述

UMAP (Uniform Manifold Approximation and Projection, 一致流形逼近与投影) 是一种降维技术,其数学框架包含以下步骤:

1. **高维表示**: 从高维数据构建模糊单纯形集
2. **低维表示**: 在低维空间构建模糊单纯形集
3. **优化**: 最小化两个模糊集之间的交叉熵

本文档详细说明标准实现中的每个数学操作。

---

## 阶段 1: 最近邻搜索

**函数**: `nearest_neighbors()` 位于 `umap/umap_.py:255`

### 输入
- `X`: 形状为 `(n_samples, n_features)` 的数据矩阵
- `n_neighbors`: 最近邻数量 k
- `metric`: 距离度量 d(x,y)

### 输出
- `knn_indices`: 形状 `(n_samples, n_neighbors)`, k个最近邻的索引
- `knn_dists`: 形状 `(n_samples, n_neighbors)`, 到k个最近邻的距离

### 算法

对于小数据集 (n < 4096), 精确计算:
```
dmat = pairwise_distances(X, metric)  # O(n²)
knn_indices[i] = argsort(dmat[i])[:k]
knn_dists[i] = dmat[i][knn_indices[i]]
```

对于大数据集, 通过 NN-Descent 近似:
```
n_trees = min(64, 5 + int(round(sqrt(n_samples) / 20)))
n_iters = max(5, int(round(log2(n_samples))))

使用 pynndescent.NNDescent, 参数:
- n_trees: 用于初始化的随机投影树
- n_iters: 邻居精炼迭代次数
- max_candidates=60: 每次迭代的候选邻居数
```

---

## 阶段 2: 模糊单纯形集构建

### 2.1 平滑 KNN 距离

**函数**: `smooth_knn_dist()` 位于 `umap/umap_.py:142`

**目的**: 将离散的 k-NN 距离转换为每个点的连续值 σ 和 ρ。

**数学模型**

对于每个点 i, 我们找到 σ_i 使得隶属度强度之和等于 log₂(k):

```
Σⱼ exp(-max(0, d(i,j) - ρ_i) / σ_i) = log₂(k)
```

其中:
- `d(i,j)`: 点 i 到邻居 j 的距离
- `ρ_i`: 到第 local_connectivity 个邻居的距离 (局部连通性参数)
- `σ_i`: 带宽参数

**算法** (二分搜索):

```python
target = log₂(k) * bandwidth  # bandwidth=1.0
for each point i:
    # 1. 计算 ρ_i (局部连通性调整)
    if non_zero_dists.shape >= local_connectivity:
        idx = floor(local_connectivity)
        interp = local_connectivity - idx
        ρ_i = non_zero_dists[idx-1] + interp * (non_zero_dists[idx] - non_zero_dists[idx-1])

    # 2. 二分搜索 σ_i
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

    # 3. 应用最小尺度约束
    if ρ_i > 0:
        result[i] = max(MIN_K_DIST_SCALE * mean_distances[i], σ_i)
    else:
        result[i] = max(MIN_K_DIST_SCALE * global_mean, σ_i)
```

**关键常量**:
- `SMOOTH_K_TOLERANCE = 1e-5`
- `MIN_K_DIST_SCALE = 1e-2`

**返回**: `sigmas` (σ), `rhos` (ρ)

---

### 2.2 计算隶属度强度

**函数**: `compute_membership_strengths()` 位于 `umap/umap_.py:350`

**目的**: 将局部模糊单纯形集的 1-骨架构建为稀疏矩阵。

**数学定义**

对于每个点 i 和每个邻居 j:
```
μ_ij = exp(-max(0, d(i,j) - ρ_i) / σ_i)
```

特殊情况:
- 如果 `j == i` (自身): `μ_ij = 0.0` (无自环)
- 如果 `d(i,j) - ρ_i <= 0` 或 `σ_i == 0`: `μ_ij = 1.0` (最大强度)

**算法**:

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

**返回**: COO 稀疏矩阵格式的 `(rows, cols, vals, dists)`

---

### 2.3 模糊单纯形集并集

**函数**: `fuzzy_simplicial_set()` 位于 `umap/umap_.py:441`

**目的**: 使用模糊集运算将局部模糊单纯形集组合成全局模糊集。

**数学运算**

设 `G` 为有向图, 其中 `G_ij = μ_ij` (从 i 到 j 的隶属度)。

**步骤 1**: 使用积 t-范数计算对称隶属度:

```
P = G · G^T  (逐元素乘积)
P_ij = μ_ij · μ_ji
```

**步骤 2**: 模糊并集和交集:

```
Union(G, G^T) = G + G^T - P
Intersection(G, G^T) = P
```

**步骤 3**: 最终组合 (set_op_mix_ratio 参数):

```
result = α · Union(G, G^T) + (1-α) · Intersection(G, G^T)
       = α(G + G^T - P) + (1-α)P
```

其中 `α = set_op_mix_ratio` (默认值: 1.0 表示纯并集)

**代码**:

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

**返回**: 表示模糊单纯形集的稀疏矩阵 `graph`, `sigmas`, `rhos`, `graph_dists`

---

## 阶段 3: 谱初始化

**函数**: `spectral_layout()` → `_spectral_layout()` 位于 `umap/spectral.py:262`

**目的**: 使用图的谱嵌入初始化低维嵌入。

**数学基础**

计算归一化图拉普拉斯算子:

```
L = I - D^(-1/2) · W · D^(-1/2)
```

其中:
- `W`: 加权邻接矩阵 (我们的模糊单纯形集)
- `D`: 度矩阵, `D_ii = Σⱼ W_ij`
- `I`: 单位矩阵

**特征值问题**:

找到对应于最小非零特征值的特征向量:

```
L · v = λ · v
```

第一个特征向量 (λ=0) 是常数, 因此我们使用第 2 到 (dim+1) 个特征向量。

**算法**:

```python
# 1. 检查连通分量
n_components, labels = connected_components(graph)
if n_components > 1:
    return multi_component_layout(...)  # 单独处理

# 2. 计算归一化拉普拉斯算子
sqrt_deg = sqrt(graph.sum(axis=0))
D = spdiags(1/sqrt_deg, 0, n, n)
I = identity(n)
L = I - D @ graph @ D

# 3. 计算特征向量
k = dim + 1
if method == "eigsh":
    eigenvalues, eigenvectors = eigsh(L, k, which="SM")
elif method == "lobpcg":
    X = 随机初始化或 TruncatedSVD(L)
    X[:, 0] = sqrt_deg / norm(sqrt_deg)  # 精确的第一个特征向量
    eigenvalues, eigenvectors = lobpcg(L, X, largest=False)

# 4. 返回特征向量 2:(dim+1)
order = argsort(eigenvalues)[1:k]
return eigenvectors[:, order]
```

**多分量处理**: 如果图有多个连通分量, 使用基于度量的布局来相对定位各分量。

---

## 阶段 4: 布局优化

### 4.1 参数发现: a 和 b

**函数**: `find_ab_params()` 位于 `umap/umap_.py:1392`

**目的**: 为低维空间中的可微曲线拟合参数。

**数学模型**

在高维空间中, 隶属度强度遵循:
```
μ_hd = exp(-d / σ)
```

在低维空间中, 我们使用可微近似:
```
μ_ld = 1 / (1 + a · d^(2b))
```

我们找到 `a` 和 `b` 来匹配分段函数:
```
f(d) = { 1.0                              if d < min_dist
       { exp(-(d - min_dist) / spread)   if d ≥ min_dist
```

**曲线拟合**:

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

**默认值**: `spread=1.0`, `min_dist=0.1`

---

### 4.2 每个样本的训练轮数

**函数**: `make_epochs_per_sample()` 位于 `umap/umap_.py:905`

**目的**: 确定在优化期间每个边的采样频率。

**数学模型**

权重更高的边应该更频繁地被采样:

```
epochs_per_sample[i] = n_epochs / (n_epochs * weights[i] / max(weights))
                    = max(weights) / weights[i]
```

**算法**:

```python
result = -1.0 * ones(n_edges)
n_samples = n_epochs * (weights / weights.max())
result[n_samples > 0] = n_epochs / n_samples[n_samples > 0]
return result
```

**解释**: 如果一条边的权重为 `w_max`, 它在每个 epoch 都被采样。权重为 `0.5 * w_max` 的边每 2 个 epoch 被采样一次。

---

### 4.3 优化目标

**函数**: `simplicial_set_embedding()` → `optimize_layout_euclidean()` 位于 `umap/layouts.py`

**数学目标**: 最小化高维和低维模糊单纯形集之间的交叉熵。

```
C = Σᵢⱼ [P_ij log(P_ij / Q_ij) + (1-P_ij) log((1-P_ij) / (1-Q_ij))]
```

其中:
- `P_ij`: 高维空间中的隶属度强度 (来自模糊单纯形集)
- `Q_ij`: 低维空间中的隶属度强度

**低维隶属度**:

对于欧几里得输出度量:
```
Q_ij = 1 / (1 + a · ||y_i - y_j||^(2b))
```

**梯度**:

对于正样本 (图中的边):
```
∂C/∂y_i = -2ab · ||y_i - y_j||^(2b-1) / (1 + a·||y_i - y_j||^(2b)) · (y_i - y_j)
```

对于负样本 (随机采样):
```
∂C/∂y_i = 2γb / (0.001 + ||y_i - y_j||²) / (1 + a·||y_i - y_j||^(2b)) · (y_i - y_j)
```

其中 `γ` 是排斥强度 (默认值: 1.0)。

---

### 4.4 优化算法

**函数**: `_optimize_layout_euclidean_single_epoch()` 位于 `umap/layouts.py:62`

**算法** (随机梯度下降):

```python
# 预处理
embedding = spectral_embedding  # 或 random/PCA 初始化
embedding = normalize_to_range(embedding, 0, 10)

# 边剪枝
graph.data[graph.data < max(graph.data) / n_epochs] = 0.0
graph.eliminate_zeros()

epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)
head = graph.row  # 源顶点
tail = graph.col  # 目标顶点

# 主优化循环
for epoch in range(n_epochs):
    alpha = initial_alpha * (1.0 - epoch / n_epochs)  # 学习率衰减

    for each edge (i, j) in graph:
        if edge_should_be_sampled(edge, epoch):
            # 正样本
            current = embedding[i]
            other = embedding[j]
            dist_squared = ||current - other||²

            # 正样本的梯度
            if dist_squared > 0:
                grad_coeff = -2ab · dist_squared^(b-1) / (1 + a·dist_squared^b)
            else:
                grad_coeff = 0.0

            for d in range(dim):
                grad_d = clip(grad_coeff * (current[d] - other[d]))
                current[d] += grad_d * alpha
                other[d] += -grad_d * alpha  # 对称更新

            # 负样本
            n_neg_samples = int((epoch - epoch_of_next_negative_sample[i]) /
                               epochs_per_negative_sample[i])

            for p in range(n_neg_samples):
                k = random_int() % n_vertices  # 随机负样本
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

**关键实现细节**:

1. **梯度裁剪**: `clip()` 函数防止数值不稳定
2. **对称更新**: 正边的两个端点都被更新
3. **负采样**: 对于每个正边, 采样 `negative_sample_rate` 个负样本
4. **学习率调度**: `alpha = initial_alpha * (1 - epoch/n_epochs)`

**常量**:
- `negative_sample_rate = 5` (默认)
- `initial_alpha = 1.0` (默认)
- `gamma = 1.0` (默认, 排斥强度)

---

## 阶段 5: 变换 (新数据)

**函数**: `transform()` 位于 `umap/umap_.py:2949`

**目的**: 将新数据点嵌入到现有嵌入中。

**算法**:

1. **在原始数据中查找最近邻**:
   - 使用训练期间构建的搜索索引
   - 获取到 k 个最近训练点的索引和距离

2. **计算隶属度强度**:
   ```python
   sigmas, rhos = smooth_knn_dist(dists, k, local_connectivity-1)
   rows, cols, vals, dists = compute_membership_strengths(
       indices, dists, sigmas, rhos, bipartite=True
   )
   graph = coo_matrix((vals, (rows, cols)), shape=(n_new, n_train))
   ```

3. **初始化新嵌入**:
   ```python
   embedding = init_graph_transform(graph, existing_embedding)
   # 邻居嵌入的加权平均
   ```

4. **优化**:
   - 保持训练嵌入固定 (`move_other=False`)
   - 运行优化 `n_epochs // 3` 个 epoch
   - 使用相同的梯度公式, 但只更新新点

---

## 完整算法流程

```
输入: X (n_samples, n_features), n_neighbors, min_dist, spread

1. 最近邻搜索
   knn_indices, knn_dists = nearest_neighbors(X, n_neighbors, metric)

2. 模糊单纯形集 (高维)
   sigmas, rhos = smooth_knn_dist(knn_dists, n_neighbors)
   graph = fuzzy_simplicial_set(knn_indices, knn_dists, sigmas, rhos)

3. 谱初始化
   embedding = spectral_layout(graph, n_components)

4. 参数拟合
   a, b = find_ab_params(spread, min_dist)

5. 优化
   epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)
   embedding = optimize_layout_euclidean(
       embedding, graph, epochs_per_sample, a, b, n_epochs
   )

输出: embedding (n_samples, n_components)
```

---

## 数学总结

### 高维模糊单纯形集

对于每个点 i 及其邻居 j ∈ N(i):
```
ρ_i = d(i, π_{local_connectivity}(i))  # 到第 local_connectivity 个邻居的距离
σ_i = 二分搜索解, 使得 Σⱼ exp(-max(0, d(i,j)-ρ_i)/σ_i) = log₂(k)
μ_ij = exp(-max(0, d(i,j) - ρ_i) / σ_i)
```

全局模糊集 (对称化):
```
P_ij = max(μ_ij, μ_ji) - μ_ij · μ_ji  # 使用积 t-范数的模糊并集
```

### 低维模糊单纯形集

```
Q_ij = 1 / (1 + a · ||y_i - y_j||^(2b))
```

### 交叉熵损失

```
L = Σᵢⱼ [P_ij log(P_ij/Q_ij) + (1-P_ij) log((1-P_ij)/(1-Q_ij))]
```

### 梯度 (欧几里得)

正边 (i,j):
```
∇ᵢL = -2ab · d²^(b-1) / (1 + a·d²^b) · (y_i - y_j)
其中 d² = ||y_i - y_j||²
```

负样本 (i,k):
```
∇ᵢL = 2γb / (0.001 + d²) / (1 + a·d²^b) · (y_i - y_k)
```

---

## 关键公式参考

| 符号 | 含义 |
|------|------|
| X | 输入数据 (n × d) |
| Y | 输出嵌入 (n × n_components) |
| k | n_neighbors (邻居数) |
| d(x,y) | 输入空间中的距离度量 |
| ||y_i - y_j|| | 输出空间中的欧几里得距离 |
| ρ_i | 点 i 的局部连通性偏移 |
| σ_i | 点 i 的带宽 |
| μ_ij | 局部隶属度强度 (高维) |
| P_ij | 全局隶属度强度 (高维, 对称化) |
| Q_ij | 隶属度强度 (低维) |
| a, b | 低维隶属度函数的参数 |
| γ | 排斥强度 (gamma) |
| α | 学习率 |
| min_dist | 输出空间中的最小距离 |
| spread | 输出分布的尺度 |

---

## 文件结构参考

- `umap/umap_.py`: UMAP 主类和核心函数
  - `smooth_knn_dist()`: σ, ρ 计算
  - `compute_membership_strengths()`: μ_ij
  - `fuzzy_simplicial_set()`: 模糊集构建
  - `simplicial_set_embedding()`: 主嵌入函数
  - `UMAP.fit()`: 主入口
  - `UMAP.transform()`: 新数据变换

- `umap/spectral.py`: 谱初始化
  - `spectral_layout()`: 图拉普拉斯算子的特征分解

- `umap/layouts.py`: 优化例程
  - `optimize_layout_euclidean()`: SGD 优化循环
  - `_optimize_layout_euclidean_single_epoch()`: 单个 epoch 计算

- `umap/distances.py`: 距离度量和梯度
- `umap/utils.py`: 工具函数 (随机采样, 子矩阵操作)
- `umap/sparse.py`: 稀疏矩阵距离计算

---

## 实现注意事项

1. **数值稳定性**:
   - 梯度裁剪防止梯度爆炸
   - 小 epsilon (0.001) 防止除零
   - 最小尺度约束防止 σ 过小

2. **性能优化**:
   - 对紧循环使用 Numba JIT 编译
   - 稀疏矩阵操作提高效率
   - 对大数据集使用近似最近邻
   - 基于权重阈值的边剪枝

3. **内存管理**:
   - NN-descent 的低内存模式
   - 全程使用稀疏矩阵
   - 尽可能就地更新

4. **可重复性**:
   - 初始化的固定随机种子
   - 设置种子时禁用并行执行
   - 确定性操作 (无竞态条件)

---

本文档涵盖了 UMAP 实现中的每个数学操作, 包含精确的公式、算法和代码引用。可作为重新实现的完整规范使用。
