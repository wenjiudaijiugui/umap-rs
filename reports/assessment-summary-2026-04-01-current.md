# umap-rs 当前工作区测评结论（2026-04-01）

## 评测口径

- 评测对象：当前工作区 `umap-rs` 重写版本
- 当前 `HEAD`：`d7352d29ec2625f48e51b2005860aa16d75f427b`
- 当前工作区为 dirty tree，本报告结论对应“当前本地状态”，不是干净提交
- 主参考实现：`umap-learn==0.5.7`
- 次参考实现：R `uwot`（使用仓库同日 `mamba` 环境工件复核）
- 线程口径：单线程约束（`OMP_NUM_THREADS=1`、`NUMBA_NUM_THREADS=1` 等）
- 随机种子：`42`

## 本次直接复核的证据

- 当前工作区重新执行 `wave1` 烟测：
  - `reports/assessment-wave1-smoke-2026-04-01-current.json`
  - 结果：`8/8` 检查全部通过
- 当前工作区重新执行完整 `release-prep` gate（含 R）：
  - `reports/release-prep-regression-2026-04-01-current.json`
  - `reports/release-prep-regression-2026-04-01-current.artifacts/`
  - 结果：`overall_pass = true`
- Python 参考环境已重新对齐到仓库定义版本：
  - `python=3.12.13`
  - `umap-learn=0.5.7`
  - `numpy=1.26.4`
  - `scipy=1.13.1`
  - `scikit-learn=1.5.2`
  - `numba=0.60.0`
- R 参考环境已重新对齐到仓库定义版本：
  - `R=4.5.3`
  - `uwot=0.2.3`
  - `jsonlite=1.8.8`

## 复用的同日全量工件

- 核心深度公平基准：
  - `benchmarks/report_real_fair.deep.json`
  - 结果：`quality_gate.overall_pass = true`
- 专项能力报告：
  - `reports/assessment-sparse-csr-2026-04-01.json`
  - `reports/assessment-inverse-2026-04-01.json`
  - `reports/assessment-binding-2026-04-01.json`

## 总体结论

如果评估对象是 `rust_umap` 核心重写本身，结论是正面的，而且证据强度足够：

- 当前树关键可靠性断言没有退化，`wave1` 重新复核通过
- 与 `umap-learn` 的质量对齐稳定，深度公平报告通过质量门
- 与 `uwot` 一起纳入时，当前工作区的 R-inclusive 一致性 gate 直接通过
- 对上一基线版本没有回归，当前工作区三种 metric 的 no-regression gate 全部通过，而且候选版本更快

真正需要保守看待的部分仍然是 `rust_umap_py` 绑定层，而不是 Rust 核心。

## 分项结论

### 1. 核心可靠性

当前工作区补跑的 `wave1` 烟测全部通过，覆盖：

- 非有限输入拒绝
- precomputed-kNN 距离契约
- ANN recall（euclidean / cosine）
- 稀疏 CSR 指标支持（manhattan / cosine）
- parametric 参数校验
- aligned warmstart

结论：当前工作区没有出现关键契约层面的明显回退。

### 2. 核心与 umap-learn 的公平对比

来自 `benchmarks/report_real_fair.deep.json` 的同日深度公平工件：

- 端到端默认路径下，`python_umap_learn / rust_umap` 时间比：
  - `breast_cancer`: `38.92x`
  - `digits`: `14.79x`
  - `california_housing`: `9.59x`
- 端到端默认路径下，`python_umap_learn / rust_umap` RSS 比：
  - `breast_cancer`: `78.04x`
  - `digits`: `52.19x`
  - `california_housing`: `28.19x`
- 共享 exact kNN 的纯算法口径下，`python_umap_learn / rust_umap` fit 时间比大致在：
  - `1.37x` 到 `3.87x`
- 共享 exact kNN 的纯算法口径下，`python_umap_learn / rust_umap` RSS 比大致在：
  - `17.86x` 到 `67.39x`
- 全局最差质量差距仍在门限内：
  - 最大 trust gap：`0.0080`
  - 最大 recall gap：`0.0195`
  - 最低 pairwise overlap@15：`0.4808`

结论：

- Rust 核心已经达到可与 `umap-learn` 正面对齐的状态
- 性能优势在端到端路径和纯算法路径上都存在
- 内存优势非常稳定，而且比时间优势更稳

### 3. R-inclusive 一致性复核

来自 `reports/release-prep-regression-2026-04-01-current.artifacts/consistency-smoke.json`：

- 参与实现：`python_umap_learn`、`r_uwot`、`rust_umap`
- `consistency_smoke` 总体通过

观测到的代表性结果：

- `breast_cancer`
  - trustworthiness：Rust `0.9164`，接近 Python `0.9179`，高于 R `0.9146`
  - recall@15：Rust `0.4101`，高于 Python `0.3979` 与 R `0.4056`
  - 最低 pairwise overlap：`0.5440`
- `digits`
  - trustworthiness：Rust `0.9651`，低于 Python `0.9706` 与 R `0.9679`，但仍在 gate 门限内
  - recall@15：Rust `0.4481`，高于 Python `0.4472`，略低于 R `0.4491`
  - 最低 pairwise overlap：`0.6245`

结论：在纳入 R 生态参考后，Rust 核心没有暴露出新的质量短板。

### 4. 基线回归检查

来自 `reports/release-prep-regression-2026-04-01-current.artifacts/no-regression-smoke-*.json`：

- `euclidean`
  - fit ratio median：`0.7987`
  - RSS ratio median：`0.9585`
- `manhattan`
  - fit ratio median：`0.7916`
  - RSS ratio median：`0.9594`
- `cosine`
  - fit ratio median：`0.8064`
  - RSS ratio median：`0.9946`

这里 ratio 是 `candidate / baseline`，全部明显低于 `1.0`，说明当前版本相对上一基线不仅没有回退，而且在三种 metric 下都有稳定速度收益，RSS 也没有恶化。

### 5. 稀疏 CSR 路径

来自 `reports/assessment-sparse-csr-2026-04-01.json`：

- `synthetic`
  - trust delta（rust - python）：`+0.001051`
  - `python / rust` fit 比：`1.086x`
  - `python / rust` RSS 比：`36.289x`
- `digits_sparse`
  - trust delta（rust - python）：`-0.000125`
  - `python / rust` fit 比：`1.212x`
  - `python / rust` RSS 比：`32.516x`

结论：稀疏 CSR fit 的质量与 `umap-learn` 基本一致，同时保留非常强的内存优势。

### 6. inverse_transform

来自 `reports/assessment-inverse-2026-04-01.json`：

- 四个数据集上都没有出现 Rust / Python 的成功失败不对称
- 四个数据集的查询集重建误差 `rust - python` 都是负值：
  - `iris`: `RMSE -0.3440`, `MAE -0.2045`
  - `breast_cancer`: `RMSE -0.2044`, `MAE -0.0923`
  - `digits`: `RMSE -0.0963`, `MAE -0.0519`
  - `california_housing`: `RMSE -0.0145`, `MAE -0.0145`

结论：当前实现至少没有在查询集逆映射质量上输给 `umap-learn`。

### 7. Python 绑定层

来自 `reports/assessment-binding-2026-04-01.json`：

- 端到端默认路径 `python_umap_learn / rust_umap_py` 时间比：
  - `breast_cancer`: `4.97x`
  - `digits`: `1.10x`
  - `california_housing`: `0.467x`
- 共享 exact kNN 纯算法口径 `python_umap_learn / rust_umap_py` fit 时间比：
  - `breast_cancer`: `0.230x`
  - `digits`: `0.242x`
  - `california_housing`: `0.235x`
- 绑定互操作审计已指出主要原因：
  - 核心仍使用 `Vec<Vec<f32>>` 行主序存储
  - Python/Rust 边界仍需至少一次复制

结论：

- 绑定层功能可用，但不是当前项目最强卖点
- 核心层的速度优势并没有稳定传导到 Python 绑定
- 下一阶段最值得优化的是绑定内存布局和跨边界复制，而不是继续卷核心算法

## 最终判断

对“`umap-rs` 核心重写是否成功”这个问题，我的判断是：

- `是`，而且已经达到可以与 `umap-learn` 和 `uwot` 做正面比较的水平
- 当前最强优势是：质量稳定、端到端速度快、RSS 明显低
- 当前最主要短板是：`rust_umap_py` 绑定层仍然会吃掉核心性能优势

如果后续要继续投入，优先级建议是：

1. 优先优化 Python 绑定的数据布局与零拷贝/少拷贝路径
2. 保持当前 `release-prep-regression` 与深度公平基准作为回归门禁
3. 稀疏与 inverse 可继续扩面，但优先级低于 binding 性能问题

## 备注

- 本次重新回钉 `uwot=0.2.3` / `jsonlite=1.8.8` 时，旧版依赖链触发了较重的源码编译，但最终安装成功。
- 因此，本报告中的 R-inclusive 一致性结论来自当前工作区直接重跑，而不是只依赖历史工件。
