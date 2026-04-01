# umap-rs 重写测评总览

## 评测范围

- 评测对象：当前工作区 `umap-rs` 重写版本（含 `rust_umap` 核心与 `rust_umap_py` 绑定）
- 参考实现：`umap-learn==0.5.7`
- 运行环境：`Python 3.12.3`，单线程约束（`OMP_NUM_THREADS=1` 等）
- 随机设置：`seed=42`
- 当前工作区存在未提交改动，因此结论对应“当前本地状态”，不是某个干净提交

## 总体结论

`rust_umap` 核心实现已经达到“可以和 `umap-learn` 正面对齐”的成熟度：在当前测评口径下，核心层公平基准全部通过质量门，且在端到端默认路径和共享 exact kNN 两种口径下都明显快于 `umap-learn`，内存占用也显著更低。质量层面，trustworthiness、原始空间 kNN recall、pairwise overlap 的差距都维持在门限内，没有看到明显的正确性退化。

真正的短板不在核心算法，而在 Python 绑定层。`rust_umap_py` 的功能正确性和接口稳定性是过关的，但性能并不稳定优于 `umap-learn`：小数据集还可以，大一些的数据集尤其是 `california_housing` 上，绑定路径明显慢于纯 Python 参考实现。仓库自己的 interop audit 也给出了直接原因：当前核心仍以 `Vec<Vec<f32>>` 存储，跨边界至少有一次行主序复制，这会吞掉相当一部分 Rust 核心本应有的速度优势。

## 分项结论

### 1. 核心可靠性

- `wave1` 烟测全部通过，8 个关键检查点全部为 `pass`
- 覆盖了输入契约、precomputed-kNN 合法性、ANN recall、稀疏度量、parametric 参数校验、aligned warmstart

对应报告：
- `reports/assessment-wave1-smoke-2026-04-01.json`

### 2. 核心与 umap-learn 的公平对比

核心深度公平报告结论是本次测评最重要的主结论：

- 默认端到端路径下，`python/rust` 时间比为：
  - `breast_cancer`: `38.34x`
  - `digits`: `12.91x`
  - `california_housing`: `10.30x`
- 默认端到端路径下，`python/rust` RSS 比为：
  - `breast_cancer`: `56.80x`
  - `digits`: `40.16x`
  - `california_housing`: `19.56x`
- 共享 exact kNN 的纯算法口径下，`python/rust` fit 时间比大致在 `1.22x` 到 `1.50x`
- 共享 exact kNN 的纯算法口径下，`python/rust` RSS 比大致在 `14.23x` 到 `51.86x`
- 全部质量门通过，最大观测：
  - trust gap `0.007271`
  - recall gap `0.013133`
  - 最低 overlap@15 `0.488467`

这说明：

- 核心算法质量与 `umap-learn` 基本对齐
- Rust 的主要优势非常稳定地体现在进程级内存占用
- 真正夸张的加速主要出现在 CLI/核心直跑的端到端路径，而不是严格算法内核口径

对应报告：
- `reports/assessment-core-fair-2026-04-01.md`
- `reports/assessment-core-fair-2026-04-01.json`
- `reports/assessment-consistency-smoke-2026-04-01.json`
- `reports/assessment-ann-e2e-smoke-2026-04-01.json`

### 3. 稀疏 CSR 路径

稀疏路径的结论也比较强：

- `synthetic` 上 `python/rust` 算法时间比 `1.086x`，RSS 比 `36.289x`
- `digits_sparse` 上 `python/rust` 算法时间比 `1.212x`，RSS 比 `32.516x`
- trustworthiness 差值都极小：
  - `synthetic`: `+0.001051`（rust 略高）
  - `digits_sparse`: `-0.000125`（rust 略低）

这说明稀疏 CSR MVP 虽然文档上仍被标为 MVP，但当前 fit 质量和资源占用已经相当有竞争力。需要注意的是，这份专项只覆盖了稀疏 fit 路径，不代表稀疏训练后的全部下游能力已经完全等价。

对应报告：
- `reports/assessment-sparse-csr-2026-04-01.json`

### 4. inverse_transform

`inverse_transform` 的结果有两个重要点：

- 四个数据集上都没有出现 Rust 与 `umap-learn` 的成功/失败不对称
- 查询集重建误差上，Rust 在四个数据集上都优于 `umap-learn`

查询集 `rust - python` 的 RMSE / MAE 差值：

- `iris`: `-0.343973 / -0.204521`
- `breast_cancer`: `-0.204365 / -0.092326`
- `digits`: `-0.096309 / -0.051944`
- `california_housing`: `-0.014543 / -0.014483`

不过这里要谨慎解释一个设计差异：

- Rust 的训练集逆映射对“精确命中的训练嵌入”会直接映回原样本，因此训练集 `train_rmse=0`
- 这不等价于说 Rust 在一般意义上的逆映射建模一定全面优于 `umap-learn`
- 更可信的结论是：当前实现至少没有在查询集重建质量上输给 `umap-learn`

对应报告：
- `reports/assessment-inverse-2026-04-01.json`

### 5. Python 绑定层

绑定层的结论与核心层明显不同：

- 绑定测试 `14/14` 通过，功能面基本稳定
- 端到端默认路径：
  - `breast_cancer`：`rust_umap_py` 更快，约为 Python 的 `4.97x`
  - `digits`：两者接近，Python/Rust 时间比 `1.10x`
  - `california_housing`：`rust_umap_py` 明显更慢，Python/Rust 时间比仅 `0.467x`
- exact shared kNN 口径下，`rust_umap_py` 在全部三个数据集上都慢于 `umap-learn`
  - `breast_cancer`: Python/Rust fit 比 `0.230x`
  - `digits`: `0.242x`
  - `california_housing`: `0.235x`

这说明：

- 绑定层没有继承核心层的大部分速度优势
- 大数据集上 binding 反而会把 Rust 内核优势吃掉，甚至变成劣势
- 目前更适合把 `rust_umap_py` 看成“功能可用、内存友好、但性能仍需优化”的接入层，而不是 `umap-learn` 的无条件快替代

绑定层仍有稳定优势的是进程 RSS，Python/Rust 的 RSS 比仍有 `3.8x` 到 `6.9x` 左右。

对应报告：
- `reports/assessment-binding-2026-04-01.md`
- `reports/assessment-binding-2026-04-01.json`

## 风险与边界

- 这次评测严格锚定 `umap-learn`，没有纳入 `uwot`，因为当前机器没有 `Rscript`
- 绑定报告里 `california_housing` 端到端默认路径不是严格同路对比：`umap-learn` 仍走 exact，`rust_umap_py` 已切到 ANN
- 文档已明确声明的边界仍然成立：
  - 不宣称 full `umap-learn` parity
  - 不宣称 full `pynndescent` parity
  - sparse-trained `inverse_transform` 仍不支持

## 最终判断

如果评估对象是 `rust_umap` 核心重写本身，结论是正面的，而且强度不低：

- 质量门通过
- 与 `umap-learn` 的差距小
- 性能和内存优势显著
- 稀疏与 inverse 等专项能力没有暴露出明显的结构性问题

如果评估对象扩大到“整个项目可否在 Python 生态里直接替代 `umap-learn`”，结论就要保守：

- 功能正确性可以接受
- 但 Python binding 还不是成熟的性能卖点
- 下一阶段最值得做的不是继续卷核心算法，而是优化绑定数据布局和跨边界复制路径
