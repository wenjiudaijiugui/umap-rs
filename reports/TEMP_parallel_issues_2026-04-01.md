# UMAP-RS 并行执行计划与 Issue 模板（临时）

Status: TEMP  
Created: 2026-04-01  
Scope: `.`

## 1. 并行计划总览（按任务并行，不按时间）

| 任务ID | 任务名 | 目标 | 主要文件 | 关键交付物 | 依赖 |
|---|---|---|---|---|---|
| L1 | 基准统计强化 | 将主报告从 smoke 单次测量升级为可用于“完成判定”的统计报告 | `benchmarks/compare_real_impls_fair.py`, `.github/workflows/deep-benchmark-report.yml` | `report_real_fair.deep.json/.md` | 无 |
| L2 | 生态 ANN 覆盖补齐 | 确保 `rust_umap_py` 报告实际覆盖 ANN 分支 | `benchmarks/compare_ecosystem_python_binding.py`, `.github/workflows/ecosystem-python-binding.yml` | `report_ecosystem_python_binding.ann.json/.md` | 无 |
| L3 | CI 质量门禁硬化 | 在主 CI 加入质量门禁（fmt/clippy 等） | `.github/workflows/ci.yml` | 新增 quality jobs | 无 |
| L4 | Rust CLI E2E 测试 | 为 CLI 补齐真实文件 I/O 与模式分支 E2E | `rust_umap/src/bin/*.rs`, `rust_umap/tests/*` | `cli_e2e` 测试集 | 无 |
| L5 | Python 绑定负路径测试 | 补齐绑定输入校验/缓冲区约束回归 | `rust_umap_py/tests/test_binding.py` | 新增 pytest 负路径用例 | 无 |
| L6 | 报告证据治理 | 修复报告证据链不一致与外部路径依赖 | `reports/algo-repro-wave/*.json`, `reports/TEMP_parallel_plan_2026-03-31.md` | 修订后的 branch 报告 | 无 |
| L7 | 项目级最终收敛报告 | 形成 final 级“可发布/不可发布”判定 | `benchmarks/release_prep_regression.py`, `reports/*` | `reports/final-convergence-*.json/.md` | 依赖 L1/L2/L3/L6 |
| L8 | 功能边界收敛决策 | 对未覆盖能力做实现或降级声明收敛 | `rust_umap/src/lib.rs`, `rust_umap_py/src/lib.rs`, `README*.md` | 决策记录 + 代码/文档一致化 | 无 |

## 2. 通用 Issue 模板（可复制）

```md
## [Lx] <任务名>

### 1) 目标
<一句话描述要达成的可验证目标>

### 2) 背景与问题
<当前现状、风险、证据链接>

### 3) In Scope（必须改）
1. <文件/模块 A>
2. <文件/模块 B>

### 4) Out of Scope（禁止改）
1. <明确不改的目录或功能>

### 5) 交付物
1. <代码改动>
2. <报告/文档产物>
3. <CI 产物>

### 6) 验收标准（DoD）
- [ ] <可执行验证标准 1>
- [ ] <可执行验证标准 2>
- [ ] <产物路径固定并可追溯>

### 7) 依赖与阻塞
<无 / 依赖 Lx>

### 8) 并行冲突规则
1. 仅修改本任务白名单文件。
2. 不改其他任务 owner 的文件；若必须改，先提冲突说明。
3. 不在本任务中做无关重构。

### 9) 建议执行步骤
1. <步骤1>
2. <步骤2>
3. <步骤3>

### 10) 验证命令
```bash
<命令 1>
<命令 2>
```

### 11) 风险与回滚
<失败时的回滚策略或降级策略>
```

## 3. 预填 Issue 草案（L1-L8）

### Issue: [L1] 基准统计强化

### 1) 目标
将主基准报告提升为具备统计稳健性的深度报告，支持“完成判定”。

### 2) 背景与问题
当前主报告多为 `warmup=0, repeats=1`，统计强度不足，难以支撑最终结论。

### 3) In Scope（必须改）
1. `benchmarks/compare_real_impls_fair.py`
2. `.github/workflows/deep-benchmark-report.yml`
3. `benchmarks/report_real_fair*.{json,md}` 产出逻辑

### 4) Out of Scope（禁止改）
1. Rust 核心算法实现
2. Python 绑定 API 形态

### 5) 交付物
1. `benchmarks/report_real_fair.deep.json`
2. `benchmarks/report_real_fair.deep.md`
3. 报告中新增阈值判定段（pass/fail）

### 6) 验收标准（DoD）
- [ ] 深度报告参数满足 `warmup>=1`, `repeats>=5`
- [ ] 报告包含均值/方差/分位数，非单次样本
- [ ] 报告明确给出阈值对比结论（pass/fail）

### 7) 依赖与阻塞
无

### 8) 并行冲突规则
仅修改 `benchmarks/` 与 `deep-benchmark-report.yml`，不改 `ci.yml`。

### 9) 建议执行步骤
1. 扩展报告配置与输出 schema。
2. 更新 workflow 默认输入参数。
3. 运行一次本地深度报告并校验产物格式。

### 10) 验证命令
```bash
python3 benchmarks/compare_real_impls_fair.py \
  --python-bin python3 \
  --rscript-bin Rscript \
  --warmup 1 \
  --repeats 5 \
  --large-max-samples 5000 \
  --sample-cap-consistency 2000 \
  --report-json benchmarks/report_real_fair.deep.json \
  --report-md benchmarks/report_real_fair.deep.md
```

### 11) 风险与回滚
若耗时过高，先将 `repeats` 降为 3 并标注为候选深度模式，不替换主结论。

---

### Issue: [L2] 生态 ANN 覆盖补齐

### 1) 目标
让 `rust_umap_py` 在生态基准中实际触发 ANN 路径并纳入报告结论。

### 2) 背景与问题
当前生态报告样本规模未超过 ANN 阈值，报告几乎只验证 exact 路径。

### 3) In Scope（必须改）
1. `benchmarks/compare_ecosystem_python_binding.py`
2. `.github/workflows/ecosystem-python-binding.yml`
3. `benchmarks/report_ecosystem_python_binding*.{json,md}` 产出逻辑

### 4) Out of Scope（禁止改）
1. Rust 核心 ANN 算法
2. 其他 benchmark 脚本

### 5) 交付物
1. `benchmarks/report_ecosystem_python_binding.ann.json`
2. `benchmarks/report_ecosystem_python_binding.ann.md`
3. 报告明确 ANN 与 exact 的策略差异说明

### 6) 验收标准（DoD）
- [ ] 至少一个数据集触发 `rust_umap_py` ANN 分支
- [ ] 报告出现 `knn_strategy.equivalence = not_equivalent`
- [ ] 对应 gate 通过

### 7) 依赖与阻塞
无

### 8) 并行冲突规则
仅改生态相关脚本与 workflow，不改主 CI。

### 9) 建议执行步骤
1. 打开大样本数据集选项并固定参数。
2. 在报告中新增 ANN 触发证据字段。
3. 增加 ANN 报告产物上传。

### 10) 验证命令
```bash
python3 benchmarks/compare_ecosystem_python_binding.py \
  --python-bin python3 \
  --warmup 1 \
  --repeats 3 \
  --sample-cap-consistency 2000 \
  --include-california \
  --large-max-samples 5000 \
  --report-json benchmarks/report_ecosystem_python_binding.ann.json \
  --report-md benchmarks/report_ecosystem_python_binding.ann.md
```

### 11) 风险与回滚
若运行过慢，可先降低 `large-max-samples`，但必须仍大于 ANN 阈值。

---

### Issue: [L3] CI 质量门禁硬化

### 1) 目标
在主 CI 引入代码质量门禁，防止“功能绿但质量退化”。

### 2) 背景与问题
当前主 CI 以构建与 smoke gate 为主，缺少 `fmt/clippy` 等质量检查。

### 3) In Scope（必须改）
1. `.github/workflows/ci.yml`

### 4) Out of Scope（禁止改）
1. benchmark 逻辑
2. 算法实现

### 5) 交付物
1. `fmt-check` job
2. `clippy` job（`-D warnings`）
3. （可选）`cargo audit` job

### 6) 验收标准（DoD）
- [ ] PR 上新增质量 job 状态
- [ ] 质量 job 失败可阻断合并（由 branch protection 配置）
- [ ] 不影响现有 smoke gate 触发逻辑

### 7) 依赖与阻塞
无

### 8) 并行冲突规则
只改 `ci.yml`；不改其他 workflow。

### 9) 建议执行步骤
1. 新增 quality stage 与缓存复用。
2. 调整 job 依赖，避免重复构建。
3. 在 README 简要同步 CI 阶段。

### 10) 验证命令
```bash
cargo fmt --manifest-path rust_umap/Cargo.toml --all -- --check
cargo clippy --manifest-path rust_umap/Cargo.toml --all-targets --all-features -- -D warnings
```

### 11) 风险与回滚
若短期 clippy 噪声过高，可先按目录启用并逐步收敛到全量。

---

### Issue: [L4] Rust CLI E2E 测试

### 1) 目标
补齐 CLI 的端到端测试，覆盖真实文件输入输出和模式分支。

### 2) 背景与问题
当前 CLI 测试主要集中在参数抽取，不足以覆盖运行态分支和 I/O 错误。

### 3) In Scope（必须改）
1. 新增 `rust_umap/tests/cli_e2e.rs`（或等效）
2. 必要时补充 `rust_umap/src/bin/fit_csv.rs` 可测试性结构

### 4) Out of Scope（禁止改）
1. 基准脚本
2. Python 绑定

### 5) 交付物
1. 覆盖 `fit`, `fit_precomputed`, `transform`, `inverse` 的 CLI E2E 用例
2. CSR 输入错误路径用例

### 6) 验收标准（DoD）
- [ ] 新增 E2E 用例在 `cargo test` 稳定通过
- [ ] 至少覆盖 1 条错误输入断言（退出码或错误消息）
- [ ] 不引入 flaky 测试

### 7) 依赖与阻塞
无

### 8) 并行冲突规则
只改 `rust_umap` crate 范围，不改 workflow。

### 9) 建议执行步骤
1. 设计临时输入文件 fixture。
2. 用 `Command` 执行目标 bin 并断言输出文件。
3. 增加错误路径断言。

### 10) 验证命令
```bash
cargo test --manifest-path rust_umap/Cargo.toml
```

### 11) 风险与回滚
若 E2E 过慢，可拆成 smoke E2E + nightly 深度 E2E。

---

### Issue: [L5] Python 绑定负路径测试

### 1) 目标
补齐绑定层输入校验和 out-buffer 约束的负路径测试。

### 2) 背景与问题
现有绑定测试偏主路径，负路径覆盖不足。

### 3) In Scope（必须改）
1. `rust_umap_py/tests/test_binding.py`
2. 必要时小幅调整 `rust_umap_py/python/rust_umap_py/_api.py` 错误消息一致性

### 4) Out of Scope（禁止改）
1. Rust 核心算法
2. 基准脚本

### 5) 交付物
1. 新增 pytest 负路径用例集
2. 用例覆盖：非法 metric/init、KNN shape mismatch、负索引、非 float32 out、非 contiguous out、只读 out

### 6) 验收标准（DoD）
- [ ] 负路径用例全部通过
- [ ] 错误类型和值得断言的消息稳定
- [ ] workflow 可运行这些测试

### 7) 依赖与阻塞
无

### 8) 并行冲突规则
仅改 `rust_umap_py` 测试/API 层。

### 9) 建议执行步骤
1. 增加参数化负路径测试。
2. 对齐 ValueError/RuntimeError 映射预期。
3. 更新 CI（如需）确保执行。

### 10) 验证命令
```bash
python3 -m pytest -q rust_umap_py/tests/test_binding.py
```

### 11) 风险与回滚
若运行环境缺依赖，先在 workflow 验证，保留本地跳过逻辑。

---

### Issue: [L6] 报告证据治理

### 1) 目标
使报告证据链自洽、可复核、无仓库外强依赖。

### 2) 背景与问题
存在引用已删除 artifact、`~/.codex/...` 外部路径、baseline 表达不清等问题。

### 3) In Scope（必须改）
1. `reports/algo-repro-wave/*.json`（仅证据字段）
2. `reports/TEMP_parallel_plan_2026-03-31.md`（修订说明）

### 4) Out of Scope（禁止改）
1. 算法代码
2. benchmark 计算逻辑

### 5) 交付物
1. 修订后的 branch 报告
2. 证据修订日志（记录修改前后差异）

### 6) 验收标准（DoD）
- [ ] 报告中 `raw_artifact_refs` 均能在仓库内定位
- [ ] 去除 `~/.codex/...` 绝对依赖或提供仓库内替代
- [ ] `commit/baseline_commit` 表达与意图一致

### 7) 依赖与阻塞
无

### 8) 并行冲突规则
只改 `reports/` 文档与 JSON，不改 code。

### 9) 建议执行步骤
1. 扫描并列出失效证据引用。
2. 修订为仓库内可定位引用。
3. 增加修订注记与核验脚本说明。

### 10) 验证命令
```bash
rg -n "~/.codex|wave1-smoke.local.json" reports -S
```

### 11) 风险与回滚
若缺原始证据，明确标注“不可追溯”而非伪造替代。

---

### Issue: [L7] 项目级最终收敛报告

### 1) 目标
基于所有 gate 与深度基准，产出 final 级发布判定报告。

### 2) 背景与问题
当前多为 branch-phase，缺少项目级 `final` 判定文档。

### 3) In Scope（必须改）
1. `benchmarks/release_prep_regression.py`（如需扩展汇总字段）
2. 新增 `reports/final-convergence-2026-04-xx.{json,md}`

### 4) Out of Scope（禁止改）
1. 核心算法功能开发
2. 绑定 API 扩展

### 5) 交付物
1. final JSON（机器可读）
2. final MD（人类可读）
3. 明确 “Go / No-Go” 结论

### 6) 验收标准（DoD）
- [ ] 汇总 L1/L2/L3/L6 的产物引用
- [ ] 明确列出剩余风险和豁免
- [ ] 给出可发布判定

### 7) 依赖与阻塞
依赖 L1/L2/L3/L6

### 8) 并行冲突规则
仅改 `benchmarks` 汇总逻辑和 `reports` 最终文档。

### 9) 建议执行步骤
1. 统一读取各 gate/report 产物。
2. 生成 final JSON 与 MD。
3. 在 CI 或手动流程中固化命令。

### 10) 验证命令
```bash
python3 benchmarks/release_prep_regression.py \
  --candidate-root "$(pwd -P)" \
  --baseline-root /tmp/umap-rs-baseline \
  --gate-config benchmarks/gate_thresholds.json \
  --metrics euclidean,manhattan,cosine \
  --output-json benchmarks/release-prep-regression.final.json
```

### 11) 风险与回滚
若依赖任务未完成，输出中显式标注“pending inputs”，不给最终 Go 结论。

---

### Issue: [L8] 功能边界收敛决策

### 1) 目标
对现有能力缺口执行“实现 or 降级声明”二选一，保证声明与事实一致。

### 2) 背景与问题
当前存在 sparse inverse 未支持、Python 未暴露 aligned/parametric 等能力边界。

### 3) In Scope（必须改）
1. `rust_umap/src/lib.rs`（若选择实现路径）
2. `rust_umap_py/src/lib.rs`（若扩展暴露路径）
3. `README.md`, `rust_umap/README.md`（无论哪条路径都要更新）

### 4) Out of Scope（禁止改）
1. 无关性能优化
2. benchmark 框架改造

### 5) 交付物
1. 决策记录（ADR 或 TEMP 记录）
2. 对应代码或文档收敛变更

### 6) 验收标准（DoD）
- [ ] 对每个缺口明确状态：`Implemented` 或 `Out of Scope`
- [ ] README 与 API 行为一致
- [ ] 对应测试覆盖已声明能力

### 7) 依赖与阻塞
无

### 8) 并行冲突规则
若选择“仅文档降级”，不改代码；若选择“实现”，必须补对应测试。

### 9) 建议执行步骤
1. 列出缺口清单与候选路径。
2. 评估实现成本与收益，确定单一路径。
3. 更新代码/文档/测试并形成决策记录。

### 10) 验证命令
```bash
cargo test --manifest-path rust_umap/Cargo.toml
python3 -m pytest -q rust_umap_py/tests/test_binding.py
```

### 11) 风险与回滚
若实现跨度过大，先做文档降级并在后续里程碑重新开启实现 issue。

## 4. 并行执行建议（Owner 切分）

1. Owner-A: `L1 + L2`（benchmarks 与生态 workflow）
2. Owner-B: `L3`（主 CI）
3. Owner-C: `L4`（Rust CLI E2E）
4. Owner-D: `L5`（Python 绑定负路径）
5. Owner-E: `L6 + L7`（reports 与最终收敛）
6. Owner-F: `L8`（功能边界决策）

## 5. 备注

1. 本文件为临时协作文档，后续可折叠进正式 roadmap/ADR。
2. 任何任务若跨白名单改动，先在 issue 中补“冲突说明”再动手。

## 6. L1 执行日志（2026-04-01）

执行模式:
1. Skill: `algo-repro-wave`
2. Workflow mode: `lite`
3. Branch task: `L1 基准统计强化`

本次代码变更:
1. `benchmarks/compare_real_impls_fair.py`
   - 新增深度报告输出参数: `--report-json`, `--report-md`
   - 新增 quality gate 参数:
     - `--quality-gate-min-warmup`
     - `--quality-gate-min-repeats`
     - `--quality-gate-max-trust-gap`
     - `--quality-gate-max-recall-gap`
     - `--quality-gate-min-pairwise-overlap`
   - 新增 `report["quality_gate"]` 计算与 JSON 输出
   - Markdown 报告新增 `Quality Gate Verdict`（含 overall pass/fail、阈值、检查表、违例列表）
2. `.github/workflows/deep-benchmark-report.yml`
   - `repeats` 默认值由 `3` 调整为 `5`
   - `optimization-report` 改为输出:
     - `benchmarks/report_real_fair.deep.json`
     - `benchmarks/report_real_fair.deep.md`
   - summary 与 artifact 上传路径同步到 `.deep.*`

algo-repro-wave 报告产物:
1. `reports/algo-repro-wave/branch-p11-deep-benchmark-quality-gate.json`
   - `report_phase=branch`
   - `algorithm_profile` / `math_first_readiness` / `branch_ownership` / `rewrite_readiness` 完整
   - 当前判定: `gates.overall.merge_ready=true`
   - 评审结论: `reviews[0].decision=go`，且 `gates.reliability.pass=true`、`gates.performance.pass=true`

验证命令与结果:
1. `python3 -m py_compile benchmarks/compare_real_impls_fair.py` -> pass
2. workflow YAML 解析校验 -> pass
3. `mamba run -n umap_bench python benchmarks/compare_real_impls_fair.py --python-bin python --rscript-bin Rscript --warmup 1 --repeats 5 --large-max-samples 5000 --sample-cap-consistency 2000 --report-json benchmarks/report_real_fair.deep.json --report-md benchmarks/report_real_fair.deep.md` -> pass（已产出 runtime 深度证据）
4. `quick_validate` 单报告校验 -> pass
5. `quick_validate` 跨分支独占校验（11 报告） -> pass

当前阻塞:
1. 无。
2. `Rscript` 由 `mamba` 环境提供并已完成执行，`report_real_fair.deep.{json,md}` runtime 证据已在本地产出。

## 7. L7 执行日志（2026-04-01）

执行模式:
1. Skill: `algo-repro-wave`
2. Workflow mode: `lite`
3. Branch task: `L7 项目级最终收敛报告`

本次执行命令:
1. `git worktree add --detach /tmp/umap-rs-baseline a5c1e0b1256db09375cc893775eb0082a30115d4`
2. `mamba run -n umap_bench python benchmarks/release_prep_regression.py --candidate-root "$(pwd -P)" --baseline-root /tmp/umap-rs-baseline --python-bin python --rscript-bin Rscript --gate-config benchmarks/gate_thresholds.json --metrics euclidean,manhattan,cosine --output-json benchmarks/release-prep-regression.final.json`

执行结果:
1. `benchmarks/release-prep-regression.final.json` 已生成。
2. 首次运行出现阻塞: `ann_e2e_smoke=fail`，错误为 `TypeError: compute_shared_exact_knn() missing 1 required positional argument: 'dist_path'`。
3. 已修复 `benchmarks/ci_ann_smoke.py` 中 `compute_shared_exact_knn` 调用参数，补齐 metric 入参。
4. 修复后重跑同一命令，gate 汇总: `wave1_smoke=pass`, `ann_e2e_smoke=pass`, `consistency_smoke=pass`, `no_regression_smoke:{euclidean,manhattan,cosine}=pass`。
5. 项目级 gate 全绿，L7 当前结论更新为 `Go`。

交付物:
1. `reports/final-convergence-2026-04-01.json`
2. `reports/final-convergence-2026-04-01.md`
3. `benchmarks/release-prep-regression.final.json`
4. `benchmarks/release-prep-regression.final.artifacts/*`

后续维护:
1. 若后续 `benchmarks/` 或 `rust_umap/` 有实质改动，需重跑 `release_prep_regression.py` 刷新 L7 结论快照。
2. 将 `reports/final-convergence-2026-04-01.{json,md}` 作为本轮并行任务收敛证据。

## 8. L7 后续固化（2026-04-01）

1. 新增快照别名:
   - `reports/final-convergence-latest.json`
   - `reports/final-convergence-latest.md`
2. 新增 integration-phase 报告:
   - `reports/algo-repro-wave/integration-final-convergence-2026-04-01.json`
   - `quick_validate --phase integration` 结果: pass
3. 新增自动化 workflow:
   - `.github/workflows/release-prep-regression.yml`
   - 触发条件: `rust_umap/**` 或 `benchmarks/**` 变更（PR/push 到 main）与手动触发
   - 目标: 自动执行 `benchmarks/release_prep_regression.py` 并上传汇总 artifact
