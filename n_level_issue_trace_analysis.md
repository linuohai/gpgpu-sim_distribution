# N-level 调度器：从 issue trace 反推逐 SM 配置（实现方案）

本文档说明如何从一次运行产生的 issue trace CSV 反推 N-level warp_allocate.csv，
并给出脚本实现、算法选择、参数含义、以及验证方式。目标是生成可被 GPGPU-Sim
直接读取的逐 SM 配置 CSV。

---

## 1. 背景与目标

N-level 调度器将每个 SM 的 warp 槽位划分为 N 组，并为每组配置时间片。
本方案通过分析 issue trace 中每个 warp 的最后一次发射周期（last_issue_cycle），
对这些结束时间进行聚类，从而推断组划分与时间片配置，输出与
`doc/n_level_warp_scheduler.md` 示例一致的 CSV。

**目标**：
- 生成每个 SM 一行的 `warp_allocate.csv`；
- CSV 格式与当前解析器完全兼容；
- 允许输出诊断日志，便于检查聚类和时间片的合理性。

---

## 2. 输入数据定义

**输入文件**：一次运行对应一个 issue trace CSV，例如：
`accel-sim-framework/result/issue_trace/fa_issue.csv`。

**必须存在的列名**：
- `cycle`：时钟周期
- `sm`：SM id
- `warp`：warp 槽 id（静态槽）
- `event`：事件类型（只使用 `ISSUE`）

**使用规则**：
1) 只统计 `event == ISSUE` 的行；
2) warp 的“执行结束时间”定义为该 warp 的最后一次 ISSUE 的 `cycle`；
3) `STALL` 行不计入结束时间，但可用于诊断；
4) 未出现 ISSUE 的 warp 视为异常 warp（outlier）。

---

## 3. 输出 CSV 格式（必须匹配示例）

单行格式如下：

```
SM_ID,(组大小列表),(时间片列表),(组0 warp 列表),(组1 warp 列表),...
```

示例：

```
0,(7,5,4),(10,5,4),(15,14,13,12,11,10,9),(8,7,6,5,4),(3,2,1,0)
```

含义：
- 第 1 列：SM id
- 第 2 列：每个组的 warp 数量（列表长度 = N）
- 第 3 列：每个组的时间片（列表长度 = N）
- 后续列：每组包含的 warp id 列表

**注意**：解析器使用“顶层逗号分割 + 括号内逗号不分割”的逻辑，因此列表必须用
`( ... )` 包裹，且不包含空格可读性更高。

---

## 4. 实现流程（概览）

对每个 SM：

1) **解析 issue trace**  
   扫描 CSV，记录每个 `(sm, warp)` 的 `last_issue_cycle`。

2) **区分正常 warp 与异常 warp**  
   - 正常 warp：出现过 ISSUE  
   - 异常 warp：从未出现 ISSUE

3) **对正常 warp 的结束时间聚类（1D）**  
   生成 N 个聚类，N 自动选择（默认使用 BIC 规则）。

4) **组内代表时间**  
   每个聚类计算代表时间（默认中位数，或可选平均值）。

5) **从代表时间映射为时间片**  
   默认使用近似 gcd（approx-gcd）策略，允许小噪声：
   `rep_i ≈ slice_i * base`，在误差容忍范围内最小化时间片总和。

6) **合并同切片组**  
   若多个组映射得到相同时间片，则合并为一个组，以减少冗余组数与超周期长度。  
   合并后时间片保持不变（不做求和）。

7) **输出 CSV**  
   组列表按代表时间升序排列，异常 warp 组追加在最后。

---

## 5. 聚类算法与 N 的选择

### 5.1 聚类算法（1D k-means 最优解）

warp 结束时间是 1D 数据，样本数最大 64。脚本实现了 **最优 1D k-means**
动态规划（DP）算法，避免随机初始化导致的不稳定：

```
SSE(i, j) = sum_{t=i..j} (x_t - mean(i..j))^2
DP[k][i] = min_{m< i} DP[k-1][m] + SSE(m, i-1)
```

优点：
- 确定性强、重复运行结果一致；
- 对小样本非常稳定；
- 便于对 k（组数）做全局搜索。

### 5.2 N 的选择（默认 BIC）

默认使用 BIC（Bayesian Information Criterion）选择组数：

```
BIC = n * ln(SSE/n) + 2 * k * ln(n)
```

选择 BIC 最小的 k。若需要，也可切换到 silhouette 评分。

---

## 6. 异常 warp 处理

异常 warp 指“从未出现 ISSUE 的 warp”。  
默认策略：**每个异常 warp 单独成组**，时间片为 1。

可选策略：
- `--outlier-mode merge`：所有异常 warp 合并成一个组；
- `--outlier-slice`：设置异常组的时间片长度（默认 1）。

---

## 7. 时间片映射规则（含示例）

默认策略：**approx-gcd**  
在允许的误差范围内，寻找一个基准 `base`，使得每个代表时间都接近
某个整数倍 `slice_i * base`，并优先最小化时间片总和（超周期 T）。
误差约束：
```
err_i = |rep_i - slice_i * base|
err_i <= max(abs_tol, rel_tol * rep_i)
```
其中 `abs_tol` 是绝对容忍（周期），`rel_tol` 是相对容忍（比例），
因此并非“只容忍 1 个周期”的修正，而是按时间尺度自适应。

如果在误差约束下找不到解，会退回到“时间片总和最小”的候选方案，
并在诊断日志中标记 `accepted=False` 以便调参。

调参建议：
- 时间片过大：降低 `--slice-max` 或适当放宽 `--slice-abs-tol/--slice-rel-tol`；
- `accepted=False`：提高 `--slice-max` 或放宽容忍阈值以获得可接受解。

**合并规则（默认开启）**  
当多个组得到相同的时间片（例如 `slice=7` 出现多次），说明这些组的代表时间
已经被映射到同一比例。此时会将这些组合并为一个组，以减少冗余并降低
时间片总和。合并后时间片保持为该值（不做求和）。

示例（噪声 1 个周期）：
```
代表时间 reps = [1001, 2000, 3000]
abs_tol = 2, rel_tol = 0.01
base ≈ 1000
时间片 = [1, 2, 3]
```

示例：
```
代表时间 reps = [120, 300, 420]
gcd = 60
时间片 = [2, 5, 7]
```

如果希望完全沿用传统 gcd，可设置 `--slice-mode gcd`。  
若绝对周期偏移很大，可用 `--slice-mode gcd-diff`：  
先减去最小值，再对差分求 gcd，并保证最小时间片为 1。

相关参数：
- `--slice-abs-tol`：绝对误差容忍（周期）
- `--slice-rel-tol`：相对误差容忍
- `--slice-max`：允许的最大时间片（同时也是搜索上限）

---

## 8. CLI 使用说明

脚本位置：
```
gpu-simulator/gpgpu-sim/n-level/issue_trace_to_n_level.py
```

常用命令：
```
python3 gpu-simulator/gpgpu-sim/n-level/issue_trace_to_n_level.py \
  --input accel-sim-framework/result/issue_trace/fa_issue.csv \
  --output accel-sim-framework/result/issue_trace \
  --algo kmeans-bic \
  --min-groups 1 \
  --max-groups 8 \
  --max-warps 64 \
  --slice-mode approx-gcd \
  --slice-abs-tol 2 \
  --slice-rel-tol 0.01 \
  --slice-max 256 \
  --summary
```

输出：
- `fa_issue_warp_allocate.csv`
- `fa_issue_diagnostics.txt`

---

## 9. 参数说明（脚本所有参数）

- `--input`：issue trace CSV 文件或目录（目录下所有 .csv 会被处理）
- `--output`：输出 CSV 文件或目录（目录模式会生成 *_warp_allocate.csv）
- `--algo`：聚类与选 k 策略（`kmeans-bic` 或 `kmeans-silhouette`）
- `--min-groups`：每个 SM 最小组数
- `--max-groups`：每个 SM 最大组数
- `--max-warps`：每个 SM 的最大 warp 槽位数（默认 64）
- `--center`：代表时间计算方式（`median` 或 `mean`）
- `--slice-mode`：时间片映射方式（`approx-gcd`/`gcd`/`gcd-diff`）
- `--slice-abs-tol`：approx-gcd 绝对误差容忍（周期）
- `--slice-rel-tol`：approx-gcd 相对误差容忍（比例）
- `--slice-max`：允许的最大时间片（也作为搜索上限）
- `--merge-same-slice`：合并同切片组（默认开启）
- `--no-merge-same-slice`：关闭合并同切片组
- `--outlier-mode`：异常 warp 处理（`separate`/`merge`）
- `--outlier-slice`：异常组时间片长度
- `--sm-filter`：只处理指定 SM（例如 `0,1,2-4`）
- `--summary`：输出简要统计到 stdout
- `--seed`：占位参数（当前算法确定性，不使用随机数）

---

## 10. 诊断日志内容

每个 SM 会记录：
- 正常 warp 数量 / 异常 warp 数量
- 选择的 k
- BIC 或 silhouette 得分（按 k 列出）
- 每组代表时间、warp id 列表、对应结束时间列表

便于检查聚类是否合理、warp 分组是否覆盖全部槽位。

---

## 11. 验证与使用建议

建议验证步骤：
1) 使用生成的 `warp_allocate.csv` 运行一次仿真；
2) 确认解析器无 warning；
3) 对比调度行为（如 IPC、stall breakdown 或 warp 完成时间分布）；
4) 若误差过大，可调整 `--max-groups` 或切换聚类策略。

---

## 12. 局限性与后续改进

已知局限：
- 只基于 “最后一次 ISSUE” 周期，不考虑 warp 实际完成/retire；
- 时间片推断是启发式（approx-gcd/gcd），未必唯一；
- 对同步/内存瓶颈导致的长尾 warp 可能敏感。

可改进方向：
- 引入“活跃周期数”或“issue 密度”等特征；
- 使用更稳健的时间片映射（如优化目标函数）；
- 加入对多次运行的统计稳定性分析。
