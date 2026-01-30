# 轻量级 Stall Reason / PC 统计（替代巨型 issue_trace）

本文档描述一种**不输出 issue_trace**、但依然能画出：

1) `STALL reason breakdown`（类似 `plot_single_issue_breakdown.py` 的横向百分比条）  
2) `STALL reason -> top-K PC + OTHER`（每个 reason 一条堆叠条，用于定位热点 SASS PC）

的实现方案。

该方案的核心思想是：在 `scheduler_unit::cycle()` 内部扫描候选 warp 时，**直接把“想发射但被阻塞的那条指令 PC”计入 (stall_reason, pc) 直方图**，并在仿真结束时一次性 dump 一个很小的 CSV（通常 KB～MB 级），从而避免 issue_trace 产生几十 GB～TB 的输出与 I/O 开销。

> 这份文档面向“准备修改 GPGPU-Sim/Accel-Sim 源码的人”。本文只讲实现逻辑与输出格式，不包含具体 patch。

---

## 0. 更新说明：waiting/BARRIER 拆分（2026-01）

早期实现里，`scheduler_unit::cycle()` 发现 `warp(warp_id).waiting()==true` 时，会把该 warp 的 stall 统一记为 `BARRIER`。但 `waiting()` 实际上包含多种等待（CTA barrier / membar / outstanding atomic / ldgsts/depbar / warp done 等），因此会出现：

- `stall_reason_pc_hist.csv` / issue_trace 里 `BARRIER` 的热点 PC 附近没有 barrier 指令
- 例如 BFS 的 `0x110` 附近是 atomic 的下一条指令，但会被归因为 `BARRIER`

为解决该歧义，现已把 `waiting()` 拆分成以下互斥 reason（PC 口径仍为 **blocked-to-issue PC**，即 `warp.get_pc()`）：

- `WAIT_CTA_BARRIER`：等待 CTA barrier（原 `warp_waiting_at_barrier()`）
- `WAIT_MEMBAR`：等待 memory barrier（原 `warp_waiting_at_mem_barrier()`）
- `WAIT_ATOMIC`：等待 outstanding atomic（原 `m_n_atomic>0`）
- `WAIT_LDGSTS`：等待 ldgsts/depbar 相关依赖（原 `m_waiting_ldgsts`）
- `WAIT_DONE`：warp 已 functional_done（等待回收/初始化）

对应地，轻量统计与 issue_trace（如果启用）都使用同一套 reason 名称，便于对齐分析。

---

## 1. 背景：为什么 issue_trace 的 `STALL.pc` 不能用于 reason->pc 归因

issue trace 的 STALL 行（参考 `issue_trace.md`）里：

- `pc`：是 **sample warp PC**（用于“举例说明该 cycle 的 stall”）
- `MEM_WAIT/REG_WAIT/...`：是扫描候选 warp 时观察到的 **stall 原因计数**（计的是“候选 warp 数”，不是 cycle 数）

因此当一个 `STALL` 行里多个原因同时非 0 时，如果用这一个 `STALL.pc` 去代表多个原因，会出现明显错配（例如：同一 PC 被“挂”到多个 reason 上）。

要修正这个问题，必须在**产生这些 reason 计数的那一刻**拿到对应 warp 的 PC，而不是事后从 `STALL.pc` 反推。

---

## 2. 目标与口径（必须先统一）

### 2.1 你要画的两类图

**A. Stall reason breakdown**

- 每个 reason 一条横向条形图
- 百分比 = `reason_total / sum(all_reason_total)`
- 数值与现有 `issue_trace` 的 breakdown 口径保持一致

**B. per-reason top-K PC + OTHER（堆叠条）**

- 每个 reason 一条堆叠条（归一化到 100%）
- 段 = topK PC + OTHER
- 用于回答：某个 stall reason 主要集中在哪些静态指令（PC）

### 2.2 推荐主口径：`stall_warp_events`（等价于现有 breakdown 的“warp-cycle 口径”）

为与现有 issue_trace breakdown 一致，建议定义：

> `stall_warp_events(reason)`：在一个 scheduler-cycle 扫描候选 warp 时，被判定为该 reason 的 warp 数（累加）

这等价于 issue_trace `STALL` 行里 `MEM_WAIT/…` 那些列累加出来的值（很多脚本叫 `stall_warp_cycles`）。

如果你只想做更粗的 `stall_rows`（某 reason 在该 cycle 出现过就 +1），也可以额外输出，但它不等价于上述 `stall_warp_events`。

---

## 3. 核心方案：在 `scheduler_unit::cycle()` 内做 (reason, pc) 直方图

### 3.1 关键点：只统计“最终没有 issue 的 cycle”

issue_trace 的行为是：只有当该 scheduler 在该 cycle **最终 `issued_inst == false`** 时才输出 `STALL` 行。

为了让“轻量统计”与 issue_trace breakdown 口径一致，需要：

1) 扫描候选 warp 的过程中，临时记录本 cycle 的 (reason, pc) 贡献  
2) cycle 结束时：
   - 若 `issued_inst == false`：把临时贡献 merge 到全局直方图  
   - 若 `issued_inst == true`：丢弃临时贡献（因为这个 cycle 最终不是 stall cycle）

否则你会把“扫描过程中的阻塞现象”统计进来，即使该 cycle 最后成功 issue 了，口径会偏离 issue_trace。

### 3.2 “PC”选取：用“想发射但被阻塞的那条指令 PC”

你已明确希望使用：
> “想发射但被阻塞的那条指令 PC”，后续可通过 SASS 定位具体指令。

在 `scheduler_unit::cycle()` 中，各 reason 可取到 PC 的典型位置（以现有代码为参考）：

- `MEM_WAIT/REG_WAIT`：scoreboard fail 分支里有 `pI->pc`（当前准备 issue 的指令）
- `PIPE_BUSY/DUAL_ISSUE_RESTRICT`：管线满/dual-issue 限制导致无法 issue 的那个 `pI->pc`（`record_pipe_busy(..., pI->pc, ...)`）
- `WAIT_*`（等待类）：`warp(warp_id).get_pc()`（该 warp 本 cycle “本该继续执行/想要取/想要发射”的 next_pc；用于 blocked-to-issue 归因）
- `IBUFFER_EMPTY`：`warp(warp_id).get_pc()`（下一条要取的 PC；但 ibuffer 为空意味着“没取到指令”，仍可作为定位点）
- `CONTROL_HAZARD`：发生 flush 时同时存在两个 PC：
  - `pI->pc`：ibuffer 里的旧指令 PC（将被 flush）
  - `pc`：pdom 栈顶给出的真实 next pc

**本实现固定选择 `pc`（真实 next pc）**，用于定位真实控制流跳转点。

### 3.3 同一 cycle 内多个 warp、多个 reason 的计入

你关心的场景：
> 同一 cycle / 同一 scheduler（subcore）内，有 2 个 warp stall，原因不同。

本方案可正确计入：

- 扫描到 warpA：`MEM_WAIT`，PC=pcA → `pending[MEM_WAIT][pcA] += 1`
- 扫描到 warpB：`WAIT_ATOMIC`，PC=pcB → `pending[WAIT_ATOMIC][pcB] += 1`
- cycle 结束若 `issued_inst == false`：把 `pending` merge 到全局

与 issue_trace 的差异是：issue_trace 只会输出一个 sample PC；而这里是对每个 reason 的每个贡献 warp 都记录对应 PC，从源头上避免错配。

---

## 4. 数据结构建议（保证“可跑”）

### 4.1 全局统计（跨 cycle 累加，最终 dump）

建议在 GPU 级别维护（例如挂到 `gpgpu_sim` 或单独的 tracer/collector 对象）：

- `total_by_reason[reason] : uint64_t`  
  - = `stall_warp_events(reason)`，用于画 reason breakdown

- `hist_by_reason_pc[reason][pc] : uint64_t`  
  - = `stall_warp_events(reason, pc)`，用于画 reason->pc topK+OTHER

原因数很少（目前约 11 个，包含 `WAIT_*` 子类），PC 的 unique 数量通常也远小于 issue_trace 行数（常见 10^2～10^4），因此内存占用很可控。

### 4.2 每-cycle 临时统计（必须可丢弃）

为了只在 `issued_inst==false` 时 commit，建议在 `scheduler_unit::cycle()` 内用临时容器记录“本 cycle 的贡献”：

**方案 A（简单可靠，推荐）**：每个 reason 一个小 `unordered_map<pc, count>`（只存本 cycle 出现过的 PC）。

**方案 B（更轻量）**：用 `vector<(reason, pc)>` 记录每次“reason 被计数”时的 (reason,pc) 对，cycle 结束时按 vector merge（可能重复，merge 时做累计）。

两者都可以，重点是：**本 cycle 能丢弃**、且 commit 时只发生一次。

---

## 5. 输出文件建议（不输出 issue_trace 时你仍能画图）

**输出目录语义：**  
`-stall_reason_pc_stats_path` 指定一个**已存在的目录**，文件名固定如下；每次 `gpgpu_sim::print_stats()` 会覆盖同名 CSV（最终文件即为全程统计）。

为了避免多算法/多次运行互相覆盖，推荐把它指向每次运行独立的子目录，例如：

```
result/issue_trace/stall_reason_pc_stats/<log_name>/
```

其中 `<log_name>` 与 `traceL1` 的 `--log-name`（或默认 log-name）一致。

### 5.1 Stall reason breakdown（用于画“reason breakdown 图”）

文件（基础文件名）：`<out_dir>/stall_reason_breakdown.csv`  
本仓库脚本 `traceL1` 会在仿真结束后把它重命名为（并只保留）：

```
<out_dir>/<log_name>_stall_reason_breakdown.csv
```

推荐字段：

```
reason,stall_warp_events,fraction_of_stall_warp_events
MEM_WAIT,10847377457,0.70057615
WAIT_ATOMIC,3439202892,0.22212037
PIPE_BUSY,1192086128,0.07699069
...
```

说明：
- `stall_warp_events` 就是 `total_by_reason[reason]`
- fraction 以所有 reason 的 total 求和为分母

### 5.2 reason+pc 直方图（用于画 topK+OTHER）

文件（基础文件名）：`<out_dir>/stall_reason_pc_hist.csv`  
本仓库脚本 `traceL1` 会在仿真结束后把它重命名为（并只保留）：

```
<out_dir>/<log_name>_stall_reason_pc_hist.csv
```

推荐字段：

```
reason,pc,stall_warp_events,fraction_of_reason_stall_warp_events
MEM_WAIT,0x690,946883720,0.08729149
MEM_WAIT,0x660,901870442,0.08314180
...
```

说明：
- `stall_warp_events` = `hist_by_reason_pc[reason][pc]`
- fraction 的分母是该 reason 的 `total_by_reason[reason]`

### 5.3 可选：直接输出 topK+OTHER（减少后处理）

文件（基础文件名）：`<out_dir>/stall_reason_pc_topk_other.csv`  
本仓库脚本 `traceL1` 会在仿真结束后把它重命名为（并只保留）：

```
<out_dir>/<log_name>_stall_reason_pc_topk_other.csv
```

推荐字段：

```
reason,segment,rank,stall_warp_events,fraction_of_reason,topk_share
MEM_WAIT,0x690,1,946883720,0.08729149,0.40579294
...
MEM_WAIT,OTHER,0,644...,0.594...,0.40579294
```

仅输出 CSV，SVG 绘图由脚本完成。

---

## 6. 与现有 issue_trace breakdown 对齐的验证方法（强烈建议做一次）

为了确认你实现的统计与现有 issue_trace 口径一致：

1) 选一个很小的 workload（或短运行），同时开启 issue_trace 与轻量统计  
   - `traceL1`：`./traceL1 --issue-trace <trace_key> <log_name>`  
   - 或直接在 config 中设置：`-issue_trace_enable 1` + `-issue_trace_path "<path>"`  
2) 用 issue_trace 的 `STALL` 行生成 breakdown（例如 `plot_single_issue_breakdown.py` 的 CSV 输出）  
3) 对比两边的 `stall_warp_events(reason)`：
   - `MEM_WAIT/WAIT_*/...` 应该完全一致（或差异极小且可解释）

如果不一致，优先检查：
- 是否在 `issued_inst==true` 的 cycle 也统计了（应丢弃）
- 是否包含了“全 0 reason 计数”的 idle-like STALL（issue_trace breakdown 默认会跳过）
- 某些 reason 的计数是否在多个分支重复累加（需要与原来的 `increment_stall_reason` 位置一致）

---

## 7. 性能与空间收益预期

**空间：**
- issue_trace：`cycle × SM × scheduler` 级别输出，极易到几十 GB～TB
- 轻量统计：`reason × unique_pc` 级别输出，通常 KB～MB

**运行时开销：**
主要来自：
- 在“启用统计”时，额外的 `unordered_map` / `vector` push
- 只有在 `issued_inst==false` 的 cycle 才 commit 到全局

总体上它应远小于 issue_trace 的字符串拼接与磁盘 I/O。

---

## 8. 局限性与解释边界（避免误用）

1) **这不是“根因归因”**  
`MEM_WAIT` 的根因可能是更早发射的某条 load/store，而这里记录的 PC 是“当前被 scoreboard 卡住、准备发射的指令 PC”。  
对你当前需求（结合 SASS 定位热点）这通常足够，但不要把它解读为“导致 miss 的那条 load 的 PC”。

2) **CONTROL_HAZARD 的 PC 定义需要明确**  
本实现固定使用真实 next pc（`pc`），与“被 flush 的旧指令 PC（`pI->pc`）”区分开。

3) **reason 计数本身是“候选 warp 扫描统计”**  
它衡量的是 scheduler 在扫描过程中看到的阻塞条件数量，不等价于硬件里严格的“stall 周期归因”，但与 issue_trace 的现有口径一致。

---

## 9. 推荐的对外接口（配置项建议）

为了能在不输出 issue_trace 的情况下启用统计，实现提供独立开关，例如：

- `-stall_reason_pc_stats_enable 1`（默认 0）
- `-stall_reason_pc_stats_path "<out_dir>"`（**输出目录**，需已存在）
- `-stall_reason_pc_stats_topk 5`（可选：直接输出 topK+OTHER）

并确保它不依赖 `issue_tracer::enabled()`。

在本仓库的运行脚本 `traceL1` 中，这个统计默认开启（无需手动改 config）；脚本会为每次运行创建
`result/issue_trace/stall_reason_pc_stats/<log_name>/` 目录，并把 `-stall_reason_pc_stats_path` 指向该目录。

### 9.1 实现落点（代码位置）

- `src/gpgpu-sim/shader.cc`：在 `scheduler_unit::cycle()` 里记录 `(reason, pc)`，并仅在 `issued_inst == false` 时提交；`CONTROL_HAZARD` 使用真实 next pc。
- `src/gpgpu-sim/stall_reason_pc_stats.{h,cc}`：全局直方图与 CSV 输出。
- `src/gpgpu-sim/gpu-sim.cc`：初始化与 `print_stats()` 时 dump。
- `src/gpgpu-sim/gpu-sim.h`：配置项与开关。

---

## 10. 你最终能画出什么（与现阶段实验的关系）

如果你只关心：

- stall reason breakdown（类似现有图）
- 每个 reason 下 top5 PC + OTHER（堆叠条）

那么启用这套轻量统计后：

> 你可以在现阶段**完全不打印 issue_trace**，只输出小 CSV + SVG，节省大量磁盘与仿真时间，并且得到更“正确”的 reason->pc 对应关系（不会被 sample PC 误导）。
