# Issue Trace：发射/停顿追踪（含 stall 归因）

issue trace 用来记录 **每个 SM、每个 scheduler、每个 cycle** 的发射（`ISSUE`）
与停顿（`STALL`）情况。它更关注 “warp 是否从 SIMT scheduler 发射出去” 这一层，
适合做调度器行为分析、stall 归因、以及与其它 trace（如 `l1_trace.md`）做时间对齐。

## 启用方式

```
-issue_trace_enable 1             # default 0
-issue_trace_path   "/path/to/issue_trace.csv"
```

当 trace 关闭时，`scheduler_unit::cycle()` 内部会快速短路，不会做字符串拼接与额外采样。

## CSV 输出格式

CSV 首行 header 由 `issue_tracer::init()` 写入，当前格式为：

```
cycle,sm,scheduler,warp,warp_group,event,pc,mask,OP,Space,One_Reason,MEM_WAIT,REG_WAIT,IBUFFER_EMPTY,BARRIER,CONTROL_HAZARD,PIPE_BUSY,DUAL_ISSUE_RESTRICT,Issue_Sector_Addresses,Issue_Sector_Lanes,hbm_bw_GBps,hbm_occupancy
```

各列含义（注意：**一行代表一个 scheduler 在一个 cycle 的结果**）：

- `cycle`：全局模拟周期（`gpu_tot_sim_cycle + gpu_sim_cycle`）
- `sm`：SM id
- `scheduler`：scheduler id（同一个 SM 内通常有多个 scheduler）
- `warp`：
  - `ISSUE`：本 cycle 成功发射的 warp 槽位 id
  - `STALL`：用于“举例说明 stall”的 sample warp id；**可能为 `-1`（见下文）**
- `warp_group`：N-level 调度器下的当前 group index；N-level 关闭时为 `NA`
- `event`：`ISSUE` 或 `STALL`
- `pc`：程序计数器（16 进制）；当无法提供 sample 时为 `NA`
- `mask`：active lane mask（16 进制）；当无法提供 sample 时为 `0x0`
- `OP`：opcode（从指令中提取的 mnemonic）；当无法提供 sample 时为 `NA`
- `Space`：访存指令的地址空间（例如 `GLOBAL/LOCAL/...`），非访存或无法判断时为 `NA`
- `One_Reason`：**只有当本 cycle 的 stall 统计里“恰好只有一种原因非零”时**，这里才会写具体原因（如 `MEM_WAIT`）；否则为 `NA`
- `MEM_WAIT/REG_WAIT/.../DUAL_ISSUE_RESTRICT`：本 cycle 里，该 scheduler 在扫描候选 warp 时观察到的 stall 原因计数（计数的是 “有多少个候选 warp 命中该原因”，不是 cycle 数）
- `Issue_Sector_Addresses / Issue_Sector_Lanes`：仅在 `ISSUE` 且为 `GLOBAL/LOCAL` 访存且扇区信息可收集时填充；其它情况为 `NA`
- `hbm_bw_GBps`：该时刻观测到的 DRAM(HBM) 近似瞬时带宽（GB/s，按 DRAM clock domain 的每次 tick 统计），为全局值（所有 channel 汇总）
- `hbm_occupancy`：对应的带宽占用率（`0~1`），= `hbm_bw / theoretical_hbm_bw`；理论带宽由（通道数 × atom 大小）/ `dram_period` 推出

## 为什么会出现 `warp = -1`（重点）

`warp = -1` **只会出现在 `event=STALL` 的行**，含义是：
> 这个 scheduler 在该 cycle 没有发射成功，同时也无法选出一个“能代表 stall 原因的 sample warp”。

对应源码逻辑：
- `scheduler_unit::cycle()` 在发射失败时，会尝试从本 cycle 采样到的 `stall_sample` 中挑一个 sample；
  如果 sample 不存在（`sample == NULL`），就会把 `sample_warp` 设为 `-1` 并写入 CSV。

最常见的触发场景是：**该 scheduler 在该 cycle 没有任何可用的候选 warp**（例如它负责的 warp 都已 `done_exit()`、或优先级列表为空），因此：
- `warp=-1`
- `pc=NA`
- `mask=0x0`
- 各种 stall 计数（`MEM_WAIT/...`）也会全部为 `0`

一个实际例子（来自 `accel-sim-framework/result/issue_trace/fa4k_nl_issue_sm0.csv`，下行补齐了新增的 `hbm_bw_GBps/hbm_occupancy` 两列）：

```
243657,0,1,-1,1,STALL,NA,0x0,NA,NA,NA,0,0,0,0,0,0,0,NA,NA,0.000000,0.000000
```

含义是：在 cycle 243657，SM 0 的 scheduler 1 没有发射成功，且该 scheduler 没有可用 warp 可作为 sample（因此用 `-1` 表示 “N/A”）。

## 使用建议（统计 stall cycle）

- issue trace 是“按 scheduler 记账”的：如果一个 SM 有 4 个 scheduler，那么同一个 `cycle` 可能出现最多 4 行记录。
- 如果你想统计“因为 `MEM_WAIT` 导致的 stall cycle”，建议不要只依赖 `One_Reason`（它在存在多种原因时会是 `NA`），而是结合计数列做口径定义，例如：
  - **严格口径**：`event=STALL` 且 `PIPE_BUSY==0 && DUAL_ISSUE_RESTRICT==0 && MEM_WAIT>0 && REG_WAIT==0`
  - **包含混合口径**：`event=STALL` 且 `PIPE_BUSY==0 && DUAL_ISSUE_RESTRICT==0 && MEM_WAIT>0`（允许同时有 `REG_WAIT`）

## 输出体量与压缩（强烈建议）

issue trace 的输出体量会随 “cycle × SM × scheduler” 线性增长，规模一大非常容易出现：
磁盘写满、I/O 拖慢仿真、以及后处理脚本读取困难。

在本仓库推荐使用 `accel-sim-framework/traceL1` 的 **边运行边压缩**（不改 C++）方案：

- gzip（系统通常自带）：
  - `./traceL1 --issue-trace-compress gzip <trace_key> <log_name>`
  - 输出：`accel-sim-framework/result/issue_trace/<log_name>_issue.csv.gz`
- zstd（压缩率通常更好；需要系统有 `zstd` 命令）：
  - `./traceL1 --issue-trace-compress zstd <trace_key> <log_name>`
  - 输出：`accel-sim-framework/result/issue_trace/<log_name>_issue.csv.zst`
- 压缩日志：`accel-sim-framework/result/log/<log_name>_issue_trace_compress.log`

## 绘图脚本对压缩输入的支持

以下脚本支持直接读取 `.csv/.csv.gz/.csv.zst`：

- `accel-sim-framework/result/issue_trace/plot/script/plot_bandwidth_over_time.py`
- `accel-sim-framework/result/issue_trace/plot/script/plot_single_issue_breakdown.py`
- `accel-sim-framework/result/issue_trace/plot/script/plot_stall_reason_pc_topk_other.py`（按 stall 原因统计 sample-PC 分布，输出 top-K+OTHER 并画堆叠条 SVG）

一键绘图入口 `accel-sim-framework/result/plot/run_plots.sh` 会自动优先使用压缩后的 issue trace（`.zst` 优先于 `.gz`，再到 `.csv`）。

如果你想显式选择某个版本（例如只画 `.csv.gz` 而不是 `.csv`），建议直接把对应文件路径作为输入传给绘图脚本（例如 `plot_single_issue_breakdown.py <path>.csv.gz`）。
