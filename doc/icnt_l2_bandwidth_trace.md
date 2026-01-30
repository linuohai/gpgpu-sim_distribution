# ICNT + L2 BW Trace（实现说明）

本文档描述在 GPGPU-Sim/Accel-Sim 中新增的两类 **window（按周期聚合）** 带宽/利用率 trace：

- **ICNT transfer（SM↔L2 互联）**：req/reply 的 `packets`、`wire GB/s`、`util`
- **L2 cache 端口（data/fill）**：`busy_cycles`、`GB/s`、`util`

它们主要用于 “L1 miss 多但 L2 hit 高” 的场景下，把瓶颈从 HBM 下钻到 **互联(ICNT)** 和 **L2 本体端口**。

相关方案推导见：`icnt_l2_bandwidth_trace_plan.md`。

---

## 1. 使用方式（GPGPU-Sim knobs）

### 1.1 ICNT BW（transfer 口径）

- `-icnt_trace_enable 0/1`（默认 `0`）
- `-icnt_trace_path "<path>"`（默认空）
- `-icnt_trace_period <N>`（默认 `500`；单位：**ICNT tick**；`N=1` 将输出每 tick，体量很大）

说明：
- 目前 **transfer packet 计数仅对 `network_mode=LOCAL_XBAR` 生效**（INTERSIM 下该计数为 0）。
- `cycle_begin/cycle_end` 使用的是全局仿真 cycle（`gpu_sim_cycle + gpu_tot_sim_cycle`），但 `period` 的窗口长度以 **ICNT tick 次数**计。

### 1.2 L2 BW（data/fill 端口）

- `-l2_bw_enable 0/1`（默认 `0`）
- `-l2_bw_path "<path>"`（默认空）
- `-l2_bw_period <N>`（默认 `500`；单位：**L2 tick**）

说明：
- `cycle_begin/cycle_end` 同样是全局仿真 cycle；`period` 的窗口长度以 **L2 tick 次数**计。
- 统计对象为所有 L2 subpartition 的汇总（全 GPU 汇总）。

---

## 2. CSV 输出格式

### 2.1 ICNT BW CSV

Header：
```
cycle_begin,cycle_end,period,req_packets,reply_packets,req_wire_GBps,reply_wire_GBps,req_util,reply_util
```

- `req_packets/reply_packets`：窗口内（period 个 tick）ICNT 实际 **forward** 的 packet 数（transfer 口径）
- `req_wire_GBps/reply_wire_GBps`：按 `packet == 1 flit` 的 wire bytes 口径换算的 GB/s
- `req_util/reply_util`：0~1 的利用率

### 2.2 L2 BW CSV

Header：
```
cycle_begin,cycle_end,period,l2_available_cycles,l2_data_busy_cycles,l2_fill_busy_cycles,l2_data_GBps,l2_fill_GBps,l2_data_util,l2_fill_util
```

- `l2_available_cycles`：窗口内端口可用 cycle 数（全 subpartition 汇总）
- `l2_data_busy_cycles/l2_fill_busy_cycles`：窗口内 data/fill 端口 busy cycle 数（全 subpartition 汇总）
- `l2_data_GBps/l2_fill_GBps`：用 `data_port_width_bytes` 换算的等效 GB/s
- `l2_data_util/l2_fill_util`：0~1 的利用率

---

## 3. 利用率与理论峰值定义（与实现一致）

### 3.1 ICNT（LOCAL_XBAR）

- `packets_per_cycle = packets_sum / period`
- `peak_packets_per_cycle = min(n_simt_clusters, n_mem_sub_partition)`
- `util = packets_per_cycle / peak_packets_per_cycle`
- `wire_GBps = packets_per_cycle * flit_bytes * f_icnt / 1e9`
- `theoretical_peak_GBps = peak_packets_per_cycle * flit_bytes * f_icnt / 1e9`

其中：
- `flit_bytes` 来自 `icnt_get_flit_size()`（LOCAL_XBAR 对应 40B/packet 的建模假设；INTERSIM 取决于其配置）
- `f_icnt = 1 / icnt_period_seconds`

### 3.2 L2 data/fill port

- `util = busy_cycles / available_cycles`
- `GBps = busy_cycles * data_port_width_bytes / (period * l2_period_seconds) / 1e9`
- `theoretical_peak_GBps ≈ (available_cycles / period) * data_port_width_bytes / l2_period_seconds / 1e9`
  - 如果 `available_cycles == n_subpartitions * period`，则 `theoretical_peak_GBps == n_subpartitions * data_port_width_bytes * f_l2 / 1e9`

---

## 4. 代码实现要点（维护者）

- ICNT tracer：`src/gpgpu-sim/icnt_tracer.{h,cc}`
  - 在 `if (clock_mask & ICNT)` 的 `icnt_transfer()` 后取本 tick 的 forward packets，并做窗口聚合写 CSV
  - forward packets 来源：`src/gpgpu-sim/local_interconnect.{h,cc}` 中 `xbar_router::last_forwarded_packets`
  - 读取接口：`src/gpgpu-sim/icnt_wrapper.{h,cc}` 新增 `icnt_get_last_transfer_packets()`

- L2 BW tracer：`src/gpgpu-sim/l2_bw_tracer.{h,cc}`
  - 在 `if (clock_mask & L2)` 的 subpartition `cache_cycle()` 循环后，汇总 `cache_sub_stats` 做 delta 并写 CSV
  - `data_port_width_bytes` 读取：`memory_sub_partition::get_L2_data_port_width()`（新增于 `src/gpgpu-sim/l2cache.{h,cc}`）

- Flush 策略：每个 window 写一行就立即 `flush_buffer()`，避免仿真因 `--max-cycle` / deadlock 早停导致 CSV 只有表头。

---

## 5. Accel-Sim traceL1 集成（便于一键使用）

`accel-sim-framework/traceL1` 默认开启两类 trace，并输出到：

- `result/icnt_bw/<log>_icnt_bw.csv`
- `result/l2_bw/<log>_l2_bw.csv`

可用参数：
- `--no-icnt-bw-trace` / `--icnt-bw-trace-period <N>`
- `--no-l2-bw-trace` / `--l2-bw-trace-period <N>`

