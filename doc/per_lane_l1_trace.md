# 每线程 L1 Cache 访问追踪功能说明

## 功能概述
为便于调试和性能分析，GPGPU-Sim 新增了一个可选的 L1 Cache 追踪器。开启后，每次有访存请求到达 L1 时，模拟器都会按照活跃线程（lane）拆分，并对同一 cache line 去重，输出包含流水周期、SM/warp 标识、lane ID、读写类型、线程原始地址、L1 访问结果、指令 PC 以及即时内存带宽/占用率的 CSV 记录。

## 启用方式
追踪功能默认关闭，可通过配置文件或命令行参数开启：

- `-l1_trace_enable 1`：开启追踪。
- `-l1_trace_path <文件路径>`：指定 CSV 输出文件路径（必须为非空字符串）。

只要任一选项缺失或 `l1_trace_enable` 仍为 0，追踪器即保持禁用。初始化逻辑位于 `gpgpu_sim::gpgpu_sim` 构造函数中，会在 GPU 初始化完成后调用 `l1_tracer::init` 并为每个 SM 准备一个缓冲区；如果无法创建输出文件，功能会自动回退为禁用。【F:src/gpgpu-sim/gpu-sim.cc†L1038-L1056】【F:src/gpgpu-sim/l1_tracer.cc†L14-L27】

## 输出格式
当追踪器启用且某次访存在 L1 Cache 上完成访问后，会为请求中各个 cache line 的首个活跃 lane 追加一行 CSV 记录：

CSV 的第一行会输出列名，后续每条记录的格式如下：

```
cycle,sm_id,warp_id,lane_id,op,space,address,l1_status,pc,hbm_bw_GBps,hbm_occupancy
```

- `cycle`：全局模拟周期（`gpu_tot_sim_cycle + gpu_sim_cycle`）。
- `sm_id` / `warp_id` / `lane_id`：请求来源的 SM、warp 与 lane 编号。
- `op`：`LD` 表示读，`ST` 表示写，由 `mem_fetch::dbg_is_store()` 决定。
- `space`：访问地址空间，取值与 issue trace 一致，如 `GLOBAL`、`LOCAL`、`SHARED`、`CONST`、`TEX`、`PARAM_KERNEL`、`PARAM_LOCAL`，未知则为 `NA`。
- `address`：该 lane 的原始线程地址，以 `0x` 前缀的 16 进制输出。
- `l1_status`：L1 Cache 访问结果字符串（例如 `HIT`、`MISS`、`RESERVATION_FAIL` 等）。
- `pc`：触发该访存的 warp 指令 PC，若缺失则输出 `NA`。
- `hbm_bw_GBps`：当前周期内内存子分区向互连回传数据的瞬时带宽（GB/s），直接复用模拟器在 ICNT 时钟域统计的回复吞吐量。
- `hbm_occupancy`：当前周期进入内存子分区的请求占用率，等于本周期收到请求的子分区数量除以总子分区数。

由于同一个 memory transaction 中多个 lane 往往映射到同一 cache line，追踪器会借助 L1 配置的 line size 对地址按行对齐，只记录首个遇到的 lane，以避免重复信息。

追踪器为每个 SM 维护独立的字符串缓冲区，并在以下时机触发刷写：

- 缓冲区大小超过 `2^16` 字节（可在 `l1_tracer.cc` 中调整）。
- 每个 kernel 结束（`gpgpu_sim::set_kernel_done`）。
- 打印统计或模拟器析构阶段（`gpgpu_sim::print_stats` 等）。【F:src/gpgpu-sim/l1_tracer.cc†L31-L56】【F:src/gpgpu-sim/gpu-sim.cc†L938-L970】【F:src/gpgpu-sim/gpu-sim.cc†L1277-L1296】

## 实现细节

1. **捕获每线程地址**：在 `shader_core_mem_fetch_allocator::alloc(const warp_inst_t&, ...)` 中，使用访问掩码枚举活跃 lane，调用 `mem_fetch::dbg_add_lane_addr` 保存 lane ID 与线程原始地址，同时用 `dbg_set_op` 记录读/写类型。`mem_fetch` 类新增了存储容器和相关访问器。【F:src/gpgpu-sim/shader.h†L2039-L2074】【F:src/gpgpu-sim/mem_fetch.h†L94-L142】【F:src/gpgpu-sim/mem_fetch.cc†L38-L78】

2. **L1 访问钩子**：在 `l1_cache::access` 成功获得 `cache_request_status` 后调用 `l1_tracer::emit`，同时传入 L1 行大小、当前 HBM 带宽与占用率等信息。追踪器会按行去重并写入 CSV。非 L1 cache 不会触发追踪。【F:src/gpgpu-sim/gpu-cache.cc†L1998-L2024】【F:src/gpgpu-sim/l1_tracer.cc†L33-L78】

3. **生命周期管理**：GPU 初始化时调用 `l1_tracer::init` 建立缓冲区，kernel 完成与最终统计输出时调用 `l1_tracer::flush_all`，确保所有数据写回文件。追踪器在禁用状态下完全为空操作，不影响正常模拟性能。【F:src/gpgpu-sim/gpu-sim.cc†L938-L970】【F:src/gpgpu-sim/gpu-sim.cc†L1038-L1056】【F:src/gpgpu-sim/l1_tracer.cc†L14-L64】

4. **HBM 即时指标**：`gpgpu_sim::cycle` 在 ICNT/L2 时钟域更新瞬时内存带宽与子分区占用率，`l1_cache::access` 通过 `get_last_hbm_bandwidth_gbps()` 与 `get_last_hbm_occupancy()` 获取这些数值并写入日志。【F:src/gpgpu-sim/gpu-sim.cc†L1909-L2070】【F:src/gpgpu-sim/gpu-cache.cc†L1998-L2024】【F:src/gpgpu-sim/gpu-sim.h†L760-L815】

## 使用建议
- 如果需要长时间运行的全局追踪，可将 `l1_trace_path` 指向磁盘上的日志目录，并定期检查文件大小。
- 对于同时运行多个 kernel 的场景，建议结合 `pc` 与 kernel 启停时间对记录进行二次分析，以精确定位访存热点。
- 追踪功能主要用于调试，因此在大规模性能实验时可保持关闭，以避免额外的 I/O 开销。
