# 1. 背景与目标（Why / Non-goals）
- Why：补齐 L2 Cache 的逐 lane 访问追踪能力，和 L1 trace 对齐输出口径，便于分析访存 pattern 与 L2 命中行为。
- 目标：
  - 仅在 trace model 下提供 L2 cache trace，输出 CSV。
  - 输出字段与 L1 trace 对齐，并支持可选打印 HBM 带宽/占用率、SM 计算单元活跃度。
  - 面向 A100，兼容 V100（不硬编码架构常数，依赖现有配置）。
- Non-goals：
  - 不改动 L1 trace 与 issue trace 的格式/逻辑。
  - 不引入新的 L2 统计项或 power model 逻辑。
  - 不在 exec/functional 模式启用该 trace。
  - 暂不新增 L2 bank/sub-partition ID 字段（如需可后续扩展）。
- Assumptions/Unknowns：
  - 假设 trace model 下 `mem_fetch::dbg_lanes()` 与 `dbg_is_store()` 已可用；若 trace 驱动路径未填充这些字段，则需要补充 `shader_core_mem_fetch_allocator::alloc(...)` 或 trace parser 侧的字段传递（需从给定代码确认具体构造路径）。

# 2. 需求规格（User-facing spec：参数、默认值、输出、边界条件）
- 参数与默认值（`src/gpgpu-sim/gpu-sim.cc` / `src/gpgpu-sim/gpu-sim.h`）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `-l2_trace_enable` | `0` | 开启 L2 per-lane trace（仅 trace model 生效） |
| `-l2_trace_path` | `""` | 输出 CSV 路径，空则禁用 |
| `-l2_trace_print_bw` | `1` | 是否输出 `hbm_bw_GBps,hbm_occupancy` 两列 |
| `-l2_trace_print_compute` | `1` | 是否输出 SM 计算单元活跃度列 |

- 输出（`src/gpgpu-sim/l2_tracer.cc`）：
  - 基础列：`cycle,sm_id,warp_id,lane_id,op,space,address,l2_status,pc`
  - 可选列：
    - 带宽组：`hbm_bw_GBps,hbm_occupancy`
    - 计算组：`sm_alu_active_lanes,sm_alu_total_lanes,sm_alu_utilization,...,sm_tensor_utilization`
  - 列头根据 `-l2_trace_print_bw/-l2_trace_print_compute` 动态拼接。

- 边界条件：
  - `-l2_trace_enable=1` 但 `-l2_trace_path` 为空/文件不可写：trace 自动关闭（同 L1 逻辑）。
  - `mem_fetch::dbg_lanes()` 为空：不输出该请求（避免无 lane 信息的内部写回干扰）。
  - L2 禁用（`m_L2_config.disabled()`）：不会走 `l2_cache::access`，因此无 trace 行。
  - 非 trace model：不调用 `l2_tracer::init`，即使参数开启也不会生成文件。

# 3. 架构与数据流（在 GPGPU-Sim 流水中挂在哪：调用路径/时序点）
- 初始化（trace model）：
  - `accel-sim.cc` 中设置 `gpgpu_sim_config::set_trace_model(true)`。
  - `trace_gpgpu_sim` 构造函数中调用 `l2_tracer::init(...)`（`trace-driven/trace_driven.h`）。
- 访问路径：
  - `gpgpu_sim::cycle` 的 L2 时钟域调用 `memory_sub_partition::cache_cycle`（`src/gpgpu-sim/l2cache.cc`）。
  - `memory_sub_partition::cache_cycle` 内部调用 `l2_cache::access`（`src/gpgpu-sim/gpu-cache.cc`）。
  - `l2_cache::access` 调用 `data_cache::access`，获得 `cache_request_status` 后调用 `l2_tracer::emit` 写入 CSV。
- 数据来源：
  - lane 地址与 op：`mem_fetch::dbg_lanes()` / `dbg_is_store()`（`src/gpgpu-sim/mem_fetch.h`）。
  - 访问空间与 PC：`mem_fetch::get_inst()` / `get_pc()`。
  - HBM 指标：`gpgpu_sim::get_last_hbm_bandwidth_gbps()` / `get_last_hbm_occupancy()`（`src/gpgpu-sim/gpu-sim.cc`）。
  - 计算单元活跃度：`gpgpu_sim::get_last_active_*`（`src/gpgpu-sim/gpu-sim.h`），由 `shader.cc` 更新。
- Flush 点：
  - `gpgpu_sim::set_kernel_done` 与 `gpgpu_sim::print_stats` 调用 `l2_tracer::flush_all`（`src/gpgpu-sim/gpu-sim.cc`）。

# 4. 设计细节（核心算法、状态机、数据结构、关键伪代码）
- 核心数据结构（`src/gpgpu-sim/l2_tracer.h`）：
  - `s_enabled`：是否启用。
  - `s_print_bw` / `s_print_compute`：控制列输出。
  - `s_buffers`：按 SM 分片的字符串缓冲区（降低锁开销）。
  - `s_flush_threshold`：缓冲区阈值（`1<<16`）。
- 核心算法：
  - 每次 `l2_cache::access` 后：
    1) 用 `m_stats.select_stats_status(...)` 得到可打印的 `l2_status`。
    2) 读取 HBM/计算单元即时指标。
    3) `l2_tracer::emit` 按 lane 拆分并按 cache line 去重（L2 line size）。
  - `emit` 在 CSV 中按配置拼接字段。

- 关键伪代码（`src/gpgpu-sim/gpu-cache.cc`）：
```
status = data_cache::access(..., &probe_status)
trace_status = m_stats.select_stats_status(probe_status, status)
cycle = gpu_sim_cycle + gpu_tot_sim_cycle
metrics = {hbm_bw, hbm_occ, active_*_lanes, total_*_lanes}
l2_tracer::emit(sid, wid, mf, trace_status, cycle, L2_line_sz, metrics)
return status
```

- Assumptions/Unknowns：
  - 若未来需要 L2 bank/sub-partition ID，请在 `l2_tracer::emit` 中追加 `mf->get_sub_partition_id()` 字段，并更新 CSV 头（需确认下游解析依赖）。

# 5. 代码改动清单（逐文件：新增/修改的类、函数签名、关键逻辑点）

| File | Symbol (class/function) | Change type (add/modify/delete) | Key logic summary (<=3 bullets) | Risk (what can break) |
|---|---|---|---|---|
| `src/gpgpu-sim/l2_tracer.h` | `class l2_tracer` | add | - 定义 L2 trace 接口与状态<br>- 支持 BW/compute 可选列 | 低：新增头文件引用错误 |
| `src/gpgpu-sim/l2_tracer.cc` | `l2_tracer::{init,emit,flush_all}` | add | - CSV 头按开关拼接<br>- lane 去重与缓冲刷写 | 中：CSV schema 变动影响解析 |
| `src/gpgpu-sim/gpu-cache.cc` | `l2_cache::access` | modify | - 获取 `trace_status`<br>- 调用 `l2_tracer::emit` | 中：访问路径额外开销 |
| `src/gpgpu-sim/gpu-sim.cc` | `gpgpu_sim_config::reg_options` | modify | - 注册 `-l2_trace_*` 参数 | 低：参数解析冲突 |
| `src/gpgpu-sim/gpu-sim.cc` | `gpgpu_sim::{set_kernel_done,print_stats}` | modify | - 添加 `l2_tracer::flush_all` | 低：重复 flush |
| `src/gpgpu-sim/gpu-sim.h` | `gpgpu_sim_config` | modify | - 新增 L2 trace 配置字段与 getter | 低：结构体变更 |
| `src/gpgpu-sim/shader.cc` | `pipelined_simd_unit::get_active_lanes_in_pipeline` | modify | - L2 compute trace 也触发 lane 统计 | 中：统计开销上升 |
| `trace-driven/trace_driven.h` | `trace_gpgpu_sim::trace_gpgpu_sim` | modify | - trace model 中初始化 L2 tracer | 中：trace model 初始化顺序 |
| `accel-sim.cc` | `gpgpu_trace_sim_init_perf_model` | modify | - 标记 trace model 以启用 L2 trace | 低：模式标记遗漏 |
| `src/gpgpu-sim/CMakeLists.txt` | `gpgpusim_SRC` | modify | - 增加 `l2_tracer.cc` | 低：构建未更新 |
| `doc/l2_cache_trace_design.md` | N/A | add | - 设计文档 | 低：无运行时风险 |

# 6. 配置与兼容性（config 迁移、stats 命名、trace 版本号等）
- 配置迁移：新增可选参数，不影响旧配置；默认关闭。
- 兼容性策略：
  - `l2_trace_enabled()` 受 trace model 标记控制，仅 trace model 生效。
  - 仅 trace model 初始化 L2 tracer（`trace_gpgpu_sim`），exec/SST 模式保持不变。
  - A100/V100：计算单元数量、L2 行大小、HBM 频率均来自现有配置 (`gpgpu_sim_config` / `memory_config`)；不硬编码架构常量。
- trace 版本号：
  - 本次不新增显式版本号，CSV 头即 schema；若未来需要稳定版本，可新增 `l2_trace_version` 伪列或在文件首行加入注释（需确认下游解析器）。

# 7. 测试与验证计划（单测/微基准/回归/对比基线；预期结果）

| Test case (microbenchmark / app / trace) | Knob settings | Expected stats/log signature | Failure patterns & debugging entry point |
|---|---|---|---|
| microbenchmark: vector add (trace model) | `-l2_trace_enable 1 -l2_trace_path <file> -l2_trace_print_bw 1 -l2_trace_print_compute 1` | CSV 头包含 BW+compute 列；行数 > 0；`l2_status` 字段存在 | 文件为空/缺失：检查 `trace-driven/trace_driven.h` 初始化与 `l2_tracer::init`；检查 `mem_fetch::dbg_lanes()` |
| microbenchmark: vector add (trace model) | `-l2_trace_enable 1 -l2_trace_path <file> -l2_trace_print_bw 0 -l2_trace_print_compute 0` | CSV 头仅 9 列（到 `pc` 为止） | 仍包含 BW/compute 列：检查 `l2_tracer::init` 头部拼接逻辑 |
| trace model vs exec model | exec 模式也加上 `-l2_trace_*` | exec 模式不生成 L2 trace 文件 | 文件出现：检查 `trace_gpgpu_sim` 之外是否调用了 `l2_tracer::init` |

# 8. 观测与调试（新增 debug print、断言、统计项、定位指南）
- 主要观测点：`-l2_trace_path` 输出的 CSV 文件，按 header 判断 schema。
- 定位指南：
  - 无输出：确认 L2 未被禁用，`mem_fetch::dbg_lanes()` 是否为空，trace model 是否生效。
  - 计算单元列全 0：确认 `-l2_trace_print_compute=1`，并检查 `shader.cc` 中 active lanes 统计路径是否执行。
  - BW 列全 0：检查 `gpgpu_sim::cycle` 中 DRAM 域是否更新 `m_last_hbm_*`。
- 可选调试手段：
  - 使用 `MEMPART_DPRINTF` / `MEM_SUBPART_DPRINTF`（`src/gpgpu-sim/l2cache_trace.h`）辅助验证 L2 请求流。

# 9. 风险与替代方案（可能踩坑、回滚开关、Plan B）
- 风险：
  - `mem_fetch::dbg_lanes()` 在 trace model 中为空导致 trace 行缺失。
  - 可选列导致下游解析脚本需要适配可变 schema。
  - 统计开销增加（特别是 compute 列启用时）。
- 回滚开关：
  - `-l2_trace_enable 0`（最小回滚）。
  - 若需要彻底关闭 trace model 逻辑，移除 `trace_gpgpu_sim` 中的 `l2_tracer::init` 调用。
- 替代方案（Plan B）：
  - 若 per-lane 数据不可用，可改为按 `mem_fetch::get_addr()` 记录每请求一行（无需 lane）。
  - 若 schema 需固定，可强制保留 BW/compute 列并填 `0/NA`。

# 10. 开发步骤与里程碑（按 PR 拆分，1/2/3 阶段交付）
- 阶段 1（PR1）：新增 `l2_tracer` 类与配置参数；trace model 初始化与 flush；确保构建通过。
- 阶段 2（PR2）：在 `l2_cache::access` 挂载 trace，并打通 BW/compute 输出与 lane 去重。
- 阶段 3（PR3）：补充文档与基础回归验证脚本/用例说明。
