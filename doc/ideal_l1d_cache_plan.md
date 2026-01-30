# Ideal L1D Cache（“所有目标 load 都 L1 hit”的上界实验）修改计划

## 目标与边界

**目标**：在 Accel-Sim/GPGPU-Sim 的**时序模型**中实现一个可开关的 “Ideal L1D”，用于评估 IMA 场景下 “如果 L1D 对目标 load 100% 命中” 的**性能上界**（IPC、issue stall 等）。

**关键边界**：
- 该改动只改变 **timing path（cache hit/miss 与依赖释放时机）**，不改变 functional correctness 的语义（寄存器/内存的值仍由 functional 模型维护）。
- “Ideal L1D” 默认只覆盖 **global read（`GLOBAL_ACC_R`）** 且 **不 bypass L1D** 的访问；不覆盖 store/atomic（理由见下文）。

## 为什么采用“强制 HIT”的实现方式

GPGPU-Sim 的时序 cache（L1/L2/DRAM）主要用 `mem_fetch` 事务建模延迟/队列/带宽，**不搬运真实 data payload**；functional layer 会在发射时执行 `ptx_exec_inst()` 并维护 `memory_space` 的真实数据。  
因此对 “上界” 而言，最稳定的做法是：在 **L1D access 入口**对目标访问 **短路返回 `HIT`**，从而：
- demand load 不再向下级发读请求（不消耗 L2/DRAM/NoC 排队与带宽）
- 依赖在 L1 hit path（含 L1 latency/bank/port 约束）下被释放

## 设计选择（默认推荐）

### 覆盖范围（建议默认）
- ✅ `GLOBAL_ACC_R` 的 load
- ✅ 仅当该访问走 L1D（即没有被 bypass L1D 的路径过滤掉）
- ❌ atomic：原子语义依赖内存子系统序列化/执行点，强制 hit 会破坏 “atomic 在 memory subsystem 执行” 的模型含义
- ❌ store：store 的 “hit” 与 “不再产生下级流量/ACK/写回” 不是一回事；把 store 也理想化会混入写路径加速，不再是 “L2→L1 prefetch 上界” 的干净定义

### 是否保留 L1 结构约束（建议默认保留）
- 保留 `-gpgpu_l1_latency`、L1 bank/port/data_port_width 等限制  
  （否则会把 L1 自身结构瓶颈也抹掉，得到过于激进的上界）

## 实现方案（代码改动点）

### 1) 新增配置开关
新增命令行参数：
- `-gpgpu_perfect_l1d`（bool，默认 0）
  - 含义：对“目标访问集合”强制 L1D HIT

预期修改文件：
- `gpu-sim.cc`：注册 option（参考 `-gpgpu_perfect_mem` / `-gpgpu_perfect_inst_const_cache`）
- `shader.h`：在 `shader_core_config` 中新增 `bool gpgpu_perfect_l1d;`

### 2) L1D 强制 HIT 的短路入口
首选插入点：
- `gpu-cache.cc`：`l1_cache::access(new_addr_type addr, mem_fetch *mf, ...)`

逻辑（伪代码）：
```
if (gpu->get_config().m_shader_config.gpgpu_perfect_l1d && is_target(mf)) {
  // 1) 不产生 miss/read_request 事件
  // 2) 更新 cache stats/tracer 以保持统计一致
  // 3) 消耗 data port（HIT 的 bandwidth accounting）
  return HIT;
}
// 否则走原始路径
```

`is_target(mf)` 默认实现：
- `mf->get_access_type() == GLOBAL_ACC_R`
- `!mf->isatomic()`
- `!mf->get_is_write()`

注意：是否 “bypass L1D” 的判定发生在 ldst path（`shader.cc` 的 `memory_cycle()`），因此在 `l1_cache::access()` 内只需要确保 “被调用时就是走 L1D 的请求”，不必重复 bypass 判定。

### 3) 统计与 trace 一致性要求
为了让现有分析脚本/图对比可用：
- L1 tracer：仍需输出一条 L1 access 记录，并显示为 hit（便于你观测 “L1 hit rate” 是否达到预期）
- cache stats：需要把该访问计入 L1 hit（而不是 miss），且不要再触发 miss queue/L2 traffic
- bandwidth：命中时应占用 L1 data port（匹配 baseline_cache::bandwidth_management 的 HIT 行为），避免把 L1 自身带宽瓶颈一起去掉

### 4) 回归/一致性检查点
建议加 3 组 sanity check（跑一个小 workload 或 BFS/SpMV 的小图即可）：
1. `baseline`：现状配置，保存 log + issue/L1/L2/bandwidth traces
2. `ideal_l1d`：开启 `-gpgpu_perfect_l1d 1`
3. （可选）`-gpgpu_perfect_mem 1`：极端上界对照，用于确认剩余瓶颈是否还在非 L1D-load 路径（bypass/store/atomic 等）

预期现象（ideal_l1d vs baseline）：
- L1D：目标访问 miss ≈ 0（miss rate 大幅下降）
- L2/L2 BW/HBM：读请求量显著下降（通常会明显少于 baseline）
- issue stall：`MEM_WAIT` 类 stall 显著下降；若仍很高，说明瓶颈在非目标路径或其它结构（reg wait / pipe busy / barrier 等）
- IPC：显著上升（上界）

## 实验数据与画图流程（对比 baseline vs ideal_l1d）

### IPC 对比（log 解析）
从 `result/log/<name>.log` 中提取每个 kernel 的：
- `kernel_name`
- `gpu_ipc` / `gpu_tot_ipc`
- `gpu_sim_insn`（用于一致性检查）

输出建议：
- `result/plot/result/ideal_l1d/ipc_compare.csv`
- `result/plot/result/ideal_l1d/ipc_compare.svg/png`

### ISSUE stall 对比
复用已有脚本：
- `result/issue_trace/plot/script/plot_single_issue_breakdown.py`

对 baseline 与 ideal 各跑一次，得到两个 `*_stall_breakdown.csv` 后，再做一个 “对比 plot”：
- 横轴：stall reason（MEM_WAIT/REG_WAIT/...）
- 两组柱：baseline vs ideal
- 可选：标注百分比变化

### 一键化（建议后续加）
新增一个 wrapper（可选）：
- `result/plot/compare_two_runs.sh --base <logA> --ideal <logB>`
  - 自动生成 IPC/issue-stall 对比图
  - 可直接调用现有 `result/plot/run_plots.sh` 生成每个 run 的 L1/L2/bandwidth 图

## 里程碑与交付物

1) **代码实现**：新增 `-gpgpu_perfect_l1d`，并在 L1D access 中短路 HIT  
2) **验证**：至少 1 个 IMA workload（BFS/SpMV/BC）跑出 baseline vs ideal 的差异  
3) **结果产出**：IPC 对比图 + ISSUE stall 对比图（baseline vs ideal）  

## 风险与注意事项

- Ideal L1D 会同时消除 demand load 对下级的读流量：这是“上界”允许的，但请在报告/结论中明确该假设（否则会把 “减少带宽干扰” 的收益也算进 “prefetch 上界”）。
- 若 workload 含大量 atomic 或写路径瓶颈，ideal L1D 可能提升有限；这也是有效结论（说明 L1 prefetch 上界不高）。

