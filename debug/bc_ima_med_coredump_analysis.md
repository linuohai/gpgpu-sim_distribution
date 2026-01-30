# bc_ima_med 在 `traceL1 --plot-background` 下的 Segmentation fault / coredump 分析

> 目标：把问题定位信息整理成**可被后续语言模型直接执行**的调试文档（复现 → 定位 → 可能根因 → 修复方向）。

## 0. 现象（Symptom）

- 复现命令（容器 `fa` 内）：
  - `cd /workspace/prefetch`
  - `./traceL1 --plot-background bc_ima_med`
- 结果：`accel-sim.out` 在仿真过程中 `SIGSEGV`，`traceL1` 报 `(core dumped)`。
- 日志位置：`/workspace/prefetch/result/log/bc_ima_med.log`
- 日志尾部关键片段（可用 `tail -n 60 result/log/bc_ima_med.log` 查看）：
  - 进入 `kernel-4` 后打印 `thread block = 0,0,0` 随即崩溃：
    - `Processing kernel .../kernel-4-ctx_...traceg.xz`
    - `launching kernel ... uid: 4`
    - `thread block = 0,0,0`
    - `Segmentation fault (core dumped) "$ACCEL_SIM_BIN" -trace "$TRACE_PATH" -config "$TRACE_CONFIG" -config "$TEMP_GPGPU_CONFIG"`

结论：**崩溃发生在 `accel-sim.out` 内部**，不是 `traceL1` 脚本本身。

## 1. 复现与定位（Recommended Workflow）

### 1.1 直接复现（确认问题仍在）

在 `fa` 容器：

```bash
cd /workspace/prefetch
./traceL1 --plot-background bc_ima_med
```

### 1.2 获取“真实的 accel-sim.out 命令行参数”（非常关键）

`traceL1` 里面的真正执行在 `traceL1:1013-1017`，但日志里只看到变量名（`$TRACE_PATH/$TEMP_GPGPU_CONFIG`）。

推荐做法：用 `bash -x` 打印展开后的变量。

```bash
cd /workspace/prefetch
bash -x ./traceL1 --plot-background bc_ima_med 2>&1 | tee /tmp/traceL1_bc_ima_med_x.log
```

你需要从输出中抓到三样东西：

- `TRACE_PATH=.../kernelslist.g`（完整路径）
- `TRACE_CONFIG=.../trace.config`
- `TEMP_GPGPU_CONFIG=/tmp/tmp.XXXXXX`（随机临时文件名）

### 1.3 保留 `TEMP_GPGPU_CONFIG`（脚本默认会删除它）

`traceL1` 会 `trap cleanup EXIT` 并 `rm -f "$TEMP_GPGPU_CONFIG"`，所以想复现/调试必须先把它复制出来。

可复用的“复制临时配置”方法（推荐）：

```bash
cd /workspace/prefetch
rm -rf /tmp/traceL1_tmp && mkdir -p /tmp/traceL1_tmp
rm -f /tmp/bc_ima_med_traceL1_temp.config
(
  for i in $(seq 1 200); do
    f=$(ls -1 /tmp/traceL1_tmp 2>/dev/null | head -n 1 || true)
    if [ -n "$f" ] && [ -f "/tmp/traceL1_tmp/$f" ]; then
      cp "/tmp/traceL1_tmp/$f" /tmp/bc_ima_med_traceL1_temp.config
      echo "[copied] /tmp/traceL1_tmp/$f -> /tmp/bc_ima_med_traceL1_temp.config"
      exit 0
    fi
    sleep 0.05
  done
) &
TMPDIR=/tmp/traceL1_tmp ./traceL1 --plot-background bc_ima_med
```

执行结束后应存在：`/tmp/bc_ima_med_traceL1_temp.config`。

### 1.4 gdb 回溯（注意：gdb 可能改变复现概率）

> 说明：此类问题疑似 use-after-free，**在 gdb 下有时不一定按同样位置崩溃**（内存布局/时序变化）。如果在 gdb 里跑很久不崩，可尝试 `set env MALLOC_PERTURB_ 165` 或改用“先跑崩溃再 attach/core”。

用保存下来的 temp config 直接跑 `accel-sim.out`：

```bash
cd /workspace/prefetch
source gpu-simulator/setup_environment.sh release

gdb -q --args gpu-simulator/bin/release/accel-sim.out \
  -trace  hw_run/traces/device-0/12.6/bc_linear_base/mtx___data_web_Google_1_0_506742/traces/kernelslist.g \
  -config gpu-simulator/configs/tested-cfgs/SM80_A100/trace.config \
  -config /tmp/bc_ima_med_traceL1_temp.config

# 可选：提升 UAF 暴露概率
# (gdb) set env MALLOC_PERTURB_ 165

(gdb) run
(gdb) bt
```

## 2. 已知崩溃点（Crash Site）

### 2.1 直接崩溃点：`ptx_sim_kernel_info(kernel=0x0)`

文件：`gpu-simulator/gpgpu-sim/src/cuda-sim/cuda-sim.cc:2053-2056`

```cpp
const struct gpgpu_ptx_sim_info *ptx_sim_kernel_info(const function_info *kernel) {
  return kernel->get_kernel_info();
}
```

当 `kernel == NULL` 时直接解引用，触发 SIGSEGV。

### 2.2 上游触发点：`shader_core_config::max_cta()`

文件：`gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc:4720-4731`

```cpp
const class function_info *kernel = k.entry();
...
const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel);
```

说明：`k` 是 `kernel_info_t` 引用；`k.entry()` 返回 `function_info*`。
当前观测到 `k.entry()` 得到的 `kernel` 变为 `NULL`（或被破坏为 0），导致后续崩溃。

## 3. 最可能的根因：kernel 生命周期管理错误（use-after-free / dangling pointer）

核心推断：**Accel-Sim 在 `cleanup()` 中提前 delete 了 kernel / function_info，但 GPGPU-Sim 的调度队列仍持有该 kernel 指针**。

### 3.1 可疑的“提前释放”位置：`accel_sim_framework::cleanup()`

文件：`gpu-simulator/accel-sim.cc:122-150`

```cpp
tracer.kernel_finalizer(k->get_trace_info());
delete k->entry();
delete k;
kernels_info.erase(kernels_info.begin() + j);
```

这会释放：
- `k`（`trace_kernel_info_t*`，本质上是 `kernel_info_t` 载体）
- `k->entry()`（`function_info*`）

如果下游仍引用旧的 `kernel_info_t*` 或其内部 `function_info*`，就可能出现：
- `k.entry()` 返回 NULL / 野指针
- 访问 `kernel->get_kernel_info()` 崩溃

### 3.2 可能持有 kernel 指针的队列：`pending_ctas`

文件：`gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.cc:5751-5797`

```cpp
kernel_info_t *kernel;
...
m_core[core]->pending_ctas.push_back(kernel);
...
kernel_info_t *pending_cta = m_core[core]->pending_ctas.front();
if (m_core[core]->can_issue_1block(*pending_cta)) {
  ...
}
```

这里把 `kernel_info_t*` 放进 per-core 的 `pending_ctas`（一个 deque）。
如果 `cleanup()` 把这个 kernel 释放了，但 `pending_ctas` 还没清空，那么 `*pending_cta` 会变成 UAF。

**与日志现象吻合：** 崩溃发生在刚开始 issue 第一个 thread block (`thread block = 0,0,0`) 时，正好会走 `can_issue_1block()` → `max_cta()`。

## 4. 修复方向（从“最小改动”到“结构性改动”）

> 目标：确保 kernel 对象在模拟器内部不再被引用之前，绝不释放。

### 4.1 最小改动（推荐优先尝试）：延后 delete 到仿真结束

- 在 `accel_sim_framework::cleanup()` 中：
  - 不要立刻 `delete k->entry(); delete k;`
  - 改为把 `k` 放到一个 `retired_kernels` 容器里（例如 `std::vector<trace_kernel_info_t*> m_retired;`）
  - 只从 `kernels_info` 中移除调度入口（避免再次被 select），但保留对象内存
- 在 simulation 完全结束（所有 kernel 都结束、所有 core/cluster 队列为空）后，再统一释放 `retired_kernels`

优点：实现快；能快速验证“生命周期问题”假设。
缺点：内存峰值略增（但 kernel 数通常可控）。

### 4.2 清理引用再释放（更严谨，但需要更多理解）

在释放 kernel 前，确保所有可能持有该 kernel 指针的结构都清空/剔除：

- per-core：`pending_ctas`（以及其他可能的 pending/running 列表）
- cluster / gpu：任何缓存了 `kernel_info_t*` 的队列/指针

优点：不会延长对象生命周期。
缺点：需要完全梳理引用关系，漏一个仍会 UAF。

### 4.3 增强健壮性：在关键解引用处加入 hard assert / defensive check

例如：
- `shader_core_config::max_cta()` 检查 `kernel != NULL`
- `ptx_sim_kernel_info()` 检查 `kernel != NULL`

注意：这只能把“随机崩溃”变成“可读的报错”，不能解决根因。

## 5. 验收标准（Definition of Done）

- `./traceL1 --plot-background bc_ima_med` 不再出现 `Segmentation fault`。
- 同目录下其他 workload（例如 `bfs_ima_med`/`sssp_ima_med`/`cc_ima_med`/`spmv_ima_high`）不引入新的崩溃。
- 若采用“延后释放”方案：仿真结束后无内存泄漏（至少 kernel 对象最终被释放）。

## 6. 附录：与 core dump 相关的注意事项

- 容器内 `ulimit -c` 为 `unlimited`，但 `core_pattern` 可能是管道（例如 `|/usr/bin/coredump_handler ...`），core 文件不一定落在当前目录。
- 如果需要核心转储文件：
  - 查看 `/proc/sys/kernel/core_pattern`
  - 或直接用 `gdb` attach 到崩溃进程（需要拿到 PID）

