# Ideal / Perfect L1D Cache 说明（`-gpgpu_perfect_l1d`）

本文档用于说明本仓库中 “Perfect L1D（理想 L1 数据缓存）” 的**真实实现语义**与**代码改动点**。  
目标是做上界实验：在不改变 functional correctness 的前提下，让所有目标访存都走 **L1D HIT 的 timing path**，并且（按你选择的 “B 方案”）**保留 write-through store 的写流量与 WRITE_ACK 回包路径**，从而还能看到写拥塞对系统的影响。

> 重要前提：GPGPU-Sim 的时序 cache 用 `mem_fetch` 建模延迟/队列/带宽，并不搬运真实 data payload；真实内存值由 functional 模型维护。Perfect L1D 只改变 timing path，不改变计算结果。

---

## 1. 如何开启

两种等价方式：

1) 在 `gpgpusim.config` 中设置：
- `-gpgpu_perfect_l1d 1`

2) 用本仓库脚本（推荐做 trace 对比时用）：
- `./traceL1 --ideal-l1d <app> <tag>`
  - 该脚本会在临时 config 里写入 `-gpgpu_perfect_l1d 1` 并自动重命名输出的 CSV。

---

## 2. 覆盖范围（什么会被强制 HIT）

当 `-gpgpu_perfect_l1d=1` 且该请求**实际走 L1D**（没有被 bypass）时：

- **强制 HIT 的访问类型**（`mem_access_type`）：
  - `GLOBAL_ACC_R / GLOBAL_ACC_W`
  - `LOCAL_ACC_R / LOCAL_ACC_W`
- **排除**：
  - **atomic**：`mf->isatomic()`（atomic 的执行点/序列化语义依赖 memory subsystem；强制 hit 会破坏原模型含义）
  - **bypass L1D 的访问**：例如 `-gpgpu_gmem_skip_L1D` 或指令级缓存策略导致的 bypass（这类请求根本不会进入 `l1_cache::access()`，因此 Perfect L1D 不会“强行覆盖”它）

换句话说：Perfect L1D 只在 “已经决定走 L1D 的 global/local 非 atomic 访存” 上生效。

---

## 3. 行为细节：Load 与 Store（为什么 store HIT 仍然会进 miss_queue）

### 3.1 Load（读）

- 直接走 **HIT path**：返回 `HIT`，不再向下级发 `READ_REQUEST_SENT`（因此 demand read 不会注入到 L2/NoC/DRAM）。
- **仍保留 L1D 自身结构约束**：继续调用 `use_data_port()`，所以 L1 的 data port/带宽限制仍会体现（不会把 “L1 自身瓶颈” 一并抹掉）。
- **统计与 trace 仍会记录为 HIT**：用于保持现有脚本/CSV 分析的一致性。

### 3.2 Store（写，B 方案：保留 write-through traffic + ACK）

Store 也会强制 HIT，但“命中”≠“不产生写流量”。  
在本实现里：**当写策略需要向下层写**时，仍然会发出写请求，以保留：

- 写流量对 NoC/L2/DRAM 的拥塞影响
- WRITE_ACK 回包路径（warp/ldst 完成时序依旧由 ACK 驱动，而不是被“瞬间完成”）

#### 哪些写策略会向下层写

满足任一条件就会发下层写请求：

- `WRITE_THROUGH`
- `WRITE_EVICT`
- `LOCAL_WB_GLOBAL_WT` 且 access_type 为 `GLOBAL_ACC_W`（global write-through，local write-back）

#### 具体怎么“保留写流量/ACK”

实现方式等价于走原本的 write-through hit 流程：把该 store 的 `mem_fetch` 通过 `send_write_request()` 推入 L1 的 `m_miss_queue`，触发事件 `WRITE_REQUEST_SENT`，从而进入下层并最终收到 ACK。

> 这也解释了你之前的疑问：把 `mf` 放进 L1 的 `miss_queue`（通过 `send_write_request`）在时序上就等价于 `wr_hit_wt()` 的关键效果——**它不是在“模拟 miss”，而是在“模拟 write-through 的下层写事务与 ACK 归还”**。

#### Perfect 也可能 stall（这是 B 方案的“真实代价”）

如果 `miss_queue_full(0)`，该 store 会返回 `RESERVATION_FAIL`。这不是 bug，而是 B 方案刻意保留的现象：  
你把读变成 100% hit，但写仍会制造下层事务，所以仍可能因为写拥塞/队列满而在 core/L1 处出现 backpressure。

---

## 4. tag/dirty/byte-mask：为什么在 perfect 下“不更新也没关系”

Perfect L1D 的实现是在 `l1_cache::access()` 入口短路：不再 probe tag array，也不会做 allocate/evict、也不会更新 dirty/byte-mask。  
在 “所有目标访问都被强制 HIT” 的前提下，tag 状态不会再影响后续命中率，因此这些元数据更新对 “上界 timing” 不再是必须项。

注意：这也意味着 Perfect L1D **不再代表容量有限、会冲突/会驱逐的真实缓存**；它代表的是 “命中率上界 + 仍保留部分结构/流量约束（尤其是 store 的写拥塞/ACK）”。

---

## 5. 源码与时序讲解（从 SM 到 L1D、以及 Perfect 的差异）

### 5.1 一条访存指令从 SM 到 L1D：真实调用链（读写都适用）

下面用 **“一条 warp 的 global load/store 指令”** 举例，解释它在 GPGPU-Sim 里是怎么走到 L1D 的。  
（注意：这里讲的是 **timing path**，不是 functional 读写真实数据；真实数据在 `ptx_exec_inst()` 的 functional 层已经更新/读取。）

#### Step 0：warp 指令里先形成 access queue（coalesced 事务）

一条 `LD`/`ST` 指令进入 `ldst_unit` 之前，会根据 active mask、地址等形成一个或多个 `mem_access_t`，存到 `warp_inst_t::accessq` 中（每个 `mem_access_t` 可以理解成一次合并后的 memory transaction）。

你在 `result/L1cache_trace/*_l1.csv` 里看到的一行（例如 `op=LD space=GLOBAL address=...`），本质上就是这些 access 的时序观测点之一。

#### Step 1：决定是否 bypass L1D

在 `ldst_unit::memory_cycle()` 会根据 cache_op、`-gpgpu_gmem_skip_L1D` 等判定 `bypassL1D`：

- `bypassL1D=true`：直接把 `mem_fetch` 推进 interconnect（`m_icnt->push(mf)`），完全绕过 L1D。
- `bypassL1D=false`：走 `process_memory_access_queue_l1cache(m_L1D, inst)`，进入 L1D。

因此 Perfect L1D **不会影响 bypass 的访问**，因为 bypass 根本不会调用到 `l1_cache::access()`。

#### Step 2：从 accessq 生成 mem_fetch，并调用 L1D access

走 L1D 的情况下（`shader.cc::ldst_unit::memory_cycle()`），会进入 `process_memory_access_queue_l1cache(m_L1D, inst)`，然后根据 `-gpgpu_l1_latency` 分两种：

- `-gpgpu_l1_latency > 0`：先把 `mem_fetch` 放进每个 bank 对应的 `l1_latency_queue`，等延迟走完后由 `L1_latency_queue_cycle()` 取出并调用 `m_L1D->access(...)`
- `-gpgpu_l1_latency == 0`：直接在当前 cycle 调用 `m_L1D->access(...)`

不管是哪种，本质流程都是：

1) 从 `inst.accessq_back()` 取一个 `mem_access_t`
2) `m_mf_allocator->alloc(...)` 生成一个 `mem_fetch *mf`
3) 在合适的时机调用 `m_L1D->access(mf->get_addr(), mf, now, events)`
4) 根据返回的 `status` 和 `events` 决定：
   - 这次访问是否“立刻完成”（HIT）
   - 还是要进入 miss/MSHR/下层（MISS / HIT_RESERVED）
   - 或者因为端口/队列满而 backpressure（RESERVATION_FAIL）

这里你可以把 `mem_fetch` 理解成：“时序模型里的一次访存事务对象”，它会在 L1/L2/DRAM/NoC 之间流动，并在合适的时机变成 `READ_REPLY` / `WRITE_ACK` 回来。

---

### 5.2 普通 L1D：HIT / MISS 时分别发生什么（用例子讲清楚）

#### 例子 A：一条 global load（`LD.GLOBAL`）

**A1) 如果 L1 HIT：**

- `l1_cache::access()` 最终返回 `HIT`。
- `ldst_unit` 这边会把对应 access 从 `accessq` pop 掉。
- 对 load 而言，会减少 `m_pending_writes`，当对应寄存器的 pending 计数归零时释放 scoreboard，使 warp 可以继续执行后续依赖指令。
- 不会产生下层读请求（不会有 `READ_REQUEST_SENT` 事件）。

**A2) 如果 L1 MISS：**

- `l1_cache::access()` 会走 tag probe + miss 处理，通常会产生一次下层读事务（`READ_REQUEST_SENT`），并返回 `MISS` 或 `HIT_RESERVED`。
- `ldst_unit` 同样会 pop 掉当前 access（表示“这个 access 已经变成一个正在进行的 mem_fetch”）。
- 但 load 的寄存器依赖不会立刻释放：warp 会一直等到这个 mem_fetch 作为 `READ_REPLY` 回来，并最终被填入 L1D（`m_L1D->fill(...)`）之后，才会释放 scoreboard/pending writes。

直观理解：MISS 就是 “accessq 里的这次访问被转换成了一个要去 L2/DRAM 排队的事务”，warp 的依赖要等它回来。

一个更“时间线”的说法（简化版）：

1) L1D miss：`READ_REQUEST_SENT`（请求出发）
2) NoC/L2/DRAM：排队、服务
3) `READ_REPLY` 回到 SM：进入 `ldst_unit` response fifo
4) 若不 bypass：`m_L1D->fill(...)`（fill 到 L1D）
5) load 依赖释放：scoreboard/pending writes 清零，warp 才能继续跑依赖指令

#### 例子 B：一条 global store（`ST.GLOBAL`）

store 比 load 更绕一点：**“L1 命中”不等于“store 立即完成”**，因为是否需要下层写、是否需要 ACK，取决于写策略。

**B1) write-through（WT）：**

- 不管 L1 命不命中，都会向下层发送 `WRITE_REQUEST_SENT`（形成写流量），并等待 `WRITE_ACK` 回来后，warp 才认为 store “完成”（`store_ack()` 递减 outstanding store 计数）。
- 源码角度看就是：`WRITE_REQUEST_SENT` → `inc_store_req()`（记录“还欠几个 ACK”）→ `WRITE_ACK` 回来后 `store_ack()`（把欠的 ACK 还掉）。

**B2) write-back（WB）：**

- 如果是 L1 hit：通常不需要下层写请求，store 可以在 L1 内部完成（不等待 `WRITE_ACK`）。
- 如果是 L1 miss：还要看 write-allocate 策略，可能需要先 fetch line（读下层）再写，或者直接写 evict 等。

---

### 5.3 Perfect L1D：我们到底改了什么（以及 miss→hit 后的差异）

Perfect L1D 的核心改动点在 `l1_cache::access(...)`：当 `-gpgpu_perfect_l1d=1` 且访问满足 “global/local + 非 atomic + 非 bypass” 时，我们在函数入口把这次访问 **短路成 HIT**。

把它和上面的普通路径对照起来看：

- 对 **load**：原来 MISS 会产生 `READ_REQUEST_SENT`，warp 要等 `READ_REPLY` 才释放依赖；现在直接 HIT，不再向下层发读事务，依赖在 L1 hit timing 下释放。
- 对 **store（B 方案）**：我们仍然保留 “需要下层写时” 的写事务与 ACK：
  - store 自己在 L1 侧仍然算 HIT（你会在 L1 trace 里看到 `HIT`）
  - 但当写策略要求向下层写时，仍会 `send_write_request()` 把该 `mem_fetch` 推入 `m_miss_queue`，触发 `WRITE_REQUEST_SENT`，最终收到 `WRITE_ACK`

因此你能实现一个“完美 L1 命中率”，但依然能观察到 **写流量引起的拥塞/backpressure**（例如 miss_queue full → `RESERVATION_FAIL`）。

补充：在代码里我们同时强制了 `probe_status=HIT` 与 `status=HIT`（除非 miss_queue 满导致 `status=RESERVATION_FAIL`）。  
这样做的目的是：

- `probe_status` 代表 “tag probe 的结果”（我们在 perfect 模式里直接替代它）
- `status` 代表 “最终能不能完成”（仍可能被资源约束卡住）

并且我们仍然调用 `use_data_port()` + 更新 stats/tracer，使 “命中但仍受结构约束” 的含义成立（而不是把整个 L1 子系统完全抹掉）。

---

### 5.4 为什么这种改法有用？会不会影响 cache 的其它部分（例如 tag）？

#### 为什么有用（研究意义）

这是一种非常“干净”的上界实验：

- 把 **demand read miss** 从系统里拿掉（不再注入到 L2/DRAM/NoC），你就能回答：
  - 这个 workload 的瓶颈到底是不是 L1D miss latency？
  - 如果 L1D 对这些访问都 hit，IPC/STALL 能提升到什么程度？
- 同时在 B 方案下保留 **WT store 的写事务与 ACK**，你还能回答：
  - 即使读都命中，写流量是否仍然是瓶颈？是否存在明显的 write-induced backpressure？

#### 为什么不会“破坏其它 cache 部分”

Perfect L1D 的设计目标不是“构造一个更真实的缓存”，而是“把命中率当成上界输入”。因此它天然会绕开一些真实缓存机制（tag/evict/dirty/byte-mask），但这在本模式下是可接受且可控的：

- **tag/dirty/byte-mask 不更新**：在 perfect 模式里我们不再依赖 tag 来决定命中率（永远 HIT），因此这些元数据对后续 timing 决策不再必要。
- **functional correctness 不受影响**：真实内存值由 functional 层维护；Perfect L1D 只是影响 “何时释放依赖/是否排队去下层”。
- **不影响非目标访问**：atomic 与 bypass 的访问仍走原来的时序路径（Perfect 不会覆盖它们）。
- **仍保留关键结构约束**：
  - 对命中访问仍调用 `use_data_port()`，保留 L1 data port/带宽约束
  - 对需要下层写的 store 仍进入 `m_miss_queue`，保留写拥塞与 `WRITE_ACK` 回包链

这也意味着一个明确的取舍：Perfect L1D **不再表示容量/冲突/替换的真实行为**，而是 “命中率上界 +（可选的）结构/拥塞约束保留”。

---

## 6. 代码改动点（对启用 Perfect L1D 的改动总结）

### 6.1 GPGPU-Sim：L1D access 强制 HIT（含 store write-through traffic/ACK 保留）

- `gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.cc`
  - `l1_cache::access(...)` 增加 Perfect L1D 短路逻辑：
    - eligible global/local 非 atomic：强制 `probe_status=HIT`、`status=HIT`
    - store 且写策略需要下层写：`send_write_request()` 入 `m_miss_queue`，保留 `WRITE_REQUEST_SENT` → 下层 → `WRITE_ACK` 的时序链
    - 保留 `use_data_port()`、stats、L1 trace 的一致性

> 说明：`-gpgpu_perfect_l1d` 选项在原始代码中就已经存在（在 `gpu-sim.cc` 注册，并存入 `shader.h` 的 `shader_core_config`），本次没有新增配置项，只是把 “eligible global reads” 的语义扩展成 “eligible global/local load/store（非 atomic）” 并补齐 store 的 B 方案行为。

### 6.2 Accel-Sim（与 Perfect L1D 无关，但跑 trace 时更稳定）

- `gpu-simulator/accel-sim.cc` / `gpu-simulator/accel-sim.h`
  - 调整 trace kernel 的释放时机：把已完成 kernel 放入 `retired_kernels`，统一在析构时释放，避免 cleanup 后续潜在的 use-after-free。

---

## 7. 验证：用 btree 检查 L1 请求都 HIT（不含 bypass/atomic）

示例（会生成 `result/L1cache_trace/<tag>_l1.csv`）：

1) 运行：
- `./traceL1 --ideal-l1d btree btree_ideal_l1d_allhit`

2) 检查 `l1_status` 列（CSV 第 8 列）是否全为 `HIT`：
- `awk -F, 'NR==1{next} $8!=\"HIT\"{bad++} END{print bad+0}' result/L1cache_trace/btree_ideal_l1d_allhit_l1.csv`
  - 期望输出：`0`

补充说明：btree trace 里通常只有 `space=GLOBAL`。如果你还想验证 `LOCAL_ACC_R/W` 路径，需选择/构造一个确实产生 local memory 访存的 workload（否则 CSV 里不会出现 `space=LOCAL`）。
