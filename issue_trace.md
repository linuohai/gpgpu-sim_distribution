# Issue Trace And Stall Attribution

The issue tracer captures every warp that leaves the SIMT scheduler as well as
every cycle where a scheduler fails to issue.  The output is a CSV stream that
mirrors the per-instruction L1 tracer (`l1_trace.md`), but focuses on the
instruction stream:

```
cycle,sm_id,event,warp_id,mask,opcode,pc,reason,detail,scheduler_id
```

* `event=ISSUE` records a successfully issued warp.  `mask` is the active lane
  bitmap, `opcode` is the PTX/SASS mnemonic (first token of the PTX string), and
  `scheduler_id` is the local scheduler that selected the warp.
* `event=STALL` records a cycle where that scheduler could not issue anything.
  `reason` is a short code (listed below) and `detail` includes a concrete
  example such as `warp3 PC=0x1c LDG.E.64 waiting on outstanding memory op`.

## Enabling the trace

```
-issue_trace_enable        1            # default 0
-issue_trace_path          "/path/to/issue_trace.csv"
```

These options mirror the existing `-l1_trace_*` flags and live in both the
option parser (`gpu-sim.cc`) and the reference configs (for example
`configs/tested-cfgs/SM7_QV100/gpgpusim.config`).  When the trace is disabled the
instrumentation short-circuits inside `scheduler_unit::cycle()` so there is no
string formatting or extra bookkeeping cost.

## CSV layout

| Column        | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| `cycle`       | `gpu_tot_sim_cycle + gpu_sim_cycle` when the event occurred                 |
| `sm_id`       | Owning SM ID                                                                |
| `event`       | `ISSUE` or `STALL`                                                          |
| `warp_id`     | Warp that issued (for `ISSUE`) or the sample warp that illustrates the stall |
| `mask`        | Active-lane bitmask (hex). `STALL` rows include a representative mask when available |
| `opcode`      | PTX mnemonic extracted from the PTX string at the warp’s PC                 |
| `pc`          | Program counter in hex                                                      |
| `reason`      | Stall category (see below). `NA` for issue rows                             |
| `detail`      | Human-readable explanation, including an example warp and PC                |
| `scheduler_id`| Local scheduler that attempted to issue                                     |

Every SM buffer maintains its own write-back queue and flushes in large chunks
(`1 << 15` bytes by default), so very long runs do not explode the log.

## Stall attribution

Each scheduler now keeps a light-weight `stall_tracker` while it walks the
prioritized warp list (`shader.cc:1259+`).  The tracker records the first warp
that encountered each blocking condition.  When the scheduler ultimately fails
to issue, that data is converted into a single `STALL` row with a precise
reason + example.

| Reason          | Detection logic (file reference)                                                                                   | Example `detail`                                                     |
|-----------------|--------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| `WAIT_BARRIER`  | All candidate warps report `shd_warp_t::waiting()` (barrier, membar or depbar) before a valid instruction exists.<br/>See `scheduler_unit::cycle()` barrier handling in `shader.cc:1313-1331`. | `warp2 PC=0x14 barrier.sync waiting on barrier/membar`               |
| `CONTROL_HAZARD`| The SIMT stack redirects a warp (PC mismatch) and its i-buffer is flushed (`shader.cc:1332-1343`).                  | `warp7 PC=0x80 BRA control hazard`                                   |
| `IBUFFER_EMPTY` | No warp had a decoded instruction ready because its instruction buffer ran dry (`shader.cc:1310-1322`).            | `warp5 PC=0x40 instruction buffer empty`                             |
| `NO_VALID`      | Fallback when no warp could present a valid instruction but none of the above sub-reasons triggered.               | `no warp had a valid instruction`                                    |
| `WAIT_MEM`      | Scoreboard collision on a register that still carries a “long-op” (global/local/TEX) dependency.<br/>The tracker queries `Scoreboard::has_pending_longop()` when `checkCollision()` fails (`shader.cc:1508-1517`). | `warp0 PC=0x1c LDG.E.64 waiting on outstanding memory op`           |
| `WAIT_DEP`      | Scoreboard collision on a short dependency (e.g., integer ALU) – same site as above but `has_pending_longop()` is false. | `warp4 PC=0xa0 IADD waiting on register dependency`                  |
| `PIPE_BUSY`     | At least one warp was scoreboard-ready but the target pipeline register set had no free slot (LD/ST, SP, INT, DP, SFU, Tensor, or a specialized unit).  Each branch records the first missing resource (e.g. `ldst pipe busy` or `tensor pipe busy`) – search for `record_pipe_busy` in `shader.cc`. | `warp1 PC=0x68 FFMA sp pipe busy`                                    |

A `STALL` row therefore reads like “Cycle X: SM 0 scheduler 1 issued nothing
because all candidate warps were waiting on outstanding memory, e.g. warp0
PC=0x1c LDG.E.64 waiting on outstanding memory op”, which matches the example
from the original request.

## Relationship to stats

The existing `shader_cycle_distro` counters are still updated exactly as before.
The issue trace simply provides the per-cycle forensic view:

* `ISSUE` rows can be counted per warp, per opcode, or per scheduler to build
  throughput charts.
* `STALL` rows can be filtered by reason to understand whether an SM is limited
  by instruction fetch, dependence chains, or pipeline occupancy.

Because all entries share a cycle-number and SM ID, downstream tooling can
correlate the issue log with L1 traces, dram queues, or any other CSV artifact
without needing to instrument the core again.
