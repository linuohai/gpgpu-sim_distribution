// GRASP Tracer — Structured CSV event log for GRASP prefetcher analysis.
//
// Follows the same static-class/per-SM-buffer pattern as l1_tracer.
// Enabled via: -grasp_trace_enable 1 -grasp_trace_path <path>
//
// CSV schema: cycle,sm_id,warp_id,event,pc,addr,detail
//
// Event types (see grasp_prefetcher.cc for emit call sites):
//   CHAIN_DETECT   — CD detected an IMA chain (idx_pc→data_pc)
//   CT_INSERT      — New CT entry created for an index PC
//   DEMAND_CT_HIT  — Demand load PC matched a CT entry
//   STRIDE_UPDATE  — CT stride learning step (delta computed)
//   STRIDE_CONVERGE— CT entry stride just became valid
//   IDX_PF_ENQUEUE — INDEX prefetch enqueued after stride confirmed
//   IDX_PF_INJECT  — INDEX prefetch mem_fetch sent to L1D
//   DATA_PF_INJECT — DATA prefetch mem_fetch sent to L1D
//   DATA_PF_ENQUEUE— DATA prefetch enqueued after INDEX_PF fill
//   L1_RESULT_HIT  — Prefetch L1 access result: HIT
//   L1_RESULT_MISS — Prefetch L1 access result: MISS/SECTOR_MISS
//   L1_RESULT_MSHR — Prefetch L1 access result: MSHR_HIT/HIT_RESERVED
//   L1_RESULT_RFAIL— Prefetch L1 access result: RESERVATION_FAIL
//   FILL_DISPATCH  — INDEX_PF fill triggered DATA_PF generation

#pragma once

#include <cstdint>
#include <string>
#include <vector>

class grasp_tracer {
 public:
  static void init(bool enable, const char *path, unsigned n_sms);

  // Emit one event row. pc and addr are hex-formatted; detail is a
  // semicolon-separated key=value string specific to each event type.
  static void emit(unsigned sm_id, unsigned warp_id, unsigned long long cycle,
                   const char *event, unsigned long long pc,
                   unsigned long long addr, const char *detail);

  static void flush_all();
  static bool enabled() { return s_enabled; }

 private:
  static void flush_sid(unsigned sid);

  static bool s_enabled;
  static std::string s_path;
  static std::vector<std::string> s_buffers;
  static size_t s_flush_threshold;
};
