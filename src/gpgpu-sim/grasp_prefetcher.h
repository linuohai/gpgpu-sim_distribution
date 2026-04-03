// GRASP (GPU Register-chain Aware Sector Prefetcher) — Top-level header
//
// Owns all GRASP sub-components: CD (Chain Detector), CT (Chain Table),
// TT (Target Table), IST (Iteration Stride Tracker), PRB (Prefetch Request
// Buffer).  Provides hooks called from ldst_unit / shader_core_ctx.
//
// Architecture overview:
//   Training path:  issue_warp() → CD detects LDG→IMAD.WIDE→LDG chains → CT/TT
//   Prefetch path:  demand_load → IST stride → index PF → pair table → PRB →
//                   fill → data PF
//
// This file concentrates all struct definitions used across GRASP components.

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "../abstract_hardware_model.h"
#include "gpu-cache.h"  // cache_reservation_fail_reason, NUM_CACHE_RESERVATION_FAIL_STATUS
#include "grasp_chain_detector.h"
#include "grasp_tables.h"

// Forward declarations
class data_cache;
class gpgpu_context;
class l1_cache;
class mem_fetch;
class mem_fetch_allocator;
class memory_config;
class shd_warp_t;
struct ima_prefetch_candidate_t;

// ============================================================================
// Configuration
// ============================================================================

struct grasp_config_t {
  bool enable = false;
  bool debug = false;

  // CD
  unsigned cd_fifo_depth = 20;  // SpMV ×16 needs 17+

  // CT / TT
  unsigned ct_size = 32;  // SpMV 30+ PC sizing
  unsigned tt_size = 8;

  // IST (Iteration Stride Tracker)
  unsigned ist_ipt_size = 64;
  unsigned ist_distance = 1;
  unsigned ist_confidence = 2;

  // PRB
  unsigned prb_capacity = 1024;  // Phase 1: large capacity probe

  // Pair Table (trace-driven)
  char *chain_csv = nullptr;

  // Throttle Control
  unsigned tc_mode = 0;             // 0=legacy, 1=dual-thr, 2=queue-cap, 3=acc-gate, 4=cooldown
  unsigned tc_mshr_threshold = 80;  // MSHR occupancy %, suppress data PF above
  unsigned tc_mshr_lo = 50;         // S1/S3: lower MSHR threshold %
  unsigned tc_mshr_hi = 90;         // S1/S3: upper MSHR threshold %
  unsigned tc_queue_cap = 0;        // S2: max data PFs in queue (0=disabled)
  unsigned tc_cooldown_cycles = 0;  // S4: cooldown duration after trigger
  unsigned tc_acc_lo = 30;          // S3: accuracy % for tight throttle
  unsigned tc_acc_hi = 60;          // S3: accuracy % for loose throttle

  // Pair Table scope (trace-driven)
  unsigned pair_table_scope = 0;  // 0=per-warp, 1=per-CTA, 2=per-kernel

  // Speculative stride: assumed stride on first observation (0=disabled)
  int speculative_stride = 0;
};

// ============================================================================
// Prefetch Request
// ============================================================================

struct grasp_prefetch_request_t {
  new_addr_type addr = 0;                // sector address (32B-aligned for INDEX_PF)
  unsigned warp_id = (unsigned)-1;
  std::vector<unsigned> seed_chain_ids;   // non-empty = INDEX_PF, empty = DATA_PF
  unsigned long long ready_cycle = 0;
  int prb_entry_id = -1;                 // >=0: pre-allocated shared PRB (v7 P1)
  std::vector<new_addr_type> lane_addrs;  // v7 P2: specific predicted addresses in this sector
};

// ============================================================================
// PRB (Prefetch Request Buffer)
// ============================================================================

struct grasp_prb_entry_t {
  bool valid = false;
  unsigned warp_id = (unsigned)-1;
  // Snapshot of CT's tt_idx[] and num_targets (decoupled from CT lifetime)
  unsigned tt_idx[3] = {(unsigned)-1, (unsigned)-1, (unsigned)-1};
  unsigned num_targets = 0;
  // v7 P1: per-sector candidate storage (replaces flat candidates vector).
  // Each sector's candidates are dispatched independently as its fill arrives.
  // dispatched flag provides idempotency for the writeback path (P3).
  struct sector_data_t {
    new_addr_type sector_addr = 0;
    std::vector<ima_prefetch_candidate_t> candidates;
    bool dispatched = false;
  };
  std::vector<sector_data_t> sector_data;
  // Sector tracking
  unsigned remaining_sectors = 0;
  unsigned long long alloc_cycle = 0;  // for lifetime stats
};

class grasp_prb_t {
 public:
  explicit grasp_prb_t(unsigned initial_capacity);

  // v7 P1: allocate without candidates (filled per-sector at inject time)
  int allocate(unsigned warp_id, unsigned num_sectors,
               unsigned long long cycle);

  // v7 P1: add candidates for a specific sector to an existing entry
  void add_sector_candidates(unsigned prb_entry_id, new_addr_type sector_addr,
                             std::vector<ima_prefetch_candidate_t> &&cands);

  // Free entry
  void free_entry(unsigned prb_entry_id);

  // Get entry (with bounds check)
  grasp_prb_entry_t &get(unsigned prb_entry_id);
  const grasp_prb_entry_t &get(unsigned prb_entry_id) const;

  // Kernel launch: clear all
  void reset();

  // Monitoring
  struct prb_stats_t {
    unsigned long long peak_occupancy = 0;
    unsigned long long current_occupancy = 0;
    unsigned long long full_stall_cycles = 0;
    unsigned long long total_allocations = 0;
    unsigned long long total_lifetime_cycles = 0;  // cumulative entry lifetime
  };
  const prb_stats_t &stats() const { return m_stats; }

 private:
  std::vector<grasp_prb_entry_t> m_entries;
  prb_stats_t m_stats;
};

// ============================================================================
// IST (Iteration Stride Tracker — replaces ima_prefetcher_t)
// ============================================================================

class grasp_ist_t {
 public:
  grasp_ist_t(unsigned sm_id, unsigned ipt_size, unsigned distance,
              unsigned conf_thresh);

  // Train stride on demand load.
  // Returns: if confidence >= threshold, predicted prefetch address list.
  std::vector<new_addr_type> on_load_access(new_addr_type pc,
                                             new_addr_type addr,
                                             unsigned long long cycle);

  void reset();
  void print_stats(FILE *fp) const;

  // Stats (externally updatable)
  unsigned long long stat_prefetch_generated = 0;
  unsigned long long stat_prefetch_issued = 0;
  unsigned long long stat_prefetch_filtered = 0;

 private:
  struct ipt_entry_t {
    bool valid = false;
    new_addr_type pc = 0;
    new_addr_type last_addr = 0;
    int64_t stride = 0;
    unsigned confidence = 0;  // 2-bit saturating (0..3)
    unsigned long long last_access_time = 0;
    static constexpr unsigned MAX_CONFIDENCE = 3;
  };

  void ipt_update(ipt_entry_t &entry, new_addr_type new_addr);
  int find_entry(new_addr_type pc) const;
  int find_lru_victim() const;

  unsigned m_sm_id;
  unsigned m_ipt_size, m_distance, m_conf_thresh;
  std::vector<ipt_entry_t> m_ipt;
  unsigned long long m_clock = 0;
};

// ============================================================================
// Comprehensive Stats
// ============================================================================

struct grasp_stats_t {
  // IST
  unsigned long long ist_prefetch_generated = 0;
  unsigned long long ist_prefetch_issued = 0;
  unsigned long long ist_prefetch_filtered = 0;

  // Index Prefetch
  unsigned long long index_pf_issued = 0;
  unsigned long long index_pf_hit = 0;
  unsigned long long index_pf_miss = 0;
  unsigned long long index_pf_mshr_merge = 0;
  unsigned long long index_pf_reservation_fail = 0;  // total (sum of per-reason)
  unsigned long long index_pf_rfail[NUM_CACHE_RESERVATION_FAIL_STATUS] = {};

  // Data Prefetch
  unsigned long long data_pf_issued = 0;
  unsigned long long data_pf_hit = 0;
  unsigned long long data_pf_miss = 0;
  unsigned long long data_pf_mshr_merge = 0;
  unsigned long long data_pf_reservation_fail = 0;  // total (sum of per-reason)
  unsigned long long data_pf_rfail[NUM_CACHE_RESERVATION_FAIL_STATUS] = {};
  unsigned long long data_pf_enqueued = 0;  // data PF candidates queued after index fill

  // Pair Table
  unsigned long long pair_table_lookup_hit = 0;
  unsigned long long pair_table_lookup_miss = 0;

  // Throttle
  unsigned long long throttle_suppressed = 0;

  // Lifecycle
  unsigned long long kernel_launches = 0;
  unsigned long long warp_exits = 0;
  unsigned long long training_frozen_cycles = 0;

  // Prefetch Queue
  unsigned long long queue_peak_depth = 0;

  // Demand load tracking (all global loads)
  unsigned long long total_demand_global_reads = 0;
  unsigned long long total_demand_global_misses = 0;

  // IMA-specific demand load tracking (only loads whose PC is in CT)
  unsigned long long ima_demand_reads = 0;
  unsigned long long ima_demand_misses = 0;
};

// ============================================================================
// Top-level GRASP Prefetcher (per-SM)
// ============================================================================

class grasp_prefetcher_t {
 public:
  // === Lifecycle ===
  grasp_prefetcher_t(unsigned sm_id, const grasp_config_t &cfg);
  ~grasp_prefetcher_t();

  // Kernel switch: clear or partially preserve state
  void on_kernel_launch(const std::string &kernel_name);

  // Warp exit: clean tracked warp state
  void on_warp_exit(unsigned warp_id);

  // === Training path (CD + CT/TT) ===

  // Called in issue_warp() for every issued instruction.
  // CD detects LDG→IMAD.WIDE→LDG register dependency chains, updates CT/TT.
  void on_instruction_issue(unsigned warp_id, const warp_inst_t &inst,
                            const std::string &sass_opcode,
                            unsigned long long cycle);

  // === Prefetch generation path (IST + IPU) ===

  // Called in L1_latency_queue_cycle() on global read path.
  // Updates stride IPT; if stride learned and CT entry exists, generates
  // index prefetch requests.
  void on_demand_load(unsigned warp_id, new_addr_type pc, new_addr_type addr,
                      unsigned long long cycle, shd_warp_t *warp,
                      const mem_fetch *mf = nullptr,
                      int l1_status = -1);

  // === Prefetch injection (IPU + DPU) ===

  // Called every cycle in ldst_unit::cycle().
  // Drains prefetch queue, issues to L1D.
  struct prefetch_result_t {
    std::vector<mem_fetch *> requests;  // mem_fetch to send to L1D
    bool stalled;                       // queue non-empty but can't issue
  };
  prefetch_result_t inject_prefetch(
      unsigned long long cycle, l1_cache *l1d,
      mem_fetch_allocator *mf_alloc, unsigned sid, unsigned tpc,
      const memory_config *mem_cfg, unsigned cache_line_sz,
      gpgpu_context *gpgpu_ctx,
      const std::function<shd_warp_t *(unsigned)> &get_warp);

  // === Fill callback (DPU) ===

  // Called in ldst_unit::cycle() response_fifo processing.
  // INDEX_PF: release PRB targets → re-enqueue as DATA_PF.
  // DATA_PF: update stats only.
  void on_fill(mem_fetch *mf, unsigned long long fill_cycle);

  // === L1 access result tracking ===
  void on_l1_access_result(mem_fetch *mf, int cache_status,
                           unsigned long long cycle,
                           enum cache_reservation_fail_reason fail_reason =
                               LINE_ALLOC_FAIL);

  // === Stats ===
  void print_config(FILE *fp) const;
  struct ima_demand_breakdown_t {
    unsigned long long idx_reads = 0, idx_hits = 0, idx_hit_reserved = 0, idx_misses = 0;
    unsigned long long data_reads = 0, data_hits = 0, data_hit_reserved = 0, data_misses = 0;
  };
  void print_stats(FILE *fp, unsigned long long pf_useful = 0,
                   unsigned long long pf_useless = 0,
                   unsigned long long pf_late = 0,
                   const ima_demand_breakdown_t *ima = nullptr) const;

  // Accessors
  bool enabled() const { return m_cfg.enable; }
  bool debug() const { return m_cfg.debug; }
  const grasp_stats_t &stats() const { return m_stats; }
  grasp_stats_t &stats_mut() { return m_stats; }
  void set_l1d(l1_cache *l1d) { m_l1d = l1d; }

 private:
  unsigned m_sm_id;
  grasp_config_t m_cfg;

  // Sub-components
  grasp_chain_detector_t m_cd;   // Chain Detector
  grasp_chain_table_t m_ct;      // Chain Table
  grasp_target_table_t m_tt;     // Target Table
  grasp_ist_t m_ist;             // Iteration Stride Tracker
  grasp_prb_t m_prb;             // Prefetch Request Buffer
  std::deque<grasp_prefetch_request_t> m_prefetch_queue;

  // Stats
  grasp_stats_t m_stats;

  // Per-warp instruction dedup for on_demand_load (avoids cross-warp
  // interleaving overwriting shared CT entry fields).
  struct warp_dedup_t {
    int ct_idx = -1;
    unsigned long long inst_uid = 0;
  };
  std::vector<warp_dedup_t> m_demand_dedup;  // indexed by warp_id

  // Per-index-PC speculative stride hints (from chain CSV stride_hint column)
  std::unordered_map<new_addr_type, int64_t> m_stride_hints;
  void load_stride_hints(const char *csv_path);

  // L1D cache pointer (set via set_l1d)
  l1_cache *m_l1d = nullptr;

  // Cross-kernel CT persistence
  std::string m_last_kernel_name;

  // Throttle Control runtime state (S4: cooldown)
  unsigned long long m_tc_cooldown_until = 0;

  // Helpers
  void queue_prefetch(new_addr_type addr, unsigned warp_id,
                      const std::vector<unsigned> &seed_chain_ids,
                      unsigned long long ready_cycle,
                      int prb_entry_id = -1,
                      const std::vector<new_addr_type> &lane_addrs = {});
  int find_ready_prefetch(unsigned long long cycle) const;
  // v7 P1: per-sector dispatch (replaces release_prb_targets)
  void release_prb_sector_targets(unsigned prb_entry_id,
                                  new_addr_type sector_addr,
                                  unsigned long long ready_cycle);
  // v7: mark sector as failed (RFAIL) — dispatched=true, remaining--, no data PF
  void mark_sector_failed(unsigned prb_entry_id, new_addr_type sector_addr);
};
