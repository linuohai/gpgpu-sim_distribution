#pragma once

#include <cstdint>
#include <vector>

#include "baseline_prefetcher.h"

class baseline_cache;

struct baseline_snake_config_t {
  bool enable = false;
  // Current implementation uses one PC-indexed table that blends paper Head
  // and promoted Tail state. In that merged design, the practical capacity
  // should track the paper Head table, not the 10-entry Tail table.
  unsigned ht_size = 32;
  unsigned training_warps = 3;
  unsigned max_chain_length = 2;  // paper depth controlled by throttle
};

class baseline_snake_prefetcher_t : public baseline_prefetcher_t {
 public:
  baseline_snake_prefetcher_t(unsigned sm_id,
                              const baseline_snake_config_t &cfg,
                              baseline_cache *l1d = nullptr);

  bool is_snake() const override { return true; }

  void on_kernel_launch() override;
  void on_warp_exit(unsigned warp_id) override;
  void on_instruction_issue(unsigned warp_id, const warp_inst_t &inst,
                            const std::string &sass_opcode,
                            unsigned long long cycle) override;
  void on_instruction_issue_with_cta(unsigned warp_id, unsigned cta_id,
                                      const warp_inst_t &inst,
                                      unsigned long long cycle);
  void on_demand_load(unsigned warp_id, new_addr_type pc, new_addr_type addr,
                      unsigned long long cycle, int cache_status,
                      shd_warp_t *warp) override;
  void print_stats(FILE *fp) const override;

 private:
  // --- HT Entry: one per load PC (matches paper's Tail Table) ---
  // Paper §3.1: indexed by PC_ld, stores PC1, PC2, strides, warpID vector
  struct ht_entry_t {
    bool valid = false;
    new_addr_type pc = 0;           // this PC (PC1 in paper)

    // Inter-thread chain: PC1 → PC2 with IT stride
    new_addr_type it_next_pc = 0;   // PC2: the consecutive PC after this one
    int64_t it_stride = 0;          // addr(PC2) - addr(PC1)
    bool it_stride_valid = false;   // stride confirmed by ≥2 observations
    unsigned it_observation_count = 0;

    // --- Head Table slots (paper §3.1, Table 3) ---
    // Per-PC, store 2 most recent (warp_id, addr) pairs.
    // "To avoid losing inter-warp strides in the presence of GTO,
    //  Snake stores information from two different warps per PC_ld."
    struct warp_slot_t {
      unsigned warp_id = static_cast<unsigned>(-1);
      unsigned cta_id = static_cast<unsigned>(-1);
      new_addr_type addr = 0;
      bool valid = false;
    };
    warp_slot_t head_slots[2];

    // IaW stride (intra-warp): same warp re-executes same PC
    int64_t iaw_stride = 0;
    bool iaw_confirmed = false;

    // IeW stride (inter-warp): different warps execute same PC
    int64_t iew_stride = 0;
    bool iew_confirmed = false;

    // Training state (paper: warpID vector + T1/T2)
    uint64_t warp_confirmed_mask = 0;
    bool training_done = false;

    unsigned training_warp_count() const {
        return static_cast<unsigned>(__builtin_popcountll(warp_confirmed_mask));
    }

    unsigned long long last_access_cycle = 0;
  };

  // --- Per-warp Head Table (paper §3.1) ---
  // Store 2 most recent (PC, addr) per warp for IT detection + IaW direct
 struct warp_pc_tracker_t {
    struct slot_t {
      new_addr_type pc = 0;
      new_addr_type addr = 0;
      bool valid = false;
    };
    slot_t slots[2];  // 2 slots per warp (paper: "doubling for GTO")
    // Most recent is slots[0], second is slots[1]
  };

  enum class uniform_gate_reason_t {
    kPassSingleActive,
    kPassAffine,
    kRejectNoActive,
    kRejectMixedStride,
    kRejectNonAffineProgression,
  };

  struct uniform_gate_result_t {
    bool accepted = false;
    new_addr_type first_addr = 0;
    unsigned active_count = 0;
    unsigned first_lane = 0;
    unsigned last_lane = 0;
    uniform_gate_reason_t reason = uniform_gate_reason_t::kRejectNoActive;
  };

  struct gate_reject_sample_t {
    new_addr_type pc = 0;
    uniform_gate_reason_t reason = uniform_gate_reason_t::kRejectNoActive;
    unsigned active_count = 0;
    unsigned first_lane = 0;
    unsigned last_lane = 0;
  };

  int find_ht_entry(new_addr_type pc) const;
  int alloc_ht_entry(new_addr_type pc, unsigned long long cycle);
  void update_head_table_slots(ht_entry_t &entry, unsigned warp_id,
                               unsigned cta_id, new_addr_type addr);
  void update_it_stride(ht_entry_t &prev_entry, new_addr_type prev_addr,
                         new_addr_type cur_pc, new_addr_type cur_addr);
  uniform_gate_result_t classify_uniform_warp_addr(
      const warp_inst_t &inst) const;
  static const char *uniform_gate_reason_name(uniform_gate_reason_t reason);
  void generate_prefetches(const ht_entry_t &entry, new_addr_type addr,
                            unsigned warp_id, unsigned long long cycle);

  baseline_snake_config_t m_cfg;
  baseline_cache *m_l1d_cache;
  std::vector<ht_entry_t> m_ht;

  // Per-warp last-PC tracker for IT stride detection
  std::vector<warp_pc_tracker_t> m_warp_trackers;  // indexed by warp_id
  static constexpr unsigned kMaxWarps = 64;

  // Throttle state (Snake paper §3.3)
  unsigned long long m_throttle_until = 0;
  static constexpr unsigned kThrottlePauseCycles = 50;
  // Paper-faithful default: IeW predicts the next future warp for this PC.
  // Larger distances, if ever needed, should come from an explicit knob rather
  // than a hard-coded heuristic baked into Snake.
  static constexpr unsigned kIeWLookahead = 1;
  static constexpr unsigned kIaWLookahead = 1;   // IaW is within same warp, degree=1

  // Per-stride-type diagnostic counters
  unsigned m_pf_iaw_issued = 0;
  unsigned m_pf_iew_issued = 0;
  unsigned m_pf_it_issued = 0;
  unsigned m_training_completions = 0;
  unsigned m_training_unique_warps = 0;
  unsigned m_training_duplicate_warps = 0;
  unsigned m_it_chain_follows = 0;
  unsigned long long m_issue_global_loads = 0;
  unsigned m_uniform_gate_passes = 0;
  unsigned m_uniform_gate_rejects = 0;
  unsigned m_uniform_gate_reject_kept_tracker = 0;
  unsigned m_uniform_gate_pass_single_active = 0;
  unsigned m_uniform_gate_pass_affine = 0;
  unsigned m_uniform_gate_reject_no_active = 0;
  unsigned m_uniform_gate_reject_mixed_stride = 0;
  unsigned m_uniform_gate_reject_non_affine = 0;
  unsigned long long m_uniform_gate_active_threads_total = 0;
  unsigned long long m_uniform_gate_multi_active_total = 0;
  unsigned m_lane_non_uniform_filtered = 0;
  unsigned m_pc_table_allocations = 0;
  unsigned m_pc_table_evictions = 0;
  unsigned m_iew_cta_mismatch = 0;
  unsigned long long m_demand_loads = 0;
  // IaW accumulation debug counters
  unsigned m_iaw_accum_checks = 0;    // times IT stride path entered
  unsigned m_iaw_it_not_valid = 0;    // blocked: IT stride not valid
  unsigned m_iaw_pc_mismatch = 0;     // blocked: it_next_pc != last_pc
  unsigned m_iaw_chain_found = 0;     // found via chain following
  unsigned m_iaw_accum_ok = 0;        // successfully computed IaW

  static constexpr unsigned kMaxGateRejectSamples = 8;
  std::vector<gate_reject_sample_t> m_gate_reject_samples;
};
