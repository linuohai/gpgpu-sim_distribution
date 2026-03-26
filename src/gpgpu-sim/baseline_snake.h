#pragma once

#include <cstdint>
#include <vector>

#include "baseline_prefetcher.h"

class baseline_cache;

struct baseline_snake_config_t {
  bool enable = false;
  unsigned ht_size = 128;
  unsigned tt_size = 256;
  unsigned training_warps = 3;
  unsigned max_chain_length = 2;  // paper depth controlled by throttle; 2 = conservative
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

    // IaW stride (intra-warp): same warp, same PC, across iterations
    new_addr_type iaw_last_addr = 0;
    int64_t iaw_stride = 0;
    bool iaw_confirmed = false;
    unsigned iaw_last_warp_id = static_cast<unsigned>(-1);

    // IeW stride (inter-warp): different warps within same CTA, same PC
    new_addr_type iew_last_addr = 0;
    int64_t iew_stride = 0;
    bool iew_confirmed = false;
    unsigned iew_last_warp_id = static_cast<unsigned>(-1);
    unsigned iew_last_cta_id = static_cast<unsigned>(-1);

    // Training state (paper: warpID vector + T1/T2)
    uint64_t warp_confirmed_mask = 0;
    bool training_done = false;

    unsigned training_warp_count() const {
        return static_cast<unsigned>(__builtin_popcountll(warp_confirmed_mask));
    }

    unsigned long long last_access_cycle = 0;
  };

  // --- Per-warp tracking for IT chain detection ---
  struct warp_pc_tracker_t {
    new_addr_type last_pc = 0;
    new_addr_type last_addr = 0;
    bool valid = false;
  };

  int find_ht_entry(new_addr_type pc) const;
  int alloc_ht_entry(new_addr_type pc, unsigned long long cycle);
  void update_iaw_stride(ht_entry_t &entry, unsigned warp_id,
                          new_addr_type addr);
  void update_iew_stride(ht_entry_t &entry, unsigned warp_id,
                          unsigned cta_id, new_addr_type addr);
  void update_it_stride(ht_entry_t &prev_entry, new_addr_type prev_addr,
                         new_addr_type cur_pc, new_addr_type cur_addr);
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
};
