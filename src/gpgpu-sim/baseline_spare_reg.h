#pragma once

#include <unordered_map>
#include <vector>

#include "baseline_prefetcher.h"

struct baseline_spare_reg_config_t {
  unsigned training_iter = 3;  // stride confidence threshold
  unsigned distance = 1;       // prefetch distance (1-4)
};

class baseline_spare_reg_prefetcher_t : public baseline_prefetcher_t {
 public:
  baseline_spare_reg_prefetcher_t(unsigned sm_id,
                                  const baseline_spare_reg_config_t &cfg);

  void on_kernel_launch() override;
  void on_demand_load(unsigned warp_id, new_addr_type pc, new_addr_type addr,
                      unsigned long long cycle, int cache_status,
                      shd_warp_t *warp) override;
  void on_fill(mem_fetch *mf, unsigned long long fill_cycle) override;
  void print_stats(FILE *fp) const override;

 private:
  // Per-PC stride state for IMA index PCs
  struct pair_entry_t {
    bool valid = false;
    new_addr_type last_addr = 0;
    int64_t stride = 0;
    unsigned confidence = 0;
  };

  baseline_spare_reg_config_t m_cfg;

  // PC -> stride state (only IMA index PCs)
  std::unordered_map<new_addr_type, pair_entry_t> m_pairs;

  // Stats
  unsigned long long m_stat_index_pf_issued = 0;
  unsigned long long m_stat_data_pf_issued = 0;
  // Diagnostics
  unsigned long long m_dbg_total_loads = 0;
  unsigned long long m_dbg_seed_hits = 0;
  unsigned long long m_dbg_cand_hits = 0;
  unsigned long long m_dbg_cand_lookups = 0;
};
