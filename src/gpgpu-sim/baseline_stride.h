#pragma once

#include <vector>

#include "baseline_prefetcher.h"

struct baseline_stride_config_t {
  bool intra_enable = false;
  bool inter_enable = false;
  unsigned table_size = 64;
  unsigned assoc = 4;
  unsigned conf_threshold = 2;
  unsigned degree = 1;
};

class baseline_stride_prefetcher_t : public baseline_prefetcher_t {
 public:
  enum mode_t { INTRA_WARP, INTER_WARP };

  baseline_stride_prefetcher_t(unsigned sm_id, mode_t mode,
                               const baseline_stride_config_t &cfg);

  void on_kernel_launch() override;
  void on_warp_exit(unsigned warp_id) override;
  void on_demand_load(unsigned warp_id, new_addr_type pc, new_addr_type addr,
                      unsigned long long cycle, int cache_status,
                      shd_warp_t *warp) override;
  void print_stats(FILE *fp) const override;

 private:
  struct stride_entry_t {
    bool valid = false;
    unsigned warp_id = static_cast<unsigned>(-1);
    new_addr_type pc = 0;
    new_addr_type last_addr = 0;
    int64_t stride = 0;
    unsigned confidence = 0;
    unsigned last_warp_id = static_cast<unsigned>(-1);
    unsigned long long last_access_cycle = 0;
  };

  unsigned set_index(unsigned warp_id, new_addr_type pc) const;
  int find_entry(unsigned warp_id, new_addr_type pc) const;
  int find_victim(unsigned set_idx) const;

  mode_t m_mode;
  baseline_stride_config_t m_cfg;
  std::vector<stride_entry_t> m_table;
  unsigned m_assoc;
  unsigned m_num_sets;
};
