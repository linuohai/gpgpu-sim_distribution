#pragma once

#include <cstdint>
#include <vector>

#include "baseline_prefetcher.h"

struct baseline_snake_config_t {
  bool enable = false;
  unsigned ht_size = 128;
  unsigned tt_size = 256;
  unsigned training_warps = 3;
  unsigned max_chain_length = 8;
};

class baseline_snake_prefetcher_t : public baseline_prefetcher_t {
 public:
  baseline_snake_prefetcher_t(unsigned sm_id,
                              const baseline_snake_config_t &cfg);

  void on_kernel_launch() override;
  void on_warp_exit(unsigned warp_id) override;
  void on_demand_load(unsigned warp_id, new_addr_type pc, new_addr_type addr,
                      unsigned long long cycle, int cache_status,
                      shd_warp_t *warp) override;
  void print_stats(FILE *fp) const override;

 private:
  // --- Tail Table: stores IT stride for each hop in a chain ---
  struct tt_entry_t {
    bool valid = false;
    int64_t it_stride = 0;       // inter-thread stride (PC_n -> PC_{n+1})
    unsigned next_tt_idx = 0xFFFF;  // linked-list pointer (0xFFFF = end)
  };

  // --- Head Table: one entry per chain-head PC ---
  struct ht_entry_t {
    bool valid = false;
    new_addr_type pc = 0;           // chain head PC
    unsigned chain_length = 0;      // number of hops (TT entries)
    unsigned tt_head_idx = 0xFFFF;  // first TT entry index

    // IaW stride (intra-warp): same warp, same PC, across iterations
    new_addr_type iaw_last_addr = 0;
    int64_t iaw_stride = 0;
    bool iaw_confirmed = false;
    unsigned iaw_last_warp_id = static_cast<unsigned>(-1);

    // IeW stride (inter-warp): different warps, same PC
    new_addr_type iew_last_addr = 0;
    int64_t iew_stride = 0;
    bool iew_confirmed = false;
    unsigned iew_last_warp_id = static_cast<unsigned>(-1);

    // Training state
    uint64_t warp_confirmed_mask = 0;  // 64-bit bitmap, bit i = warp i confirmed
    bool training_done = false;

    unsigned training_warp_count() const {
        return static_cast<unsigned>(__builtin_popcountll(warp_confirmed_mask));
    }

    // IT chain training: track last PC's addr per warp for IT stride
    new_addr_type last_it_addr = 0;
    new_addr_type last_it_pc = 0;
    bool last_it_valid = false;

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
  int alloc_tt_entry();
  void update_iaw_stride(ht_entry_t &entry, unsigned warp_id,
                          new_addr_type addr);
  void update_iew_stride(ht_entry_t &entry, unsigned warp_id,
                          new_addr_type addr);
  void try_extend_chain(ht_entry_t &head, new_addr_type prev_pc,
                         new_addr_type prev_addr, new_addr_type cur_pc,
                         new_addr_type cur_addr);
  void generate_prefetches(const ht_entry_t &entry, new_addr_type addr,
                            unsigned warp_id, unsigned long long cycle);

  baseline_snake_config_t m_cfg;
  std::vector<ht_entry_t> m_ht;
  std::vector<tt_entry_t> m_tt;
  unsigned m_tt_free_head;  // simple free-list for TT allocation

  // Per-warp last-PC tracker for IT stride detection
  std::vector<warp_pc_tracker_t> m_warp_trackers;  // indexed by warp_id
  static constexpr unsigned kMaxWarps = 64;
};
