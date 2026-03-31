#pragma once

#include <cstdint>
#include <vector>

#include "baseline_prefetcher.h"

// Maximum concurrent CTAs per SM (matches abstract_hardware_model.h).
#ifndef MAX_CTA_PER_SHADER
#define MAX_CTA_PER_SHADER 32
#endif

struct caps_config_t {
  unsigned percta_entries = 4;          // entries per CTA (paper: 4)
  unsigned dist_entries = 4;            // shared DIST table entries (paper: 4)
  unsigned mispredict_threshold = 128;  // misprediction counter ceiling (paper: 128)
};

// Per-CTA table entry: tracks base address of one load PC for one CTA.
struct caps_percta_entry_t {
  bool valid = false;
  new_addr_type pc = 0;
  unsigned leading_warp_id = static_cast<unsigned>(-1);
  new_addr_type base_addr = 0;  // first coalesced address from leading warp
  unsigned long long last_access_cycle = 0;
};

// DIST (Distance) table entry: tracks intra-CTA stride for one load PC,
// shared across all CTAs.
struct caps_dist_entry_t {
  bool valid = false;
  new_addr_type pc = 0;
  int64_t stride = 0;
  bool stride_valid = false;
  uint8_t mispredict_counter = 0;
  unsigned long long last_access_cycle = 0;
};

class baseline_caps_prefetcher_t : public baseline_prefetcher_t {
 public:
  baseline_caps_prefetcher_t(unsigned sm_id, const caps_config_t &cfg);

  void on_kernel_launch() override;
  void on_warp_exit(unsigned warp_id) override;
  void on_demand_load(unsigned warp_id, new_addr_type pc, new_addr_type addr,
                      unsigned long long cycle, int cache_status,
                      shd_warp_t *warp) override;
  void print_stats(FILE *fp) const override;

 private:
  // PerCTA table helpers.
  int find_percta_entry(unsigned cta_id, new_addr_type pc) const;
  int alloc_percta_entry(unsigned cta_id, unsigned long long cycle);

  // DIST table helpers.
  int find_dist_entry(new_addr_type pc) const;
  int alloc_dist_entry(unsigned long long cycle);

  // Convert flat index → (cta_id, slot).
  unsigned percta_base(unsigned cta_id) const {
    return cta_id * m_cfg.percta_entries;
  }

  caps_config_t m_cfg;

  // PerCTA table: m_percta[cta_id * percta_entries + slot]
  std::vector<caps_percta_entry_t> m_percta;

  // DIST table: m_dist[slot]
  std::vector<caps_dist_entry_t> m_dist;

  // Per-SM kernel transition detection (since on_kernel_launch only fires
  // on SM0 due to ctaid==0 check in shader.cc).
  unsigned m_last_kernel_uid = static_cast<unsigned>(-1);
  void detect_kernel_transition(const shd_warp_t *warp);

  // Extended stats beyond what baseline_stats_t tracks.
  unsigned long long m_total_demands = 0;
  unsigned long long m_leading_warp_set = 0;
  unsigned long long m_leading_warp_revisit = 0;
  unsigned long long m_stride_detected = 0;
  unsigned long long m_mispredict_suppressed = 0;
  unsigned long long m_indirect_filtered = 0;
  unsigned long long m_cross_cta_prefetches = 0;
};
