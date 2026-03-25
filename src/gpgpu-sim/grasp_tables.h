// GRASP Tables — Chain Table (CT) and Target Table (TT)
//
// CT: Maps index_pc → {data_pc, imad_pc, tt_idx[], stride state}.
//     Supports one-to-many targets (up to K=3).
//     Integrates IST-style stride learning per entry.
//
// TT: CAM storing (base_addr, scale) pairs.  In trace-driven mode these are
//     placeholders since pair table substitutes for address computation.

#pragma once

#include <cstdint>
#include <vector>

#include "../abstract_hardware_model.h"

// ============================================================================
// CT Entry
// ============================================================================

struct ct_entry_t {
  bool valid = false;
  new_addr_type index_pc = 0;  // key
  new_addr_type data_pc = 0;
  new_addr_type imad_pc = 0;
  static constexpr unsigned MAX_TARGETS = 3;  // K=3
  unsigned tt_idx[MAX_TARGETS] = {(unsigned)-1, (unsigned)-1, (unsigned)-1};
  unsigned num_targets = 0;  // 0..3
  // Stride learning: per-warp first-lane observations.
  // Each warp that visits this PC records one lane's address.
  // When ANY warp visits a second time, stride is computed from its own delta.
  int64_t iter_stride = 0;
  bool stride_valid = false;
  static constexpr unsigned MAX_STRIDE_OBS = 16;
  struct stride_obs_t {
    unsigned warp_id = (unsigned)-1;
    unsigned lane_id = (unsigned)-1;
    new_addr_type last_addr = 0;
    unsigned long long last_cycle = 0;  // dedup same instruction
  };
  stride_obs_t stride_obs[MAX_STRIDE_OBS];
  unsigned num_stride_obs = 0;
  // LRU
  unsigned long long last_access_time = 0;
  // Prefetch generation dedup (one PF set per warp instruction)
  unsigned last_pf_warp_id = (unsigned)-1;
  unsigned long long last_pf_inst_uid = 0;
};

// ============================================================================
// TT Entry
// ============================================================================

struct tt_entry_t {
  bool valid = false;
  // Key: (base_addr, scale) — placeholders in trace-driven mode
  new_addr_type base_addr = 0;
  unsigned scale = 0;
};

// ============================================================================
// Chain Table
// ============================================================================

class grasp_chain_table_t {
 public:
  explicit grasp_chain_table_t(unsigned num_entries);

  // Find PC's entry, return index or -1
  int find(new_addr_type index_pc) const;

  // Insert new entry (prefer evicting stride_valid=false)
  int insert(new_addr_type index_pc, new_addr_type data_pc,
             new_addr_type imad_pc, unsigned tt_idx);

  // Append one-to-many target
  bool append_target(int ct_idx, unsigned tt_idx);

  // Stride learning: record per-warp lane address observation.
  // Returns true if stride just became valid.
  bool update_stride(int ct_idx, unsigned warp_id, unsigned lane_id,
                     new_addr_type current_addr, unsigned long long cycle,
                     int64_t *out_delta = nullptr);

  // All valid entries have stride_valid? (for CD freeze check)
  bool all_stride_valid() const;

  // Kernel launch: clear all
  void reset();
  // Same-kernel re-launch: keep stride, reset tracking state
  void reset_tracking();

  // Accessors
  ct_entry_t &entry(int idx);
  const ct_entry_t &entry(int idx) const;
  unsigned capacity() const { return static_cast<unsigned>(m_entries.size()); }

  // Stats
  struct ct_stats_t {
    unsigned long long peak_occupancy = 0;
    unsigned long long eviction_count = 0;
    unsigned long long eviction_stride_valid_count = 0;
  };
  const ct_stats_t &stats() const { return m_stats; }

 private:
  std::vector<ct_entry_t> m_entries;
  ct_stats_t m_stats;
  int find_victim() const;  // prefer stride_valid=false, then LRU
  void update_peak_occupancy();
};

// ============================================================================
// Target Table
// ============================================================================

class grasp_target_table_t {
 public:
  explicit grasp_target_table_t(unsigned num_entries);

  // CAM lookup by (base, scale)
  int find(new_addr_type base_addr, unsigned scale) const;

  // Allocate new entry
  int allocate(new_addr_type base_addr, unsigned scale);

  // Kernel launch: clear all
  void reset();

  tt_entry_t &entry(int idx);
  const tt_entry_t &entry(int idx) const;
  unsigned capacity() const { return static_cast<unsigned>(m_entries.size()); }

 private:
  std::vector<tt_entry_t> m_entries;
};
