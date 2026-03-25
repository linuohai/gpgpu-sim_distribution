// GRASP Tables — CT and TT implementation

#include "grasp_tables.h"

#include <algorithm>
#include <cassert>

// ============================================================================
// Chain Table (CT)
// ============================================================================

grasp_chain_table_t::grasp_chain_table_t(unsigned num_entries)
    : m_entries(num_entries) {}

int grasp_chain_table_t::find(new_addr_type index_pc) const {
  for (int i = 0; i < (int)m_entries.size(); ++i) {
    if (m_entries[i].valid && m_entries[i].index_pc == index_pc) return i;
  }
  return -1;
}

int grasp_chain_table_t::find_victim() const {
  // Priority 1: invalid entry
  for (int i = 0; i < (int)m_entries.size(); ++i) {
    if (!m_entries[i].valid) return i;
  }
  // Priority 2: entry with stride_valid=false (less useful)
  int victim = -1;
  unsigned long long oldest_time = (unsigned long long)-1;
  for (int i = 0; i < (int)m_entries.size(); ++i) {
    if (!m_entries[i].stride_valid &&
        m_entries[i].last_access_time < oldest_time) {
      victim = i;
      oldest_time = m_entries[i].last_access_time;
    }
  }
  if (victim >= 0) return victim;
  // Priority 3: LRU among all entries
  victim = 0;
  for (int i = 1; i < (int)m_entries.size(); ++i) {
    if (m_entries[i].last_access_time < m_entries[victim].last_access_time) {
      victim = i;
    }
  }
  return victim;
}

void grasp_chain_table_t::update_peak_occupancy() {
  unsigned count = 0;
  for (const auto &e : m_entries) {
    if (e.valid) ++count;
  }
  if (count > m_stats.peak_occupancy) m_stats.peak_occupancy = count;
}

int grasp_chain_table_t::insert(new_addr_type index_pc, new_addr_type data_pc,
                                new_addr_type imad_pc, unsigned tt_idx) {
  int idx = find_victim();
  assert(idx >= 0 && idx < (int)m_entries.size());
  if (m_entries[idx].valid) {
    ++m_stats.eviction_count;
    if (m_entries[idx].stride_valid) ++m_stats.eviction_stride_valid_count;
  }
  ct_entry_t &e = m_entries[idx];
  e.valid = true;
  e.index_pc = index_pc;
  e.data_pc = data_pc;
  e.imad_pc = imad_pc;
  e.tt_idx[0] = tt_idx;
  e.tt_idx[1] = (unsigned)-1;
  e.tt_idx[2] = (unsigned)-1;
  e.num_targets = 1;
  e.iter_stride = 0;
  e.stride_valid = false;
  for (unsigned i = 0; i < ct_entry_t::MAX_STRIDE_OBS; ++i)
    e.stride_obs[i] = ct_entry_t::stride_obs_t();
  e.num_stride_obs = 0;
  e.last_access_time = 0;
  e.last_pf_warp_id = (unsigned)-1;
  e.last_pf_inst_uid = 0;
  update_peak_occupancy();
  return idx;
}

bool grasp_chain_table_t::append_target(int ct_idx, unsigned tt_idx) {
  assert(ct_idx >= 0 && ct_idx < (int)m_entries.size());
  ct_entry_t &e = m_entries[ct_idx];
  assert(e.valid);
  if (e.num_targets >= ct_entry_t::MAX_TARGETS) {
    // K=3 hardware limit reached
    return false;
  }
  e.tt_idx[e.num_targets] = tt_idx;
  ++e.num_targets;
  return true;
}

bool grasp_chain_table_t::update_stride(int ct_idx, unsigned warp_id,
                                        unsigned lane_id,
                                        new_addr_type current_addr,
                                        unsigned long long cycle,
                                        int64_t *out_delta) {
  assert(ct_idx >= 0 && ct_idx < (int)m_entries.size());
  ct_entry_t &e = m_entries[ct_idx];
  assert(e.valid);

  // Find existing observation for this warp
  ct_entry_t::stride_obs_t *obs = nullptr;
  for (unsigned i = 0; i < e.num_stride_obs; ++i) {
    if (e.stride_obs[i].warp_id == warp_id) {
      obs = &e.stride_obs[i];
      break;
    }
  }

  if (!obs) {
    // First visit from this warp: record observation
    if (e.num_stride_obs < ct_entry_t::MAX_STRIDE_OBS) {
      obs = &e.stride_obs[e.num_stride_obs++];
      obs->warp_id = warp_id;
      obs->lane_id = lane_id;
      obs->last_addr = current_addr;
      obs->last_cycle = cycle;
    }
    return false;
  }

  // Dedup: same instruction produces multiple mem_fetches in same cycle
  if (obs->last_cycle == cycle) return false;
  obs->last_cycle = cycle;

  // Must use the same lane for consistent address sequence
  if (obs->lane_id != lane_id) return false;

  int64_t delta =
      static_cast<int64_t>(current_addr) - static_cast<int64_t>(obs->last_addr);
  obs->last_addr = current_addr;
  if (out_delta) *out_delta = delta;

  if (delta == 0) return false;

  if (!e.stride_valid) {
    // Any warp's second observation with non-zero delta → stride valid
    e.iter_stride = delta;
    e.stride_valid = true;
    return true;
  }

  // Already valid: check consistency
  if (delta != e.iter_stride) {
    e.iter_stride = delta;
    e.stride_valid = true;
  }
  return false;
}

bool grasp_chain_table_t::all_stride_valid() const {
  bool has_valid = false;
  for (const auto &e : m_entries) {
    if (!e.valid) continue;
    has_valid = true;
    if (!e.stride_valid) return false;
  }
  return has_valid;  // false if table is empty
}

void grasp_chain_table_t::reset() {
  for (auto &e : m_entries) {
    e.valid = false;
    e.index_pc = 0;
    e.data_pc = 0;
    e.imad_pc = 0;
    for (unsigned i = 0; i < ct_entry_t::MAX_TARGETS; ++i)
      e.tt_idx[i] = (unsigned)-1;
    e.num_targets = 0;
    e.iter_stride = 0;
    e.stride_valid = false;
    for (unsigned j = 0; j < ct_entry_t::MAX_STRIDE_OBS; ++j)
      e.stride_obs[j] = ct_entry_t::stride_obs_t();
    e.num_stride_obs = 0;
    e.last_access_time = 0;
    e.last_pf_warp_id = (unsigned)-1;
    e.last_pf_inst_uid = 0;
  }
}

ct_entry_t &grasp_chain_table_t::entry(int idx) {
  assert(idx >= 0 && idx < (int)m_entries.size());
  return m_entries[idx];
}

const ct_entry_t &grasp_chain_table_t::entry(int idx) const {
  assert(idx >= 0 && idx < (int)m_entries.size());
  return m_entries[idx];
}

void grasp_chain_table_t::reset_tracking() {
  // Same-kernel re-launch: preserve stride knowledge, reset per-warp state
  for (auto &e : m_entries) {
    if (!e.valid) continue;
    for (unsigned j = 0; j < ct_entry_t::MAX_STRIDE_OBS; ++j)
      e.stride_obs[j] = ct_entry_t::stride_obs_t();
    e.num_stride_obs = 0;
    e.last_access_time = 0;
    e.last_pf_warp_id = (unsigned)-1;
    e.last_pf_inst_uid = 0;
  }
}

// ============================================================================
// Target Table (TT)
// ============================================================================

grasp_target_table_t::grasp_target_table_t(unsigned num_entries)
    : m_entries(num_entries) {}

int grasp_target_table_t::find(new_addr_type base_addr, unsigned scale) const {
  for (int i = 0; i < (int)m_entries.size(); ++i) {
    if (m_entries[i].valid && m_entries[i].base_addr == base_addr &&
        m_entries[i].scale == scale)
      return i;
  }
  return -1;
}

int grasp_target_table_t::allocate(new_addr_type base_addr, unsigned scale) {
  // Find first invalid slot
  for (int i = 0; i < (int)m_entries.size(); ++i) {
    if (!m_entries[i].valid) {
      m_entries[i].valid = true;
      m_entries[i].base_addr = base_addr;
      m_entries[i].scale = scale;
      return i;
    }
  }
  return -1;  // Full
}

void grasp_target_table_t::reset() {
  for (auto &e : m_entries) {
    e.valid = false;
    e.base_addr = 0;
    e.scale = 0;
  }
}

tt_entry_t &grasp_target_table_t::entry(int idx) {
  assert(idx >= 0 && idx < (int)m_entries.size());
  return m_entries[idx];
}

const tt_entry_t &grasp_target_table_t::entry(int idx) const {
  assert(idx >= 0 && idx < (int)m_entries.size());
  return m_entries[idx];
}
