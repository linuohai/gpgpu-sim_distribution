#include "baseline_snake.h"

#include <algorithm>
#include <cstring>

#include "shader.h"

baseline_snake_prefetcher_t::baseline_snake_prefetcher_t(
    unsigned sm_id, const baseline_snake_config_t &cfg)
    : baseline_prefetcher_t(sm_id),
      m_cfg(cfg),
      m_ht(cfg.ht_size),
      m_tt(cfg.tt_size),
      m_tt_free_head(0),
      m_warp_trackers(kMaxWarps) {}

void baseline_snake_prefetcher_t::on_kernel_launch() {
  baseline_prefetcher_t::on_kernel_launch();
  for (auto &e : m_ht) e = ht_entry_t();
  for (unsigned i = 0; i < m_tt.size(); ++i) {
    m_tt[i] = tt_entry_t();
    m_tt[i].next_tt_idx = i + 1;  // free list chain
  }
  if (!m_tt.empty()) m_tt.back().next_tt_idx = 0xFFFF;
  m_tt_free_head = m_tt.empty() ? 0xFFFF : 0;
  for (auto &w : m_warp_trackers) w = warp_pc_tracker_t();
}

void baseline_snake_prefetcher_t::on_warp_exit(unsigned warp_id) {
  if (warp_id < m_warp_trackers.size()) {
    m_warp_trackers[warp_id] = warp_pc_tracker_t();
  }
}

// --- HT lookup / allocation ---

int baseline_snake_prefetcher_t::find_ht_entry(new_addr_type pc) const {
  for (unsigned i = 0; i < m_ht.size(); ++i) {
    if (m_ht[i].valid && m_ht[i].pc == pc) return static_cast<int>(i);
  }
  return -1;
}

int baseline_snake_prefetcher_t::alloc_ht_entry(new_addr_type pc,
                                                 unsigned long long cycle) {
  // Find invalid slot first
  for (unsigned i = 0; i < m_ht.size(); ++i) {
    if (!m_ht[i].valid) {
      m_ht[i] = ht_entry_t();
      m_ht[i].valid = true;
      m_ht[i].pc = pc;
      m_ht[i].last_access_cycle = cycle;
      return static_cast<int>(i);
    }
  }
  // LRU eviction: prefer non-trained entries
  unsigned victim = 0;
  for (unsigned i = 1; i < m_ht.size(); ++i) {
    const ht_entry_t &vi = m_ht[victim];
    const ht_entry_t &ci = m_ht[i];
    // Non-trained beats trained
    if (!ci.training_done && vi.training_done) { victim = i; continue; }
    if (ci.training_done != vi.training_done) continue;
    // Fewer confirmed warps = less useful
    if (ci.training_warp_count() < vi.training_warp_count()) {
      victim = i; continue;
    }
    if (ci.training_warp_count() != vi.training_warp_count()) continue;
    // LRU tiebreak
    if (ci.last_access_cycle < vi.last_access_cycle) { victim = i; }
  }
  // Free any TT chain owned by victim
  unsigned tt_idx = m_ht[victim].tt_head_idx;
  while (tt_idx != 0xFFFF && tt_idx < m_tt.size()) {
    unsigned next = m_tt[tt_idx].next_tt_idx;
    m_tt[tt_idx] = tt_entry_t();
    m_tt[tt_idx].next_tt_idx = m_tt_free_head;
    m_tt_free_head = tt_idx;
    tt_idx = next;
  }
  m_ht[victim] = ht_entry_t();
  m_ht[victim].valid = true;
  m_ht[victim].pc = pc;
  m_ht[victim].last_access_cycle = cycle;
  return static_cast<int>(victim);
}

int baseline_snake_prefetcher_t::alloc_tt_entry() {
  if (m_tt_free_head == 0xFFFF) return -1;
  unsigned idx = m_tt_free_head;
  m_tt_free_head = m_tt[idx].next_tt_idx;
  m_tt[idx] = tt_entry_t();
  m_tt[idx].valid = true;
  m_tt[idx].next_tt_idx = 0xFFFF;
  return static_cast<int>(idx);
}

// --- Stride learning ---

void baseline_snake_prefetcher_t::update_iaw_stride(ht_entry_t &entry,
                                                     unsigned warp_id,
                                                     new_addr_type addr) {
  // IaW: same warp, same PC, across loop iterations
  if (entry.iaw_last_warp_id == warp_id && entry.iaw_last_addr != 0) {
    int64_t delta = static_cast<int64_t>(addr) -
                    static_cast<int64_t>(entry.iaw_last_addr);
    if (delta != 0) {
      if (delta == entry.iaw_stride) {
        entry.iaw_confirmed = true;
      } else {
        entry.iaw_stride = delta;
        entry.iaw_confirmed = false;
      }
    }
  }
  entry.iaw_last_addr = addr;
  entry.iaw_last_warp_id = warp_id;
}

void baseline_snake_prefetcher_t::update_iew_stride(ht_entry_t &entry,
                                                     unsigned warp_id,
                                                     new_addr_type addr) {
  // IeW: different warp, same PC
  if (entry.iew_last_warp_id != static_cast<unsigned>(-1) &&
      entry.iew_last_warp_id != warp_id) {
    int64_t delta = static_cast<int64_t>(addr) -
                    static_cast<int64_t>(entry.iew_last_addr);
    if (delta != 0) {
      if (delta == entry.iew_stride) {
        entry.iew_confirmed = true;
      } else {
        entry.iew_stride = delta;
        entry.iew_confirmed = false;
      }
    }
  }
  entry.iew_last_addr = addr;
  entry.iew_last_warp_id = warp_id;
}

// --- IT chain extension ---

void baseline_snake_prefetcher_t::try_extend_chain(
    ht_entry_t &head, new_addr_type prev_pc, new_addr_type prev_addr,
    new_addr_type cur_pc, new_addr_type cur_addr) {
  if (head.chain_length >= m_cfg.max_chain_length) return;
  if (prev_pc == cur_pc) return;  // not inter-thread (same PC = intra)

  int64_t it_stride =
      static_cast<int64_t>(cur_addr) - static_cast<int64_t>(prev_addr);
  if (it_stride == 0) return;

  // Check if this hop already exists at the chain tail
  unsigned tail_idx = head.tt_head_idx;
  unsigned prev_idx = 0xFFFF;
  unsigned hop_count = 0;
  while (tail_idx != 0xFFFF && tail_idx < m_tt.size()) {
    if (hop_count == head.chain_length - 1) break;  // found tail
    prev_idx = tail_idx;
    tail_idx = m_tt[tail_idx].next_tt_idx;
    ++hop_count;
  }

  // If chain is empty, this is the first hop
  if (head.chain_length == 0) {
    int new_idx = alloc_tt_entry();
    if (new_idx < 0) return;
    m_tt[new_idx].it_stride = it_stride;
    head.tt_head_idx = static_cast<unsigned>(new_idx);
    head.chain_length = 1;
    return;
  }

  // Append new hop at end
  if (tail_idx != 0xFFFF && tail_idx < m_tt.size()) {
    int new_idx = alloc_tt_entry();
    if (new_idx < 0) return;
    m_tt[new_idx].it_stride = it_stride;
    m_tt[tail_idx].next_tt_idx = static_cast<unsigned>(new_idx);
    head.chain_length++;
  }
}

// --- Prefetch generation ---

void baseline_snake_prefetcher_t::generate_prefetches(
    const ht_entry_t &entry, new_addr_type addr, unsigned warp_id,
    unsigned long long cycle) {
  // 1. IaW prefetch
  if (entry.iaw_confirmed && entry.iaw_stride != 0) {
    new_addr_type pf_addr = static_cast<new_addr_type>(
        static_cast<int64_t>(addr) + entry.iaw_stride);
    queue_prefetch(pf_addr, warp_id, cycle);
  }

  // 2. IeW prefetch
  if (entry.iew_confirmed && entry.iew_stride != 0) {
    new_addr_type pf_addr = static_cast<new_addr_type>(
        static_cast<int64_t>(addr) + entry.iew_stride);
    queue_prefetch(pf_addr, warp_id, cycle);
  }

  // 3. Chain prefetch: walk TT chain, accumulate IT strides
  new_addr_type pf_addr = addr;
  unsigned tt_idx = entry.tt_head_idx;
  for (unsigned hop = 0; hop < entry.chain_length; ++hop) {
    if (tt_idx == 0xFFFF || tt_idx >= m_tt.size()) break;
    if (!m_tt[tt_idx].valid) break;
    pf_addr = static_cast<new_addr_type>(
        static_cast<int64_t>(pf_addr) + m_tt[tt_idx].it_stride);
    queue_prefetch(pf_addr, warp_id, cycle);
    tt_idx = m_tt[tt_idx].next_tt_idx;
  }
}

// --- Main entry point ---

void baseline_snake_prefetcher_t::on_demand_load(
    unsigned warp_id, new_addr_type pc, new_addr_type addr,
    unsigned long long cycle, int cache_status, shd_warp_t *warp) {
  (void)warp;
  note_demand_access(addr, cache_status);

  warp_pc_tracker_t *tracker = nullptr;
  if (warp_id < m_warp_trackers.size()) {
    tracker = &m_warp_trackers[warp_id];
  }

  int ht_idx = find_ht_entry(pc);
  if (ht_idx < 0) {
    ht_idx = alloc_ht_entry(pc, cycle);
  }
  ht_entry_t &entry = m_ht[ht_idx];
  entry.last_access_cycle = cycle;

  // --- Training-phase bookkeeping (warp count + IT chain) ---
  // Warp counting MUST happen before update_*_stride, which overwrites
  // the last_warp_id fields used for the distinct-warp check.
  if (!entry.training_done) {
    // Record this warp in the confirmation bitmap
    if (warp_id < 64 &&
        !(entry.warp_confirmed_mask & (1ULL << warp_id))) {
      entry.warp_confirmed_mask |= (1ULL << warp_id);
      if (entry.training_warp_count() >= m_cfg.training_warps) {
        entry.training_done = true;
      }
    }

    // IT chain: if this warp just accessed a different PC, try to extend
    if (tracker != nullptr && tracker->valid && tracker->last_pc != pc) {
      int prev_ht = find_ht_entry(tracker->last_pc);
      if (prev_ht >= 0) {
        try_extend_chain(m_ht[prev_ht], tracker->last_pc, tracker->last_addr,
                         pc, addr);
      }
    }
  }

  // --- ALWAYS: stride learning (independent of training gate) ---
  update_iaw_stride(entry, warp_id, addr);
  update_iew_stride(entry, warp_id, addr);

  // --- ALWAYS: prefetch generation ---
  generate_prefetches(entry, addr, warp_id, cycle);

  // Update per-warp tracker
  if (tracker != nullptr) {
    tracker->last_pc = pc;
    tracker->last_addr = addr;
    tracker->valid = true;
  }
}

void baseline_snake_prefetcher_t::print_stats(FILE *fp) const {
  print_common_stats(fp, "BASELINE_SNAKE");
}
