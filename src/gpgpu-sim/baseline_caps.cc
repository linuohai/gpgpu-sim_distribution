#include "baseline_caps.h"

#include <algorithm>
#include <cassert>

#include "shader.h"

// ---------------------------------------------------------------------------
// Construction / kernel lifecycle
// ---------------------------------------------------------------------------

baseline_caps_prefetcher_t::baseline_caps_prefetcher_t(unsigned sm_id,
                                                       const caps_config_t &cfg)
    : baseline_prefetcher_t(sm_id),
      m_cfg(cfg),
      m_percta(static_cast<size_t>(MAX_CTA_PER_SHADER) * cfg.percta_entries),
      m_dist(cfg.dist_entries) {}

void baseline_caps_prefetcher_t::on_kernel_launch() {
  baseline_prefetcher_t::on_kernel_launch();

  // Clear PerCTA entries — CTA slot assignments change between kernels.
  for (auto &e : m_percta) e = caps_percta_entry_t();

  // PERSIST the DIST table across kernels: stride is a property of the PC,
  // not of any particular CTA instance.  BFS/SSSP re-launch the same kernel
  // per iteration, reusing the same load PCs with the same stride.
  // Only reset misprediction counters so stale suppression doesn't carry over.
  for (auto &e : m_dist) {
    if (e.valid) e.mispredict_counter = 0;
  }

  m_total_demands = 0;
  m_leading_warp_set = 0;
  m_leading_warp_revisit = 0;
  m_stride_detected = 0;
  m_mispredict_suppressed = 0;
  m_indirect_filtered = 0;
  m_cross_cta_prefetches = 0;
}

// ---------------------------------------------------------------------------
// Warp exit — clear PerCTA entries where exiting warp was leading warp.
// This handles CTA slot reuse: when a CTA finishes all its warps exit,
// clearing their PerCTA entries so the next CTA assigned to the same
// hardware slot starts fresh.
// ---------------------------------------------------------------------------

void baseline_caps_prefetcher_t::on_warp_exit(unsigned warp_id) {
  for (auto &e : m_percta) {
    if (e.valid && e.leading_warp_id == warp_id) {
      e = caps_percta_entry_t();
    }
  }
}

// ---------------------------------------------------------------------------
// Kernel transition detection — called from on_demand_load.
// Since on_kernel_launch() only fires on SM0 (ctaid==0 check in shader.cc),
// other SMs detect kernel transitions here and perform the equivalent reset.
// ---------------------------------------------------------------------------

void baseline_caps_prefetcher_t::detect_kernel_transition(
    const shd_warp_t *warp) {
  if (warp == nullptr) return;
  kernel_info_t *ki = warp->get_kernel_info();
  if (ki == nullptr) return;

  const unsigned kid = ki->get_uid();
  if (kid == m_last_kernel_uid) return;

  m_last_kernel_uid = kid;

  // Perform the same reset as on_kernel_launch().  We call this from
  // on_demand_load because on_kernel_launch only fires on SM0 (ctaid==0
  // guard in shader.cc:1143).
  on_kernel_launch();
}

// ---------------------------------------------------------------------------
// PerCTA table helpers
// ---------------------------------------------------------------------------

int baseline_caps_prefetcher_t::find_percta_entry(unsigned cta_id,
                                                   new_addr_type pc) const {
  if (cta_id >= MAX_CTA_PER_SHADER) return -1;
  const unsigned base = percta_base(cta_id);
  const unsigned end = base + m_cfg.percta_entries;
  for (unsigned i = base; i < end; ++i) {
    if (m_percta[i].valid && m_percta[i].pc == pc) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

int baseline_caps_prefetcher_t::alloc_percta_entry(unsigned cta_id,
                                                    unsigned long long cycle) {
  const unsigned base = percta_base(cta_id);
  const unsigned end = base + m_cfg.percta_entries;

  // Prefer an invalid slot.
  for (unsigned i = base; i < end; ++i) {
    if (!m_percta[i].valid) return static_cast<int>(i);
  }

  // LRU victim.
  unsigned victim = base;
  for (unsigned i = base + 1; i < end; ++i) {
    if (m_percta[i].last_access_cycle < m_percta[victim].last_access_cycle) {
      victim = i;
    }
  }
  m_percta[victim] = caps_percta_entry_t();
  return static_cast<int>(victim);
}

// ---------------------------------------------------------------------------
// DIST table helpers
// ---------------------------------------------------------------------------

int baseline_caps_prefetcher_t::find_dist_entry(new_addr_type pc) const {
  for (unsigned i = 0; i < m_cfg.dist_entries; ++i) {
    if (m_dist[i].valid && m_dist[i].pc == pc) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

int baseline_caps_prefetcher_t::alloc_dist_entry(unsigned long long cycle) {
  // Prefer an invalid slot.
  for (unsigned i = 0; i < m_cfg.dist_entries; ++i) {
    if (!m_dist[i].valid) return static_cast<int>(i);
  }

  // LRU victim.
  unsigned victim = 0;
  for (unsigned i = 1; i < m_cfg.dist_entries; ++i) {
    if (m_dist[i].last_access_cycle < m_dist[victim].last_access_cycle) {
      victim = i;
    }
  }
  m_dist[victim] = caps_dist_entry_t();
  return static_cast<int>(victim);
}

// ---------------------------------------------------------------------------
// Core algorithm — on_demand_load
//
// Paper reference: Koo et al., "CTA-Aware Prefetching and Scheduling for GPU",
// IPDPS 2018, Section V-B.
//
// Steps:
//   1. Indirect access filter (skip IMA data PCs)
//   2. PerCTA table lookup
//   3. Leading warp detection (first warp from this CTA for this PC)
//   4. Leading warp revisit (update base address)
//   5. Stride detection (trailing warp in same CTA)
//   6. Misprediction check
//   7. Cross-CTA prefetch generation
// ---------------------------------------------------------------------------

void baseline_caps_prefetcher_t::on_demand_load(
    unsigned warp_id, new_addr_type pc, new_addr_type addr,
    unsigned long long cycle, int cache_status, shd_warp_t *warp) {
  // Detect kernel transition on this SM (workaround for on_kernel_launch
  // only firing on SM0 via ctaid==0 check in shader.cc:1143).
  detect_kernel_transition(warp);

  note_demand_access(addr, cache_status);
  ++m_total_demands;

  // --- Step 1: indirect access filter ---
  if (warp != nullptr && warp->is_ima_data_pc(static_cast<address_type>(pc))) {
    ++m_indirect_filtered;
    return;
  }

  // Get hardware CTA id from the warp.
  if (warp == nullptr) return;
  const unsigned cta_id = warp->get_cta_id();
  if (cta_id >= MAX_CTA_PER_SHADER) return;

  // --- Step 2: PerCTA table lookup ---
  int pidx = find_percta_entry(cta_id, pc);

  // --- Step 3: leading warp detection ---
  if (pidx < 0) {
    // First warp in this CTA to access this PC → become leading warp.
    pidx = alloc_percta_entry(cta_id, cycle);
    caps_percta_entry_t &entry = m_percta[pidx];
    entry.valid = true;
    entry.pc = pc;
    entry.leading_warp_id = warp_id;
    entry.base_addr = addr;
    entry.last_access_cycle = cycle;
    ++m_leading_warp_set;

    // --- Step 3b: self-CTA prefetch (key optimization) ---
    const int didx_self = find_dist_entry(pc);
    if (didx_self >= 0 && m_dist[didx_self].stride_valid &&
        m_dist[didx_self].mispredict_counter <= m_cfg.mispredict_threshold) {
      const int64_t s = m_dist[didx_self].stride;
      // Prefetch for trailing warps 1..degree ahead.
      for (unsigned d = 1; d <= 4; ++d) {
        const new_addr_type pf_addr = static_cast<new_addr_type>(
            static_cast<int64_t>(addr) + s * static_cast<int64_t>(d));
        queue_prefetch(pf_addr, warp_id, cycle);
        ++m_cross_cta_prefetches;
      }
    }
    return;
  }

  caps_percta_entry_t &pentry = m_percta[pidx];
  pentry.last_access_cycle = cycle;

  // --- Step 4: leading warp revisit ---
  if (warp_id == pentry.leading_warp_id) {
    // Same leading warp executing again — update base address for next
    // iteration.
    pentry.base_addr = addr;
    ++m_leading_warp_revisit;
    return;
  }

  // --- Step 5: trailing warp in same CTA — detect / validate stride ---
  // Compute per-warp stride: the address distance between consecutive warps.
  // Within a CTA, warp IDs are sequential (assigned by init_warps).
  // warp_offset = how many warps ahead this warp is from the leading warp.
  const int warp_offset = static_cast<int>(warp_id) -
                          static_cast<int>(pentry.leading_warp_id);
  if (warp_offset == 0) return;  // shouldn't happen (caught by Step 4)

  const int64_t total_delta =
      static_cast<int64_t>(addr) - static_cast<int64_t>(pentry.base_addr);
  // Per-warp stride = total_delta / warp_offset.
  // Only valid if evenly divisible (all warps have same stride).
  const int64_t per_warp_stride =
      (warp_offset != 0) ? (total_delta / static_cast<int64_t>(warp_offset))
                         : 0;

  int didx = find_dist_entry(pc);

  if (didx < 0) {
    // No DIST entry yet for this PC — allocate and store stride.
    if (per_warp_stride == 0) return;  // zero stride is useless
    didx = alloc_dist_entry(cycle);
    caps_dist_entry_t &dentry = m_dist[didx];
    dentry.valid = true;
    dentry.pc = pc;
    dentry.stride = per_warp_stride;
    dentry.stride_valid = true;
    dentry.mispredict_counter = 0;
    dentry.last_access_cycle = cycle;
    ++m_stride_detected;
    return;
  }

  caps_dist_entry_t &dentry = m_dist[didx];
  dentry.last_access_cycle = cycle;

  if (!dentry.stride_valid) {
    // First stride observation for this DIST entry.
    if (per_warp_stride == 0) return;
    dentry.stride = per_warp_stride;
    dentry.stride_valid = true;
    ++m_stride_detected;
    return;
  }

  // --- Step 6: misprediction check ---
  // Predicted address: base + stride × warp_offset.
  const new_addr_type predicted = static_cast<new_addr_type>(
      static_cast<int64_t>(pentry.base_addr) +
      dentry.stride * static_cast<int64_t>(warp_offset));

  // Compare at sector granularity (32 B).
  if (normalize_sector_addr(predicted) != normalize_sector_addr(addr)) {
    // Mismatch — increment counter.
    if (dentry.mispredict_counter < 255) ++dentry.mispredict_counter;
  } else {
    // Match — decrement counter.
    if (dentry.mispredict_counter > 0) --dentry.mispredict_counter;
  }

  if (dentry.mispredict_counter > m_cfg.mispredict_threshold) {
    ++m_mispredict_suppressed;
    return;
  }

  // --- Step 7: cross-CTA prefetch generation ---
  // For every OTHER active CTA that has a base address for this PC,
  // generate a prefetch for the first trailing warp: other_base + stride.
  for (unsigned other_cta = 0; other_cta < MAX_CTA_PER_SHADER; ++other_cta) {
    if (other_cta == cta_id) continue;

    const int other_pidx = find_percta_entry(other_cta, pc);
    if (other_pidx < 0) continue;

    const caps_percta_entry_t &other = m_percta[other_pidx];
    const new_addr_type pf_addr = static_cast<new_addr_type>(
        static_cast<int64_t>(other.base_addr) + dentry.stride);

    queue_prefetch(pf_addr, warp_id, cycle);
    ++m_cross_cta_prefetches;
  }
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

void baseline_caps_prefetcher_t::print_stats(FILE *fp) const {
  print_common_stats(fp, "BASELINE_CAPS");

  const baseline_stats_t stats = finalized_stats();
  const unsigned long long total_pf =
      stats.prefetch_issued > 0 ? stats.prefetch_issued : 1;

  fprintf(fp,
          "BASELINE_CAPS_EXT_SM%u: total_demands=%llu "
          "leading_set=%llu leading_revisit=%llu stride_detected=%llu "
          "mispredict_suppressed=%llu indirect_filtered=%llu "
          "cross_cta_prefetches=%llu\n",
          m_sm_id, m_total_demands, m_leading_warp_set,
          m_leading_warp_revisit, m_stride_detected,
          m_mispredict_suppressed, m_indirect_filtered,
          m_cross_cta_prefetches);
}
