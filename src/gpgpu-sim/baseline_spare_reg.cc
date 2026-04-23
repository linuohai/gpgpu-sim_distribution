#include "baseline_spare_reg.h"

#include <algorithm>
#include <unordered_set>

#include "mem_fetch.h"
#include "shader.h"

baseline_spare_reg_prefetcher_t::baseline_spare_reg_prefetcher_t(
    unsigned sm_id, const baseline_spare_reg_config_t &cfg)
    : baseline_prefetcher_t(sm_id), m_cfg(cfg) {}

void baseline_spare_reg_prefetcher_t::on_kernel_launch() {
  baseline_prefetcher_t::on_kernel_launch();
  m_pairs.clear();
  m_stat_index_pf_issued = 0;
  m_stat_data_pf_issued = 0;
  m_dbg_total_loads = 0;
  m_dbg_seed_hits = 0;
  m_dbg_cand_hits = 0;
  m_dbg_cand_lookups = 0;
}

// ---------------------------------------------------------------------------
// Approach B (closest to paper):
//   (1) Index prefetch via stride for FUTURE iterations
//   (2) Data prefetch via pair-table lookup for CURRENT iteration
// ---------------------------------------------------------------------------
void baseline_spare_reg_prefetcher_t::on_demand_load(
    unsigned warp_id, new_addr_type pc, new_addr_type addr,
    unsigned long long cycle, int cache_status, shd_warp_t *warp) {
  note_demand_access(addr, cache_status);
  if (warp == nullptr) return;

  ++m_dbg_total_loads;

  // Only operate on IMA index PCs (identified via chain CSV)
  const std::vector<unsigned> seed_chain_ids =
      warp->get_ima_seed_chain_ids(static_cast<address_type>(pc));
  if (seed_chain_ids.empty()) return;
  ++m_dbg_seed_hits;

  // --- (2) Data prefetch for CURRENT iteration ---
  {
    const std::vector<ima_prefetch_candidate_t> cands =
        warp->lookup_ima_prefetch_candidates(addr, seed_chain_ids,
                                             /*exact_match_only=*/true);
    ++m_dbg_cand_lookups;
    if (!cands.empty()) ++m_dbg_cand_hits;

    std::unordered_set<new_addr_type> seen_lines;
    for (const ima_prefetch_candidate_t &cand : cands) {
      const new_addr_type line_addr =
          cand.data_addr & ~static_cast<new_addr_type>(127);
      if (!seen_lines.insert(line_addr).second) continue;
      queue_prefetch(cand.data_addr, warp_id, cycle);
      ++m_stat_data_pf_issued;
    }
  }

  // --- Stride learning ---
  pair_entry_t &entry = m_pairs[pc];
  if (!entry.valid) {
    entry.valid = true;
    entry.last_addr = addr;
    return;
  }

  const int64_t delta =
      static_cast<int64_t>(addr) - static_cast<int64_t>(entry.last_addr);
  if (delta != 0 && delta == entry.stride) {
    entry.confidence = std::min(entry.confidence + 1U,
                                m_cfg.training_iter);
  } else {
    entry.stride = delta;
    entry.confidence = delta == 0 ? 0U : 1U;
  }
  entry.last_addr = addr;

  if (entry.stride == 0 || entry.confidence < m_cfg.training_iter) return;

  // --- (1) Index prefetch for FUTURE iterations ---
  for (unsigned d = 1; d <= m_cfg.distance; ++d) {
    const new_addr_type future_head_addr = static_cast<new_addr_type>(
        static_cast<int64_t>(addr) +
        static_cast<int64_t>(d) * entry.stride);
    queue_prefetch(future_head_addr, warp_id, cycle);
    ++m_stat_index_pf_issued;
  }
}

void baseline_spare_reg_prefetcher_t::on_fill(
    mem_fetch *mf, unsigned long long fill_cycle) {
  baseline_prefetcher_t::on_fill(mf, fill_cycle);
}

void baseline_spare_reg_prefetcher_t::print_stats(FILE *fp) const {
  print_common_stats(fp, "BASELINE_SPARE_REG");
  fprintf(fp, "BASELINE_SPARE_REG_SM%u: index_pf=%llu data_pf=%llu "
              "total_loads=%llu seed_hits=%llu cand_lookups=%llu "
              "cand_hits=%llu\n",
          m_sm_id, m_stat_index_pf_issued, m_stat_data_pf_issued,
          m_dbg_total_loads, m_dbg_seed_hits,
          m_dbg_cand_lookups, m_dbg_cand_hits);
}
