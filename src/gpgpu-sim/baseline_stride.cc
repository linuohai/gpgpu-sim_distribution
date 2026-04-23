#include "baseline_stride.h"

#include <algorithm>

#include "shader.h"

namespace {

unsigned sanitize_table_size(unsigned table_size) {
  return table_size == 0 ? 1 : table_size;
}

unsigned sanitize_assoc(unsigned table_size, unsigned assoc) {
  if (assoc == 0) return 1;
  return std::min(table_size, assoc);
}

}  // namespace

baseline_stride_prefetcher_t::baseline_stride_prefetcher_t(
    unsigned sm_id, mode_t mode, const baseline_stride_config_t &cfg)
    : baseline_prefetcher_t(sm_id),
      m_mode(mode),
      m_cfg(cfg),
      m_table(sanitize_table_size(cfg.table_size)),
      m_assoc(sanitize_assoc(sanitize_table_size(cfg.table_size), cfg.assoc)),
      m_num_sets(std::max(
          1U, (sanitize_table_size(cfg.table_size) + m_assoc - 1) / m_assoc)) {}

void baseline_stride_prefetcher_t::on_kernel_launch() {
  baseline_prefetcher_t::on_kernel_launch();
  for (auto &entry : m_table) entry = stride_entry_t();
}

void baseline_stride_prefetcher_t::on_warp_exit(unsigned warp_id) {
  if (m_mode != INTRA_WARP) return;
  for (auto &entry : m_table) {
    if (entry.valid && entry.warp_id == warp_id) {
      entry = stride_entry_t();
    }
  }
}

unsigned baseline_stride_prefetcher_t::set_index(unsigned warp_id,
                                                 new_addr_type pc) const {
  // SASS PCs are instruction-aligned, so hashing raw low bits collapses hot
  // load PCs into the same set. Drop alignment bits before indexing.
  const unsigned long long pc_tag = static_cast<unsigned long long>(pc >> 4);
  const unsigned long long key =
      m_mode == INTRA_WARP
          ? ((static_cast<unsigned long long>(warp_id) << 32) ^ pc_tag)
          : pc_tag;
  return static_cast<unsigned>(key % m_num_sets);
}

int baseline_stride_prefetcher_t::find_entry(unsigned warp_id,
                                             new_addr_type pc) const {
  const unsigned set_idx = set_index(warp_id, pc);
  const unsigned begin = set_idx * m_assoc;
  const unsigned end = std::min(begin + m_assoc,
                                static_cast<unsigned>(m_table.size()));
  for (unsigned idx = begin; idx < end; ++idx) {
    const stride_entry_t &entry = m_table[idx];
    if (!entry.valid || entry.pc != pc) continue;
    if (m_mode == INTRA_WARP && entry.warp_id != warp_id) continue;
    return static_cast<int>(idx);
  }
  return -1;
}

int baseline_stride_prefetcher_t::find_victim(unsigned set_idx) const {
  const unsigned begin = set_idx * m_assoc;
  const unsigned end = std::min(begin + m_assoc,
                                static_cast<unsigned>(m_table.size()));
  for (unsigned idx = begin; idx < end; ++idx) {
    if (!m_table[idx].valid) return static_cast<int>(idx);
  }

  unsigned victim = begin;
  for (unsigned idx = begin + 1; idx < end; ++idx) {
    if (m_table[idx].last_access_cycle < m_table[victim].last_access_cycle) {
      victim = idx;
    }
  }
  return static_cast<int>(victim);
}

void baseline_stride_prefetcher_t::on_demand_load(
    unsigned warp_id, new_addr_type pc, new_addr_type addr,
    unsigned long long cycle, int cache_status, shd_warp_t *warp) {
  (void)warp;
  note_demand_access(addr, cache_status);

  int idx = find_entry(warp_id, pc);
  if (idx < 0) {
    idx = find_victim(set_index(warp_id, pc));
    stride_entry_t &entry = m_table[idx];
    entry.valid = true;
    entry.warp_id = warp_id;
    entry.pc = pc;
    entry.last_addr = addr;
    entry.stride = 0;
    entry.confidence = 0;
    entry.last_warp_id = warp_id;
    entry.last_access_cycle = cycle;
    return;
  }

  stride_entry_t &entry = m_table[idx];
  entry.last_access_cycle = cycle;

  bool trained = false;
  int64_t delta = 0;
  if (m_mode == INTRA_WARP) {
    delta = static_cast<int64_t>(addr) - static_cast<int64_t>(entry.last_addr);
    trained = true;
  } else if (entry.last_warp_id != warp_id) {
    delta = static_cast<int64_t>(addr) - static_cast<int64_t>(entry.last_addr);
    trained = true;
  }

  if (trained) {
    if (delta != 0 && delta == entry.stride) {
      entry.confidence =
          std::min(entry.confidence + 1U, m_cfg.conf_threshold);
    } else {
      entry.stride = delta;
      entry.confidence = delta == 0 ? 0U : 1U;
    }
  }

  entry.last_addr = addr;
  entry.last_warp_id = warp_id;

  if (entry.stride == 0 || entry.confidence < m_cfg.conf_threshold) return;

  for (unsigned degree = 1; degree <= m_cfg.degree; ++degree) {
    const new_addr_type pf_addr = static_cast<new_addr_type>(
        static_cast<int64_t>(addr) +
        static_cast<int64_t>(degree) * entry.stride);
    queue_prefetch(pf_addr, warp_id, cycle);
  }
}

void baseline_stride_prefetcher_t::print_stats(FILE *fp) const {
  print_common_stats(fp, m_mode == INTRA_WARP ? "BASELINE_INTRA"
                                              : "BASELINE_INTER");
}
