// GRASP Prefetcher — Top-level implementation
//
// Orchestrates all sub-components: CD, CT, TT, IST, PRB.
// Provides hooks called from ldst_unit and shader_core_ctx.

#include "grasp_prefetcher.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <set>
#include <tuple>

#include "gpu-cache.h"
#include "grasp_tracer.h"
#include "mem_fetch.h"
#include "shader.h"

// ============================================================================
// PRB Implementation
// ============================================================================

grasp_prb_t::grasp_prb_t(unsigned initial_capacity)
    : m_entries(initial_capacity), m_capacity(initial_capacity) {}

// v7 P1: allocate without candidates (filled per-sector at inject time)
int grasp_prb_t::allocate(unsigned warp_id, unsigned num_sectors,
                           unsigned long long cycle) {
  auto init_entry = [&](grasp_prb_entry_t &e) {
    e.valid = true;
    e.warp_id = warp_id;
    e.sector_data.clear();
    e.remaining_sectors = num_sectors;
    e.alloc_cycle = cycle;
  };
  for (size_t i = 0; i < m_entries.size(); ++i) {
    if (!m_entries[i].valid) {
      init_entry(m_entries[i]);
      ++m_stats.current_occupancy;
      ++m_stats.total_allocations;
      if (m_stats.current_occupancy > m_stats.peak_occupancy) {
        m_stats.peak_occupancy = m_stats.current_occupancy;
      }
      return static_cast<int>(i);
    }
  }
  // Hard cap: reject if all slots occupied
  ++m_stats.prb_drop_count;
  return -1;
}

// v7 P1: add candidates for a specific sector to an existing PRB entry
void grasp_prb_t::add_sector_candidates(
    unsigned prb_entry_id, new_addr_type sector_addr,
    std::vector<ima_prefetch_candidate_t> &&cands) {
  assert(prb_entry_id < m_entries.size());
  assert(m_entries[prb_entry_id].valid);
  grasp_prb_entry_t::sector_data_t sd;
  sd.sector_addr = sector_addr;
  sd.candidates = std::move(cands);
  sd.dispatched = false;
  m_entries[prb_entry_id].sector_data.push_back(std::move(sd));
}

void grasp_prb_t::free_entry(unsigned prb_entry_id) {
  // C2: PRB entry must be valid before free
  assert(prb_entry_id < m_entries.size());
  assert(m_entries[prb_entry_id].valid);
  unsigned long long lifetime =
      0;  // Would need current cycle for exact lifetime
  m_entries[prb_entry_id].valid = false;
  m_entries[prb_entry_id].warp_id = (unsigned)-1;
  m_entries[prb_entry_id].sector_data.clear();
  assert(m_stats.current_occupancy > 0);
  --m_stats.current_occupancy;
}

grasp_prb_entry_t &grasp_prb_t::get(unsigned prb_entry_id) {
  // C3: bounds check
  assert(prb_entry_id < m_entries.size());
  return m_entries[prb_entry_id];
}

const grasp_prb_entry_t &grasp_prb_t::get(unsigned prb_entry_id) const {
  assert(prb_entry_id < m_entries.size());
  return m_entries[prb_entry_id];
}

void grasp_prb_t::reset() {
  for (auto &e : m_entries) {
    e.valid = false;
    e.warp_id = (unsigned)-1;
    e.sector_data.clear();
  }
  m_stats.current_occupancy = 0;
}

// ============================================================================
// IST Implementation (migrated from ima_prefetcher_t)
// ============================================================================

grasp_ist_t::grasp_ist_t(unsigned sm_id, unsigned ipt_size, unsigned distance,
                         unsigned conf_thresh)
    : m_sm_id(sm_id),
      m_ipt_size(ipt_size),
      m_distance(distance),
      m_conf_thresh(conf_thresh),
      m_ipt(ipt_size) {}

int grasp_ist_t::find_entry(new_addr_type pc) const {
  for (int i = 0; i < (int)m_ipt_size; ++i) {
    if (m_ipt[i].valid && m_ipt[i].pc == pc) return i;
  }
  return -1;
}

int grasp_ist_t::find_lru_victim() const {
  for (int i = 0; i < (int)m_ipt_size; ++i) {
    if (!m_ipt[i].valid) return i;
  }
  int victim = 0;
  for (int i = 1; i < (int)m_ipt_size; ++i) {
    if (m_ipt[i].last_access_time < m_ipt[victim].last_access_time) {
      victim = i;
    }
  }
  return victim;
}

void grasp_ist_t::ipt_update(ipt_entry_t &entry, new_addr_type new_addr) {
  int64_t new_stride =
      static_cast<int64_t>(new_addr) - static_cast<int64_t>(entry.last_addr);
  if (entry.stride == 0) {
    entry.stride = new_stride;
    entry.confidence = 0;
  } else if (new_stride == entry.stride) {
    entry.confidence =
        std::min(entry.confidence + 1, ipt_entry_t::MAX_CONFIDENCE);
  } else {
    entry.stride = new_stride;
    entry.confidence = 0;
  }
  entry.last_addr = new_addr;
}

std::vector<new_addr_type> grasp_ist_t::on_load_access(new_addr_type pc,
                                                        new_addr_type addr,
                                                        unsigned long long cycle) {
  m_clock = cycle;
  std::vector<new_addr_type> result;

  int idx = find_entry(pc);
  if (idx < 0) {
    idx = find_lru_victim();
    ipt_entry_t &e = m_ipt[idx];
    e.valid = true;
    e.pc = pc;
    e.last_addr = addr;
    e.stride = 0;
    e.confidence = 0;
    e.last_access_time = cycle;
    return result;
  }

  ipt_entry_t &entry = m_ipt[idx];
  entry.last_access_time = cycle;
  ipt_update(entry, addr);

  if (entry.stride != 0 && entry.confidence >= m_conf_thresh) {
    for (unsigned n = 1; n <= m_distance; ++n) {
      new_addr_type paddr = static_cast<new_addr_type>(
          static_cast<int64_t>(addr) +
          static_cast<int64_t>(n) * entry.stride);
      result.push_back(paddr);
      ++stat_prefetch_generated;
    }
  }
  return result;
}

void grasp_ist_t::reset() {
  for (auto &e : m_ipt) {
    e.valid = false;
    e.pc = 0;
    e.last_addr = 0;
    e.stride = 0;
    e.confidence = 0;
    e.last_access_time = 0;
  }
  m_clock = 0;
}

void grasp_ist_t::print_stats(FILE *fp) const {
  fprintf(fp,
          "GRASP_IST SM%u: generated=%llu issued=%llu filtered=%llu\n",
          m_sm_id, stat_prefetch_generated, stat_prefetch_issued,
          stat_prefetch_filtered);
}

// ============================================================================
// Top-level GRASP Prefetcher
// ============================================================================

grasp_prefetcher_t::grasp_prefetcher_t(unsigned sm_id,
                                       const grasp_config_t &cfg)
    : m_sm_id(sm_id),
      m_cfg(cfg),
      m_cd(cfg.cd_fifo_depth),
      m_ct(cfg.ct_size, cfg.speculative_stride),
      m_tt(cfg.tt_size),
      m_ist(sm_id, cfg.ist_ipt_size, cfg.ist_distance, cfg.ist_confidence),
      m_prb(cfg.prb_capacity),
      m_demand_dedup(64) {  // 64 warps per SM max
  if (cfg.chain_csv && strlen(cfg.chain_csv) > 0) {
    load_stride_hints(cfg.chain_csv);
  }
}

grasp_prefetcher_t::~grasp_prefetcher_t() = default;

void grasp_prefetcher_t::load_stride_hints(const char *csv_path) {
  FILE *f = fopen(csv_path, "r");
  if (!f) return;
  char line[4096];
  // Parse header
  if (!fgets(line, sizeof(line), f)) { fclose(f); return; }
  std::string hdr(line);
  // Find column indices
  int idx_pc_col = -1, stride_col = -1;
  int col = 0;
  size_t start = 0;
  while (start < hdr.size()) {
    size_t end = hdr.find(',', start);
    if (end == std::string::npos) end = hdr.size();
    std::string name = hdr.substr(start, end - start);
    // trim whitespace/newline
    while (!name.empty() && (name.back() == '\n' || name.back() == '\r' || name.back() == ' '))
      name.pop_back();
    if (name == "index_pc") idx_pc_col = col;
    if (name == "stride_hint") stride_col = col;
    start = end + 1;
    ++col;
  }
  if (idx_pc_col < 0 || stride_col < 0) { fclose(f); return; }
  // Parse rows
  while (fgets(line, sizeof(line), f)) {
    std::string row(line);
    col = 0; start = 0;
    unsigned pc = 0; int hint = 0;
    bool got_pc = false, got_hint = false;
    while (start < row.size()) {
      size_t end = row.find(',', start);
      if (end == std::string::npos) end = row.size();
      std::string val = row.substr(start, end - start);
      while (!val.empty() && (val.back() == '\n' || val.back() == '\r' || val.back() == ' '))
        val.pop_back();
      if (col == idx_pc_col) {
        pc = (unsigned)strtoul(val.c_str(), nullptr, 16);
        got_pc = true;
      }
      if (col == stride_col) {
        hint = atoi(val.c_str());
        got_hint = true;
      }
      start = end + 1;
      ++col;
    }
    if (got_pc && got_hint && hint != 0 && pc != 0) {
      m_stride_hints[pc] = static_cast<int64_t>(hint);
    }
  }
  fclose(f);
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void grasp_prefetcher_t::on_kernel_launch(const std::string &kernel_name) {
  bool same_kernel = (!kernel_name.empty() &&
                      kernel_name == m_last_kernel_name);
  m_last_kernel_name = kernel_name;
  // CD always reset (warp assignments change across launches)
  m_cd.reset();

  if (same_kernel) {
    // Same kernel function re-launched (e.g., BFS level iteration):
    // preserve CT/TT stride knowledge, reset per-warp tracking state only
    m_ct.reset_tracking();
  } else {
    // Different kernel: full reset
    m_ct.reset();
    m_tt.reset();
  }

  m_ist.reset();
  m_prb.reset();
  m_prefetch_queue.clear();
  for (auto &dd : m_demand_dedup) dd = warp_dedup_t();
  ++m_stats.kernel_launches;
}

void grasp_prefetcher_t::on_warp_exit(unsigned warp_id) {
  ++m_stats.warp_exits;
  // CD handles freeze internally — no warp succession.
  // Tracked warp exit → training frozen permanently for this kernel.
  m_cd.on_warp_exit(warp_id);

  // Clear per-warp stride observations in CT to prevent stale addresses
  // from polluting delta computation when the warp slot is reused by a
  // new CTA.  (Stride itself is frozen after convergence, but keeping
  // obs clean avoids wasted slots and misleading trace output.)
  for (unsigned i = 0; i < m_ct.capacity(); ++i) {
    ct_entry_t &e = m_ct.entry(i);
    if (!e.valid) continue;
    for (unsigned j = 0; j < e.num_stride_obs; ++j) {
      if (e.stride_obs[j].warp_id == warp_id) {
        e.stride_obs[j] = e.stride_obs[e.num_stride_obs - 1];
        e.stride_obs[e.num_stride_obs - 1] = ct_entry_t::stride_obs_t();
        --e.num_stride_obs;
        break;  // at most one obs per warp per CT entry
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Training Path (CD → CT/TT)
// ---------------------------------------------------------------------------

void grasp_prefetcher_t::on_instruction_issue(unsigned warp_id,
                                               const warp_inst_t &inst,
                                               const std::string &sass_opcode,
                                               unsigned long long cycle) {
  if (!m_cfg.enable) return;

  int dst_reg = inst.arch_reg.dst[0];
  const int *src_regs = inst.arch_reg.src;
  unsigned num_src = 0;
  for (unsigned i = 0; i < MAX_REG_OPERANDS; ++i) {
    if (inst.arch_reg.src[i] >= 0) ++num_src;
    else break;
  }

  // Extract first active lane address (for stride seed at CT_INSERT)
  // Guard: per-thread info must be valid, otherwise get_addr() asserts
  // (seen on LS PTA kernel 11: some load insts reach here before addresses
  // are computed, 2026-04-07).
  new_addr_type first_lane_addr = 0;
  unsigned first_lane_id = 0;
  if (!inst.empty() && inst.is_load() && inst.has_per_thread_info()) {
    for (unsigned l = 0; l < inst.warp_size(); ++l) {
      if (inst.active(l)) {
        first_lane_addr = inst.get_addr(l);
        first_lane_id = l;
        break;
      }
    }
  }

  grasp_chain_detector_t::detected_chain_t chain;
  bool detected = m_cd.on_instruction_issue(warp_id, inst.pc, inst.op,
                                             sass_opcode, dst_reg, src_regs,
                                             num_src, cycle,
                                             first_lane_addr, first_lane_id,
                                             inst.data_size,
                                             &chain);

  if (detected) {
    // TT allocation (placeholder in trace-driven mode)
    int tt_idx = m_tt.find(chain.base_placeholder, chain.scale_placeholder);
    if (tt_idx < 0) {
      tt_idx = m_tt.allocate(chain.base_placeholder, chain.scale_placeholder);
    }
    if (tt_idx < 0) {
      // TT full — skip this chain
      if (m_cfg.debug) {
        printf(
            "GRASP CD: TT full, dropping chain idx_pc=0x%llx data_pc=0x%llx "
            "sm=%u\n",
            (unsigned long long)chain.index_pc,
            (unsigned long long)chain.data_pc, m_sm_id);
      }
      return;
    }

    // CT insertion or target append
    int ct_idx = m_ct.find(chain.index_pc);
    bool ct_is_new = (ct_idx < 0);
    if (ct_idx >= 0) {
      // Existing entry: append target if room
      m_ct.append_target(ct_idx, (unsigned)tt_idx);
      m_ct.entry(ct_idx).last_access_time = cycle;
    } else {
      ct_idx = m_ct.insert(chain.index_pc, chain.data_pc, 0 /*imad_pc*/,
                           (unsigned)tt_idx);
      m_ct.entry(ct_idx).last_access_time = cycle;
      // Per-entry speculative stride: CSV stride_hint > data_size > global default
      // data_size fallback only activates when speculative stride is globally enabled
      {
        auto it = m_stride_hints.find(chain.index_pc);
        if (it != m_stride_hints.end()) {
          m_ct.entry(ct_idx).speculative_stride_hint = it->second;
        } else if (m_cfg.speculative_stride != 0 && chain.index_data_size > 0) {
          m_ct.entry(ct_idx).speculative_stride_hint =
              static_cast<int64_t>(chain.index_data_size);
        }
      }

      // Seed first stride observation: record the index load address from the
      // iteration that triggered CD.  This allows stride to converge one
      // iteration earlier (at the very next demand load for this PC).
      if (chain.index_addr != 0) {
        m_ct.update_stride(ct_idx, warp_id, chain.index_lane_id,
                           chain.index_addr, cycle, nullptr);
      }
    }

    if (m_cfg.debug) {
      printf(
          "GRASP CD: chain detected idx_pc=0x%llx data_pc=0x%llx "
          "ct_idx=%d tt_idx=%d sm=%u warp=%u cycle=%llu\n",
          (unsigned long long)chain.index_pc,
          (unsigned long long)chain.data_pc, ct_idx, tt_idx, m_sm_id,
          warp_id, cycle);
    }
    if (grasp_tracer::enabled()) {
      char det[160];
      snprintf(det, sizeof(det),
               "idx_pc=0x%llx;data_pc=0x%llx;ct_idx=%d;tt_idx=%d;ct_new=%d",
               (unsigned long long)chain.index_pc,
               (unsigned long long)chain.data_pc, ct_idx, tt_idx,
               ct_is_new ? 1 : 0);
      grasp_tracer::emit(m_sm_id, warp_id, cycle,
                         ct_is_new ? "CT_INSERT" : "CHAIN_DETECT",
                         (unsigned long long)chain.index_pc,
                         (unsigned long long)chain.data_pc, det);
    }
  }
}

// ---------------------------------------------------------------------------
// Prefetch Generation (IST → queue)
// ---------------------------------------------------------------------------

void grasp_prefetcher_t::on_demand_load(unsigned warp_id, new_addr_type pc,
                                         new_addr_type addr,
                                         unsigned long long cycle,
                                         shd_warp_t *warp,
                                         const mem_fetch *mf,
                                         int l1_status) {
  if (!m_cfg.enable) return;

  // Check if this PC is an IMA index load (registered in CT by CD)
  int ct_idx = m_ct.find(pc);
  if (ct_idx < 0) {
    // Trace CT miss for known IMA PCs (helps debug late chain detection)
    if (grasp_tracer::enabled() && warp != nullptr &&
        !warp->get_ima_seed_chain_ids(static_cast<address_type>(pc)).empty()) {
      char det[128];
      snprintf(det, sizeof(det), "pc=0x%x;reason=CT_NOT_FOUND",
               (unsigned)pc);
      grasp_tracer::emit(m_sm_id, warp_id, cycle, "DEMAND_CT_MISS",
                         (unsigned long long)pc, (unsigned long long)addr,
                         det);
    }
    return;
  }

  // IMA-specific demand load tracking (accuracy/coverage/timeliness denominator)
  ++m_stats.ima_demand_reads;
  if (l1_status == MISS || l1_status == SECTOR_MISS) {
    ++m_stats.ima_demand_misses;
  }

  // ── Per-warp stride learning ──
  // Every warp records its first-lane address on first visit. When any warp
  // revisits, stride is computed from that warp's own address delta.
  // This is much more robust than locking to a single warp: whichever warp
  // iterates first determines the stride.
  ct_entry_t &cte_mut = m_ct.entry(ct_idx);

  // Extract lowest active lane and its address from the warp instruction
  // Guard: per-thread info must be valid (see PTA fix 2026-04-07)
  new_addr_type lane_addr = addr;  // fallback
  unsigned lane_id = (unsigned)-1;
  bool has_lane_addr = false;
  if (mf) {
    const warp_inst_t &inst = mf->get_inst();
    if (!inst.empty() && inst.is_load() && inst.has_per_thread_info()) {
      for (unsigned l = 0; l < inst.warp_size(); ++l) {
        if (inst.active(l)) {
          lane_addr = inst.get_addr(l);
          lane_id = l;
          has_lane_addr = true;
          break;
        }
      }
    }
  }

  // Dedup: extract inst_uid early so we can skip redundant trace + stride
  // updates for the 2nd-4th sectors of the same warp instruction.
  // Uses per-warp dedup vector (not per-CT-entry) to avoid cross-warp
  // interleaving invalidating the stored (warp_id, inst_uid) pair.
  unsigned long long inst_uid = 0;
  if (mf) {
    const warp_inst_t &uid_inst = mf->get_inst();
    if (!uid_inst.empty()) inst_uid = uid_inst.get_uid();
  }
  bool is_first_sector = true;
  if (inst_uid != 0 && warp_id < m_demand_dedup.size()) {
    auto &dd = m_demand_dedup[warp_id];
    is_first_sector = !(dd.ct_idx == ct_idx && dd.inst_uid == inst_uid);
    if (is_first_sector) {
      dd.ct_idx = ct_idx;
      dd.inst_uid = inst_uid;
    }
  }

  // Trace: demand load hit CT entry (once per instruction, not per sector)
  if (grasp_tracer::enabled() && is_first_sector) {
    char det[192];
    snprintf(det, sizeof(det),
             "ct_idx=%d;stride_valid=%d;iter_stride=%lld;sector_addr=0x%llx",
             ct_idx, cte_mut.stride_valid ? 1 : 0,
             (long long)cte_mut.iter_stride,
             (unsigned long long)(addr & ~((new_addr_type)31)));
    grasp_tracer::emit(m_sm_id, warp_id, cycle, "DEMAND_CT_HIT",
                       (unsigned long long)pc, (unsigned long long)lane_addr,
                       det);
  }

  // Stride learning: update_stride handles per-warp observation table internally
  int64_t stride_delta = 0;
  bool newly_valid = false;
  if (has_lane_addr) {
    newly_valid = m_ct.update_stride(ct_idx, warp_id, lane_id, lane_addr,
                                     cycle, &stride_delta);
  }
  m_ct.entry(ct_idx).last_access_time = cycle;

  // Trace: stride learning step (only when update_stride actually computed a
  // delta; skip deduped calls where stride_delta stays at 0)
  if (grasp_tracer::enabled() && has_lane_addr && stride_delta != 0) {
    char det[128];
    snprintf(det, sizeof(det),
             "ct_idx=%d;delta=%lld;iter_stride=%lld;stride_valid=%d;lane=%u",
             ct_idx, (long long)stride_delta, (long long)cte_mut.iter_stride,
             cte_mut.stride_valid ? 1 : 0, lane_id);
    grasp_tracer::emit(m_sm_id, warp_id, cycle, "STRIDE_UPDATE",
                       (unsigned long long)pc, (unsigned long long)lane_addr,
                       det);
  }

  if (newly_valid) {
    if (m_cfg.debug) {
      printf(
          "GRASP CT: stride converged ct_idx=%d pc=0x%llx stride=%lld sm=%u\n",
          ct_idx, (unsigned long long)pc,
          (long long)m_ct.entry(ct_idx).iter_stride, m_sm_id);
    }
    if (grasp_tracer::enabled()) {
      char det[96];
      snprintf(det, sizeof(det), "ct_idx=%d;stride=%lld",
               ct_idx, (long long)m_ct.entry(ct_idx).iter_stride);
      grasp_tracer::emit(m_sm_id, warp_id, cycle, "STRIDE_CONVERGE",
                         (unsigned long long)pc, (unsigned long long)lane_addr,
                         det);
    }
    // Freeze CD when all CT entries have learned strides.
    // This prevents wasted FIFO operations and spurious CHAIN_DETECT events.
    m_cd.check_freeze(m_ct);
  }

  // Generate prefetch only if CT stride is learned
  if (!cte_mut.stride_valid || cte_mut.iter_stride == 0) return;

  // Dedup: only generate prefetch once per warp instruction (first sector).
  if (!is_first_sector) return;

  // Get seed chain IDs for this PC (needed for pair table lookup)
  std::vector<unsigned> seed_chain_ids;
  if (warp != nullptr) {
    seed_chain_ids =
        warp->get_ima_seed_chain_ids(static_cast<address_type>(pc));
  }
  if (seed_chain_ids.empty()) return;  // No chain mapping → can't do pair lookup

  // v7 P1+P2: Warp-level coalesced prefetch with shared PRB.
  // For every active lane, compute predicted_addr = lane_addr + distance * stride,
  // group by sector (32B), allocate ONE shared PRB, queue per-sector with lane_addrs.
  if (mf) {
    const warp_inst_t &pf_inst = mf->get_inst();
    if (!pf_inst.empty()) {
      // Guard: skip if per-lane addresses are not valid (e.g., barrier /
      // control-flow inst reaching this path, or inst constructed before
      // address computation). Without this check, get_addr(l) below asserts
      // in abstract_hardware_model.h:1222 and crashes the simulator — first
      // surfaced on LS PTA kernel 11 (2026-04-07).
      if (!pf_inst.has_per_thread_info()) return;
      const unsigned n = m_cfg.ist_distance;
      static const new_addr_type SECTOR_MASK = ~((new_addr_type)31);

      // Compute predicted lane addresses (shared by IPU and Data-Only paths)
      std::vector<new_addr_type> predicted_addrs;
      for (unsigned l = 0; l < pf_inst.warp_size(); ++l) {
        if (!pf_inst.active(l)) continue;
        predicted_addrs.push_back(static_cast<new_addr_type>(
            static_cast<int64_t>(pf_inst.get_addr(l)) +
            static_cast<int64_t>(n) * cte_mut.iter_stride));
      }
      if (predicted_addrs.empty()) return;

      if (m_cfg.ipu_enable) {
        // === IPU path: enqueue INDEX_PF ===
        // 2a. Group predicted lane addresses by sector
        std::vector<std::pair<new_addr_type, std::vector<new_addr_type>>>
            sector_to_lanes;
        for (auto predicted : predicted_addrs) {
          new_addr_type sector = predicted & SECTOR_MASK;
          bool found = false;
          for (auto &p : sector_to_lanes) {
            if (p.first == sector) {
              bool dup = false;
              for (auto a : p.second) {
                if (a == predicted) { dup = true; break; }
              }
              if (!dup) p.second.push_back(predicted);
              found = true;
              break;
            }
          }
          if (!found) {
            sector_to_lanes.push_back({sector, {predicted}});
          }
        }

        if (sector_to_lanes.empty()) return;

        // 2b. Allocate ONE shared PRB entry
        unsigned num_sectors = static_cast<unsigned>(sector_to_lanes.size());
        int prb_id = m_prb.allocate(warp_id, num_sectors, cycle);
        if (prb_id < 0) return;  // PRB full — skip entire instruction

        // 2c. Queue one INDEX_PF per sector with shared prb_id + lane_addrs
        for (auto &p : sector_to_lanes) {
          queue_prefetch(p.first, warp_id, seed_chain_ids, cycle,
                         prb_id, p.second);
          ++m_stats.ist_prefetch_generated;
          if (grasp_tracer::enabled()) {
            char det[128];
            snprintf(det, sizeof(det),
                     "ct_idx=%d;distance=%u;stride=%lld;n_sectors=%u;prb=%d;"
                     "lanes=%zu",
                     ct_idx, n, (long long)cte_mut.iter_stride,
                     num_sectors, prb_id, p.second.size());
            grasp_tracer::emit(m_sm_id, warp_id, cycle, "IDX_PF_ENQUEUE",
                               (unsigned long long)pc,
                               (unsigned long long)p.first, det);
          }
        }
      } else if (m_cfg.dpu_enable) {
        // === Data-Only path: skip INDEX_PF, generate DATA_PF directly ===
        // Pair table lookup on predicted addresses → enqueue data prefetches
        if (warp) {
          std::set<new_addr_type> seen_sectors;
          for (auto pa : predicted_addrs) {
            auto cands = warp->lookup_ima_prefetch_candidates(
                pa, seed_chain_ids, /*exact_match_only=*/true);
            for (auto &c : cands) {
              new_addr_type data_sector = c.data_addr & SECTOR_MASK;
              if (seen_sectors.insert(data_sector).second) {
                queue_prefetch(data_sector, warp_id, c.successor_chain_ids,
                               cycle);
                ++m_stats.data_pf_enqueued;
              }
            }
          }
        }
      }
      return;
    }
  }

  // Fallback (no mf): single-address prefetch at target distance
  if (m_cfg.ipu_enable) {
    new_addr_type paddr = static_cast<new_addr_type>(
        static_cast<int64_t>(addr) +
        static_cast<int64_t>(m_cfg.ist_distance) * cte_mut.iter_stride);
    new_addr_type sector = paddr & ~((new_addr_type)31);
    int prb_id = m_prb.allocate(warp_id, 1, cycle);
    if (prb_id >= 0) {
      queue_prefetch(sector, warp_id, seed_chain_ids, cycle, prb_id, {paddr});
      ++m_stats.ist_prefetch_generated;
    }
  }
}

// ---------------------------------------------------------------------------
// Prefetch Injection
// ---------------------------------------------------------------------------

grasp_prefetcher_t::prefetch_result_t grasp_prefetcher_t::inject_prefetch(
    unsigned long long cycle, l1_cache *l1d,
    mem_fetch_allocator *mf_alloc, unsigned sid, unsigned tpc,
    const memory_config *mem_cfg, unsigned cache_line_sz,
    gpgpu_context *gpgpu_ctx,
    const std::function<shd_warp_t *(unsigned)> &get_warp) {
  prefetch_result_t result;
  result.stalled = false;

  if (!m_cfg.enable || !l1d || m_prefetch_queue.empty()) return result;

  // Inject up to 4 sectors per cycle (matching L1D bank bandwidth).
  static const unsigned MAX_INJECT_PER_CYCLE = 4;
  for (unsigned injected = 0; injected < MAX_INJECT_PER_CYCLE; ++injected) {
    if (m_prefetch_queue.empty()) break;

    int ready_idx = find_ready_prefetch(cycle);
    if (ready_idx < 0) {
      result.stalled = !m_prefetch_queue.empty();
      break;
    }

    grasp_prefetch_request_t req = m_prefetch_queue[ready_idx];

    // Throttle Control: suppress DATA_PF based on tc_mode strategy.
    // INDEX_PF is NOT throttled (critical for pipeline correctness).
    // Bottleneck mode (no_throttle=true) bypasses the entire gate.
    bool is_data_pf = req.seed_chain_ids.empty();
    if (is_data_pf && l1d && !m_cfg.no_throttle) {
      bool suppress = false;

      if (m_cfg.tc_mode == 0) {
        // Legacy: binary MSHR threshold (exact current behavior)
        if (m_cfg.tc_mshr_threshold > 0) {
          float pct = l1d->mshr_occupancy_ratio() * 100.0f;
          suppress = (pct >= (float)m_cfg.tc_mshr_threshold);
        }
      } else if (m_cfg.tc_mode == 1) {
        // S1: Dual-threshold proportional
        float pct = l1d->mshr_occupancy_ratio() * 100.0f;
        if (pct >= (float)m_cfg.tc_mshr_hi) {
          suppress = true;
        } else if (pct > (float)m_cfg.tc_mshr_lo) {
          float drop_prob = (pct - (float)m_cfg.tc_mshr_lo) /
                            (float)(m_cfg.tc_mshr_hi - m_cfg.tc_mshr_lo);
          suppress = ((cycle % 100) < (unsigned long long)(drop_prob * 100));
        }
      } else if (m_cfg.tc_mode == 2) {
        // S2: Queue-depth cap
        if (m_cfg.tc_queue_cap > 0) {
          unsigned data_count = 0;
          for (const auto &e : m_prefetch_queue)
            if (e.seed_chain_ids.empty()) ++data_count;
          suppress = (data_count >= m_cfg.tc_queue_cap);
        }
      } else if (m_cfg.tc_mode == 3) {
        // S3: Accuracy-gated adaptive threshold
        float pct = l1d->mshr_occupancy_ratio() * 100.0f;
        unsigned long long total_pt = m_stats.pair_table_lookup_hit +
                                      m_stats.pair_table_lookup_miss;
        float acc = total_pt > 0
            ? 100.0f * (float)m_stats.pair_table_lookup_hit / (float)total_pt
            : 50.0f;
        float eff_thr;
        if (acc <= (float)m_cfg.tc_acc_lo)
          eff_thr = (float)m_cfg.tc_mshr_lo;
        else if (acc >= (float)m_cfg.tc_acc_hi)
          eff_thr = (float)m_cfg.tc_mshr_hi;
        else {
          float t = (acc - (float)m_cfg.tc_acc_lo) /
                    (float)(m_cfg.tc_acc_hi - m_cfg.tc_acc_lo);
          eff_thr = (float)m_cfg.tc_mshr_lo +
                    t * (float)(m_cfg.tc_mshr_hi - m_cfg.tc_mshr_lo);
        }
        suppress = (pct >= eff_thr);
      } else if (m_cfg.tc_mode == 4) {
        // S4: Cooldown timer
        float pct = l1d->mshr_occupancy_ratio() * 100.0f;
        if (cycle < m_tc_cooldown_until) {
          suppress = true;
        } else if (m_cfg.tc_mshr_threshold > 0 &&
                   pct >= (float)m_cfg.tc_mshr_threshold) {
          suppress = true;
          m_tc_cooldown_until = cycle + m_cfg.tc_cooldown_cycles;
        }
      } else if (m_cfg.tc_mode == 5) {
        // S5: Dynamic cooldown with sliding window accuracy feedback
        // Update window periodically using L1 pf_useful/pf_useless deltas
        if (cycle >= m_tc_last_snapshot_cycle + m_cfg.tc_window_cycles) {
          unsigned long long cur_useful = l1d->pf_useful_count();
          unsigned long long cur_useless = l1d->pf_useless_count();
          unsigned long long d_useful = cur_useful - m_tc_snapshot_useful;
          unsigned long long d_useless = cur_useless - m_tc_snapshot_useless;
          unsigned long long d_total = d_useful + d_useless;
          m_tc_window_accuracy = d_total > 0
              ? (float)d_useful * 100.0f / (float)d_total : 50.0f;
          m_tc_snapshot_useful = cur_useful;
          m_tc_snapshot_useless = cur_useless;
          m_tc_last_snapshot_cycle = cycle;
        }
        // Interpolate effective MSHR threshold from window accuracy
        float eff_thr;
        if (m_tc_window_accuracy <= (float)m_cfg.tc_acc_lo)
          eff_thr = (float)m_cfg.tc_mshr_lo;
        else if (m_tc_window_accuracy >= (float)m_cfg.tc_acc_hi)
          eff_thr = (float)m_cfg.tc_mshr_hi;
        else {
          float t = (m_tc_window_accuracy - (float)m_cfg.tc_acc_lo) /
                    (float)(m_cfg.tc_acc_hi - m_cfg.tc_acc_lo);
          eff_thr = (float)m_cfg.tc_mshr_lo +
                    t * (float)(m_cfg.tc_mshr_hi - m_cfg.tc_mshr_lo);
        }
        // Cooldown + dynamic threshold
        float pct = l1d->mshr_occupancy_ratio() * 100.0f;
        if (cycle < m_tc_cooldown_until) {
          suppress = true;
        } else if (pct >= eff_thr) {
          suppress = true;
          m_tc_cooldown_until = cycle + m_cfg.tc_cooldown_cycles;
        }
      }

      if (suppress) {
        m_prefetch_queue.erase(m_prefetch_queue.begin() + ready_idx);
        ++m_stats.throttle_suppressed;
        continue;
      }
    }

    m_prefetch_queue.erase(m_prefetch_queue.begin() + ready_idx);

    new_addr_type paddr = req.addr;

  // Compute sector mask (128B cache line, 32B sector)
  const unsigned line_sz = cache_line_sz;
  const unsigned sector_sz = SECTOR_SIZE;
  unsigned sector_idx = (paddr % line_sz) / sector_sz;
  mem_access_sector_mask_t sector_mask;
  sector_mask.set(sector_idx);
  mem_access_byte_mask_t byte_mask;
  byte_mask.set();
  active_mask_t all_active;
  all_active.set();

  mem_access_t prefetch_acc(GLOBAL_ACC_R, paddr, sector_sz, /*wr=*/false,
                             all_active, byte_mask, sector_mask,
                             gpgpu_ctx);

  int prb_entry_id = -1;
  ima_prefetch_kind_t pf_kind =
      req.seed_chain_ids.empty() ? IMA_PREFETCH_DATA : IMA_PREFETCH_INDEX;

  // Bottleneck mode: retry requests skip the pair-table lookup + PRB
  // candidate add path — candidates were stored on the first attempt.
  // Just propagate prb_entry_id so on_l1_access_result can find the PRB entry.
  if (req.is_retry && !req.seed_chain_ids.empty()) {
    prb_entry_id = req.prb_entry_id;
  } else if (!req.seed_chain_ids.empty()) {
    // v7 P1: PRB was pre-allocated in on_demand_load
    assert(req.prb_entry_id >= 0);
    prb_entry_id = req.prb_entry_id;

    // v7 P2: exact pair table lookup for each predicted lane address
    shd_warp_t *warp = get_warp(req.warp_id);
    assert(warp != nullptr);
    std::vector<ima_prefetch_candidate_t> cands;
    std::set<std::tuple<unsigned, new_addr_type, new_addr_type>> dedupe;
    for (new_addr_type la : req.lane_addrs) {
      auto partial = warp->lookup_ima_prefetch_candidates(
          la, req.seed_chain_ids, /*exact_match_only=*/true);
      for (auto &c : partial) {
        auto key = std::make_tuple(c.source_chain_id, c.idx_addr, c.data_addr);
        if (dedupe.insert(key).second) {
          cands.push_back(c);
        }
      }
    }
    warp->record_ima_verify_predictions(cands, cycle, m_cfg.debug);

    // Stats and trace (before move)
    size_t num_cands = cands.size();
    if (num_cands > 0) {
      ++m_stats.pair_table_lookup_hit;
    } else {
      ++m_stats.pair_table_lookup_miss;
      if (m_cfg.debug) {
        printf("GRASP_PT_MISS: sm=%u warp=%u sector=0x%llx lane_addrs=[",
               m_sm_id, req.warp_id, (unsigned long long)paddr);
        for (size_t i = 0; i < req.lane_addrs.size(); ++i) {
          if (i > 0) printf(",");
          printf("0x%llx", (unsigned long long)req.lane_addrs[i]);
        }
        printf("] chains=[");
        for (size_t i = 0; i < req.seed_chain_ids.size(); ++i) {
          if (i > 0) printf(",");
          printf("%u", req.seed_chain_ids[i]);
        }
        printf("]\n");
      }
    }
    if (m_cfg.debug) {
      printf(
          "GRASP inject INDEX_PF: sid=%u warp=%u addr=0x%llx lanes=%zu "
          "hits=%zu prb=%d cycle=%llu\n",
          sid, req.warp_id, (unsigned long long)req.addr,
          req.lane_addrs.size(), num_cands, prb_entry_id, cycle);
    }
    if (grasp_tracer::enabled()) {
      char det[128];
      snprintf(det, sizeof(det), "pair_hits=%zu;prb_id=%d;lanes=%zu",
               num_cands, prb_entry_id, req.lane_addrs.size());
      grasp_tracer::emit(sid, req.warp_id, cycle, "IDX_PF_INJECT",
                         0, (unsigned long long)paddr, det);
    }
    // Always register sector in PRB (even if empty) so fill/RFAIL can
    // find it and decrement remaining_sectors.
    m_prb.add_sector_candidates(prb_entry_id, req.addr, std::move(cands));
  }
  if (pf_kind == IMA_PREFETCH_DATA && grasp_tracer::enabled()) {
    grasp_tracer::emit(sid, req.warp_id, cycle, "DATA_PF_INJECT",
                       0, (unsigned long long)paddr, "");
  }

  // Create mem_fetch
  mem_fetch *pf_mf =
      new mem_fetch(prefetch_acc, /*inst=*/nullptr, /*streamID=*/0,
                    READ_PACKET_SIZE, /*wid=*/req.warp_id, sid, tpc,
                    mem_cfg, cycle);
  pf_mf->set_ima_metadata(pf_kind,
                           prb_entry_id >= 0 ? (unsigned)prb_entry_id
                                             : (unsigned)-1);

  result.requests.push_back(pf_mf);
  }  // end for (injected)
  return result;
}

// ---------------------------------------------------------------------------
// L1 Access Result Tracking
// ---------------------------------------------------------------------------

void grasp_prefetcher_t::on_l1_access_result(
    mem_fetch *mf, int cache_status, unsigned long long cycle,
    enum cache_reservation_fail_reason fail_reason) {
  if (!mf->is_ima_prefetch()) return;

  bool is_index = (mf->get_ima_kind() == IMA_PREFETCH_INDEX);
  unsigned prb_entry_id = mf->get_ima_prb_entry_id();

  // Helper lambda for L1 result trace emit
  auto emit_l1_result = [&](const char *ev) {
    if (!grasp_tracer::enabled()) return;
    char det[96];
    snprintf(det, sizeof(det), "kind=%s;prb_id=%u",
             is_index ? "INDEX" : "DATA", prb_entry_id);
    grasp_tracer::emit(m_sm_id, mf->get_wid(), cycle, ev,
                       0, (unsigned long long)mf->get_addr(), det);
  };

  static const new_addr_type SECTOR_MASK = ~((new_addr_type)31);

  if (cache_status == HIT) {
    emit_l1_result("L1_RESULT_HIT");
    // v7 P1: per-sector dispatch on HIT
    if (is_index && prb_entry_id != (unsigned)-1 &&
        m_prb.get(prb_entry_id).valid) {
      new_addr_type sector = mf->get_addr() & SECTOR_MASK;
      release_prb_sector_targets(prb_entry_id, sector, cycle + 1);
      ++m_stats.index_pf_hit;
    } else if (!is_index) {
      ++m_stats.data_pf_hit;
    }
    ++m_ist.stat_prefetch_filtered;
    delete mf;  // HIT: cache did NOT take ownership
  } else if (cache_status == MSHR_HIT || cache_status == HIT_RESERVED) {
    emit_l1_result("L1_RESULT_MSHR");
    // v7 P3: cache already took ownership of mf via m_mshrs.add().
    // Keep PRB alive. When the MSHR fill completes, the merged mf
    // comes through writeback case 4 → on_fill() → sector dispatch.
    if (is_index) {
      ++m_stats.index_pf_mshr_merge;
    } else {
      ++m_stats.data_pf_mshr_merge;
    }
    // DO NOT delete mf — cache owns it (stored in MSHR entry's m_list).
    // DO NOT free PRB — on_fill will handle it.
  } else if (cache_status == RESERVATION_FAIL) {
    emit_l1_result("L1_RESULT_RFAIL");

    // Bottleneck mode: retry by re-enqueueing the request for next cycle.
    // Must NOT call mark_sector_failed — PRB entry still owns this sector.
    // Safety cap (16384) prevents unbounded queue growth under pathological load.
    static const size_t RETRY_QUEUE_CAP = 16384;
    if (m_cfg.no_throttle && m_prefetch_queue.size() < RETRY_QUEUE_CAP) {
      grasp_prefetch_request_t retry_req;
      retry_req.addr = mf->get_addr();
      retry_req.warp_id = mf->get_wid();
      retry_req.ready_cycle = cycle + 1;
      retry_req.is_retry = true;
      if (is_index) {
        // Preserve INDEX kind via dummy non-empty seed_chain_ids (content
        // ignored because is_retry=true bypasses pair-table lookup).
        retry_req.seed_chain_ids.push_back(0);
        retry_req.prb_entry_id = (int)prb_entry_id;
        ++m_stats.index_pf_rfail_retried;
        ++m_stats.index_pf_reservation_fail;
        ++m_stats.index_pf_rfail[fail_reason];
      } else {
        // DATA_PF: empty seed_chain_ids preserves DATA kind
        retry_req.prb_entry_id = -1;
        ++m_stats.data_pf_rfail_retried;
        ++m_stats.data_pf_reservation_fail;
        ++m_stats.data_pf_rfail[fail_reason];
      }
      m_prefetch_queue.push_back(retry_req);
      delete mf;  // RFAIL: cache did NOT take ownership; retry rebuilds mf
      return;
    }

    // v7 P1: per-sector RFAIL — mark failed + decrement remaining_sectors
    if (is_index && prb_entry_id != (unsigned)-1 &&
        m_prb.get(prb_entry_id).valid) {
      new_addr_type sector = mf->get_addr() & SECTOR_MASK;
      mark_sector_failed(prb_entry_id, sector);
      ++m_stats.index_pf_reservation_fail;
      ++m_stats.index_pf_rfail[fail_reason];
    } else {
      ++m_stats.data_pf_reservation_fail;
      ++m_stats.data_pf_rfail[fail_reason];
    }
    ++m_ist.stat_prefetch_filtered;
    delete mf;  // RFAIL: cache did NOT take ownership
  } else {
    emit_l1_result("L1_RESULT_MISS");
    // MISS or SECTOR_MISS: cache took ownership, will fetch from L2.
    // on_fill() will dispatch this sector's targets when data returns.
    if (is_index) {
      ++m_stats.index_pf_issued;
    } else {
      ++m_stats.data_pf_issued;
    }
    ++m_ist.stat_prefetch_issued;
    // DO NOT delete mf — cache owns it (in MSHR + miss queue).
  }
}

// ---------------------------------------------------------------------------
// Fill Callback
// ---------------------------------------------------------------------------

void grasp_prefetcher_t::on_fill(mem_fetch *mf, unsigned long long fill_cycle) {
  if (mf == nullptr || !mf->is_ima_prefetch()) return;

  if (mf->get_ima_kind() == IMA_PREFETCH_INDEX) {
    unsigned prb_entry_id = mf->get_ima_prb_entry_id();
    if (prb_entry_id == (unsigned)-1) return;
    // Guard: PRB may have been reset by kernel switch while PF was in-flight
    if (!m_prb.get(prb_entry_id).valid) return;

    // v7 P1: per-sector dispatch. release_prb_sector_targets is idempotent
    // (dispatched flag), so duplicate calls from response_fifo + writeback are safe.
    static const new_addr_type SECTOR_MASK = ~((new_addr_type)31);
    new_addr_type sector = mf->get_addr() & SECTOR_MASK;
    release_prb_sector_targets(prb_entry_id, sector, fill_cycle + 1);
    // PRB free handled inside release_prb_sector_targets when remaining_sectors==0
  }
  // DATA_PF: fire-and-forget, stats already updated on issue
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

void grasp_prefetcher_t::print_config(FILE *fp) const {
  fprintf(fp,
          "GRASP_CONFIG: cd_fifo=%u ct_size=%u tt_size=%u "
          "ist_ipt=%u ist_dist=%u ist_conf=%u "
          "prb_cap=%u tc_mode=%u tc_mshr_thr=%u "
          "tc_mshr_lo=%u tc_mshr_hi=%u tc_queue_cap=%u tc_cooldown=%u "
          "tc_acc_lo=%u tc_acc_hi=%u tc_window=%u "
          "pt_scope=%u spec_stride=%lld ipu=%d dpu=%d no_throttle=%d\n",
          m_cfg.cd_fifo_depth, m_cfg.ct_size, m_cfg.tt_size,
          m_cfg.ist_ipt_size, m_cfg.ist_distance, m_cfg.ist_confidence,
          m_cfg.prb_capacity, m_cfg.tc_mode, m_cfg.tc_mshr_threshold,
          m_cfg.tc_mshr_lo, m_cfg.tc_mshr_hi, m_cfg.tc_queue_cap,
          m_cfg.tc_cooldown_cycles, m_cfg.tc_acc_lo, m_cfg.tc_acc_hi,
          m_cfg.tc_window_cycles,
          m_cfg.pair_table_scope, (long long)m_cfg.speculative_stride,
          (int)m_cfg.ipu_enable, (int)m_cfg.dpu_enable,
          (int)m_cfg.no_throttle);
}

void grasp_prefetcher_t::print_stats(FILE *fp, unsigned long long pf_useful,
                                     unsigned long long pf_useless,
                                     unsigned long long pf_late,
                                     const ima_demand_breakdown_t *ima) const {
  fprintf(fp, "GRASP_SM%u: ", m_sm_id);
  fprintf(fp, "kernels=%llu warp_exits=%llu ", m_stats.kernel_launches,
          m_stats.warp_exits);
  fprintf(fp, "idx_pf(issued=%llu hit=%llu miss=%llu mshr=%llu rfail=%llu) ",
          m_stats.index_pf_issued, m_stats.index_pf_hit,
          m_stats.index_pf_miss, m_stats.index_pf_mshr_merge,
          m_stats.index_pf_reservation_fail);
  fprintf(fp,
          "idx_rfail(line_alloc=%llu missq=%llu mshr_entry=%llu "
          "mshr_merge=%llu rw_pending=%llu) ",
          m_stats.index_pf_rfail[LINE_ALLOC_FAIL],
          m_stats.index_pf_rfail[MISS_QUEUE_FULL],
          m_stats.index_pf_rfail[MSHR_ENRTY_FAIL],
          m_stats.index_pf_rfail[MSHR_MERGE_ENRTY_FAIL],
          m_stats.index_pf_rfail[MSHR_RW_PENDING]);
  fprintf(fp, "data_pf(issued=%llu hit=%llu miss=%llu mshr=%llu rfail=%llu) ",
          m_stats.data_pf_issued, m_stats.data_pf_hit,
          m_stats.data_pf_miss, m_stats.data_pf_mshr_merge,
          m_stats.data_pf_reservation_fail);
  fprintf(fp,
          "data_rfail(line_alloc=%llu missq=%llu mshr_entry=%llu "
          "mshr_merge=%llu rw_pending=%llu) ",
          m_stats.data_pf_rfail[LINE_ALLOC_FAIL],
          m_stats.data_pf_rfail[MISS_QUEUE_FULL],
          m_stats.data_pf_rfail[MSHR_ENRTY_FAIL],
          m_stats.data_pf_rfail[MSHR_MERGE_ENRTY_FAIL],
          m_stats.data_pf_rfail[MSHR_RW_PENDING]);
  fprintf(fp, "pair_table(hit=%llu miss=%llu) ", m_stats.pair_table_lookup_hit,
          m_stats.pair_table_lookup_miss);
  fprintf(fp, "queue_peak=%llu ", m_stats.queue_peak_depth);
  fprintf(fp, "\n");

  // Sub-component stats
  m_ist.print_stats(fp);

  const auto &cd_s = m_cd.stats();
  fprintf(fp,
          "GRASP_CD SM%u: chains=%llu fifo_peak=%llu drops=%llu "
          "read_inv=%llu write_inv=%llu frozen=%s\n",
          m_sm_id, cd_s.chains_detected, cd_s.fifo_peak_occupancy,
          cd_s.fifo_drop_count, cd_s.fifo_read_invalidations,
          cd_s.fifo_write_invalidations,
          m_cd.is_frozen() ? "yes" : "no");

  const auto &ct_s = m_ct.stats();
  fprintf(fp,
          "GRASP_CT SM%u: peak_occ=%llu evictions=%llu evict_stride_valid=%llu "
          "all_stride_valid=%s\n",
          m_sm_id, ct_s.peak_occupancy, ct_s.eviction_count,
          ct_s.eviction_stride_valid_count,
          m_ct.all_stride_valid() ? "yes" : "no");

  const auto &prb_s = m_prb.stats();
  fprintf(fp,
          "GRASP_PRB SM%u: peak_occ=%llu allocs=%llu stall_cycles=%llu "
          "drops=%llu\n",
          m_sm_id, prb_s.peak_occupancy, prb_s.total_allocations,
          prb_s.full_stall_cycles, prb_s.prb_drop_count);

  // Storage utilization summary (peak occupancy vs configured capacity)
  {
    double prb_util = m_cfg.prb_capacity > 0
                          ? 100.0 * prb_s.peak_occupancy / m_cfg.prb_capacity
                          : 0.0;
    double ct_util = m_cfg.ct_size > 0
                         ? 100.0 * ct_s.peak_occupancy / m_cfg.ct_size
                         : 0.0;
    double cd_util = m_cfg.cd_fifo_depth > 0
                         ? 100.0 * cd_s.fifo_peak_occupancy / m_cfg.cd_fifo_depth
                         : 0.0;
    fprintf(fp,
            "GRASP_STORAGE SM%u: prb=%llu/%u(%.1f%%) ct=%llu/%u(%.1f%%) "
            "cd_fifo=%llu/%u(%.1f%%) pf_queue_peak=%llu\n",
            m_sm_id, prb_s.peak_occupancy, m_cfg.prb_capacity, prb_util,
            ct_s.peak_occupancy, m_cfg.ct_size, ct_util,
            cd_s.fifo_peak_occupancy, m_cfg.cd_fifo_depth, cd_util,
            m_stats.queue_peak_depth);
  }

  // Demand load stats (tracked in grasp_stats_t)
  fprintf(fp,
          "GRASP_DEMAND SM%u: global_reads=%llu global_misses=%llu "
          "throttle_suppressed=%llu\n",
          m_sm_id, m_stats.total_demand_global_reads,
          m_stats.total_demand_global_misses,
          m_stats.throttle_suppressed);

  // IMA demand breakdown per SM (from ldst_unit, chain CSV-based PC filter)
  if (ima) {
    fprintf(fp,
            "GRASP_IMA_DEMAND SM%u:"
            " idx(reads=%llu hits=%llu hit_res=%llu misses=%llu)"
            " data(reads=%llu hits=%llu hit_res=%llu misses=%llu)\n",
            m_sm_id,
            ima->idx_reads, ima->idx_hits, ima->idx_hit_reserved, ima->idx_misses,
            ima->data_reads, ima->data_hits, ima->data_hit_reserved, ima->data_misses);
    // Timeliness = hits / (hits + hit_reserved)
    unsigned long long idx_a = ima->idx_hits + ima->idx_hit_reserved;
    unsigned long long data_a = ima->data_hits + ima->data_hit_reserved;
    fprintf(fp, "GRASP_TIMELINESS SM%u:", m_sm_id);
    if (idx_a > 0)
      fprintf(fp, " index=%.2f%%", 100.0 * ima->idx_hits / idx_a);
    if (data_a > 0)
      fprintf(fp, " data=%.2f%%", 100.0 * ima->data_hits / data_a);
    fprintf(fp, "\n");
  }

  // Legacy prefetch effectiveness metrics (kept for backward compat / debug)
  fprintf(fp,
          "GRASP_EFFECT SM%u: pf_useful=%llu pf_useless=%llu pf_late=%llu",
          m_sm_id, pf_useful, pf_useless, pf_late);
  unsigned long long total_classified = pf_useful + pf_useless;
  if (total_classified > 0) {
    fprintf(fp, " accuracy=%.2f%%", 100.0 * pf_useful / total_classified);
  }
  fprintf(fp, "\n");

  // Reservation failure breakdown: merge index+data, report per-reason %
  {
    unsigned long long rfail_combined[NUM_CACHE_RESERVATION_FAIL_STATUS];
    unsigned long long rfail_total = 0;
    for (int r = 0; r < NUM_CACHE_RESERVATION_FAIL_STATUS; ++r) {
      rfail_combined[r] =
          m_stats.index_pf_rfail[r] + m_stats.data_pf_rfail[r];
      rfail_total += rfail_combined[r];
    }
    fprintf(fp, "GRASP_RFAIL SM%u: total=%llu", m_sm_id, rfail_total);
    if (rfail_total > 0) {
      fprintf(fp,
              " line_alloc=%.1f%% missq=%.1f%% mshr_entry=%.1f%%"
              " mshr_merge=%.1f%% rw_pending=%.1f%%",
              100.0 * rfail_combined[LINE_ALLOC_FAIL] / rfail_total,
              100.0 * rfail_combined[MISS_QUEUE_FULL] / rfail_total,
              100.0 * rfail_combined[MSHR_ENRTY_FAIL] / rfail_total,
              100.0 * rfail_combined[MSHR_MERGE_ENRTY_FAIL] / rfail_total,
              100.0 * rfail_combined[MSHR_RW_PENDING] / rfail_total);
    }
    fprintf(fp, "\n");
  }

  // Prefetch pipeline funnel: index → data conversion tracking
  {
    auto idx_attempted = m_stats.index_pf_issued + m_stats.index_pf_hit +
                         m_stats.index_pf_mshr_merge +
                         m_stats.index_pf_reservation_fail;
    auto idx_rfail = m_stats.index_pf_reservation_fail;
    auto idx_got_data = idx_attempted - idx_rfail;
    auto data_enqueued = m_stats.data_pf_enqueued;
    auto data_throttled = m_stats.throttle_suppressed;
    auto data_attempted = m_stats.data_pf_issued + m_stats.data_pf_hit +
                          m_stats.data_pf_mshr_merge +
                          m_stats.data_pf_reservation_fail;
    auto data_rfail = m_stats.data_pf_reservation_fail;
    auto data_got_data = data_attempted - data_rfail;
    fprintf(fp,
            "GRASP_FUNNEL SM%u: idx_attempted=%llu idx_rfail=%llu",
            m_sm_id, idx_attempted, idx_rfail);
    if (idx_attempted > 0) {
      fprintf(fp, "(%.1f%%)", 100.0 * idx_rfail / idx_attempted);
    }
    fprintf(fp, " idx_got_data=%llu", idx_got_data);
    fprintf(fp,
            " | data_enqueued=%llu data_throttled=%llu data_attempted=%llu"
            " data_rfail=%llu",
            data_enqueued, data_throttled, data_attempted, data_rfail);
    if (data_attempted > 0) {
      fprintf(fp, "(%.1f%%)", 100.0 * data_rfail / data_attempted);
    }
    fprintf(fp, " data_got_data=%llu\n", data_got_data);
  }

  // Bottleneck analysis: theoretical demand vs actual issue vs rfail/retry
  // Uses ima_demand_breakdown_t (from ldst_unit) as theoretical denominator.
  if (ima) {
    unsigned long long idx_accepted = m_stats.index_pf_hit +
                                      m_stats.index_pf_mshr_merge +
                                      m_stats.index_pf_issued;
    unsigned long long data_accepted = m_stats.data_pf_hit +
                                       m_stats.data_pf_mshr_merge +
                                       m_stats.data_pf_issued;
    fprintf(fp,
            "GRASP_BOTTLENECK SM%u: "
            "idx_theory=%llu idx_generated=%llu idx_accepted=%llu "
            "idx_rfail=%llu idx_retried=%llu | "
            "data_theory=%llu data_enqueued=%llu data_accepted=%llu "
            "data_rfail=%llu data_retried=%llu | "
            "tc_suppressed=%llu no_throttle=%d\n",
            m_sm_id,
            ima->idx_reads, m_stats.ist_prefetch_generated, idx_accepted,
            m_stats.index_pf_reservation_fail, m_stats.index_pf_rfail_retried,
            ima->data_reads, m_stats.data_pf_enqueued, data_accepted,
            m_stats.data_pf_reservation_fail, m_stats.data_pf_rfail_retried,
            m_stats.throttle_suppressed, (int)m_cfg.no_throttle);
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

void grasp_prefetcher_t::queue_prefetch(
    new_addr_type addr, unsigned warp_id,
    const std::vector<unsigned> &seed_chain_ids,
    unsigned long long ready_cycle,
    int prb_entry_id,
    const std::vector<new_addr_type> &lane_addrs) {
  // Dedup: skip if an identical (addr, warp_id, same kind) request is already
  // pending.  This prevents flooding the queue with duplicate prefetch requests.
  bool is_index = !seed_chain_ids.empty();
  for (const auto &existing : m_prefetch_queue) {
    if (existing.addr == addr && existing.warp_id == warp_id &&
        !existing.seed_chain_ids.empty() == is_index) {
      return;  // duplicate — skip
    }
  }

  grasp_prefetch_request_t req;
  req.addr = addr;
  req.warp_id = warp_id;
  req.seed_chain_ids = seed_chain_ids;
  req.ready_cycle = ready_cycle;
  req.prb_entry_id = prb_entry_id;
  req.lane_addrs = lane_addrs;
  m_prefetch_queue.push_back(req);

  // Track queue depth
  unsigned long long depth = m_prefetch_queue.size();
  if (depth > m_stats.queue_peak_depth) {
    m_stats.queue_peak_depth = depth;
  }
}

int grasp_prefetcher_t::find_ready_prefetch(unsigned long long cycle) const {
  for (size_t i = 0; i < m_prefetch_queue.size(); ++i) {
    if (m_prefetch_queue[i].ready_cycle <= cycle) return static_cast<int>(i);
  }
  return -1;
}

// v7 P1: per-sector dispatch — release candidates for a single sector.
// Idempotent: if sector already dispatched, returns immediately.
// Decrements remaining_sectors; frees PRB when it reaches 0.
void grasp_prefetcher_t::release_prb_sector_targets(
    unsigned prb_entry_id, new_addr_type sector_addr,
    unsigned long long ready_cycle) {
  grasp_prb_entry_t &entry = m_prb.get(prb_entry_id);
  if (!entry.valid) return;

  // Find the sector in this PRB entry
  for (auto &sd : entry.sector_data) {
    if (sd.sector_addr != sector_addr) continue;
    if (sd.dispatched) return;  // idempotent — already dispatched
    sd.dispatched = true;

    if (m_cfg.debug) {
      printf(
          "GRASP fill dispatch: sm=%u prb=%u warp=%u sector=0x%llx "
          "targets=%zu remaining=%u ready=%llu\n",
          m_sm_id, prb_entry_id, entry.warp_id,
          (unsigned long long)sector_addr, sd.candidates.size(),
          entry.remaining_sectors - 1, ready_cycle);
    }
    if (grasp_tracer::enabled()) {
      char det[128];
      snprintf(det, sizeof(det), "prb_id=%u;num_targets=%zu;remaining=%u",
               prb_entry_id, sd.candidates.size(),
               entry.remaining_sectors - 1);
      grasp_tracer::emit(m_sm_id, entry.warp_id, ready_cycle,
                         "FILL_DISPATCH",
                         0, (unsigned long long)sector_addr, det);
    }

    // Coalesce DATA_PF by 32B sector; INDEX_PF (non-empty successors) not coalesced
    if (m_cfg.dpu_enable) {
      static const new_addr_type SECTOR_MASK = ~((new_addr_type)31);
      std::vector<new_addr_type> seen;
      for (const auto &cand : sd.candidates) {
        new_addr_type data_sector = cand.data_addr & SECTOR_MASK;
        if (cand.successor_chain_ids.empty()) {
          bool dup = false;
          for (auto s : seen) {
            if (s == data_sector) { dup = true; break; }
          }
          if (dup) continue;
          seen.push_back(data_sector);
        }
        queue_prefetch(data_sector, entry.warp_id, cand.successor_chain_ids,
                       ready_cycle);
        ++m_stats.data_pf_enqueued;
        if (grasp_tracer::enabled()) {
          char det2[128];
          snprintf(det2, sizeof(det2), "chain_id=%u;prb_id=%u;succ=%zu",
                   cand.source_chain_id, prb_entry_id,
                   cand.successor_chain_ids.size());
          grasp_tracer::emit(m_sm_id, entry.warp_id, ready_cycle,
                             "DATA_PF_ENQUEUE",
                             0, (unsigned long long)data_sector, det2);
        }
      }
    }  // dpu_enable

    assert(entry.remaining_sectors > 0);
    --entry.remaining_sectors;
    if (entry.remaining_sectors == 0) {
      m_prb.free_entry(prb_entry_id);
    }
    return;
  }
  // Sector not found in PRB — no candidates were registered for it.
  // This can happen if the inject for this sector hasn't run yet
  // (e.g., MSHR_HIT for a sector whose inject is still in queue).
}

// v7: mark sector as failed (RFAIL) — no data PF dispatch, just bookkeeping.
void grasp_prefetcher_t::mark_sector_failed(unsigned prb_entry_id,
                                             new_addr_type sector_addr) {
  grasp_prb_entry_t &entry = m_prb.get(prb_entry_id);
  if (!entry.valid) return;

  for (auto &sd : entry.sector_data) {
    if (sd.sector_addr == sector_addr && !sd.dispatched) {
      sd.dispatched = true;
      assert(entry.remaining_sectors > 0);
      --entry.remaining_sectors;
      if (entry.remaining_sectors == 0) {
        m_prb.free_entry(prb_entry_id);
      }
      return;
    }
  }
}
