// GRASP Prefetcher — Top-level implementation
//
// Orchestrates all sub-components: CD, CT, TT, IST, PRB.
// Provides hooks called from ldst_unit and shader_core_ctx.

#include "grasp_prefetcher.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>

#include "gpu-cache.h"
#include "grasp_tracer.h"
#include "mem_fetch.h"
#include "shader.h"

// ============================================================================
// PRB Implementation
// ============================================================================

grasp_prb_t::grasp_prb_t(unsigned initial_capacity)
    : m_entries(initial_capacity) {}

int grasp_prb_t::allocate(unsigned warp_id,
                           const std::vector<ima_prefetch_candidate_t> &cands,
                           unsigned num_sectors,
                           unsigned long long cycle) {
  for (size_t i = 0; i < m_entries.size(); ++i) {
    if (!m_entries[i].valid) {
      m_entries[i].valid = true;
      m_entries[i].warp_id = warp_id;
      m_entries[i].candidates = cands;
      m_entries[i].remaining_sectors = num_sectors;
      m_entries[i].alloc_cycle = cycle;
      ++m_stats.current_occupancy;
      ++m_stats.total_allocations;
      if (m_stats.current_occupancy > m_stats.peak_occupancy) {
        m_stats.peak_occupancy = m_stats.current_occupancy;
      }
      return static_cast<int>(i);
    }
  }
  // Expand if capacity allows (Phase 1: dynamic growth)
  grasp_prb_entry_t entry;
  entry.valid = true;
  entry.warp_id = warp_id;
  entry.candidates = cands;
  entry.remaining_sectors = num_sectors;
  entry.alloc_cycle = cycle;
  m_entries.push_back(entry);
  ++m_stats.current_occupancy;
  ++m_stats.total_allocations;
  if (m_stats.current_occupancy > m_stats.peak_occupancy) {
    m_stats.peak_occupancy = m_stats.current_occupancy;
  }
  return static_cast<int>(m_entries.size() - 1);
}

void grasp_prb_t::free_entry(unsigned prb_entry_id) {
  // C2: PRB entry must be valid before free
  assert(prb_entry_id < m_entries.size());
  assert(m_entries[prb_entry_id].valid);
  unsigned long long lifetime =
      0;  // Would need current cycle for exact lifetime
  m_entries[prb_entry_id].valid = false;
  m_entries[prb_entry_id].warp_id = (unsigned)-1;
  m_entries[prb_entry_id].candidates.clear();
  m_entries[prb_entry_id].retry_count = 0;
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
    e.candidates.clear();
    e.retry_count = 0;
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
      m_ct(cfg.ct_size),
      m_tt(cfg.tt_size),
      m_ist(sm_id, cfg.ist_ipt_size, cfg.ist_distance, cfg.ist_confidence),
      m_prb(cfg.prb_capacity),
      m_demand_dedup(64) {}  // 64 warps per SM max

grasp_prefetcher_t::~grasp_prefetcher_t() = default;

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void grasp_prefetcher_t::on_kernel_launch(const void *kernel_entry) {
  bool same_kernel = (kernel_entry != nullptr &&
                      kernel_entry == m_last_kernel_entry);
  m_last_kernel_entry = kernel_entry;

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
  new_addr_type first_lane_addr = 0;
  unsigned first_lane_id = 0;
  if (!inst.empty() && inst.is_load()) {
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
                                         const mem_fetch *mf) {
  if (!m_cfg.enable) return;

  // Check if this PC is an IMA index load (registered in CT by CD)
  int ct_idx = m_ct.find(pc);
  if (ct_idx < 0) return;  // Not an IMA load — skip

  // ── Per-warp stride learning ──
  // Every warp records its first-lane address on first visit. When any warp
  // revisits, stride is computed from that warp's own address delta.
  // This is much more robust than locking to a single warp: whichever warp
  // iterates first determines the stride.
  ct_entry_t &cte_mut = m_ct.entry(ct_idx);

  // Extract lowest active lane and its address from the warp instruction
  new_addr_type lane_addr = addr;  // fallback
  unsigned lane_id = (unsigned)-1;
  bool has_lane_addr = false;
  if (mf) {
    const warp_inst_t &inst = mf->get_inst();
    if (!inst.empty() && inst.is_load()) {
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

  // Warp-level coalesced prefetch: for every active lane, compute
  // predicted_addr = lane_addr + distance * stride, then coalesce by
  // sector (32B) and queue one INDEX_PF per unique sector.
  // Only prefetch at the target distance (not 1..distance).
  if (mf) {
    const warp_inst_t &pf_inst = mf->get_inst();
    if (!pf_inst.empty()) {
      {
        const unsigned n = m_cfg.ist_distance;
        // Collect predicted addresses, deduplicate by sector (32B)
        std::vector<new_addr_type> sector_addrs;
        for (unsigned l = 0; l < pf_inst.warp_size(); ++l) {
          if (!pf_inst.active(l)) continue;
          new_addr_type predicted = static_cast<new_addr_type>(
              static_cast<int64_t>(pf_inst.get_addr(l)) +
              static_cast<int64_t>(n) * cte_mut.iter_stride);
          new_addr_type sector = predicted & ~((new_addr_type)31);
          bool dup = false;
          for (auto s : sector_addrs) {
            if (s == sector) { dup = true; break; }
          }
          if (!dup) sector_addrs.push_back(sector);
        }
        // Queue one INDEX_PF per unique sector
        for (auto sector_addr : sector_addrs) {
          queue_prefetch(sector_addr, warp_id, seed_chain_ids, cycle);
          ++m_stats.ist_prefetch_generated;
          if (grasp_tracer::enabled()) {
            char det[128];
            snprintf(det, sizeof(det),
                     "ct_idx=%d;distance=%u;stride=%lld;n_sectors=%zu",
                     ct_idx, n, (long long)cte_mut.iter_stride,
                     sector_addrs.size());
            grasp_tracer::emit(m_sm_id, warp_id, cycle, "IDX_PF_ENQUEUE",
                               (unsigned long long)pc,
                               (unsigned long long)sector_addr, det);
          }
        }
      }
      return;
    }
  }

  // Fallback (no mf): single-address prefetch at target distance
  {
    new_addr_type paddr = static_cast<new_addr_type>(
        static_cast<int64_t>(addr) +
        static_cast<int64_t>(m_cfg.ist_distance) * cte_mut.iter_stride);
    queue_prefetch(paddr, warp_id, seed_chain_ids, cycle);
    ++m_stats.ist_prefetch_generated;
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

    // Throttle Control: suppress DATA_PF when MSHR is congested.
    // INDEX_PF is NOT throttled (critical for pipeline correctness).
    bool is_data_pf = req.seed_chain_ids.empty();
    if (is_data_pf && l1d && m_cfg.tc_mshr_threshold > 0) {
      float mshr_ratio = l1d->mshr_occupancy_ratio() * 100.0f;
      if (mshr_ratio >= (float)m_cfg.tc_mshr_threshold) {
        // MSHR congested: discard this DATA_PF
        m_prefetch_queue.erase(m_prefetch_queue.begin() + ready_idx);
        ++m_stats.throttle_suppressed;
        continue;  // try next request
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

  if (!req.seed_chain_ids.empty()) {
    if (req.prb_entry_id >= 0) {
      // RETRY: reuse existing PRB entry, skip pair table lookup
      prb_entry_id = req.prb_entry_id;
      if (m_cfg.debug) {
        printf(
            "GRASP inject INDEX_PF RETRY: sid=%u warp=%u addr=0x%llx "
            "prb=%d retry=%u cycle=%llu\n",
            sid, req.warp_id, (unsigned long long)req.addr,
            prb_entry_id, req.retry_count, cycle);
      }
    } else {
      // FRESH: lookup pair table for data targets
      shd_warp_t *warp = get_warp(req.warp_id);
      assert(warp != nullptr);
      std::vector<ima_prefetch_candidate_t> cands =
          warp->lookup_ima_prefetch_candidates(req.addr, req.seed_chain_ids);
      warp->record_ima_verify_predictions(cands, cycle, m_cfg.debug);
      if (!cands.empty()) {
        prb_entry_id = m_prb.allocate(req.warp_id, cands, 1, cycle);
        if (prb_entry_id >= 0) {
          ++m_stats.pair_table_lookup_hit;
        }
      } else {
        ++m_stats.pair_table_lookup_miss;
      }
      if (m_cfg.debug) {
        printf(
            "GRASP inject INDEX_PF: sid=%u warp=%u addr=0x%llx chains=%zu "
            "hits=%zu prb=%d cycle=%llu\n",
            sid, req.warp_id, (unsigned long long)req.addr,
            req.seed_chain_ids.size(), cands.size(), prb_entry_id, cycle);
      }
      if (grasp_tracer::enabled()) {
        char det[128];
        snprintf(det, sizeof(det), "pair_hits=%zu;prb_id=%d",
                 cands.size(), prb_entry_id);
        grasp_tracer::emit(sid, req.warp_id, cycle, "IDX_PF_INJECT",
                           0, (unsigned long long)paddr, det);
      }
    }
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

  if (cache_status == HIT) {
    emit_l1_result("L1_RESULT_HIT");
    // Data already in L1 — release targets immediately
    if (is_index && prb_entry_id != (unsigned)-1 && m_prb.get(prb_entry_id).valid) {
      release_prb_targets(prb_entry_id, cycle + 1);
      m_prb.free_entry(prb_entry_id);
      ++m_stats.index_pf_hit;
    } else if (!is_index) {
      ++m_stats.data_pf_hit;
    }
    ++m_ist.stat_prefetch_filtered;
    delete mf;
  } else if (cache_status == MSHR_HIT || cache_status == HIT_RESERVED) {
    emit_l1_result("L1_RESULT_MSHR");
    if (is_index && prb_entry_id != (unsigned)-1 && m_prb.get(prb_entry_id).valid) {
      // INDEX_PF hit existing MSHR: retry after delay so fill completes
      grasp_prb_entry_t &prb_entry = m_prb.get(prb_entry_id);
      if (prb_entry.retry_count < MAX_RETRIES) {
        ++prb_entry.retry_count;
        grasp_prefetch_request_t retry;
        retry.addr = mf->get_addr();
        retry.warp_id = mf->get_wid();
        retry.seed_chain_ids = {0};  // INDEX_PF marker
        retry.prb_entry_id = static_cast<int>(prb_entry_id);
        retry.retry_count = prb_entry.retry_count;
        retry.ready_cycle = cycle + MSHR_RETRY_DELAY;
        m_prefetch_queue.push_back(retry);
      } else {
        // Max retries exhausted: drop
        m_prb.free_entry(prb_entry_id);
      }
      ++m_stats.index_pf_mshr_merge;
    } else {
      ++m_stats.data_pf_mshr_merge;
    }
    ++m_ist.stat_prefetch_filtered;
    delete mf;
  } else if (cache_status == RESERVATION_FAIL) {
    emit_l1_result("L1_RESULT_RFAIL");
    // Resource full: discard and record per-reason breakdown
    if (is_index && prb_entry_id != (unsigned)-1 && m_prb.get(prb_entry_id).valid) {
      m_prb.free_entry(prb_entry_id);
      ++m_stats.index_pf_reservation_fail;
      ++m_stats.index_pf_rfail[fail_reason];
    } else {
      ++m_stats.data_pf_reservation_fail;
      ++m_stats.data_pf_rfail[fail_reason];
    }
    ++m_ist.stat_prefetch_filtered;
    delete mf;
  } else {
    emit_l1_result("L1_RESULT_MISS");
    // MISS or SECTOR_MISS: cache took ownership, will fetch from L2
    if (is_index) {
      ++m_stats.index_pf_issued;
    } else {
      ++m_stats.data_pf_issued;
    }
    ++m_ist.stat_prefetch_issued;
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
    const grasp_prb_entry_t &entry = m_prb.get(prb_entry_id);
    if (!entry.valid) return;
    release_prb_targets(prb_entry_id, fill_cycle + 1);  // T2: 1-cycle ACU delay
    m_prb.free_entry(prb_entry_id);
  }
  // DATA_PF: fire-and-forget, stats already updated on issue
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

void grasp_prefetcher_t::print_stats(FILE *fp) const {
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
          "GRASP_PRB SM%u: peak_occ=%llu allocs=%llu stall_cycles=%llu\n",
          m_sm_id, prb_s.peak_occupancy, prb_s.total_allocations,
          prb_s.full_stall_cycles);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

void grasp_prefetcher_t::queue_prefetch(
    new_addr_type addr, unsigned warp_id,
    const std::vector<unsigned> &seed_chain_ids,
    unsigned long long ready_cycle) {
  // Dedup: skip if an identical (addr, warp_id, same kind) request is already
  // pending.  This prevents RESERVATION_FAIL retries from flooding the queue
  // with duplicate prefetch requests.
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

void grasp_prefetcher_t::release_prb_targets(unsigned prb_entry_id,
                                              unsigned long long ready_cycle) {
  const grasp_prb_entry_t &entry = m_prb.get(prb_entry_id);
  if (!entry.valid) return;  // PRB was reset by kernel switch while PF was in-flight

  if (m_cfg.debug) {
    printf(
        "GRASP fill dispatch: sm=%u prb=%u warp=%u targets=%zu ready=%llu\n",
        m_sm_id, prb_entry_id, entry.warp_id, entry.candidates.size(),
        ready_cycle);
  }
  if (grasp_tracer::enabled()) {
    char det[96];
    snprintf(det, sizeof(det), "prb_id=%u;num_targets=%zu",
             prb_entry_id, entry.candidates.size());
    grasp_tracer::emit(m_sm_id, entry.warp_id, ready_cycle, "FILL_DISPATCH",
                       0, 0, det);
  }

  // Coalesce DATA_PF candidates targeting the same sector (32B aligned),
  // matching the L1 sector size and IDX_PF aggregation granularity.
  // INDEX_PF (non-empty successors) are NOT coalesced — each triggers an
  // independent pair table lookup at the next level.
  static const new_addr_type SECTOR_MASK = ~((new_addr_type)31);
  std::vector<new_addr_type> seen_data_sectors;
  unsigned coalesced_count = 0;

  for (const ima_prefetch_candidate_t &cand : entry.candidates) {
    new_addr_type sector_addr = cand.data_addr & SECTOR_MASK;
    if (cand.successor_chain_ids.empty()) {
      // Terminal DATA_PF: coalesce by sector
      bool dup = false;
      for (auto s : seen_data_sectors) {
        if (s == sector_addr) { dup = true; break; }
      }
      if (dup) {
        ++coalesced_count;
        continue;
      }
      seen_data_sectors.push_back(sector_addr);
    }

    queue_prefetch(sector_addr, entry.warp_id, cand.successor_chain_ids,
                   ready_cycle);
    if (m_cfg.debug) {
      printf("  target chain=%u idx=0x%llx data=0x%llx succ=%zu\n",
             cand.source_chain_id, (unsigned long long)cand.idx_addr,
             (unsigned long long)cand.data_addr,
             cand.successor_chain_ids.size());
    }
    if (grasp_tracer::enabled()) {
      char det[128];
      snprintf(det, sizeof(det), "chain_id=%u;prb_id=%u;succ=%zu",
               cand.source_chain_id, prb_entry_id,
               cand.successor_chain_ids.size());
      grasp_tracer::emit(m_sm_id, entry.warp_id, ready_cycle,
                         "DATA_PF_ENQUEUE",
                         0, (unsigned long long)sector_addr, det);
    }
  }

  if (m_cfg.debug && coalesced_count > 0) {
    printf("  coalesced %u redundant DATA_PF (same cache line)\n",
           coalesced_count);
  }
}
