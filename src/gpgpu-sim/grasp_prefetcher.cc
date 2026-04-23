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
      m_prb(cfg.prb_capacity) {}

grasp_prefetcher_t::~grasp_prefetcher_t() = default;

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

void grasp_prefetcher_t::on_kernel_launch() {
  m_cd.reset();
  m_ct.reset();
  m_tt.reset();
  m_ist.reset();
  m_prb.reset();
  m_prefetch_queue.clear();
  ++m_stats.kernel_launches;

  // C8: verify all components cleared
  // (PRB current_occupancy == 0 after reset)
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

  grasp_chain_detector_t::detected_chain_t chain;
  bool detected = m_cd.on_instruction_issue(warp_id, inst.pc, inst.op,
                                             sass_opcode, dst_reg, src_regs,
                                             num_src, cycle, &chain);

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
                                         shd_warp_t *warp) {
  if (!m_cfg.enable) return;

  // Check if this PC is an IMA index load (registered in CT by CD)
  int ct_idx = m_ct.find(pc);
  if (ct_idx < 0) return;  // Not an IMA load — skip

  // Trace: demand load hit CT entry
  if (grasp_tracer::enabled()) {
    const ct_entry_t &cte = m_ct.entry(ct_idx);
    char det[128];
    snprintf(det, sizeof(det), "ct_idx=%d;stride_valid=%d;iter_stride=%lld",
             ct_idx, cte.stride_valid ? 1 : 0, (long long)cte.iter_stride);
    grasp_tracer::emit(m_sm_id, warp_id, cycle, "DEMAND_CT_HIT",
                       (unsigned long long)pc, (unsigned long long)addr, det);
  }

  // CT stride learning: only the tracked warp contributes to stride training.
  // Once stride_valid, ALL warps' demand loads at this PC generate prefetches.
  int64_t stride_delta = 0;
  bool newly_valid = m_ct.update_stride(ct_idx, warp_id,
                                         m_cd.tracked_warp_id(), addr,
                                         &stride_delta);
  m_ct.entry(ct_idx).last_access_time = cycle;

  // Trace: stride learning step (only if this warp contributes)
  if (grasp_tracer::enabled() && warp_id == m_cd.tracked_warp_id()) {
    const ct_entry_t &cte = m_ct.entry(ct_idx);
    char det[128];
    snprintf(det, sizeof(det),
             "ct_idx=%d;delta=%lld;iter_stride=%lld;stride_valid=%d",
             ct_idx, (long long)stride_delta, (long long)cte.iter_stride,
             cte.stride_valid ? 1 : 0);
    grasp_tracer::emit(m_sm_id, warp_id, cycle, "STRIDE_UPDATE",
                       (unsigned long long)pc, (unsigned long long)addr, det);
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
                         (unsigned long long)pc, (unsigned long long)addr, det);
    }
  }

  // Generate prefetch only if CT stride is learned
  const ct_entry_t &ct = m_ct.entry(ct_idx);
  if (!ct.stride_valid || ct.iter_stride == 0) return;

  // Get seed chain IDs for this PC (needed for pair table lookup)
  std::vector<unsigned> seed_chain_ids;
  if (warp != nullptr) {
    seed_chain_ids =
        warp->get_ima_seed_chain_ids(static_cast<address_type>(pc));
  }
  if (seed_chain_ids.empty()) return;  // No chain mapping → can't do pair lookup

  // Generate index prefetch at predicted next address(es)
  for (unsigned n = 1; n <= m_cfg.ist_distance; ++n) {
    new_addr_type paddr = static_cast<new_addr_type>(
        static_cast<int64_t>(addr) +
        static_cast<int64_t>(n) * ct.iter_stride);
    queue_prefetch(paddr, warp_id, seed_chain_ids, cycle);
    ++m_stats.ist_prefetch_generated;
    if (grasp_tracer::enabled()) {
      char det[128];
      snprintf(det, sizeof(det), "ct_idx=%d;distance=%u;stride=%lld",
               ct_idx, n, (long long)ct.iter_stride);
      grasp_tracer::emit(m_sm_id, warp_id, cycle, "IDX_PF_ENQUEUE",
                         (unsigned long long)pc, (unsigned long long)paddr,
                         det);
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

  int ready_idx = find_ready_prefetch(cycle);
  if (ready_idx < 0) {
    result.stalled = !m_prefetch_queue.empty();
    return result;
  }

  grasp_prefetch_request_t req = m_prefetch_queue[ready_idx];

  // TODO(human): Throttle Control — suppress DATA_PF when MSHR is congested.
  // Check l1d's MSHR occupancy against m_cfg.tc_mshr_threshold (percentage).
  // If this is a DATA_PF (seed_chain_ids empty) and MSHR is above threshold,
  // discard the request, increment m_stats.throttle_suppressed, and return.
  // INDEX_PF should NOT be throttled (critical for pipeline correctness).
  // Hint: l1_cache inherits from baseline_cache which has MSHR access methods.

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
    // INDEX_PF: lookup pair table for data targets
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
  return result;
}

// ---------------------------------------------------------------------------
// L1 Access Result Tracking
// ---------------------------------------------------------------------------

void grasp_prefetcher_t::on_l1_access_result(mem_fetch *mf, int cache_status,
                                              unsigned long long cycle) {
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
    if (is_index && prb_entry_id != (unsigned)-1) {
      release_prb_targets(prb_entry_id, cycle + 1);
      m_prb.free_entry(prb_entry_id);
      ++m_stats.index_pf_hit;
    } else {
      ++m_stats.data_pf_hit;
    }
    ++m_ist.stat_prefetch_filtered;
    delete mf;
  } else if (cache_status == MSHR_HIT || cache_status == HIT_RESERVED) {
    emit_l1_result("L1_RESULT_MSHR");
    // Already in-flight
    if (is_index && prb_entry_id != (unsigned)-1) {
      m_prb.free_entry(prb_entry_id);
      ++m_stats.index_pf_mshr_merge;
    } else {
      ++m_stats.data_pf_mshr_merge;
    }
    ++m_ist.stat_prefetch_filtered;
    delete mf;
  } else if (cache_status == RESERVATION_FAIL) {
    emit_l1_result("L1_RESULT_RFAIL");
    // MSHR/miss-queue full: discard
    if (is_index && prb_entry_id != (unsigned)-1) {
      m_prb.free_entry(prb_entry_id);
      ++m_stats.index_pf_reservation_fail;
    } else {
      ++m_stats.data_pf_reservation_fail;
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
    // C3: bounds check
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
  fprintf(fp, "data_pf(issued=%llu hit=%llu miss=%llu mshr=%llu rfail=%llu) ",
          m_stats.data_pf_issued, m_stats.data_pf_hit,
          m_stats.data_pf_miss, m_stats.data_pf_mshr_merge,
          m_stats.data_pf_reservation_fail);
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
  assert(entry.valid);

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

  // Coalesce DATA_PF candidates targeting the same cache line (128B aligned).
  // INDEX_PF (non-empty successors) are NOT coalesced — each triggers an
  // independent pair table lookup at the next level.
  static const new_addr_type LINE_MASK = ~((new_addr_type)127);
  std::vector<new_addr_type> seen_data_lines;
  unsigned coalesced_count = 0;

  for (const ima_prefetch_candidate_t &cand : entry.candidates) {
    if (cand.successor_chain_ids.empty()) {
      // Terminal DATA_PF: coalesce by cache line
      new_addr_type line_addr = cand.data_addr & LINE_MASK;
      bool dup = false;
      for (auto la : seen_data_lines) {
        if (la == line_addr) { dup = true; break; }
      }
      if (dup) {
        ++coalesced_count;
        continue;
      }
      seen_data_lines.push_back(line_addr);
    }

    queue_prefetch(cand.data_addr, entry.warp_id, cand.successor_chain_ids,
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
                         0, (unsigned long long)cand.data_addr, det);
    }
  }

  if (m_cfg.debug && coalesced_count > 0) {
    printf("  coalesced %u redundant DATA_PF (same cache line)\n",
           coalesced_count);
  }
}
