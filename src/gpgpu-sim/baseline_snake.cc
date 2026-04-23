#include "baseline_snake.h"

#include <algorithm>
#include <cstring>

#include "gpu-cache.h"
#include "shader.h"

const char *baseline_snake_prefetcher_t::uniform_gate_reason_name(
    uniform_gate_reason_t reason) {
  switch (reason) {
    case uniform_gate_reason_t::kPassSingleActive:
      return "single_active";
    case uniform_gate_reason_t::kPassAffine:
      return "affine";
    case uniform_gate_reason_t::kRejectNoActive:
      return "no_active";
    case uniform_gate_reason_t::kRejectMixedStride:
      return "mixed_stride";
    case uniform_gate_reason_t::kRejectNonAffineProgression:
      return "non_affine_progression";
  }
  return "unknown";
}

baseline_snake_prefetcher_t::baseline_snake_prefetcher_t(
    unsigned sm_id, const baseline_snake_config_t &cfg, baseline_cache *l1d)
    : baseline_prefetcher_t(sm_id),
      m_cfg(cfg),
      m_l1d_cache(l1d),
      m_ht(cfg.ht_size),
      m_warp_trackers(kMaxWarps) {}

void baseline_snake_prefetcher_t::on_kernel_launch() {
  baseline_prefetcher_t::on_kernel_launch();
  for (auto &e : m_ht) e = ht_entry_t();
  for (auto &w : m_warp_trackers) w = warp_pc_tracker_t();
  m_throttle_until = 0;
  m_pf_iaw_issued = 0;
  m_pf_iew_issued = 0;
  m_pf_it_issued = 0;
  m_training_completions = 0;
  m_training_unique_warps = 0;
  m_training_duplicate_warps = 0;
  m_it_chain_follows = 0;
  m_issue_global_loads = 0;
  m_uniform_gate_passes = 0;
  m_uniform_gate_rejects = 0;
  m_uniform_gate_reject_kept_tracker = 0;
  m_uniform_gate_pass_single_active = 0;
  m_uniform_gate_pass_affine = 0;
  m_uniform_gate_reject_no_active = 0;
  m_uniform_gate_reject_mixed_stride = 0;
  m_uniform_gate_reject_non_affine = 0;
  m_uniform_gate_active_threads_total = 0;
  m_uniform_gate_multi_active_total = 0;
  m_lane_non_uniform_filtered = 0;
  m_pc_table_allocations = 0;
  m_pc_table_evictions = 0;
  m_iew_cta_mismatch = 0;
  m_demand_loads = 0;
  m_iaw_accum_checks = 0;
  m_iaw_it_not_valid = 0;
  m_iaw_pc_mismatch = 0;
  m_iaw_chain_found = 0;
  m_iaw_accum_ok = 0;
  m_gate_reject_samples.clear();
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
      m_pc_table_allocations++;
      return static_cast<int>(i);
    }
  }
  // LRU eviction: prefer non-trained, then lowest warp count, then LRU
  unsigned victim = 0;
  for (unsigned i = 1; i < m_ht.size(); ++i) {
    const ht_entry_t &vi = m_ht[victim];
    const ht_entry_t &ci = m_ht[i];
    if (!ci.training_done && vi.training_done) { victim = i; continue; }
    if (ci.training_done != vi.training_done) continue;
    if (ci.training_warp_count() < vi.training_warp_count()) {
      victim = i; continue;
    }
    if (ci.training_warp_count() != vi.training_warp_count()) continue;
    if (ci.last_access_cycle < vi.last_access_cycle) { victim = i; }
  }
  m_pc_table_evictions++;
  m_ht[victim] = ht_entry_t();
  m_ht[victim].valid = true;
  m_ht[victim].pc = pc;
  m_ht[victim].last_access_cycle = cycle;
  m_pc_table_allocations++;
  return static_cast<int>(victim);
}

// --- Head Table slot update (paper §3.1, Table 3: 2 warps per PC_ld) ---
// Each PC entry stores 2 (warp_id, addr) slots.
// When warp W accesses PC at addr A:
//   - If W matches a slot → IaW: stride = A - slot.addr (same warp, same PC)
//   - If W is new → IeW: stride = A - other_slot.addr (different warp, same PC)
//   - Replace LRU slot with (W, A)

void baseline_snake_prefetcher_t::update_head_table_slots(
    ht_entry_t &entry, unsigned warp_id, unsigned cta_id,
    new_addr_type addr) {
  auto &s0 = entry.head_slots[0];
  auto &s1 = entry.head_slots[1];

  // Check if this warp already has a slot → update addr only (no IaW here,
  // IaW is computed via IT stride accumulation in case 2)
  int my_slot = -1;
  if (s0.valid && s0.warp_id == warp_id) my_slot = 0;
  else if (s1.valid && s1.warp_id == warp_id) my_slot = 1;

  if (my_slot >= 0) {
    // Same warp revisiting this PC → just update address
    entry.head_slots[my_slot].addr = addr;
    entry.head_slots[my_slot].cta_id = cta_id;
  } else {
    // IeW: different warp accessing same PC
    // Compare with the most recent same-CTA slot. Paper IeW only trains
    // across warps from the same CTA.
    int other = -1;
    if (s0.valid && s0.cta_id == cta_id) other = 0;
    if (s1.valid && s1.cta_id == cta_id) other = 1;  // prefer slot 1 (more recent)

    if (other >= 0) {
      auto &slot = entry.head_slots[other];
      int64_t delta = static_cast<int64_t>(addr) -
                      static_cast<int64_t>(slot.addr);
      if (delta != 0) {
        if (delta == entry.iew_stride) {
          entry.iew_confirmed = true;
        } else {
          entry.iew_stride = delta;
          entry.iew_confirmed = false;
        }
      }
    } else if (s0.valid || s1.valid) {
      m_iew_cta_mismatch++;
    }

    // Replace the oldest/invalid slot with this warp
    ht_entry_t::warp_slot_t new_slot;
    new_slot.warp_id = warp_id;
    new_slot.cta_id = cta_id;
    new_slot.addr = addr;
    new_slot.valid = true;

    if (!s0.valid) {
      s0 = new_slot;
    } else if (!s1.valid) {
      s1 = new_slot;
    } else {
      // Both valid, replace slot 0 (older), shift slot 1 → slot 0
      s0 = s1;
      s1 = new_slot;
    }
  }
}

// --- IT stride update (paper §3.1 Detection Step, Figure 12) ---
// When warp executes PC_prev then PC_cur consecutively,
// update PC_prev's entry with: next_pc = PC_cur, stride = addr_cur - addr_prev.
// If the same (prev→cur) pair is observed again, CONFIRM (don't duplicate).

void baseline_snake_prefetcher_t::update_it_stride(
    ht_entry_t &prev_entry, new_addr_type prev_addr,
    new_addr_type cur_pc, new_addr_type cur_addr) {
  int64_t it_stride =
      static_cast<int64_t>(cur_addr) - static_cast<int64_t>(prev_addr);
  if (it_stride == 0) return;

  if (prev_entry.it_next_pc == cur_pc && prev_entry.it_stride == it_stride) {
    // Same (PC2, stride) pair observed again → increment confirmation
    prev_entry.it_observation_count++;
  } else if (prev_entry.it_next_pc == cur_pc &&
             prev_entry.it_stride != it_stride) {
    // Same PC2 but stride changed → update stride, keep 1 observation
    prev_entry.it_stride = it_stride;
    prev_entry.it_observation_count = 1;
    prev_entry.it_stride_valid = false;
  } else if (prev_entry.it_observation_count == 0 ||
             prev_entry.it_next_pc == 0) {
    // No prior observation → first record
    prev_entry.it_next_pc = cur_pc;
    prev_entry.it_stride = it_stride;
    prev_entry.it_observation_count = 1;
    prev_entry.it_stride_valid = false;
  } else {
    // Different PC2 → replace only if current pair has weak evidence
    if (prev_entry.it_observation_count <= 1) {
      prev_entry.it_next_pc = cur_pc;
      prev_entry.it_stride = it_stride;
      prev_entry.it_observation_count = 1;
      prev_entry.it_stride_valid = false;
    }
    // Otherwise keep the stronger existing pair
  }

  // Confirm after 2 consistent observations
  if (!prev_entry.it_stride_valid &&
      prev_entry.it_observation_count >= 2) {
    prev_entry.it_stride_valid = true;
  }
}

baseline_snake_prefetcher_t::uniform_gate_result_t
baseline_snake_prefetcher_t::classify_uniform_warp_addr(
    const warp_inst_t &inst) const {
  uniform_gate_result_t result;
  bool have_first = false;
  bool have_stride = false;
  unsigned prev_lane = 0;
  new_addr_type prev_addr = 0;
  int64_t lane_stride = 0;

  for (unsigned lane = 0; lane < inst.warp_size(); ++lane) {
    if (!inst.active(lane)) continue;
    result.active_count++;
    if (result.active_count == 1) {
      result.first_lane = lane;
    }
    result.last_lane = lane;
    const new_addr_type lane_addr = inst.get_addr(lane);
    if (!have_first) {
      result.first_addr = lane_addr;
      prev_addr = lane_addr;
      prev_lane = lane;
      have_first = true;
      continue;
    }

    const unsigned lane_delta = lane - prev_lane;
    if (lane_delta == 0) {
      result.reason = uniform_gate_reason_t::kRejectNonAffineProgression;
      return result;
    }
    const int64_t addr_delta =
        static_cast<int64_t>(lane_addr) - static_cast<int64_t>(prev_addr);
    if (addr_delta % static_cast<int64_t>(lane_delta) != 0) {
      result.reason = uniform_gate_reason_t::kRejectNonAffineProgression;
      return result;
    }
    const int64_t cur_stride = addr_delta / static_cast<int64_t>(lane_delta);
    if (!have_stride) {
      lane_stride = cur_stride;
      have_stride = true;
    } else if (lane_stride != cur_stride) {
      result.reason = uniform_gate_reason_t::kRejectMixedStride;
      return result;
    }
    prev_addr = lane_addr;
    prev_lane = lane;
  }

  if (!have_first) {
    result.reason = uniform_gate_reason_t::kRejectNoActive;
    return result;
  }
  result.accepted = true;
  result.reason =
      result.active_count == 1 ? uniform_gate_reason_t::kPassSingleActive
                               : uniform_gate_reason_t::kPassAffine;
  return result;
}

// --- Prefetch generation (paper §3.2) ---
// Three types of prefetch: IaW, IeW, and IT chain.
// IT chain follows HT entries: entry[PC_A].it_next_pc → find entry[PC_B] → ...
// Depth controlled by max_chain_length.

void baseline_snake_prefetcher_t::generate_prefetches(
    const ht_entry_t &entry, new_addr_type addr, unsigned warp_id,
    unsigned long long cycle) {
  // Throttle
  if (cycle < m_throttle_until) return;

  if (m_l1d_cache) {
    unsigned pf_count = m_l1d_cache->count_snake_prefetch_lines();
    unsigned total = m_l1d_cache->get_total_lines();
    if (pf_count >= total / 2) {
      m_throttle_until = cycle + kThrottlePauseCycles;
      return;
    }
    unsigned mshr_used = m_l1d_cache->get_mshr_used();
    unsigned mshr_total = m_l1d_cache->get_mshr_entries();
    if (mshr_total > 0 && (mshr_used * 100 / mshr_total) > 70) {
      m_throttle_until = cycle + kThrottlePauseCycles;
      return;
    }
  }

  // 1. IaW prefetch (same warp, same PC, across iterations)
  if (entry.iaw_confirmed && entry.iaw_stride != 0) {
    new_addr_type pf_addr = static_cast<new_addr_type>(
        static_cast<int64_t>(addr) +
        static_cast<int64_t>(entry.iaw_stride) * kIaWLookahead);
    queue_prefetch(pf_addr, warp_id, cycle);
    m_pf_iaw_issued++;
  }

  // 2. IT chain prefetch: follow PC1 -> PC2 -> ... using trained IT strides.
  // Keep IT ahead of IeW when both are possible for the same trigger.
  bool issued_it_chain = false;
  const ht_entry_t *chain_entry = &entry;
  new_addr_type chain_addr = addr;
  for (unsigned depth = 0; depth < m_cfg.max_chain_length; ++depth) {
    if (!chain_entry->it_stride_valid || chain_entry->it_next_pc == 0) break;

    chain_addr = static_cast<new_addr_type>(
        static_cast<int64_t>(chain_addr) +
        static_cast<int64_t>(chain_entry->it_stride));
    queue_prefetch(chain_addr, warp_id, cycle);
    m_pf_it_issued++;
    issued_it_chain = true;

    const int next_idx = find_ht_entry(chain_entry->it_next_pc);
    if (next_idx < 0) break;
    const ht_entry_t &next_entry = m_ht[next_idx];
    if (!next_entry.training_done) break;
    chain_entry = &next_entry;
    if (depth + 1 < m_cfg.max_chain_length) {
      m_it_chain_follows++;
    }
  }

  // 3. IeW prefetch: only when IT chain did not already claim the trigger.
  // Paper §3.2 prefers chain-based lookahead over future-warp prediction.
  if (!issued_it_chain && entry.iew_confirmed && entry.iew_stride != 0) {
    new_addr_type pf_addr = static_cast<new_addr_type>(
        static_cast<int64_t>(addr) +
        static_cast<int64_t>(entry.iew_stride) * kIeWLookahead);
    queue_prefetch(pf_addr, warp_id, cycle);
    m_pf_iew_issued++;
  }
}

// --- Early detection at instruction issue (paper §3.1, Figure 14) ---
// Called at issue time BEFORE coalescing. Has the full warp_inst_t with
// per-thread addresses. This is where IT chain detection and warp tracker
// updates happen — much earlier than L1 access, giving prefetches more
// timeliness.

void baseline_snake_prefetcher_t::on_instruction_issue(
    unsigned warp_id, const warp_inst_t &inst, const std::string &sass_opcode,
    unsigned long long cycle) {
  // Fallback: called when CTA ID is not available
  on_instruction_issue_with_cta(warp_id, static_cast<unsigned>(-1),
                                 inst, cycle);
}

void baseline_snake_prefetcher_t::on_instruction_issue_with_cta(
    unsigned warp_id, unsigned cta_id, const warp_inst_t &inst,
    unsigned long long cycle) {
  // Only process global loads
  if (!inst.is_load() || inst.space.get_type() != global_space)
    return;
  m_issue_global_loads++;

  new_addr_type pc = inst.pc;
  new_addr_type addr = 0;
  warp_pc_tracker_t *tracker = nullptr;
  if (warp_id < m_warp_trackers.size()) {
    tracker = &m_warp_trackers[warp_id];
  }

  // Paper §3.2: only keep the first-thread address when the warp exhibits a
  // uniform lane stride; otherwise exclude this issue from Snake training.
  const uniform_gate_result_t gate_result = classify_uniform_warp_addr(inst);
  m_uniform_gate_active_threads_total += gate_result.active_count;
  if (gate_result.active_count > 1) {
    m_uniform_gate_multi_active_total++;
  }
  if (!gate_result.accepted) {
    m_uniform_gate_rejects++;
    switch (gate_result.reason) {
      case uniform_gate_reason_t::kRejectNoActive:
        m_uniform_gate_reject_no_active++;
        break;
      case uniform_gate_reason_t::kRejectMixedStride:
        m_uniform_gate_reject_mixed_stride++;
        break;
      case uniform_gate_reason_t::kRejectNonAffineProgression:
        m_uniform_gate_reject_non_affine++;
        break;
      case uniform_gate_reason_t::kPassSingleActive:
      case uniform_gate_reason_t::kPassAffine:
        break;
    }
    if (m_gate_reject_samples.size() < kMaxGateRejectSamples) {
      gate_reject_sample_t sample;
      sample.pc = pc;
      sample.reason = gate_result.reason;
      sample.active_count = gate_result.active_count;
      sample.first_lane = gate_result.first_lane;
      sample.last_lane = gate_result.last_lane;
      m_gate_reject_samples.push_back(sample);
    }
    if (tracker != nullptr &&
        (tracker->slots[0].valid || tracker->slots[1].valid)) {
      m_uniform_gate_reject_kept_tracker++;
    }
    m_lane_non_uniform_filtered++;
    return;
  }
  m_uniform_gate_passes++;
  switch (gate_result.reason) {
    case uniform_gate_reason_t::kPassSingleActive:
      m_uniform_gate_pass_single_active++;
      break;
    case uniform_gate_reason_t::kPassAffine:
      m_uniform_gate_pass_affine++;
      break;
    case uniform_gate_reason_t::kRejectNoActive:
    case uniform_gate_reason_t::kRejectMixedStride:
    case uniform_gate_reason_t::kRejectNonAffineProgression:
      break;
  }
  addr = gate_result.first_addr;

  int ht_idx = find_ht_entry(pc);
  if (ht_idx < 0) {
    ht_idx = alloc_ht_entry(pc, cycle);
  }
  ht_entry_t &entry = m_ht[ht_idx];
  entry.last_access_cycle = cycle;

  // --- Warp confirmation ---
  if (!entry.training_done) {
    if (warp_id < 64 &&
        !(entry.warp_confirmed_mask & (1ULL << warp_id))) {
      entry.warp_confirmed_mask |= (1ULL << warp_id);
      m_training_unique_warps++;
      if (entry.training_warp_count() >= m_cfg.training_warps) {
        entry.training_done = true;
        m_training_completions++;
      }
    } else if (warp_id < 64) {
      m_training_duplicate_warps++;
    }
  }

  // --- Per-warp 2-slot Head Table: IaW direct + IT stride detection ---
  if (tracker != nullptr) {
    // Check if this PC is already in one of the warp's slots → IaW (direct)
    int iaw_slot = -1;
    for (int s = 0; s < 2; ++s) {
      if (tracker->slots[s].valid && tracker->slots[s].pc == pc) {
        iaw_slot = s;
        break;
      }
    }

    if (iaw_slot >= 0) {
      // IaW: same warp revisiting same PC (loop iteration)
      int64_t delta = static_cast<int64_t>(addr) -
                      static_cast<int64_t>(tracker->slots[iaw_slot].addr);
      if (delta != 0) {
        // Debug first 5 IaW stride observations on SM0
        static unsigned s_iaw_learn_count = 0;
        if (m_sm_id == 0 && s_iaw_learn_count < 5) {
          printf("SNAKE_LEARN SM0: IaW warp=%u pc=0x%lx old_addr=0x%lx "
                 "new_addr=0x%lx delta=%ld cycle=%llu\n",
                 warp_id, (unsigned long)pc,
                 (unsigned long)tracker->slots[iaw_slot].addr,
                 (unsigned long)addr, (long)delta, cycle);
          s_iaw_learn_count++;
        }
        if (delta == entry.iaw_stride) {
          entry.iaw_confirmed = true;
        } else {
          entry.iaw_stride = delta;
          entry.iaw_confirmed = false;
        }
      }
      // Update the slot's addr
      tracker->slots[iaw_slot].addr = addr;
    }

    // IT stride detection: from most recent slot (if different PC)
    if (tracker->slots[0].valid && tracker->slots[0].pc != pc) {
      int prev_ht = find_ht_entry(tracker->slots[0].pc);
      if (prev_ht >= 0) {
        update_it_stride(m_ht[prev_ht], tracker->slots[0].addr, pc, addr);
      }
    }

    // Update per-warp tracker: push current (pc, addr) as most recent
    if (iaw_slot < 0) {
      // New PC for this warp → shift slots
      tracker->slots[1] = tracker->slots[0];
      tracker->slots[0].pc = pc;
      tracker->slots[0].addr = addr;
      tracker->slots[0].valid = true;
    } else {
      // Existing PC → move it to slot[0] (most recent)
      if (iaw_slot == 1) {
        tracker->slots[1] = tracker->slots[0];
        tracker->slots[0].pc = pc;
        tracker->slots[0].addr = addr;
        tracker->slots[0].valid = true;
      }
      // If iaw_slot == 0, addr already updated above
    }
  }

  // --- Head Table per-PC slots: IeW detection ---
  update_head_table_slots(entry, warp_id, cta_id, addr);

  // --- Prefetch generation ---
  if (entry.training_done) {
    generate_prefetches(entry, addr, warp_id, cycle);
  }

  // Per-warp tracker is now updated inside the detection section above
}

// --- L1 access hook (only for demand miss tracking) ---

void baseline_snake_prefetcher_t::on_demand_load(
    unsigned warp_id, new_addr_type pc, new_addr_type addr,
    unsigned long long cycle, int cache_status, shd_warp_t *warp) {
  (void)warp;
  (void)warp_id;
  (void)pc;
  (void)addr;
  (void)cycle;
  m_demand_loads++;
  // Only track demand access stats for accuracy/coverage computation.
  // All stride detection and prefetch generation now happens at issue time
  // in on_instruction_issue() for better timeliness.
  note_demand_access(addr, cache_status);
}

void baseline_snake_prefetcher_t::print_stats(FILE *fp) const {
  print_common_stats(fp, "BASELINE_SNAKE");
  const baseline_stats_t stats = finalized_stats();
  const unsigned long long predicted_requests =
      m_pf_iaw_issued + m_pf_iew_issued + m_pf_it_issued;
  const unsigned long long timely_correct = stats.prefetch_useful;
  const double coverage_paper =
      m_demand_loads == 0
          ? 0.0
          : static_cast<double>(timely_correct) /
                static_cast<double>(m_demand_loads);
  const double accuracy_paper =
      predicted_requests == 0
          ? 0.0
          : static_cast<double>(timely_correct) /
                static_cast<double>(predicted_requests);
  fprintf(fp,
          "BASELINE_SNAKE_DETAIL SM%u: "
          "pf_iaw_issued=%u pf_iew_issued=%u pf_it_issued=%u "
          "it_chain_follows=%u training_completions=%u "
          "training_unique_warps=%u training_duplicate_warps=%u "
          "issue_global_loads=%llu uniform_gate_passes=%u "
          "uniform_gate_rejects=%u uniform_gate_reject_kept_tracker=%u "
          "uniform_gate_pass_single_active=%u uniform_gate_pass_affine=%u "
          "uniform_gate_reject_no_active=%u "
          "uniform_gate_reject_mixed_stride=%u "
          "uniform_gate_reject_non_affine=%u "
          "uniform_gate_active_threads_total=%llu "
          "uniform_gate_multi_active_total=%llu "
          "lane_non_uniform_filtered=%u "
          "table_allocations=%u table_evictions=%u iew_cta_mismatch=%u "
          "coverage_paper=%.6f accuracy_paper=%.6f "
          "demand_loads=%llu predicted_requests=%llu timely_correct=%llu "
          "iaw_checks=%u iaw_it_nv=%u iaw_pc_mm=%u iaw_ok=%u\n",
          m_sm_id, m_pf_iaw_issued, m_pf_iew_issued, m_pf_it_issued,
          m_it_chain_follows, m_training_completions,
          m_training_unique_warps, m_training_duplicate_warps,
          m_issue_global_loads, m_uniform_gate_passes,
          m_uniform_gate_rejects, m_uniform_gate_reject_kept_tracker,
          m_uniform_gate_pass_single_active, m_uniform_gate_pass_affine,
          m_uniform_gate_reject_no_active, m_uniform_gate_reject_mixed_stride,
          m_uniform_gate_reject_non_affine,
          m_uniform_gate_active_threads_total,
          m_uniform_gate_multi_active_total,
          m_lane_non_uniform_filtered,
          m_pc_table_allocations, m_pc_table_evictions, m_iew_cta_mismatch,
          coverage_paper, accuracy_paper, m_demand_loads, predicted_requests,
          timely_correct,
          m_iaw_accum_checks, m_iaw_it_not_valid, m_iaw_pc_mismatch,
          m_iaw_accum_ok);
  for (const auto &sample : m_gate_reject_samples) {
    fprintf(fp,
            "BASELINE_SNAKE_GATE_SAMPLE SM%u: pc=0x%llx reason=%s "
            "active_count=%u first_lane=%u last_lane=%u\n",
            m_sm_id, static_cast<unsigned long long>(sample.pc),
            uniform_gate_reason_name(sample.reason), sample.active_count,
            sample.first_lane, sample.last_lane);
  }
}
