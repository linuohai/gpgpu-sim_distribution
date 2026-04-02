// GRASP Chain Detector (CD) — Implementation
//
// Detects LDG → IMAD.WIDE → LDG chains via a circular FIFO that tracks
// register producers.  Uses Read Detection (invalidation when a tracked
// register is consumed by a non-IMA instruction) and Write Invalidation
// (when a tracked register is overwritten).

#include "grasp_chain_detector.h"

#include <algorithm>
#include <cassert>
#include <cstdio>

#include "grasp_tables.h"

// ============================================================================
// Constructor
// ============================================================================

grasp_chain_detector_t::grasp_chain_detector_t(unsigned fifo_depth)
    : m_fifo(fifo_depth), m_depth(fifo_depth) {}

// ============================================================================
// FIFO Operations (circular buffer)
// ============================================================================

void grasp_chain_detector_t::push(const fifo_entry_t &entry) {
  if (m_count >= m_depth) {
    // FIFO full: overwrite oldest entry
    ++m_stats.fifo_drop_count;
    m_fifo[m_head] = entry;
    m_head = (m_head + 1) % m_depth;
    // m_tail advances to next slot
    m_tail = (m_tail + 1) % m_depth;
    // count stays at m_depth
  } else {
    // Write Invalidation: if new entry's dst_reg matches an existing entry,
    // invalidate the old one (register overwritten)
    if (entry.dst_reg >= 0) {
      invalidate_by_dst(entry.dst_reg);
    }
    m_fifo[m_tail] = entry;
    m_tail = (m_tail + 1) % m_depth;
    ++m_count;
  }
  // Update peak occupancy
  if (m_count > m_stats.fifo_peak_occupancy) {
    m_stats.fifo_peak_occupancy = m_count;
  }
}

const grasp_chain_detector_t::fifo_entry_t *
grasp_chain_detector_t::lookup_by_reg(int reg, unsigned warp_id) const {
  if (reg < 0) return nullptr;
  // Search from newest to oldest for most recent producer from the SAME warp.
  // Different warps share register names but have independent register files.
  for (unsigned i = 0; i < m_count; ++i) {
    unsigned idx = (m_tail + m_depth - 1 - i) % m_depth;
    if (m_fifo[idx].valid && m_fifo[idx].dst_reg == reg &&
        m_fifo[idx].warp_id == warp_id) {
      return &m_fifo[idx];
    }
  }
  return nullptr;
}

void grasp_chain_detector_t::invalidate_by_src(const int *src_regs,
                                                unsigned num_src) {
  // Read Detection: when a tracked register is consumed by a non-IMA
  // instruction, invalidate FIFO entries whose dst_reg matches any src_reg.
  for (unsigned s = 0; s < num_src; ++s) {
    if (src_regs[s] < 0) continue;
    for (unsigned i = 0; i < m_count; ++i) {
      unsigned idx = (m_head + i) % m_depth;
      if (m_fifo[idx].valid && m_fifo[idx].dst_reg == src_regs[s]) {
        m_fifo[idx].valid = false;
        ++m_stats.fifo_read_invalidations;
      }
    }
  }
}

void grasp_chain_detector_t::invalidate_by_dst(int dst_reg) {
  // Write Invalidation: when a new instruction writes the same register,
  // invalidate old entries tracking that register.
  if (dst_reg < 0) return;
  for (unsigned i = 0; i < m_count; ++i) {
    unsigned idx = (m_head + i) % m_depth;
    if (m_fifo[idx].valid && m_fifo[idx].dst_reg == dst_reg) {
      m_fifo[idx].valid = false;
      ++m_stats.fifo_write_invalidations;
    }
  }
}

// ============================================================================
// Main Detection Logic
// ============================================================================

bool grasp_chain_detector_t::on_instruction_issue(
    unsigned warp_id, new_addr_type pc, op_type op,
    const std::string &sass_opcode, int dst_reg, const int *src_regs,
    unsigned num_src_regs, unsigned long long cycle,
    new_addr_type first_lane_addr, unsigned first_lane_id,
    unsigned data_size,
    detected_chain_t *out_chain) {
  // C7: if training is frozen, skip all FIFO operations
  if (m_training_frozen) return false;

  // P4: Use sass_opcode to identify LDG, not op == LOAD_OP.
  // LOAD_OP includes ATOM instructions (ATOMG, ATOM, RED) which are not
  // valid IMA data loads and must be excluded from chain detection.
  bool is_ldg = (sass_opcode.find("LDG") != std::string::npos);

  // Tracked warp selection: first 2 LDGs from different warps become tracked
  if (m_tracked_warp_ids[0] == UNSET_WARP) {
    if (is_ldg) {
      m_tracked_warp_ids[0] = warp_id;
    } else {
      return false;
    }
  } else if (m_tracked_warp_ids[1] == UNSET_WARP &&
             warp_id != m_tracked_warp_ids[0] && is_ldg) {
    m_tracked_warp_ids[1] = warp_id;
  }

  // Only process tracked warps
  if (!is_tracked_warp(warp_id)) return false;

  // Check if this is a load instruction (LDG, not ATOM)
  bool is_load = is_ldg;

  // Check if this is IMAD.WIDE (address computation for IMA)
  bool is_imad_wide =
      (sass_opcode.find("IMAD.WIDE") != std::string::npos);

  if (is_load) {
    // Check if address register was produced by an IMAD.WIDE (→ data load)
    for (unsigned s = 0; s < num_src_regs; ++s) {
      if (src_regs[s] < 0) continue;
      const fifo_entry_t *producer = lookup_by_reg(src_regs[s], warp_id);
      if (producer && producer->type == IMA_ADDR_COMPUTE) {
        // Complete chain detected: index_PC → IMAD.WIDE → data_PC
        if (out_chain) {
          out_chain->index_pc = producer->index_pc;
          out_chain->data_pc = pc;
          out_chain->scale_placeholder = producer->scale;
          out_chain->base_placeholder = producer->base;
          out_chain->index_addr = producer->index_addr;
          out_chain->index_lane_id = producer->index_lane_id;
          out_chain->index_data_size = producer->data_size;
        }
        ++m_stats.chains_detected;
        // Track unique index PCs for CT coverage metric
        if (m_unique_index_pcs.insert(producer->index_pc).second) {
          ++m_stats.unique_index_pcs_detected;
        }

        // Push data load result into FIFO (it might be used as index for
        // a subsequent chain)
        if (dst_reg >= 0) {
          fifo_entry_t entry;
          entry.valid = true;
          entry.dst_reg = dst_reg;
          entry.warp_id = warp_id;
          entry.type = LOAD_RESULT;
          entry.pc = pc;
          entry.addr = first_lane_addr;
          entry.lane_id = first_lane_id;
          entry.data_size = data_size;
          push(entry);
        }
        return true;
      }
    }
    // Not a data load (no IMAD.WIDE producer) — regular load, track it
    if (dst_reg >= 0) {
      fifo_entry_t entry;
      entry.valid = true;
      entry.dst_reg = dst_reg;
      entry.warp_id = warp_id;
      entry.type = LOAD_RESULT;
      entry.pc = pc;
      entry.addr = first_lane_addr;
      entry.lane_id = first_lane_id;
      entry.data_size = data_size;
      push(entry);
    }
  } else if (is_imad_wide) {
    // IMAD.WIDE: check if source register was produced by a load (index load)
    for (unsigned s = 0; s < num_src_regs; ++s) {
      if (src_regs[s] < 0) continue;
      const fifo_entry_t *producer = lookup_by_reg(src_regs[s], warp_id);
      if (producer && producer->type == LOAD_RESULT) {
        // IMAD.WIDE consuming a load result → address computation
        if (dst_reg >= 0) {
          fifo_entry_t entry;
          entry.valid = true;
          entry.dst_reg = dst_reg;
          entry.warp_id = warp_id;
          entry.type = IMA_ADDR_COMPUTE;
          entry.pc = pc;
          entry.index_pc = producer->pc;  // PC of the index load
          entry.index_addr = producer->addr;  // address of the index load
          entry.index_lane_id = producer->lane_id;
          entry.data_size = producer->data_size;  // propagate index LDG width
          entry.scale = 0;  // placeholder in trace-driven mode
          entry.base = 0;   // placeholder in trace-driven mode
          push(entry);
        }
        return false;  // Chain not complete yet
      }
    }
    // IMAD.WIDE but source isn't a tracked load — ignore
  } else {
    // Other instruction: Read Detection invalidation
    // If any source register matches a FIFO entry, invalidate it
    // (the register value is consumed by non-IMA computation)
    invalidate_by_src(src_regs, num_src_regs);

    // Write Invalidation for dst_reg is handled by push() when we don't
    // push a new entry. Do it explicitly here.
    if (dst_reg >= 0) {
      invalidate_by_dst(dst_reg);
    }
  }

  return false;
}

// ============================================================================
// Lifecycle
// ============================================================================

void grasp_chain_detector_t::on_warp_exit(unsigned warp_id) {
  if (!is_tracked_warp(warp_id)) return;
  // Clear FIFO
  for (unsigned i = 0; i < m_depth; ++i) {
    m_fifo[i].valid = false;
  }
  m_head = 0;
  m_tail = 0;
  m_count = 0;
  // Freeze training when a tracked warp exits.
  // m_tracked_warp_ids preserved (not reset) for diagnostics.
  m_training_frozen = true;
}

void grasp_chain_detector_t::reset() {
  for (unsigned i = 0; i < m_depth; ++i) {
    m_fifo[i].valid = false;
  }
  m_head = 0;
  m_tail = 0;
  m_count = 0;
  m_tracked_warp_ids[0] = UNSET_WARP;
  m_tracked_warp_ids[1] = UNSET_WARP;
  m_training_frozen = false;
  m_stats = cd_stats_t();
  m_unique_index_pcs.clear();
}

void grasp_chain_detector_t::check_freeze(const grasp_chain_table_t &ct) {
  // Freeze only when CT is full AND every entry has learned its stride.
  // If CT has empty slots, new IMA chains may still appear and need detection.
  if (ct.is_full() && ct.all_stride_valid()) {
    m_training_frozen = true;
  }
}
