// GRASP Chain Detector (CD) — Header
//
// Detects IMA (Indirect Memory Access) chains in the instruction stream:
//   LDG (index load) → IMAD.WIDE (address computation) → LDG (data load)
//
// Uses a FIFO to track register producers. When a data LDG's address register
// was produced by an IMAD.WIDE whose source was an index LDG, a chain is
// detected.
//
// Trace-driven mode: uses SASS opcode string to identify IMAD.WIDE; pair table
// substitutes for register value computation.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "../abstract_hardware_model.h"

// Forward declaration
class grasp_chain_table_t;

class grasp_chain_detector_t {
 public:
  explicit grasp_chain_detector_t(unsigned fifo_depth);

  // Detected chain descriptor
  struct detected_chain_t {
    new_addr_type index_pc;
    new_addr_type data_pc;
    // Trace-driven mode: scale/base are placeholders
    unsigned scale_placeholder;
    unsigned base_placeholder;
    // Index load address (first active lane) — used to seed stride observation
    new_addr_type index_addr;
    unsigned index_lane_id;
    unsigned index_data_size;  // index LDG memory width (bytes), for speculative stride
  };

  // Called on each instruction issue (tracked warp only).
  // If chain detected, writes to *out_chain and returns true.
  bool on_instruction_issue(unsigned warp_id, new_addr_type pc, op_type op,
                            const std::string &sass_opcode, int dst_reg,
                            const int *src_regs,  // inst.arch_reg.src[]
                            unsigned num_src_regs, unsigned long long cycle,
                            new_addr_type first_lane_addr,
                            unsigned first_lane_id,
                            unsigned data_size,  // memory access width (bytes)
                            detected_chain_t *out_chain);

  // Warp exit: clear FIFO
  void on_warp_exit(unsigned warp_id);

  // Kernel launch: full reset
  void reset();

  // Training freeze status
  bool is_frozen() const { return m_training_frozen; }

  // Force check: should freeze? (all CT entries stride_valid)
  void check_freeze(const grasp_chain_table_t &ct);

  // Note: no unfreeze() — once tracked warp exits, training is permanently
  // frozen for this kernel. Reset on next kernel launch.

  // Get tracked warp IDs (up to 2)
  unsigned tracked_warp_id() const { return m_tracked_warp_ids[0]; }
  unsigned tracked_warp_id_2() const { return m_tracked_warp_ids[1]; }
  bool is_tracked_warp(unsigned warp_id) const {
    return warp_id == m_tracked_warp_ids[0] || warp_id == m_tracked_warp_ids[1];
  }

  // Stats
  struct cd_stats_t {
    unsigned long long chains_detected = 0;
    unsigned long long fifo_peak_occupancy = 0;
    unsigned long long fifo_drop_count = 0;
    unsigned long long fifo_read_invalidations = 0;
    unsigned long long fifo_write_invalidations = 0;
    // Metrics: distinct index PCs ever detected (for CT coverage denominator)
    unsigned long long unique_index_pcs_detected = 0;
  };
  const cd_stats_t &stats() const { return m_stats; }

 private:
  static constexpr unsigned UNSET_WARP = (unsigned)-1;

  // FIFO entry types
  enum fifo_type_t { LOAD_RESULT, IMA_ADDR_COMPUTE };

  struct fifo_entry_t {
    bool valid = false;
    int dst_reg = -1;
    unsigned warp_id = (unsigned)-1;  // owning warp
    fifo_type_t type = LOAD_RESULT;
    new_addr_type pc = 0;
    // Memory address (first active lane) — for LOAD_RESULT
    new_addr_type addr = 0;
    unsigned lane_id = 0;
    unsigned data_size = 0;  // memory access width in bytes
    // Only for IMA_ADDR_COMPUTE type
    new_addr_type index_pc = 0;
    new_addr_type index_addr = 0;
    unsigned index_lane_id = 0;
    unsigned scale = 0;
    unsigned base = 0;
  };

  // Circular buffer FIFO operations
  void push(const fifo_entry_t &entry);
  const fifo_entry_t *lookup_by_reg(int reg, unsigned warp_id) const;
  void invalidate_by_src(const int *src_regs,
                         unsigned num_src);  // Read Detection
  void invalidate_by_dst(int dst_reg);       // Write Invalidation

  std::vector<fifo_entry_t> m_fifo;
  unsigned m_depth;
  unsigned m_head = 0, m_tail = 0, m_count = 0;  // circular buffer

  unsigned m_tracked_warp_ids[2] = {UNSET_WARP, UNSET_WARP};
  bool m_training_frozen = false;

  cd_stats_t m_stats;

  // Metrics: track unique index PCs for CT coverage ratio
  std::unordered_set<new_addr_type> m_unique_index_pcs;
};
