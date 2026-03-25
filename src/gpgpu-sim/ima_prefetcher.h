// IMA (Indirect Memory Access) Prefetcher — v0.1
//
// Implements stride-based index-array prefetching via a per-SM IMA Pattern
// Table (IPT).  The prefetcher observes global load addresses at each PC,
// detects a regular stride in the address stream, and speculatively issues
// prefetch requests for upcoming sectors of the index array.
//
// Architecture:
//   Component A — IMA Pattern Table (IPT):
//     Per-SM, fully-associative table mapping load PC → stride/confidence.
//     Training: on every global load, ipt_update() is called to refine the
//     stride estimate and update a 2-bit saturating confidence counter.
//
//   Component B — Index Prefetch:
//     When confidence >= threshold, addresses (addr + n*stride) for
//     n=1..prefetch_distance are enqueued into ldst_unit::m_prefetch_queue.
//     The ldst_unit drains this queue each cycle, filtering requests that
//     would be L1 HIT or MSHR_HIT (already in-flight).
//
// Data prefetch (Component C) requires actual index values and is deferred
// to a future version requiring functional simulation support.

#pragma once

#include <cstdint>
#include <cstdio>
#include <vector>
#include "../abstract_hardware_model.h"

// One entry in the IMA Pattern Table.
struct ipt_entry_t {
  bool valid = false;
  new_addr_type pc = 0;          // PC of the tracked global load
  new_addr_type last_addr = 0;   // most recently observed address
  int64_t stride = 0;            // detected byte stride in the address stream
  unsigned confidence = 0;       // 2-bit saturating confidence counter (0..3)
  unsigned long long last_access_time = 0;  // for LRU replacement

  static constexpr unsigned MAX_CONFIDENCE = 3;
};

// Per-SM IMA prefetcher.
class ima_prefetcher_t {
 public:
  ima_prefetcher_t(unsigned sm_id, unsigned ipt_size, unsigned distance,
                   unsigned conf_thresh);
  ~ima_prefetcher_t() = default;

  // Called on every global load (GLOBAL_ACC_R) access.
  // Updates the IPT entry for `pc` at `addr`, and returns a list of prefetch
  // addresses to enqueue if confidence is sufficient.
  std::vector<new_addr_type> on_load_access(new_addr_type pc,
                                             new_addr_type addr,
                                             unsigned long long cycle);

  // Print per-SM prefetch statistics.
  void print_stats(FILE *fp) const;

  // Counters updated externally (by ldst_unit) after filter decisions.
  unsigned long long stat_prefetch_generated = 0;  // addresses queued
  unsigned long long stat_prefetch_issued = 0;      // L2 prefetches sent
  unsigned long long stat_prefetch_filtered = 0;    // dropped (L1 hit or MSHR)

 private:
  // TODO(human): Implement ipt_update() below.
  // This is the core training function for the IPT entry.
  void ipt_update(ipt_entry_t &entry, new_addr_type new_addr);

  // Return index of IPT entry for `pc`, or -1 if not found.
  int find_entry(new_addr_type pc) const;

  // Return index of a victim entry to replace (LRU policy).
  int find_lru_victim() const;

  unsigned m_sm_id;
  unsigned m_ipt_size;     // number of IPT entries
  unsigned m_distance;     // prefetch lookahead (in strides)
  unsigned m_conf_thresh;  // minimum confidence to trigger prefetch

  std::vector<ipt_entry_t> m_ipt;
  unsigned long long m_clock = 0;  // current cycle (for LRU timestamps)
};
