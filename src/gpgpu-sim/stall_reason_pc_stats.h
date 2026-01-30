#ifndef STALL_REASON_PC_STATS_H
#define STALL_REASON_PC_STATS_H

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>

#include "../abstract_hardware_model.h"

class stall_reason_pc_stats {
 public:
  enum class reason {
    MEM_WAIT = 0,
    REG_WAIT,
    IBUFFER_EMPTY,
    WAIT_CTA_BARRIER,
    WAIT_MEMBAR,
    WAIT_ATOMIC,
    WAIT_LDGSTS,
    WAIT_DONE,
    CONTROL_HAZARD,
    PIPE_BUSY,
    DUAL_ISSUE_RESTRICT,
    COUNT
  };

  static void init(bool enable, const char *out_dir, unsigned topk);
  static bool enabled() { return s_enabled; }
  static void add(reason reason_id, address_type pc, uint64_t count);
  static void dump();
  static const char *reason_name(reason reason_id);

 private:
  static constexpr size_t kReasonCount =
      static_cast<size_t>(reason::COUNT);
  using PcHistogram = std::unordered_map<address_type, uint64_t>;

  static bool s_enabled;
  static std::string s_out_dir;
  static unsigned s_topk;
  static std::array<uint64_t, kReasonCount> s_total_by_reason;
  static std::array<PcHistogram, kReasonCount> s_hist_by_reason_pc;
};

#endif  // STALL_REASON_PC_STATS_H
