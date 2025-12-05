#ifndef ISSUE_TRACER_H
#define ISSUE_TRACER_H

#include <cstddef>
#include <string>
#include <vector>

#include "../abstract_hardware_model.h"

struct issue_stall_counts {
  unsigned mem_wait = 0;
  unsigned reg_wait = 0;
  unsigned ibuffer_empty = 0;
  unsigned barrier = 0;
  unsigned control_hazard = 0;
  unsigned pipe_busy = 0;
  unsigned dual_issue_restrict = 0;
};

class issue_tracer {
 public:
 static void init(bool enable, const char *path, unsigned n_sms);
  static void emit_issue(unsigned sid, unsigned wid, unsigned sch_id,
                         unsigned long long cycle, const active_mask_t &mask,
                         const std::string &opcode,
                         const std::string &space, address_type pc,
                         const std::string &sector_addresses,
                         const std::string &sector_lane_ids);
  static void emit_stall(unsigned sid, int sample_warp,
                         unsigned long long cycle, const active_mask_t *mask,
                         const std::string &opcode,
                         const std::string &space, address_type pc,
                         int scheduler_id, const issue_stall_counts &counts,
                         const std::string &one_reason);
  static void flush_all();
  static bool enabled() { return s_enabled; }

 private:
  static void flush_sid(unsigned sid);
  static void append(unsigned sid, const std::string &line);
  static std::string mask_to_hex(const active_mask_t &mask);
  static std::string hexify(address_type value);
  static std::string escape(const std::string &field);

  static bool s_enabled;
  static std::string s_path;
  static std::vector<std::string> s_buffers;
  static size_t s_flush_threshold;
};

#endif  // ISSUE_TRACER_H
