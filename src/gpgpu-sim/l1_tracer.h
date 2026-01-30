#ifndef L1_TRACER_H
#define L1_TRACER_H

#include <cstddef>
#include <string>
#include <vector>

#include "gpu-cache.h"

class mem_fetch;

class l1_tracer {
 public:
  static void init(bool enable, const char *path, unsigned n_sms,
                   bool print_bw, bool print_compute);
  static void emit(unsigned sid, unsigned wid, const mem_fetch *mf,
                   enum cache_request_status status,
                   unsigned long long cycle, unsigned line_sz,
                   double hbm_bandwidth_gbps, double hbm_occupancy,
                   unsigned active_alu_lanes, unsigned total_alu_lanes,
                   unsigned active_sp_lanes, unsigned total_sp_lanes,
                   unsigned active_int_lanes, unsigned total_int_lanes,
                   unsigned active_dp_lanes, unsigned total_dp_lanes,
                   unsigned active_sfu_lanes, unsigned total_sfu_lanes,
                   unsigned active_tensor_lanes, unsigned total_tensor_lanes);
  static void flush_all();

 private:
  static void flush_sid(unsigned sid);

  static bool s_enabled;
  static bool s_print_bw;
  static bool s_print_compute;
  static std::string s_path;
  static std::vector<std::string> s_buffers;
  static size_t s_flush_threshold;
};

#endif  // L1_TRACER_H
