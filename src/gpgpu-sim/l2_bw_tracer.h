#ifndef L2_BW_TRACER_H
#define L2_BW_TRACER_H

#include <cstddef>
#include <string>

#include "gpu-cache.h"

class memory_sub_partition;

class l2_bw_tracer {
 public:
  static void init(bool enable, const char *path, unsigned n_subpartitions,
                   unsigned data_port_width_bytes, double l2_period_seconds,
                   unsigned period);
  static void on_l2_tick(unsigned long long cycle,
                         memory_sub_partition **sub_partitions,
                         unsigned n_subpartitions);
  static void flush();
  static bool enabled() { return s_enabled; }

 private:
  static void append(const std::string &line);
  static void flush_buffer();
  static void reset_window();

  static bool s_enabled;
  static std::string s_path;
  static std::string s_buffer;
  static size_t s_flush_threshold;

  static unsigned s_period;
  static double s_l2_period;
  static unsigned s_n_subparts;
  static unsigned s_port_width_bytes;

  static unsigned s_tick_in_window;
  static unsigned long long s_cycle_begin;
  static unsigned long long s_cycle_end;
  static cache_sub_stats s_last_total;
  static bool s_has_window;
};

#endif  // L2_BW_TRACER_H
