#ifndef HBM_PARTITION_TRACER_H
#define HBM_PARTITION_TRACER_H

#include <cstddef>
#include <string>
#include <vector>

class hbm_partition_tracer {
 public:
  static void init(bool enable, const char *path, unsigned num_partitions,
                   unsigned period);
  static void emit(unsigned long long cycle, double total_bw_gbps,
                   double total_occ, const std::vector<double> &part_bw_gbps);
  static void flush();
  static bool enabled() { return s_enabled; }

 private:
  static void append(const std::string &line);
  static void flush_buffer();

  static bool s_enabled;
  static std::string s_path;
  static std::string s_buffer;
  static size_t s_flush_threshold;
  static unsigned s_period;
  static unsigned s_num_partitions;
  static unsigned long long s_dram_tick;
};

#endif  // HBM_PARTITION_TRACER_H
