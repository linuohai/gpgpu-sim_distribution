#include "hbm_partition_tracer.h"

#include <fstream>
#include <iomanip>
#include <sstream>

bool hbm_partition_tracer::s_enabled = false;
std::string hbm_partition_tracer::s_path;
std::string hbm_partition_tracer::s_buffer;
size_t hbm_partition_tracer::s_flush_threshold = 1u << 15;
unsigned hbm_partition_tracer::s_period = 1;
unsigned hbm_partition_tracer::s_num_partitions = 0;
unsigned long long hbm_partition_tracer::s_dram_tick = 0;

void hbm_partition_tracer::init(bool enable, const char *path,
                                unsigned num_partitions, unsigned period) {
  s_enabled = enable && path && *path;
  s_path.clear();
  s_buffer.clear();
  s_num_partitions = 0;
  s_dram_tick = 0;
  s_period = period ? period : 1;
  if (!s_enabled) return;

  s_path = path;
  s_num_partitions = num_partitions;

  std::ofstream file(s_path.c_str(), std::ios::out | std::ios::trunc);
  if (!file.good()) {
    s_enabled = false;
    s_path.clear();
    s_num_partitions = 0;
    return;
  }

  file << "cycle,hbm_bw_GBps,hbm_occupancy";
  for (unsigned i = 0; i < s_num_partitions; ++i) {
    file << ",part" << i << "_bw_GBps";
  }
  file << '\n';
}

void hbm_partition_tracer::emit(
    unsigned long long cycle, double total_bw_gbps, double total_occ,
    const std::vector<double> &part_bw_gbps) {
  if (!s_enabled) return;
  ++s_dram_tick;
  if ((s_dram_tick - 1) % s_period != 0) return;

  std::ostringstream oss;
  oss << cycle << ',' << std::fixed << std::setprecision(6) << total_bw_gbps
      << ',' << total_occ;
  for (unsigned i = 0; i < s_num_partitions; ++i) {
    double value = (i < part_bw_gbps.size()) ? part_bw_gbps[i] : 0.0;
    oss << ',' << value;
  }
  oss << '\n';
  append(oss.str());
}

void hbm_partition_tracer::flush() {
  if (!s_enabled) return;
  flush_buffer();
}

void hbm_partition_tracer::append(const std::string &line) {
  s_buffer.append(line);
  if (s_buffer.size() >= s_flush_threshold) {
    flush_buffer();
  }
}

void hbm_partition_tracer::flush_buffer() {
  if (s_buffer.empty()) return;
  std::ofstream file(s_path.c_str(), std::ios::out | std::ios::app);
  if (!file.good()) return;
  file << s_buffer;
  s_buffer.clear();
}
