#include "l2_bw_tracer.h"

#include <fstream>
#include <iomanip>
#include <sstream>

#include "l2cache.h"

bool l2_bw_tracer::s_enabled = false;
std::string l2_bw_tracer::s_path;
std::string l2_bw_tracer::s_buffer;
size_t l2_bw_tracer::s_flush_threshold = 1u << 15;

unsigned l2_bw_tracer::s_period = 500;
double l2_bw_tracer::s_l2_period = 0.0;
unsigned l2_bw_tracer::s_n_subparts = 0;
unsigned l2_bw_tracer::s_port_width_bytes = 0;

unsigned l2_bw_tracer::s_tick_in_window = 0;
unsigned long long l2_bw_tracer::s_cycle_begin = 0;
unsigned long long l2_bw_tracer::s_cycle_end = 0;
cache_sub_stats l2_bw_tracer::s_last_total;
bool l2_bw_tracer::s_has_window = false;

namespace {

double clamp01(double value) {
  if (value < 0.0) return 0.0;
  if (value > 1.0) return 1.0;
  return value;
}

cache_sub_stats diff_css(const cache_sub_stats &cur,
                         const cache_sub_stats &prev) {
  cache_sub_stats out;
  out.accesses = cur.accesses - prev.accesses;
  out.misses = cur.misses - prev.misses;
  out.pending_hits = cur.pending_hits - prev.pending_hits;
  out.res_fails = cur.res_fails - prev.res_fails;
  out.port_available_cycles =
      cur.port_available_cycles - prev.port_available_cycles;
  out.data_port_busy_cycles =
      cur.data_port_busy_cycles - prev.data_port_busy_cycles;
  out.fill_port_busy_cycles =
      cur.fill_port_busy_cycles - prev.fill_port_busy_cycles;
  return out;
}

}  // namespace

void l2_bw_tracer::init(bool enable, const char *path, unsigned n_subpartitions,
                        unsigned data_port_width_bytes, double l2_period_seconds,
                        unsigned period) {
  s_enabled = enable && path && *path;
  s_path.clear();
  s_buffer.clear();
  s_period = period ? period : 1;
  s_l2_period = l2_period_seconds;
  s_n_subparts = n_subpartitions;
  s_port_width_bytes = data_port_width_bytes;
  s_last_total.clear();
  reset_window();

  if (!s_enabled) return;
  if (s_n_subparts == 0 || s_port_width_bytes == 0 || s_l2_period <= 0.0) {
    s_enabled = false;
    return;
  }

  s_path = path;
  std::ofstream file(s_path.c_str(), std::ios::out | std::ios::trunc);
  if (!file.good()) {
    s_enabled = false;
    s_path.clear();
    return;
  }

  file << "cycle_begin,cycle_end,period,l2_available_cycles,l2_data_busy_cycles,"
          "l2_fill_busy_cycles,l2_data_GBps,l2_fill_GBps,l2_data_util,l2_fill_util\n";
}

void l2_bw_tracer::on_l2_tick(unsigned long long cycle,
                              memory_sub_partition **sub_partitions,
                              unsigned n_subpartitions) {
  if (!s_enabled) return;
  if (!sub_partitions) return;

  unsigned count = n_subpartitions;
  if (s_n_subparts && count > s_n_subparts) count = s_n_subparts;

  cache_sub_stats total;
  total.clear();
  for (unsigned i = 0; i < count; ++i) {
    cache_sub_stats sub;
    sub.clear();
    if (sub_partitions[i]) {
      sub_partitions[i]->get_L2cache_sub_stats(sub);
    }
    total += sub;
  }

  if (!s_has_window) {
    s_has_window = true;
    s_cycle_begin = cycle;
    s_cycle_end = cycle;
    s_tick_in_window = 0;
  }

  s_cycle_end = cycle;
  ++s_tick_in_window;
  if (s_tick_in_window < s_period) return;

  cache_sub_stats delta = diff_css(total, s_last_total);
  s_last_total = total;

  const unsigned long long avail = delta.port_available_cycles;
  const unsigned long long data_busy = delta.data_port_busy_cycles;
  const unsigned long long fill_busy = delta.fill_port_busy_cycles;

  const double window_time = static_cast<double>(s_period) * s_l2_period;
  double data_gbps = 0.0;
  double fill_gbps = 0.0;
  if (window_time > 0.0) {
    data_gbps = (static_cast<double>(data_busy) *
                 static_cast<double>(s_port_width_bytes)) /
                window_time / 1.0e9;
    fill_gbps = (static_cast<double>(fill_busy) *
                 static_cast<double>(s_port_width_bytes)) /
                window_time / 1.0e9;
  }

  double data_util = 0.0;
  double fill_util = 0.0;
  if (avail) {
    data_util =
        clamp01(static_cast<double>(data_busy) / static_cast<double>(avail));
    fill_util =
        clamp01(static_cast<double>(fill_busy) / static_cast<double>(avail));
  }

  std::ostringstream oss;
  oss << s_cycle_begin << ',' << s_cycle_end << ',' << s_period << ',' << avail
      << ',' << data_busy << ',' << fill_busy << ',' << std::fixed
      << std::setprecision(6) << data_gbps << ',' << fill_gbps << ','
      << data_util << ',' << fill_util << '\n';
  append(oss.str());
  flush_buffer();
  reset_window();
}

void l2_bw_tracer::flush() {
  if (!s_enabled) return;
  flush_buffer();
}

void l2_bw_tracer::append(const std::string &line) {
  s_buffer.append(line);
  if (s_buffer.size() >= s_flush_threshold) {
    flush_buffer();
  }
}

void l2_bw_tracer::flush_buffer() {
  if (s_buffer.empty()) return;
  std::ofstream file(s_path.c_str(), std::ios::out | std::ios::app);
  if (!file.good()) return;
  file << s_buffer;
  s_buffer.clear();
}

void l2_bw_tracer::reset_window() {
  s_tick_in_window = 0;
  s_has_window = false;
  s_cycle_begin = 0;
  s_cycle_end = 0;
}
