#include "icnt_tracer.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>

bool icnt_tracer::s_enabled = false;
std::string icnt_tracer::s_path;
std::string icnt_tracer::s_buffer;
size_t icnt_tracer::s_flush_threshold = 1u << 15;

unsigned icnt_tracer::s_period = 500;
double icnt_tracer::s_icnt_period = 0.0;
unsigned icnt_tracer::s_flit_bytes = 0;
unsigned icnt_tracer::s_peak_req_packets_per_cycle = 0;
unsigned icnt_tracer::s_peak_reply_packets_per_cycle = 0;

unsigned icnt_tracer::s_tick_in_window = 0;
unsigned long long icnt_tracer::s_cycle_begin = 0;
unsigned long long icnt_tracer::s_cycle_end = 0;
unsigned long long icnt_tracer::s_req_packets_sum = 0;
unsigned long long icnt_tracer::s_reply_packets_sum = 0;
bool icnt_tracer::s_has_window = false;

namespace {

double clamp01(double value) {
  if (value < 0.0) return 0.0;
  if (value > 1.0) return 1.0;
  return value;
}

}  // namespace

void icnt_tracer::init(bool enable, const char *path, unsigned n_shader_inputs,
                       unsigned n_mem_outputs, double icnt_period_seconds,
                       unsigned period, unsigned flit_size_bytes) {
  s_enabled = enable && path && *path;
  s_path.clear();
  s_buffer.clear();
  s_period = period ? period : 1;
  s_icnt_period = icnt_period_seconds;
  s_flit_bytes = flit_size_bytes;
  s_peak_req_packets_per_cycle =
      std::min(n_shader_inputs, n_mem_outputs);
  s_peak_reply_packets_per_cycle = s_peak_req_packets_per_cycle;
  reset_window();

  if (!s_enabled) return;

  s_path = path;
  std::ofstream file(s_path.c_str(), std::ios::out | std::ios::trunc);
  if (!file.good()) {
    s_enabled = false;
    s_path.clear();
    return;
  }

  file << "cycle_begin,cycle_end,period,req_packets,reply_packets,req_wire_GBps,"
          "reply_wire_GBps,req_util,reply_util\n";
}

void icnt_tracer::on_icnt_tick(unsigned long long cycle, unsigned forwarded_req,
                               unsigned forwarded_reply) {
  if (!s_enabled) return;

  if (!s_has_window) {
    s_has_window = true;
    s_cycle_begin = cycle;
    s_cycle_end = cycle;
    s_tick_in_window = 0;
    s_req_packets_sum = 0;
    s_reply_packets_sum = 0;
  }

  s_cycle_end = cycle;
  s_req_packets_sum += forwarded_req;
  s_reply_packets_sum += forwarded_reply;
  ++s_tick_in_window;

  if (s_tick_in_window < s_period) return;

  const double freq = (s_icnt_period > 0.0) ? (1.0 / s_icnt_period) : 0.0;
  const double req_ppc =
      static_cast<double>(s_req_packets_sum) / static_cast<double>(s_period);
  const double reply_ppc =
      static_cast<double>(s_reply_packets_sum) / static_cast<double>(s_period);
  const double req_wire_gbps =
      req_ppc * static_cast<double>(s_flit_bytes) * freq / 1.0e9;
  const double reply_wire_gbps =
      reply_ppc * static_cast<double>(s_flit_bytes) * freq / 1.0e9;

  double req_util = 0.0;
  double reply_util = 0.0;
  if (s_peak_req_packets_per_cycle) {
    req_util = clamp01(req_ppc /
                       static_cast<double>(s_peak_req_packets_per_cycle));
  }
  if (s_peak_reply_packets_per_cycle) {
    reply_util = clamp01(
        reply_ppc / static_cast<double>(s_peak_reply_packets_per_cycle));
  }

  std::ostringstream oss;
  oss << s_cycle_begin << ',' << s_cycle_end << ',' << s_period << ','
      << s_req_packets_sum << ',' << s_reply_packets_sum << ','
      << std::fixed << std::setprecision(6) << req_wire_gbps << ','
      << reply_wire_gbps << ',' << req_util << ',' << reply_util << '\n';
  append(oss.str());
  flush_buffer();
  reset_window();
}

void icnt_tracer::flush() {
  if (!s_enabled) return;
  flush_buffer();
}

void icnt_tracer::append(const std::string &line) {
  s_buffer.append(line);
  if (s_buffer.size() >= s_flush_threshold) {
    flush_buffer();
  }
}

void icnt_tracer::flush_buffer() {
  if (s_buffer.empty()) return;
  std::ofstream file(s_path.c_str(), std::ios::out | std::ios::app);
  if (!file.good()) return;
  file << s_buffer;
  s_buffer.clear();
}

void icnt_tracer::reset_window() {
  s_tick_in_window = 0;
  s_req_packets_sum = 0;
  s_reply_packets_sum = 0;
  s_has_window = false;
  s_cycle_begin = 0;
  s_cycle_end = 0;
}
