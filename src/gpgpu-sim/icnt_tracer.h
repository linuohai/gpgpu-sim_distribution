#ifndef ICNT_TRACER_H
#define ICNT_TRACER_H

#include <cstddef>
#include <string>

class icnt_tracer {
 public:
  static void init(bool enable, const char *path, unsigned n_shader_inputs,
                   unsigned n_mem_outputs, double icnt_period_seconds,
                   unsigned period, unsigned flit_size_bytes);
  static void on_icnt_tick(unsigned long long cycle, unsigned forwarded_req,
                           unsigned forwarded_reply);
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
  static double s_icnt_period;
  static unsigned s_flit_bytes;
  static unsigned s_peak_req_packets_per_cycle;
  static unsigned s_peak_reply_packets_per_cycle;

  static unsigned s_tick_in_window;
  static unsigned long long s_cycle_begin;
  static unsigned long long s_cycle_end;
  static unsigned long long s_req_packets_sum;
  static unsigned long long s_reply_packets_sum;
  static bool s_has_window;
};

#endif  // ICNT_TRACER_H
