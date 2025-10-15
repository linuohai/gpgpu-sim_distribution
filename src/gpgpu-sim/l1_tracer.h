#ifndef L1_TRACER_H
#define L1_TRACER_H

#include <cstddef>
#include <string>
#include <vector>

#include "gpu-cache.h"

class mem_fetch;

class l1_tracer {
 public:
  static void init(bool enable, const char *path, unsigned n_sms);
  static void emit(unsigned sid, unsigned wid, const mem_fetch *mf,
                   enum cache_request_status status,
                   unsigned long long cycle);
  static void flush_all();

 private:
  static void flush_sid(unsigned sid);

  static bool s_enabled;
  static std::string s_path;
  static std::vector<std::string> s_buffers;
  static size_t s_flush_threshold;
};

#endif  // L1_TRACER_H
