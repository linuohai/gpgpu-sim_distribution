#include "grasp_tracer.h"

#include <fstream>
#include <sstream>

bool grasp_tracer::s_enabled = false;
std::string grasp_tracer::s_path;
std::vector<std::string> grasp_tracer::s_buffers;
size_t grasp_tracer::s_flush_threshold = 1u << 16;  // 64 KB per-SM buffer

void grasp_tracer::init(bool enable, const char *path, unsigned n_sms) {
  s_enabled = enable && path && *path;
  s_path.clear();
  s_buffers.clear();
  if (!s_enabled) return;

  s_path = path;
  s_buffers.assign(n_sms, std::string());

  std::ofstream file(s_path.c_str(), std::ios::out | std::ios::trunc);
  if (!file.good()) {
    s_enabled = false;
    s_path.clear();
    s_buffers.clear();
    return;
  }
  file << "cycle,sm_id,warp_id,event,pc,addr,detail\n";
}

void grasp_tracer::emit(unsigned sm_id, unsigned warp_id,
                        unsigned long long cycle, const char *event,
                        unsigned long long pc, unsigned long long addr,
                        const char *detail) {
  if (!s_enabled) return;
  if (sm_id >= s_buffers.size()) return;

  std::ostringstream oss;
  oss << cycle << ',' << sm_id << ',' << warp_id << ',' << event << ','
      << "0x" << std::hex << pc << ','
      << "0x" << std::hex << addr << std::dec << ','
      << (detail ? detail : "") << '\n';

  std::string &buf = s_buffers[sm_id];
  buf.append(oss.str());
  if (buf.size() >= s_flush_threshold) flush_sid(sm_id);
}

void grasp_tracer::flush_all() {
  if (!s_enabled) return;
  for (unsigned i = 0; i < static_cast<unsigned>(s_buffers.size()); ++i) {
    flush_sid(i);
  }
}

void grasp_tracer::flush_sid(unsigned sid) {
  if (!s_enabled) return;
  if (sid >= s_buffers.size()) return;
  std::string &buf = s_buffers[sid];
  if (buf.empty()) return;

  std::ofstream file(s_path.c_str(), std::ios::out | std::ios::app);
  if (!file.good()) return;

  file << buf;
  buf.clear();
}
