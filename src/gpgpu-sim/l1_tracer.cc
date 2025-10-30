#include "l1_tracer.h"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <unordered_set>

#include "mem_fetch.h"

namespace {

std::string format_double(double value) {
  std::ostringstream tmp;
  tmp.setf(std::ios::fixed);
  tmp << std::setprecision(6) << value;
  return tmp.str();
}

}  // namespace

bool l1_tracer::s_enabled = false;
std::string l1_tracer::s_path;
std::vector<std::string> l1_tracer::s_buffers;
size_t l1_tracer::s_flush_threshold = 1u << 16;

void l1_tracer::init(bool enable, const char *path, unsigned n_sms) {
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

  file <<
      "cycle,sm_id,warp_id,lane_id,op,address,l1_status,pc,hbm_bw_GBps,";
  file << "hbm_occupancy\n";
}

void l1_tracer::emit(unsigned sid, unsigned wid, const mem_fetch *mf,
                     enum cache_request_status status,
                     unsigned long long cycle, unsigned line_sz,
                     double hbm_bandwidth_gbps, double hbm_occupancy) {
  if (!s_enabled || !mf) return;
  if (sid >= s_buffers.size()) return;
  const std::vector<std::pair<unsigned, addr_t>> &lanes = mf->dbg_lanes();
  if (lanes.empty()) return;

  const char *status_name = cache_request_status_str(status);
  const char *op = mf->dbg_is_store() ? "ST" : "LD";

  std::ostringstream oss;
  address_type pc = mf->get_pc();
  bool has_pc = pc != static_cast<address_type>(-1);
  const std::string bw_str = format_double(hbm_bandwidth_gbps);
  const std::string occ_str = format_double(hbm_occupancy);

  std::unordered_set<addr_t> seen_lines;
  seen_lines.reserve(lanes.size());

  for (const auto &entry : lanes) {
    addr_t line_addr = entry.second;
    if (line_sz) {
      line_addr = entry.second &
                  ~static_cast<addr_t>(static_cast<addr_t>(line_sz) - 1);
    }
    if (!seen_lines.insert(line_addr).second) continue;

    oss << cycle << ',' << sid << ',' << wid << ',' << entry.first << ',' << op
        << ',';
    oss << "0x" << std::hex << entry.second << std::dec << ',' << status_name
        << ',';
    if (has_pc) {
      oss << "0x" << std::hex << pc << std::dec;
    } else {
      oss << "NA";
    }
    oss << ',' << bw_str << ',' << occ_str;
    oss << '\n';
  }

  std::string &buffer = s_buffers[sid];
  buffer.append(oss.str());
  if (buffer.size() >= s_flush_threshold) {
    flush_sid(sid);
  }
}

void l1_tracer::flush_all() {
  if (!s_enabled) return;
  for (unsigned sid = 0; sid < s_buffers.size(); ++sid) {
    flush_sid(sid);
  }
}

void l1_tracer::flush_sid(unsigned sid) {
  if (!s_enabled) return;
  if (sid >= s_buffers.size()) return;
  std::string &buffer = s_buffers[sid];
  if (buffer.empty()) return;

  std::ofstream file(s_path.c_str(), std::ios::out | std::ios::app);
  if (!file.good()) return;

  file << buffer;
  buffer.clear();
}
