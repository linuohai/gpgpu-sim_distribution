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

  file << "cycle,sm_id,warp_id,lane_id,op,address,l1_status,pc,hbm_bw_GBps,";
  file << "hbm_occupancy,sm_alu_active_lanes,sm_alu_total_lanes,";
  file << "sm_alu_utilization,sm_sp_active_lanes,sm_sp_total_lanes,"
       << "sm_sp_utilization,sm_int_active_lanes,sm_int_total_lanes,"
       << "sm_int_utilization,sm_dp_active_lanes,sm_dp_total_lanes,"
       << "sm_dp_utilization,sm_sfu_active_lanes,sm_sfu_total_lanes,"
       << "sm_sfu_utilization,sm_tensor_active_lanes,sm_tensor_total_lanes,"
       << "sm_tensor_utilization\n";
}

void l1_tracer::emit(unsigned sid, unsigned wid, const mem_fetch *mf,
                     enum cache_request_status status,
                     unsigned long long cycle, unsigned line_sz,
                     double hbm_bandwidth_gbps, double hbm_occupancy,
                     unsigned active_alu_lanes, unsigned total_alu_lanes,
                     unsigned active_sp_lanes, unsigned total_sp_lanes,
                     unsigned active_int_lanes, unsigned total_int_lanes,
                     unsigned active_dp_lanes, unsigned total_dp_lanes,
                     unsigned active_sfu_lanes, unsigned total_sfu_lanes,
                     unsigned active_tensor_lanes, unsigned total_tensor_lanes) {
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
  double alu_util = 0.0;
  if (total_alu_lanes) {
    alu_util =
        static_cast<double>(active_alu_lanes) / static_cast<double>(total_alu_lanes);
    if (alu_util > 1.0) alu_util = 1.0;
  }
  const std::string alu_util_str = format_double(alu_util);
  auto util_string = [](unsigned active, unsigned total) -> std::string {
    double util = 0.0;
    if (total) {
      util = static_cast<double>(active) / static_cast<double>(total);
      if (util > 1.0) util = 1.0;
    }
    return format_double(util);
  };
  const std::string sp_util_str = util_string(active_sp_lanes, total_sp_lanes);
  const std::string int_util_str =
      util_string(active_int_lanes, total_int_lanes);
  const std::string dp_util_str = util_string(active_dp_lanes, total_dp_lanes);
  const std::string sfu_util_str =
      util_string(active_sfu_lanes, total_sfu_lanes);
  const std::string tensor_util_str =
      util_string(active_tensor_lanes, total_tensor_lanes);

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
    oss << ',' << active_alu_lanes << ',' << total_alu_lanes << ','
        << alu_util_str;
    oss << ',' << active_sp_lanes << ',' << total_sp_lanes << ','
        << sp_util_str;
    oss << ',' << active_int_lanes << ',' << total_int_lanes << ','
        << int_util_str;
    oss << ',' << active_dp_lanes << ',' << total_dp_lanes << ','
        << dp_util_str;
    oss << ',' << active_sfu_lanes << ',' << total_sfu_lanes << ','
        << sfu_util_str;
    oss << ',' << active_tensor_lanes << ',' << total_tensor_lanes << ','
        << tensor_util_str;
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
