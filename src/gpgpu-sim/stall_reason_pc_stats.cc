#include "stall_reason_pc_stats.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <vector>

namespace {

std::string join_path(const std::string &dir, const char *name) {
  if (dir.empty()) return std::string(name);
  char tail = dir.back();
  if (tail == '/' || tail == '\\') return dir + name;
  return dir + "/" + name;
}

bool is_dir(const std::string &path) {
  if (path.empty()) return false;
  struct stat st;
  if (stat(path.c_str(), &st) != 0) return false;
  return S_ISDIR(st.st_mode);
}

std::string format_double(double value) {
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss << std::setprecision(8) << value;
  return oss.str();
}

std::string hexify(address_type value) {
  std::ostringstream oss;
  oss << "0x" << std::hex << value << std::dec;
  return oss.str();
}

}  // namespace

bool stall_reason_pc_stats::s_enabled = false;
std::string stall_reason_pc_stats::s_out_dir;
unsigned stall_reason_pc_stats::s_topk = 0;
std::array<uint64_t, stall_reason_pc_stats::kReasonCount>
    stall_reason_pc_stats::s_total_by_reason = {};
std::array<stall_reason_pc_stats::PcHistogram,
           stall_reason_pc_stats::kReasonCount>
    stall_reason_pc_stats::s_hist_by_reason_pc = {};

void stall_reason_pc_stats::init(bool enable, const char *out_dir,
                                 unsigned topk) {
  s_enabled = enable && out_dir && *out_dir;
  s_out_dir.clear();
  s_topk = topk;
  s_total_by_reason.fill(0);
  for (auto &hist : s_hist_by_reason_pc) hist.clear();

  if (!s_enabled) return;
  s_out_dir = out_dir;
  if (!is_dir(s_out_dir)) {
    s_enabled = false;
    s_out_dir.clear();
    return;
  }
}

void stall_reason_pc_stats::add(reason reason_id, address_type pc,
                                uint64_t count) {
  if (!s_enabled || count == 0) return;
  size_t idx = static_cast<size_t>(reason_id);
  if (idx >= kReasonCount) return;
  s_total_by_reason[idx] += count;
  s_hist_by_reason_pc[idx][pc] += count;
}

const char *stall_reason_pc_stats::reason_name(reason reason_id) {
  switch (reason_id) {
    case reason::MEM_WAIT:
      return "MEM_WAIT";
    case reason::REG_WAIT:
      return "REG_WAIT";
    case reason::IBUFFER_EMPTY:
      return "IBUFFER_EMPTY";
    case reason::WAIT_CTA_BARRIER:
      return "WAIT_CTA_BARRIER";
    case reason::WAIT_MEMBAR:
      return "WAIT_MEMBAR";
    case reason::WAIT_ATOMIC:
      return "WAIT_ATOMIC";
    case reason::WAIT_LDGSTS:
      return "WAIT_LDGSTS";
    case reason::WAIT_DONE:
      return "WAIT_DONE";
    case reason::CONTROL_HAZARD:
      return "CONTROL_HAZARD";
    case reason::PIPE_BUSY:
      return "PIPE_BUSY";
    case reason::DUAL_ISSUE_RESTRICT:
      return "DUAL_ISSUE_RESTRICT";
    case reason::COUNT:
      break;
  }
  return "UNKNOWN";
}

void stall_reason_pc_stats::dump() {
  if (!s_enabled) return;

  uint64_t total_all = 0;
  for (auto total : s_total_by_reason) total_all += total;

  std::string breakdown_path =
      join_path(s_out_dir, "stall_reason_breakdown.csv");
  std::string hist_path =
      join_path(s_out_dir, "stall_reason_pc_hist.csv");
  std::string topk_path =
      join_path(s_out_dir, "stall_reason_pc_topk_other.csv");

  std::ofstream breakdown(breakdown_path.c_str(),
                          std::ios::out | std::ios::trunc);
  if (!breakdown.good()) return;
  breakdown << "reason,stall_warp_events,fraction_of_stall_warp_events\n";
  for (size_t idx = 0; idx < kReasonCount; ++idx) {
    uint64_t total = s_total_by_reason[idx];
    double fraction =
        total_all ? static_cast<double>(total) / total_all : 0.0;
    breakdown << reason_name(static_cast<reason>(idx)) << ',' << total << ','
              << format_double(fraction) << '\n';
  }

  std::ofstream hist(hist_path.c_str(), std::ios::out | std::ios::trunc);
  if (!hist.good()) return;
  hist << "reason,pc,stall_warp_events,"
          "fraction_of_reason_stall_warp_events\n";
  for (size_t idx = 0; idx < kReasonCount; ++idx) {
    const auto &map = s_hist_by_reason_pc[idx];
    if (map.empty()) continue;

    std::vector<std::pair<address_type, uint64_t>> entries(map.begin(),
                                                           map.end());
    std::sort(entries.begin(), entries.end(),
              [](const std::pair<address_type, uint64_t> &lhs,
                 const std::pair<address_type, uint64_t> &rhs) {
                if (lhs.second != rhs.second) return lhs.second > rhs.second;
                return lhs.first < rhs.first;
              });

    uint64_t reason_total = s_total_by_reason[idx];
    for (const auto &entry : entries) {
      double fraction =
          reason_total ? static_cast<double>(entry.second) / reason_total
                       : 0.0;
      hist << reason_name(static_cast<reason>(idx)) << ',' << hexify(entry.first)
           << ',' << entry.second << ',' << format_double(fraction) << '\n';
    }
  }

  if (s_topk == 0) return;
  std::ofstream topk(topk_path.c_str(), std::ios::out | std::ios::trunc);
  if (!topk.good()) return;
  topk << "reason,segment,rank,stall_warp_events,fraction_of_reason,topk_share\n";

  for (size_t idx = 0; idx < kReasonCount; ++idx) {
    const auto &map = s_hist_by_reason_pc[idx];
    uint64_t reason_total = s_total_by_reason[idx];
    if (reason_total == 0 || map.empty()) continue;

    std::vector<std::pair<address_type, uint64_t>> entries(map.begin(),
                                                           map.end());
    std::sort(entries.begin(), entries.end(),
              [](const std::pair<address_type, uint64_t> &lhs,
                 const std::pair<address_type, uint64_t> &rhs) {
                if (lhs.second != rhs.second) return lhs.second > rhs.second;
                return lhs.first < rhs.first;
              });

    size_t k = std::min<size_t>(s_topk, entries.size());
    uint64_t sum_topk = 0;
    for (size_t i = 0; i < k; ++i) sum_topk += entries[i].second;
    double topk_share =
        reason_total ? static_cast<double>(sum_topk) / reason_total : 0.0;

    for (size_t i = 0; i < k; ++i) {
      double fraction =
          reason_total ? static_cast<double>(entries[i].second) / reason_total
                       : 0.0;
      topk << reason_name(static_cast<reason>(idx)) << ','
           << hexify(entries[i].first) << ',' << (i + 1) << ','
           << entries[i].second << ',' << format_double(fraction) << ','
           << format_double(topk_share) << '\n';
    }

    uint64_t other = reason_total - sum_topk;
    double other_fraction =
        reason_total ? static_cast<double>(other) / reason_total : 0.0;
    topk << reason_name(static_cast<reason>(idx)) << ",OTHER,0," << other
         << ',' << format_double(other_fraction) << ','
         << format_double(topk_share) << '\n';
  }
}
