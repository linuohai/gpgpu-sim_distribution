#include "issue_tracer.h"

#include <fstream>
#include <iomanip>
#include <sstream>

bool issue_tracer::s_enabled = false;
std::string issue_tracer::s_path;
std::vector<std::string> issue_tracer::s_buffers;
size_t issue_tracer::s_flush_threshold = 1u << 15;

namespace {

char nibble_to_hex(unsigned value) {
  static const char *digits = "0123456789abcdef";
  return digits[value & 0xF];
}

}  // namespace

void issue_tracer::init(bool enable, const char *path, unsigned n_sms) {
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

  file << "cycle,sm,scheduler,warp,event,pc,mask,OP,Space,One_Reason";
  file << ",MEM_WAIT,REG_WAIT,IBUFFER_EMPTY,BARRIER,CONTROL_HAZARD";
  file << ",PIPE_BUSY,DUAL_ISSUE_RESTRICT,Issue_Sector_Addresses";
  file << ",Issue_Sector_Lanes\n";
}

void issue_tracer::emit_issue(unsigned sid, unsigned wid, unsigned sch_id,
                              unsigned long long cycle,
                              const active_mask_t &mask,
                              const std::string &opcode,
                              const std::string &space, address_type pc,
                              const std::string &sector_addresses,
                              const std::string &sector_lane_ids) {
  if (!s_enabled) return;
  if (sid >= s_buffers.size()) return;

  std::ostringstream oss;
  const std::string opcode_field = opcode.empty() ? "NA" : opcode;
  const std::string space_field = space.empty() ? "NA" : space;
  const std::string sector_addr_field =
      sector_addresses.empty() ? "NA" : sector_addresses;
  const std::string sector_lane_field =
      sector_lane_ids.empty() ? "NA" : sector_lane_ids;
  oss << cycle << ',' << sid << ',' << sch_id << ',' << wid << ",ISSUE,"
      << hexify(pc) << ',' << mask_to_hex(mask) << ',' << escape(opcode_field)
      << ',' << escape(space_field)
      << ",NA,0,0,0,0,0,0,0," << escape(sector_addr_field) << ','
      << escape(sector_lane_field) << '\n';
  append(sid, oss.str());
}

void issue_tracer::emit_stall(unsigned sid, int sample_warp,
                              unsigned long long cycle,
                              const active_mask_t *mask,
                              const std::string &opcode,
                              const std::string &space, address_type pc,
                              int scheduler_id,
                              const issue_stall_counts &counts,
                              const std::string &one_reason) {
  if (!s_enabled) return;
  if (sid >= s_buffers.size()) return;

  std::ostringstream oss;
  const std::string opcode_field = opcode.empty() ? "NA" : opcode;
  const std::string reason_field = one_reason.empty() ? "NA" : one_reason;
  const std::string space_field = space.empty() ? "NA" : space;
  oss << cycle << ',' << sid << ',' << scheduler_id << ',' << sample_warp
      << ",STALL,";
  if (pc == static_cast<address_type>(-1))
    oss << "NA";
  else
    oss << hexify(pc);
  oss << ',';
  if (mask)
    oss << mask_to_hex(*mask);
  else
    oss << "0x0";
  oss << ',' << escape(opcode_field) << ',' << escape(space_field) << ','
      << escape(reason_field) << ','
      << counts.mem_wait << ',' << counts.reg_wait << ','
      << counts.ibuffer_empty << ',' << counts.barrier << ','
      << counts.control_hazard << ',' << counts.pipe_busy << ','
      << counts.dual_issue_restrict << ",NA,NA\n";
  append(sid, oss.str());
}

void issue_tracer::flush_all() {
  if (!s_enabled) return;
  for (unsigned sid = 0; sid < s_buffers.size(); ++sid) {
    flush_sid(sid);
  }
}

void issue_tracer::flush_sid(unsigned sid) {
  if (!s_enabled) return;
  if (sid >= s_buffers.size()) return;
  std::string &buffer = s_buffers[sid];
  if (buffer.empty()) return;

  std::ofstream file(s_path.c_str(), std::ios::out | std::ios::app);
  if (!file.good()) return;
  file << buffer;
  buffer.clear();
}

void issue_tracer::append(unsigned sid, const std::string &line) {
  if (sid >= s_buffers.size()) return;
  std::string &buffer = s_buffers[sid];
  buffer.append(line);
  if (buffer.size() >= s_flush_threshold) {
    flush_sid(sid);
  }
}

std::string issue_tracer::mask_to_hex(const active_mask_t &mask) {
  std::string digits;
  digits.reserve(mask.size() / 4 + 1);
  unsigned nibble_value = 0;
  unsigned nibble_bits = 0;
  for (int bit = static_cast<int>(mask.size()) - 1; bit >= 0; --bit) {
    nibble_value = (nibble_value << 1) | (mask.test(bit) ? 1 : 0);
    ++nibble_bits;
    if (nibble_bits == 4) {
      digits.push_back(nibble_to_hex(nibble_value));
      nibble_value = 0;
      nibble_bits = 0;
    }
  }
  if (nibble_bits) {
    nibble_value <<= (4 - nibble_bits);
    digits.push_back(nibble_to_hex(nibble_value));
  }
  size_t pos = digits.find_first_not_of('0');
  std::string trimmed =
      pos == std::string::npos ? std::string("0") : digits.substr(pos);
  return "0x" + trimmed;
}

std::string issue_tracer::hexify(address_type value) {
  std::ostringstream oss;
  oss << "0x" << std::hex << value << std::dec;
  return oss.str();
}

std::string issue_tracer::escape(const std::string &field) {
  if (field.empty()) return "NA";
  bool needs_quotes = false;
  for (char c : field) {
    if (c == ',' || c == '"' || c == '\n' || c == '\r') {
      needs_quotes = true;
      break;
    }
  }
  if (!needs_quotes) return field;
  std::string escaped;
  escaped.reserve(field.size() + 2);
  escaped.push_back('"');
  for (char c : field) {
    if (c == '"') escaped.push_back('"');
    escaped.push_back(c);
  }
  escaped.push_back('"');
  return escaped;
}
