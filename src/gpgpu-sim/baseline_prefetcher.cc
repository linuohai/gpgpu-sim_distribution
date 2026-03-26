#include "baseline_prefetcher.h"

#include <algorithm>

#include "gpu-cache.h"
#include "mem_fetch.h"

namespace {

constexpr unsigned kBaselineSectorSize = 32;
constexpr unsigned kBaselineLineSize = 128;

bool is_cache_miss_status(int cache_status) {
  return cache_status != HIT && cache_status != HIT_RESERVED &&
         cache_status != MSHR_HIT && cache_status != RESERVATION_FAIL;
}

}  // namespace

baseline_prefetcher_t::baseline_prefetcher_t(unsigned sm_id) : m_sm_id(sm_id) {}

baseline_prefetcher_t::~baseline_prefetcher_t() = default;

void baseline_prefetcher_t::on_kernel_launch() {
  m_prefetch_queue.clear();
  m_pending_fill_by_uid.clear();
  m_live_prefetch_sectors.clear();
  m_queued_prefetch_sectors.clear();
  m_stats = baseline_stats_t();
}

void baseline_prefetcher_t::on_warp_exit(unsigned warp_id) { (void)warp_id; }

void baseline_prefetcher_t::on_instruction_issue(
    unsigned warp_id, const warp_inst_t &inst, const std::string &sass_opcode,
    unsigned long long cycle) {
  (void)warp_id;
  (void)inst;
  (void)sass_opcode;
  (void)cycle;
}

void baseline_prefetcher_t::queue_prefetch(new_addr_type addr, unsigned warp_id,
                                           unsigned long long ready_cycle) {
  const new_addr_type sector_addr = normalize_sector_addr(addr);
  if (m_live_prefetch_sectors.count(sector_addr) ||
      m_queued_prefetch_sectors.count(sector_addr)) {
    return;
  }
  for (const auto &entry : m_pending_fill_by_uid) {
    if (entry.second == sector_addr) return;
  }

  baseline_prefetch_request_t req;
  req.addr = addr;
  req.warp_id = warp_id;
  req.ready_cycle = ready_cycle;
  m_prefetch_queue.push_back(req);
  m_queued_prefetch_sectors.insert(sector_addr);
}

void baseline_prefetcher_t::note_demand_access(new_addr_type addr,
                                               int cache_status) {
  if (is_cache_miss_status(cache_status)) {
    ++m_stats.total_demand_miss;
  }

  const new_addr_type sector_addr = normalize_sector_addr(addr);
  auto it = m_live_prefetch_sectors.find(sector_addr);
  if (it != m_live_prefetch_sectors.end()) {
    ++m_stats.prefetch_useful;
    m_live_prefetch_sectors.erase(it);
  }
}

int baseline_prefetcher_t::find_ready_prefetch(unsigned long long cycle) const {
  for (size_t i = 0; i < m_prefetch_queue.size(); ++i) {
    if (m_prefetch_queue[i].ready_cycle <= cycle) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

baseline_prefetcher_t::inject_result_t baseline_prefetcher_t::inject_prefetch(
    unsigned long long cycle, mem_fetch_allocator *mf_alloc, unsigned sid,
    unsigned tpc, const memory_config *mem_cfg, gpgpu_context *gpgpu_ctx) {
  inject_result_t result;
  if (m_prefetch_queue.empty()) return result;

  const int ready_idx = find_ready_prefetch(cycle);
  if (ready_idx < 0) {
    result.stalled = true;
    return result;
  }

  const baseline_prefetch_request_t req = m_prefetch_queue[ready_idx];
  m_prefetch_queue.erase(m_prefetch_queue.begin() + ready_idx);
  m_queued_prefetch_sectors.erase(normalize_sector_addr(req.addr));

  mem_access_sector_mask_t sector_mask;
  sector_mask.set((req.addr % kBaselineLineSize) / kBaselineSectorSize);
  mem_access_byte_mask_t byte_mask;
  byte_mask.set();
  active_mask_t active_mask;
  active_mask.set();

  mem_fetch *pf_mf =
      mf_alloc->alloc(req.addr, GLOBAL_ACC_R, active_mask, byte_mask,
                      sector_mask, kBaselineSectorSize, false, cycle,
                      req.warp_id, sid, tpc, nullptr, 0);
  if (is_snake()) pf_mf->set_snake_prefetch(true);
  result.requests.push_back(pf_mf);
  (void)mem_cfg;
  (void)gpgpu_ctx;
  return result;
}

void baseline_prefetcher_t::mark_live_sector(new_addr_type sector_addr) {
  m_live_prefetch_sectors.insert(sector_addr);
}

void baseline_prefetcher_t::on_l1_access_result(mem_fetch *mf, int cache_status,
                                                unsigned long long cycle) {
  (void)cycle;

  ++m_stats.prefetch_issued;
  const new_addr_type sector_addr = normalize_sector_addr(mf->get_addr());

  if (cache_status == HIT) {
    ++m_stats.prefetch_hit;
    mark_live_sector(sector_addr);
    delete mf;
    return;
  }

  if (cache_status == MSHR_HIT || cache_status == HIT_RESERVED) {
    delete mf;
    return;
  }

  if (cache_status == RESERVATION_FAIL) {
    ++m_stats.prefetch_rfail;
    ++m_stats.prefetch_useless;
    delete mf;
    return;
  }

  m_pending_fill_by_uid[mf->get_request_uid()] = sector_addr;
}

void baseline_prefetcher_t::on_fill(mem_fetch *mf,
                                    unsigned long long fill_cycle) {
  (void)fill_cycle;
  auto it = m_pending_fill_by_uid.find(mf->get_request_uid());
  if (it == m_pending_fill_by_uid.end()) return;
  mark_live_sector(it->second);
  m_pending_fill_by_uid.erase(it);
}

baseline_stats_t baseline_prefetcher_t::finalized_stats() const {
  baseline_stats_t stats = m_stats;
  stats.prefetch_useless += m_live_prefetch_sectors.size();
  stats.prefetch_useless += m_pending_fill_by_uid.size();
  return stats;
}

void baseline_prefetcher_t::print_common_stats(FILE *fp,
                                               const char *name) const {
  const baseline_stats_t stats = finalized_stats();
  const double coverage =
      stats.total_demand_miss == 0
          ? 0.0
          : static_cast<double>(stats.prefetch_useful) /
                static_cast<double>(stats.total_demand_miss);
  const double accuracy =
      stats.prefetch_issued == 0
          ? 0.0
          : static_cast<double>(stats.prefetch_useful) /
                static_cast<double>(stats.prefetch_issued);

  fprintf(fp,
          "%s_SM%u: prefetch_issued=%llu prefetch_hit=%llu "
          "prefetch_useful=%llu prefetch_useless=%llu prefetch_rfail=%llu "
          "coverage=%.6f accuracy=%.6f demand_miss=%llu\n",
          name, m_sm_id, stats.prefetch_issued, stats.prefetch_hit,
          stats.prefetch_useful, stats.prefetch_useless, stats.prefetch_rfail,
          coverage, accuracy, stats.total_demand_miss);
}
