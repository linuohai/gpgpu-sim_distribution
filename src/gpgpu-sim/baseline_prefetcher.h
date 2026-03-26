#pragma once

#include <cstdio>
#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../abstract_hardware_model.h"

class gpgpu_context;
class mem_fetch;
class mem_fetch_allocator;
class memory_config;
class shd_warp_t;
class warp_inst_t;

struct baseline_stats_t {
  unsigned long long prefetch_issued = 0;
  unsigned long long prefetch_hit = 0;
  unsigned long long prefetch_useful = 0;
  unsigned long long prefetch_useless = 0;
  unsigned long long prefetch_rfail = 0;
  unsigned long long total_demand_miss = 0;
};

struct baseline_prefetch_request_t {
  new_addr_type addr = 0;
  unsigned warp_id = static_cast<unsigned>(-1);
  unsigned long long ready_cycle = 0;
};

class baseline_prefetcher_t {
 public:
  explicit baseline_prefetcher_t(unsigned sm_id);
  virtual ~baseline_prefetcher_t();

  virtual void on_kernel_launch();
  virtual void on_warp_exit(unsigned warp_id);
  virtual void on_instruction_issue(unsigned warp_id, const warp_inst_t &inst,
                                    const std::string &sass_opcode,
                                    unsigned long long cycle);
  virtual void on_demand_load(unsigned warp_id, new_addr_type pc,
                              new_addr_type addr, unsigned long long cycle,
                              int cache_status, shd_warp_t *warp) = 0;

  struct inject_result_t {
    std::vector<mem_fetch *> requests;
    bool stalled = false;
  };

  inject_result_t inject_prefetch(
      unsigned long long cycle, mem_fetch_allocator *mf_alloc, unsigned sid,
      unsigned tpc, const memory_config *mem_cfg, gpgpu_context *gpgpu_ctx);

  void on_l1_access_result(mem_fetch *mf, int cache_status,
                           unsigned long long cycle);
  virtual void on_fill(mem_fetch *mf, unsigned long long fill_cycle);

  virtual void print_stats(FILE *fp) const = 0;
  virtual bool is_snake() const { return false; }

 protected:
  void queue_prefetch(new_addr_type addr, unsigned warp_id,
                      unsigned long long ready_cycle);
  void note_demand_access(new_addr_type addr, int cache_status);
  void print_common_stats(FILE *fp, const char *name) const;
  baseline_stats_t finalized_stats() const;

  static new_addr_type normalize_sector_addr(new_addr_type addr) {
    return addr & ~static_cast<new_addr_type>(31);
  }

  unsigned m_sm_id;
  std::deque<baseline_prefetch_request_t> m_prefetch_queue;
  baseline_stats_t m_stats;

 private:
  int find_ready_prefetch(unsigned long long cycle) const;
  void mark_live_sector(new_addr_type sector_addr);

  std::unordered_map<unsigned, new_addr_type> m_pending_fill_by_uid;
  std::unordered_set<new_addr_type> m_live_prefetch_sectors;
  std::unordered_set<new_addr_type> m_queued_prefetch_sectors;
};
