# Exclude power_interface.cc if no power model
list(APPEND gpgpusim_SRC addrdec.cc
    dram.cc
    dram_sched.cc
    gpu-cache.cc
    gpu-misc.cc
    gpu-sim.cc
    hashing.cc
    histogram.cc
    icnt_wrapper.cc
    icnt_tracer.cc
    hbm_partition_tracer.cc
    issue_tracer.cc
    l1_tracer.cc
    l2_bw_tracer.cc
    l2_tracer.cc
    l2cache.cc
    grasp_prefetcher.cc
    grasp_chain_detector.cc
    grasp_tables.cc
    grasp_tracer.cc
    ima_prefetcher.cc
    stall_reason_pc_stats.cc
    local_interconnect.cc
    mem_fetch.cc
    mem_latency_stat.cc
    power_interface.cc
    power_stat.cc
    scoreboard.cc
    shader.cc
    stack.cc
    stat-tool.cc
    traffic_breakdown.cc
    visualizer.cc)
if(NOT GPGPUSIM_USE_POWER_MODEL)
    list(REMOVE_ITEM ${gpgpusim_SRC} power_interface.cc)
endif()

# Create libgpgpusim.a
add_library(gpgpusim STATIC ${gpgpusim_SRC})
target_include_directories(gpgpusim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(gpgpusim PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

if(GPGPUSIM_USE_POWER_MODEL)
target_compile_definitions(gpgpusim PRIVATE GPGPUSIM_POWER_MODEL)
target_include_directories(gpgpusim PRIVATE ${GPGPUSIM_POWER_MODEL})
endif()
