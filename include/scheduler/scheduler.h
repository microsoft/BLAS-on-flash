// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "../bof_types.h"
#include "../pointers/pointer.h"
#include "../queue.h"
#include "../tasks/task.h"
#include "../vector.h"
#include "cache.h"
#include "io_executor.h"
#include "prioritizer.h"

namespace flash {
  class CompletionRecord {
    // access to `complete` is lock-less
    // only Scheduler::sched_thread_fn uses this structure
    std::vector<bool> complete;

    void resize() {
      FBLAS_UINT req_size = this->complete.size() * 2;
      this->complete.resize(req_size, false);
    }

   public:
    CompletionRecord(FBLAS_UINT start_size = 1024) {
      this->complete = std::vector<bool>(start_size, false);
    }

    bool is_complete(FBLAS_UINT tsk_id) {
      // if completion tracker is smaller than tsk_id:
      //  * expand to 2x initial size
      //  * return false
      if (complete.size() <= tsk_id) {
        this->resize();
        return false;
      } else {
        return this->complete[tsk_id];
      }
    }

    // removes completed task ids from `tsk_ids`
    // retains incomplete task ids in the same vector
    void remove_complete(std::vector<FBLAS_UINT>& tsk_ids) {
      auto it = tsk_ids.begin();
      while (it != tsk_ids.end()) {
        // if `tsk_id` is complete, erase
        if (is_complete(*it)) {
          it = tsk_ids.erase(it);
        } else {
          it++;
        }
      }
    }

    void mark_complete(FBLAS_UINT tsk_id) {
      if (complete.size() <= tsk_id) {
        this->resize();
      }
      GLOG_DEBUG("COMPLETE:tsk_id=", tsk_id);
      this->complete[tsk_id] = true;
    }
  };

  struct SchedulerOptions {
    // aggressively schedules tasks whose buffer requirements are partly
    // satisfied WARNING:: May incur overhead, use only if required
    bool enable_prioritizer = true;

    // checks for overlap between every write operation
    // very low overhead, but reduces scalability with number of I/O threads
    bool enable_overlap_check = true;

    // if true, each buffer is evicted when released
    // set for high-performance\ data-streaming from disk
    // disable for caching and re-using (may incur caching overhead)
    bool single_use_discard = false;
  };

  class Scheduler {
    // scheduler hyper-parameters
    const FBLAS_UINT        max_mem;
    std::atomic<FBLAS_UINT> n_compute_thr;

    Cache      cache;
    IoExecutor io_exec;

    // wait, read, compute, complete task containers
    typedef std::pair<bool, BaseTask*> bool_tsk_t;

    // parents not complete
    ConcurrentVector<BaseTask*> wait_tsks;
    // parents complete
    Prioritizer prio;
    // ready AND alloc'ed AND I/O NOT complete
    ConcurrentVector<BaseTask*> alloced_tsks;
    // I/O complete AND compute NOT complete
    ConcurrentQueue<BaseTask*> compute_queue;
    // compute complete
    ConcurrentQueue<BaseTask*> complete_queue;

    // completion recorder
    CompletionRecord c_rec;

    // thread functions
    void sched_thread_fn();
    void compute_thread_fn();

    // thread shutdown signals
    std::atomic<bool> shutdown;

    // threads
    std::thread              sched_thread;
    std::vector<std::thread> compute_threads;

    // helper functions
    bool alloc_ready(BaseTask* tsk);

   public:
    Scheduler(FBLAS_UINT n_io_threads, FBLAS_UINT n_compute_thr,
              FBLAS_UINT max_mem);
    ~Scheduler();

    // adds a task to the scheduler
    void add_task(BaseTask* tsk);

    // flushes the cache
    // NOTE:: use only if you need result persistence before program exit
    void flush_cache();

    void set_options(SchedulerOptions& sched_opts);

    void set_num_compute_threads(FBLAS_UINT new_num);
    const FBLAS_UINT get_num_compute_threads() const {
      return this->n_compute_thr;
    }
  };
}  // namespace flash
