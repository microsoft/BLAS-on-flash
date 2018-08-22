// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <algorithm>
#include "../tasks/task.h"

namespace flash {
  class Cache;
  struct Key;

  struct TaskInfo {
    // Task pointer
    BaseTask *tsk;

    // List of all memory accesses needed
    // Using unordered_set keeps this list unique
    std::unordered_set<Key> all_keys;

    // Extra memory required to execute this task
    // Equal to sum of buffer sizes for buffers not in
    // `Prioritizer::in_mem_keys`
    FBLAS_UINT mem_reqd;
  };

  class Prioritizer {
    // keep track of all buffers that are in memory
    // each buffer is uniquely associated with a cache key, so tracking the keys
    // is sufficient
    std::unordered_set<Key> in_mem_keys;

    // obtain a list of tasks to create a partial ordering
    std::deque<TaskInfo> tsks;
    Cache &              cache;

    // if `use_prio` is set to `false`, priorities are not enforced - FCFS
    // equivalent
    bool use_prio;

    // compute `mem_reqd` using current prioritizer state
    void fill_memreqd(TaskInfo &tsk_info) {
      for (auto &k : tsk_info.all_keys) {
        if (this->in_mem_keys.find(k) == this->in_mem_keys.end()) {
          tsk_info.mem_reqd += buf_size(k.sinfo);
        }
      }
    }

   public:
    Prioritizer(Cache &cache) : cache(cache) {
      this->use_prio = true;
    }

    ~Prioritizer() {
      GLOG_ASSERT(this->tsks.empty(), "tasks left");
    }

    // insert a batch of tasks into the prioritizer queue
    // NOTE :: This doesn't force an UPDATE; use `update()` to force ordering
    // update
    void insert(std::vector<BaseTask *> new_tsks) {
      for (auto &tsk : new_tsks) {
        // fill in `tsk` and `all_keys`
        TaskInfo tsk_info;
        tsk_info.tsk = tsk;
        for (auto &fptr_sinfo : tsk->read_list) {
          Key k(fptr_sinfo.first, fptr_sinfo.second);
          tsk_info.all_keys.insert(k);
        }
        for (auto &fptr_sinfo : tsk->write_list) {
          Key k(fptr_sinfo.first, fptr_sinfo.second);
          tsk_info.all_keys.insert(k);
        }

        tsk_info.mem_reqd = 0;

        // compute `memreqd` only if using priority
        if (this->use_prio) {
          // compute `mem_reqd` based on known state
          for (auto &k : tsk_info.all_keys) {
            if (this->in_mem_keys.find(k) == this->in_mem_keys.end()) {
              tsk_info.mem_reqd += buf_size(k.sinfo);
            }
          }
        }

        this->tsks.push_back(tsk_info);
      }
    }

    bool empty() {
      return this->tsks.empty();
    }

    FBLAS_UINT size() {
      return this->tsks.size();
    }

    // returns the highest priority task struct
    TaskInfo get_prio() {
      GLOG_ASSERT(!this->empty(), "bad check");
      TaskInfo tsk_info = this->tsks.front();
      this->tsks.pop_front();
      return tsk_info;
    }

    // a high priority task struct is returned to priority queue
    // push_front as priority is high
    void return_prio(TaskInfo tsk_info) {
      this->tsks.push_front(tsk_info);
    }

    // update in-memory keys info
    // recompute new IO and memory requirements
    // sort based on newly obtained `mem_reqd` values
    // when this call finishes, priority ordering is fresh and up-to-date
    void update() {
      // refresh in-mem keys
      this->in_mem_keys.clear();
      for (auto &tsk_sinfo : this->tsks) {
        for (auto &k : tsk_sinfo.all_keys) {
          this->in_mem_keys.insert(k);
        }
      }

      // query cache
      this->cache.keep_if_in_cache(this->in_mem_keys);

      // recompute `mem_reqd`
      for (auto &tsk_sinfo : this->tsks) {
        this->fill_memreqd(tsk_sinfo);
      }

      // sort `tsks` in decreasing order of priority (increasing order of
      // `mem_reqd`)
      std::sort(this->tsks.begin(), this->tsks.end(),
                [](const TaskInfo &left, const TaskInfo &right) {
                  return left.mem_reqd < right.mem_reqd;
                });
    }

    friend class Scheduler;
  };
}  // namespace flash
