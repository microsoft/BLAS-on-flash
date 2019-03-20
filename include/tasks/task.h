// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <atomic>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "bof_logger.h"
#include "pointers/pointer.h"

namespace flash {
  extern std::atomic<FBLAS_UINT> global_task_counter;
  // Different states of a Task
  enum TaskStatus {
    Wait,          // task not yet started
    AllocReady,    // task is ready to be alloc'ed, but not alloc'ed
    Alloc,         // task has been alloc'ed, but not compute-ready
    ComputeReady,  // task has been alloc'ed, AND compute-ready
    Compute,       // task is in compute
    Complete       // task has finished compute
  };

  // Task interface
  class BaseTask {
   protected:
    // I/O list
    // <ptr, strides, R|W>
    std::vector<std::pair<flash_ptr<void>, StrideInfo>> read_list;
    std::vector<std::pair<flash_ptr<void>, StrideInfo>> write_list;
    std::vector<FBLAS_UINT> parents;

    std::unordered_map<flash_ptr<void>, void*, FlashPtrHasher, FlashPtrEq>
        in_mem_ptrs;

    // Status of Task
    std::atomic<TaskStatus> st;

    // Continuation
    BaseTask* next;

    // Unique Task ID
    FBLAS_UINT task_id;

   public:
    BaseTask() {
      this->st.store(Wait);
      this->next = nullptr;
      this->task_id = global_task_counter.fetch_add(1);
    }

    virtual ~BaseTask() {
    }
    // NOTE :: Make sure all reads are done
    // before invoking ::execute()
    virtual void execute() = 0;
    void add_read(flash_ptr<void> fptr, StrideInfo& sinfo) {
      GLOG_DEBUG("adding read=", std::string(sinfo));
      GLOG_ASSERT_LT(sinfo.len_per_stride, ((FBLAS_UINT) 1 << 35));
      this->read_list.push_back(std::make_pair(fptr, sinfo));
    }

    void add_write(flash_ptr<void> fptr, StrideInfo& sinfo) {
      GLOG_DEBUG("adding write=", std::string(sinfo));
      GLOG_ASSERT_LT(sinfo.len_per_stride, ((FBLAS_UINT) 1 << 35));
      this->write_list.push_back(std::make_pair(fptr, sinfo));
    }

    virtual FBLAS_UINT size() = 0;

    void add_parent(FBLAS_UINT id) {
      this->parents.push_back(id);
    }
    std::vector<FBLAS_UINT>& get_parents() {
      return this->parents;
    }

    void add_next(BaseTask* nxt) {
      this->next = nxt;
    }

    BaseTask* get_next() {
      return this->next;
    }

    virtual TaskStatus get_status() {
      return this->st.load();
    }

    virtual void set_status(TaskStatus sts) {
      this->st.store(sts);
    }

    FBLAS_UINT get_id() {
      return this->task_id;
    }

    friend class Scheduler;
    friend class Cache;
    friend class Prioritizer;
  };

}  // namespace flash
