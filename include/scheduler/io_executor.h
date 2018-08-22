// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <atomic>
#include <functional>
#include <thread>
#include <vector>
#include "../file_handles/file_handle.h"
#include "../pointers/pointer.h"
#include "../queue.h"

namespace flash {
  // Wrapper struct for all work to be done by an IO thread
  struct IoTask {
    flash_ptr<void> fptr;   // fptr to R/W from
    StrideInfo      sinfo;  // access pattern
    void*           buf;    // buf for I/O
    bool is_write;          // if `true`, indicates a write operation, else read
    std::function<void(void)> callback;  // `fn` to call after I/O is done

    // constructor to return null IoTask
    IoTask() {
      fptr.foffset = 0;
      fptr.fop = 0;
      sinfo = {0, 0, 0};
      buf = 0;
      is_write = false;
    }

    // one-shot initialization of all member variables
    IoTask(flash_ptr<void> fptr, StrideInfo sinfo, void* buf, bool is_write,
           std::function<void(void)> fn)
        : fptr(fptr), sinfo(sinfo), buf(buf), is_write(is_write), callback(fn) {
    }

    // for DEBUG purposes
    operator std::string() const {
      return std::string(fptr) + std::string("-") + std::string(sinfo);
    }
  };

  class IoExecutor {
    // List of all IO thread objects
    std::vector<std::thread> io_threads;

    // Number of IO threads spawned
    FBLAS_UINT n_threads;

    // List of all IO tasks being executed by each thread
    std::vector<IoTask*> thread_tsks;

    // List of access specifiers to modify thread_tsks
    typedef std::unique_lock<std::mutex> mutex_locker;
    std::mutex*                          thread_muts;

    // Indicator variables as to whether each task executed by an IO thread is a
    // write/read
    std::atomic<bool>* is_writes;

    // Control Variable - Checks for overlap between IO tasks if set
    // DEFAULT: `true`
    bool overlap_check;

    // Task queue for IO threads
    ConcurrentQueue<IoTask*> tsk_queue;

    // Atomic boolean to signal shutdown of library
    std::atomic<bool> shutdown;

    // Thread function executed by each IO thread
    void io_thread_fn(FBLAS_UINT thread_idx);

    // Overlap Check function
    // returns `true` if `thread_tsks[t1_idx]` and `thread_tsks[t2_idx]` overlap
    bool overlap(FBLAS_UINT t1_idx, FBLAS_UINT t2_idx);

    // Advertise `tsk` as `thread_idx`'s new task
    void set_task(FBLAS_UINT thread_idx, IoTask* tsk);

    // Advertise NULL task for thread-`thread_idx`
    void set_null(FBLAS_UINT thread_idx);

    // helper function to execute task
    // also deletes `tsk`
    void execute_task(IoTask* tsk);

   public:
    // Constructor - Spawns `n_threads` number of IO threads
    IoExecutor(FBLAS_UINT n_threads);

    // Cleanup - Shutdown `this->n_threads` number of threads
    ~IoExecutor();

    // NOTE :: if `sinfo.n_strides==1`, pre-align `buf, fptr, sinfo` for better
    // performance
    // creates an IoTask object and adds to the task queue
    void add_read(flash_ptr<void> fptr, StrideInfo sinfo, void* buf,
                  std::function<void(void)> callback);

    // Same semantics as `add_read()`, but creates a `write` task instead
    void add_write(flash_ptr<void> fptr, StrideInfo sinfo, void* buf,
                   std::function<void(void)> callback);

    friend class Scheduler;
  };
}  // namespace flash
