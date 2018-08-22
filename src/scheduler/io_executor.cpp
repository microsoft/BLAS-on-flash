// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "scheduler/io_executor.h"
#include <malloc.h>
#include "file_handles/flash_file_handle.h"
#include "timer.h"

namespace {
  bool strip_overlap(FBLAS_UINT start1, FBLAS_UINT end1, FBLAS_UINT start2,
                     FBLAS_UINT end2) {
    start1 = ROUND_DOWN(start1, SECTOR_LEN);
    start2 = ROUND_DOWN(start2, SECTOR_LEN);
    end1 = ROUND_UP(end1, SECTOR_LEN);
    end2 = ROUND_UP(end2, SECTOR_LEN);
    return !((end2 <= start1) || (end1 <= start2));
  }
  std::string to_string(const flash::IoTask& tsk) {
    return std::to_string((FBLAS_UINT) tsk.fptr.fop) + ":" +
           std::to_string(tsk.fptr.foffset) + "+" + std::string(tsk.sinfo);
  }
  void print_conflict(const flash::IoTask& tsk1, const flash::IoTask& tsk2) {
    std::string first = to_string(tsk1);
    std::string second = to_string(tsk2);
    GLOG_WARN("CONFLICT:[", first, "] <--> [", second, "]");
  }

  bool same_stride_overlap(FBLAS_UINT o1, FBLAS_UINT l1, FBLAS_UINT n1,
                           FBLAS_UINT o2, FBLAS_UINT l2, FBLAS_UINT n2,
                           FBLAS_UINT s) {
    // if all params are aligned, then no overlap
    if (IS_ALIGNED(o1) && IS_ALIGNED(o2) && IS_ALIGNED(l1) && IS_ALIGNED(l2) &&
        IS_ALIGNED(s)) {
      return false;
    }

    GLOG_ASSERT(o1 <= o2, "bad offset ordering");
    // ---1|2--- overlap
    if (strip_overlap(o1, o1 + l1, o2, o2 + l2)) {
      return true;
    }

    // ---2|1--- overlap
    if (strip_overlap(o1 + s, o1 + s + l1, o2, o2 + l2)) {
      return true;
    }

    // # bytes between end of first strip and start of second strip
    FBLAS_UINT delta = (o2 - (o1 + l1));
    if (delta < SECTOR_LEN) {
      if (IS_ALIGNED(o1) && IS_ALIGNED(o2) && IS_ALIGNED(s)) {
        return false;
      } else {
        return true;
      }
    } else {
      // no overlap
      return false;
    }
  }
  bool is_overlap(const flash::IoTask& tsk1, const flash::IoTask& tsk2) {
    /*
    GLOG_DEBUG("in_ptrs=", std::to_string((FBLAS_UINT) tsk1.fptr.fop), ", ",
               std::to_string((FBLAS_UINT) tsk2.fptr.fop));
               */
    // if not even the same file, return false
    if (tsk1.fptr.fop != tsk2.fptr.fop) {
      return false;
    }

    // reads never conflict
    if (!tsk1.is_write && !tsk2.is_write) {
      return false;
    }

    FBLAS_UINT o1 = tsk1.fptr.foffset;
    FBLAS_UINT n1 = tsk1.sinfo.n_strides;
    FBLAS_UINT l1 = tsk1.sinfo.len_per_stride;
    FBLAS_UINT s1 = tsk1.sinfo.stride;
    FBLAS_UINT o2 = tsk2.fptr.foffset;
    FBLAS_UINT n2 = tsk2.sinfo.n_strides;
    FBLAS_UINT l2 = tsk2.sinfo.len_per_stride;
    FBLAS_UINT s2 = tsk2.sinfo.stride;

    // 3 unique cases
    if (n1 == 1 && n2 == 1) {
      if (strip_overlap(o1, o1 + l1, o2, o2 + l2)) {
        return true;
      } else {
        return false;
      }
    }
    // swap to keep first access as strided
    if (n2 != 1 && n1 == 1) {
      std::swap(n1, n2);
      std::swap(l1, l2);
      std::swap(o1, o2);
      std::swap(s1, s2);
    }
    if (n1 != 1 && n2 == 1) {
      FBLAS_UINT e2 = o2 + l2;
      // check if entirely disjoint
      bool overlap = strip_overlap(o1, o1 + n1 * s1, o2, e2);
      if (!overlap) {
        return false;
      }
      // if o2 comes first, then there has to be an overlap
      if (o2 <= o1) {
        print_conflict(tsk1, tsk2);
        return true;
      }

      FBLAS_UINT k_low = (o2 - o1) / s1;  // floor((o2-o1) / s1);
      // check overlap between `k_low` strip and o2:l2
      FBLAS_UINT k_start = o1 + k_low * s1;
      overlap = strip_overlap(k_start, k_start + l1, o2, e2);
      if (overlap) {
        print_conflict(tsk1, tsk2);
        return true;
      }
      // check overlap between `k_low + 1` strip and o2:l2
      // if no overlap, then the two accesses don't overlap
      k_start += s1;
      overlap = strip_overlap(k_start, k_start + l1, o2, e2);
      if (overlap) {
        print_conflict(tsk1, tsk2);
        return true;
      } else {
        return false;
      }
    } else {
      // both strided accesses
      // special case when both use same strides (matrix blocking)
      if (s1 == s2) {
        if (o2 < o1) {
          std::swap(n1, n2);
          std::swap(l1, l2);
          std::swap(o1, o2);
          std::swap(s1, s2);
        }

        return same_stride_overlap(o1, l1, n1, o2, l2, n2, s1);
      } else {
        // different strides
        // no idea why this will happen, but I'll support it regardless
        bool overlap = strip_overlap(o1, o1 + n1 * s1, o2, o2 + n2 * s2);
        if (!overlap) {
          return false;
        }
        print_conflict(tsk1, tsk2);
        // TODO :: implement this
        GLOG_FATAL("non-homogenous overlap operator not implemented");
        return true;
      }
    }
  }
}  // namespace

namespace flash {
  void IoExecutor::execute_task(IoTask* tsk_ptr) {
    Timer   timer;
    IoTask& tsk = *tsk_ptr;
    GLOG_DEBUG("I/O:START:", to_string(tsk));
    flash_ptr<void> fptr = tsk.fptr;
    GLOG_ASSERT(fptr.fop != nullptr, "bad fptr");
    StrideInfo                     sinfo = tsk.sinfo;
    void*                          buf = tsk.buf;
    auto                           callback_fn = tsk.callback;
    static std::atomic<FBLAS_UINT> write_count(1);
    if (tsk.is_write) {
      GLOG_DEBUG("write #", write_count.fetch_add(1),
                 ", sinfo=", std::string(sinfo));
    }
/*
GLOG_ASSERT(malloc_usable_size(buf) >= buf_size(sinfo),
            "no mem; expected=", buf_size(sinfo),
            ", got=", malloc_usable_size(buf));
            */
#ifdef DEBUG
    FlashFileHandle* ffh = dynamic_cast<FlashFileHandle*>(fptr.fop);
    GLOG_ASSERT(ffh != nullptr, "bad fop");
#endif
    if (sinfo.n_strides == 1) {
      GLOG_DEBUG("args:offset=", fptr.foffset, ", lps=", sinfo.len_per_stride,
                 ", buf=", buf);
      if (tsk.is_write) {
        fptr.fop->write(fptr.foffset, sinfo.len_per_stride, buf,
                        dummy_std_func);
      } else {
        fptr.fop->read(fptr.foffset, sinfo.len_per_stride, buf, dummy_std_func);
      }
    } else {
      if (tsk.is_write) {
        fptr.fop->swrite(fptr.foffset, sinfo, buf);
      } else {
        fptr.fop->sread(fptr.foffset, sinfo, buf, dummy_std_func);
      }
    }

    callback_fn();
    GLOG_DEBUG("I/O:END:", to_string(tsk), ", time taken = ", timer.elapsed(),
               "ms");
  }  // namespace flash

  bool IoExecutor::overlap(FBLAS_UINT t1_idx, FBLAS_UINT t2_idx) {
    Timer        timer;
    mutex_locker lk1(thread_muts[t1_idx], std::defer_lock);
    mutex_locker lk2(thread_muts[t2_idx], std::defer_lock);

    // try locking both using a deadlock-avoidance algorithm
    std::lock(lk1, lk2);

    // take care of the case when either of them is a null task
    if (thread_tsks[t1_idx] == nullptr || thread_tsks[t2_idx] == nullptr) {
      GLOG_ERROR("shouldn't check for overlap because is_writes=false");
      return false;
    }

    bool result_f = ::is_overlap(*thread_tsks[t1_idx], *thread_tsks[t2_idx]);
    bool result_b = ::is_overlap(*thread_tsks[t2_idx], *thread_tsks[t1_idx]);
    GLOG_ASSERT(result_f == result_b, "bidirectional comparator required");

    // give up both locks
    lk1.unlock();
    lk2.unlock();
    GLOG_DEBUG("Overlap Check: ", timer.elapsed(), "ms");

    return result_f || result_b;
  }

  void IoExecutor::set_task(FBLAS_UINT thread_idx, IoTask* tsk) {
    if (this->overlap_check) {
      mutex_locker lk(thread_muts[thread_idx]);
      thread_tsks[thread_idx] = tsk;
      this->is_writes[thread_idx] = tsk->is_write;
      lk.unlock();
    }
  }

  void IoExecutor::set_null(FBLAS_UINT thread_idx) {
    if (this->overlap_check) {
      mutex_locker lk(thread_muts[thread_idx]);
      thread_tsks[thread_idx] = nullptr;
      this->is_writes[thread_idx] = false;
      lk.unlock();
    }
  }

  void IoExecutor::io_thread_fn(FBLAS_UINT thread_idx) {
    // backlog of overlapping I/Os
    std::queue<IoTask*> backlog;

    // register thread
    FlashFileHandle::register_thread();

    while (true) {
      // give higher priority to `backlog` tasks
      // try to execute each task in `backlog` before giving up
      IoTask* tsk = nullptr;

      if (this->overlap_check) {
        FBLAS_UINT n_tries = backlog.size();
        while (n_tries > 0) {
          n_tries--;

          // get a `backlog` task
          tsk = backlog.front();
          backlog.pop();

          // announce my task
          set_task(thread_idx, tsk);

          // check if conflict has been resolved
          bool conflict = false;
          if (this->overlap_check) {
            bool thread_writes = this->is_writes[thread_idx].load();
            for (FBLAS_UINT i = 0; i < this->n_threads; i++) {
              if (i != thread_idx) {
                bool other_writes = this->is_writes[i].load();
                // data race only if both write
                // (R | W) and (W | R) is a WAR or RAW hazard that should be
                // taken care of by adding dependencies in the task DAG
                // (R | R) presents no hazard
                if (thread_writes && other_writes) {
                  bool overlap = this->overlap(thread_idx, i);
                  if (overlap) {
                    // conflict
                    GLOG_WARN("conflict");
                    set_null(thread_idx);
                    backlog.push(tsk);
                    conflict = true;
                    break;
                  }
                }
              }
            }
          }

          // execute if no conflict
          if (!conflict) {
            this->execute_task(tsk);
            set_null(thread_idx);
            delete tsk;
          }
        }
      }

      // now service `tsk_queue`
      tsk = this->tsk_queue.pop();
      // can be null if taken from `tsk_queue`
      if (tsk == nullptr) {
        // shutdown mechanism
        if (this->shutdown.load() && backlog.empty()) {
          break;
        } else {
          // wait for a push
          this->tsk_queue.wait_for_push_notify();
        }
      } else {
        // announce my task
        set_task(thread_idx, tsk);

        // check if conflict exists between `tsk` and other threads
        bool conflict = false;
        if (this->overlap_check) {
          bool thread_writes = this->is_writes[thread_idx].load();
          for (FBLAS_UINT i = 0; i < this->n_threads; i++) {
            if (i != thread_idx) {
              bool other_writes = this->is_writes[i].load();
              // data race only if both write
              // (R | W) and (W | R) is a WAR or RAW hazard that should be
              // taken care of by adding dependencies in the task DAG
              // (R | R) presents no hazard
              if (thread_writes && other_writes) {
                bool overlap = this->overlap(thread_idx, i);
                if (overlap) {
                  // conflict
                  GLOG_WARN("conflict");
                  set_null(thread_idx);
                  backlog.push(tsk);
                  conflict = true;
                  break;
                }
              }
            }
          }
        }

        // execute if no conflict
        if (!conflict) {
          this->execute_task(tsk);
          set_null(thread_idx);
          delete tsk;
        }
      }
    }

    FlashFileHandle::deregister_thread();
    GLOG_DEBUG("IO thread #", thread_idx, " down");
    return;
  }

  IoExecutor::IoExecutor(FBLAS_UINT n_threads)
      : n_threads(n_threads), tsk_queue(nullptr) {
    GLOG_DEBUG("init IO startup");
    this->shutdown.store(false);
    this->thread_tsks.resize(n_threads);
    this->thread_muts = new std::mutex[n_threads];
    this->is_writes = new std::atomic<bool>[n_threads];
    for (FBLAS_UINT i = 0; i < n_threads; i++) {
      set_null(i);
      this->io_threads.push_back(
          std::thread(&IoExecutor::io_thread_fn, this, i));
      this->is_writes[i] = ATOMIC_FLAG_INIT;
    }

    this->overlap_check = true;

    GLOG_DEBUG("IO startup complete");
  }

  IoExecutor::~IoExecutor() {
    GLOG_DEBUG("init IO shutdown");

    this->shutdown.store(true);
    this->tsk_queue.push_notify_all();
    for (auto& thr : this->io_threads) {
      thr.join();
    }

    delete[] this->thread_muts;
    delete[] this->is_writes;
    GLOG_DEBUG("IO shutdown complete");
  }

  void IoExecutor::add_read(flash_ptr<void> fptr, StrideInfo sinfo, void* buf,
                            std::function<void(void)> callback) {
    GLOG_DEBUG("adding read");
    IoTask* tsk = new IoTask(fptr, sinfo, buf, false, callback);
    this->tsk_queue.push(tsk);
    this->tsk_queue.push_notify_one();
  }

  void IoExecutor::add_write(flash_ptr<void> fptr, StrideInfo sinfo, void* buf,
                             std::function<void(void)> callback) {
    GLOG_DEBUG("adding write");
    IoTask* tsk = new IoTask(fptr, sinfo, buf, true, callback);
    this->tsk_queue.push(tsk);
    this->tsk_queue.push_notify_one();
  }
}  // namespace flash
