// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "scheduler/scheduler.h"
#include <cassert>
#include "timer.h"

namespace flash {
  Scheduler::Scheduler(FBLAS_UINT n_io_threads, FBLAS_UINT n_compute_thr,
                       FBLAS_UINT max_mem)
      : n_compute_thr(0), max_mem(max_mem), io_exec(n_io_threads),
        cache(io_exec, max_mem), prio(cache) {
    this->shutdown.store(false);
    this->set_num_compute_threads(n_compute_thr);
    this->sched_thread = std::thread(&Scheduler::sched_thread_fn, this);
  }

  Scheduler::~Scheduler() {
    GLOG_DEBUG("Destroying scheduler");
    this->shutdown.store(true);
    this->sched_thread.join();

    for (auto& thr : compute_threads) {
      this->complete_queue.push_notify_all();
      thr.join();
    }

    GLOG_ASSERT(this->prio.empty(), "non-empty ready tasks list");

    GLOG_DEBUG("Flushing cache");
    this->cache.flush();

    GLOG_DEBUG("All Scheduler threads down");
    GLOG_ASSERT(this->wait_tsks.empty(), "non-empty");
    GLOG_ASSERT(this->prio.empty(), "non-empty");
    GLOG_ASSERT(this->alloced_tsks.empty(), "non-empty");
    GLOG_ASSERT(this->compute_queue.empty(), "non-empty");
    GLOG_ASSERT(this->complete_queue.empty(), "non-empty");
  }

  void Scheduler::flush_cache() {
    this->cache.flush();
  }

  // returns `true` if all buffers required by `tsk` are in `cache`
  bool Scheduler::alloc_ready(BaseTask* tsk) {
    bool ready = true;
    for (auto& fptr_sinfo : tsk->read_list) {
      if (tsk->in_mem_ptrs.find(fptr_sinfo.first) == tsk->in_mem_ptrs.end()) {
        void* ptr = cache.get_buf(fptr_sinfo.first, fptr_sinfo.second, false);
        if (ptr == nullptr) {
          ready = false;
        } else {
          tsk->in_mem_ptrs[fptr_sinfo.first] = ptr;
        }
      }
    }

    for (auto& fptr_sinfo : tsk->write_list) {
      if (tsk->in_mem_ptrs.find(fptr_sinfo.first) == tsk->in_mem_ptrs.end()) {
        void* ptr = cache.get_buf(fptr_sinfo.first, fptr_sinfo.second, true);
        if (ptr == nullptr) {
          ready = false;
        } else {
          tsk->in_mem_ptrs[fptr_sinfo.first] = ptr;
        }
      }
    }

    return ready;
  }

  void Scheduler::sched_thread_fn() {
    GLOG_DEBUG("Scheduler Thread Up");

    // max # of tasks in memory completely
    // keep this at least (N_COMPUTE_THR * 3) for optimal pipelining
    const FBLAS_UINT max_in_mem_tsks = N_COMPUTE_THR * 4;

    // returns `true` if all parents are not complete
    const std::function<bool(BaseTask*)> wait_keep_fn = [this](BaseTask* tsk) {
      this->c_rec.remove_complete(tsk->get_parents());
      return !(tsk->get_parents().empty());
    };

    const std::function<bool(BaseTask*)> alloc_keep_fn = [this](BaseTask* tsk) {
      return !alloc_ready(tsk);
    };

    Timer            timer;
    FBLAS_UINT       sleep_ms = 0;
    FPTYPE           max_sleep_ms = 100;
    FPTYPE           min_sleep_ms = 50;
    FBLAS_UINT       tsks_in_mem = 0;
    FPTYPE           total_sched_time = 0.0f;
    const FBLAS_UINT update_every = 1;
    FBLAS_UINT       update_in = update_every;
    while (true) {
      timer.reset();
      /*
      GLOG_PASS("Scheduler state:complete_size=", complete_queue.size(),
                ", wait_size=", wait_tsks.size(),
                ", ready_size=", ready_tsks.size(),
                ", alloced_size=", alloced_tsks.size(),
                ", compute_size=", compute_queue.size());
      GLOG_DEBUG("Scheduler state:shutdown=", shutdown.load(),
                 ", complete_empty=", complete_queue.empty(),
                 ", wait_empty=", wait_tsks.empty(),
                 ", ready_empty=", ready_tsks.empty(),
                 ", alloced_empty=", alloced_tsks.empty(),
                 ", compute_empty=", compute_queue.empty());
      */
      if (shutdown.load() && this->wait_tsks.empty() && this->prio.empty() &&
          this->alloced_tsks.empty() && this->compute_queue.empty()) {
        break;
      }

      // * mark complete all in `complete_queue`
      // * add `tsk->next` into `wait_tsks` if `tsk->next` is `NOT nullptr`
      FBLAS_UINT n_completions = 0;
      BaseTask*  tsk = this->complete_queue.pop();
      while (tsk != nullptr) {
        n_completions++;
        tsks_in_mem--;
        this->c_rec.mark_complete(tsk->get_id());
        this->cache.release(tsk);
        tsk->set_status(Complete);
        BaseTask* next = tsk->next;
        if (next != nullptr) {
          GLOG_ASSERT(next->get_status() < AllocReady,
                      "bad next status, expected ", Wait, ", got ",
                      next->get_status());
          next->set_status(Wait);
          this->wait_tsks.push_back(next);
        }
        tsk = this->complete_queue.pop();
      }

      // 2. Process `wait_tsks`
      // * `update_in_place` over `wait_tsks`
      // * filter ready tasks and move to `ready_tsks`
      std::vector<BaseTask*> cur_ready_tsks =
          this->wait_tsks.filter(std::ref(wait_keep_fn));
      for (auto& tsk : cur_ready_tsks) {
        tsk->set_status(AllocReady);
        GLOG_DEBUG("READY:tsk_id=", tsk->get_id());
      }

      FBLAS_UINT ready_delta = cur_ready_tsks.size();

      if (ready_delta > 0) {
        this->prio.insert(cur_ready_tsks);
        GLOG_DEBUG("update_in=", update_in);

        // updation track
        update_in--;
        if (!update_in) {
          Timer t;
          this->prio.update();
          GLOG_DEBUG("Prioritizer Update Latency = ", t.elapsed(), "ms for ",
                     this->prio.size(), " tasks");
          update_in = update_every;
        }
      }

      FBLAS_UINT num_tsks_to_alloc = max_in_mem_tsks - tsks_in_mem;
      FBLAS_UINT num_ready_tsks = this->prio.size();

      while (num_tsks_to_alloc > 0 && !this->prio.empty()) {
        // alloc this task
        TaskInfo  tsk_info = this->prio.get_prio();
        BaseTask* tsk = tsk_info.tsk;
        GLOG_ASSERT(tsk != nullptr, "bad while condition");

        if (this->cache.allocate(tsk)) {
          tsks_in_mem++;
          alloced_tsks.push_back(tsk);
          tsk->set_status(Alloc);
          num_ready_tsks--;
        } else {
          this->prio.return_prio(tsk_info);
          break;
        }
      }

      // update alloced tsks, filter out compute ready tsks & add to compute
      // queue
      auto compute_ready_tsks = this->alloced_tsks.filter(alloc_keep_fn);

      // if added new compute tasks for compute threads, wake all of them up
      if (!compute_ready_tsks.empty()) {
        for (auto& tsk : compute_ready_tsks) {
          tsk->set_status(ComputeReady);
        }
        this->compute_queue.insert(compute_ready_tsks.begin(),
                                   compute_ready_tsks.end());
        this->compute_queue.push_notify_all();
      }

      // service backlogs from all decisions made now
      this->cache.service_backlog();

      // Metrics
      FPTYPE elapsed_ms = timer.elapsed();
      total_sched_time += elapsed_ms;
      sleep_ms = std::max(max_sleep_ms - elapsed_ms, min_sleep_ms);
      if ((FBLAS_UINT) elapsed_ms > 0) {
        GLOG_DEBUG("SCHED: took ", elapsed_ms, "ms, sleeping for ", sleep_ms,
                   "ms");
      }

      usleep(sleep_ms * 1000);
    }
    GLOG_DEBUG("Total Scheduling overhead=", total_sched_time, "ms");
    GLOG_DEBUG("Scheduler Thread Down");
  }

  void Scheduler::compute_thread_fn() {
    FBLAS_UINT cthread_id = this->n_compute_thr.fetch_add(1);
    GLOG_INFO("Compute Thread #", cthread_id, " Up");
    while (true) {
      if (cthread_id >= this->n_compute_thr.load()) {
        if (this->shutdown.load() && this->wait_tsks.empty() &&
            this->prio.empty() && this->alloced_tsks.empty() &&
            this->compute_queue.empty()) {
          break;
        } else {
          // if(!wait_tsks.empty()){
          //   GLOG_INFO("Disabling compute thread #", cthread_id, " because
          //   wait_tsks not empty");
          // } else if(!prio.empty()){
          //   GLOG_INFO("Disabling compute thread #", cthread_id, " because
          //   prio not empty");
          // } else if(!alloced_tsks.empty()){
          //   GLOG_INFO("Disabling compute thread #", cthread_id, " because
          //   alloced_tsks not empty");
          // } else if(!compute_queue.empty()){
          //   GLOG_INFO("Disabling compute thread #", cthread_id, " because
          //   compute_queue not empty");
          // } else{
          //   GLOG_INFO("Disabling compute thread #", cthread_id, " because not
          //   asked to shutdown");
          // }
          // disable thread
          ::usleep(100000);  // 100ms
        }
      } else {
        BaseTask* tsk = this->compute_queue.pop();
        if (tsk == nullptr) {
          if (this->shutdown.load() && this->wait_tsks.empty() &&
              this->prio.empty() && this->alloced_tsks.empty() &&
              this->compute_queue.empty()) {
            break;
          } else {
            this->compute_queue.wait_for_push_notify();
          }
        } else {
          GLOG_DEBUG("executing tsk_id=", tsk->get_id());
          tsk->set_status(Compute);
          tsk->execute();
          this->complete_queue.push(tsk);
        }
      }
    }
    GLOG_INFO("Compute Thread #", cthread_id, " Down");
    this->n_compute_thr--;
  }

  void Scheduler::add_task(BaseTask* tsk) {
    GLOG_DEBUG("adding tsk_id=", tsk->get_id(), " to wait");
    tsk->set_status(Wait);
    this->wait_tsks.push_back(tsk);
  }

  void Scheduler::set_options(SchedulerOptions& sched_opts) {
    this->io_exec.overlap_check = sched_opts.enable_overlap_check;
    this->cache.single_use_discard = sched_opts.single_use_discard;
    if (!this->prio.use_prio && sched_opts.enable_prioritizer) {
      this->prio.use_prio = sched_opts.enable_prioritizer;
      this->prio.update();
    }
  }

  void Scheduler::set_num_compute_threads(FBLAS_UINT new_num) {
    if (new_num == this->n_compute_thr) {
      return;
    } else if (new_num > this->n_compute_thr) {
      FBLAS_UINT n_thr_to_add = new_num - this->n_compute_thr;
      for (FBLAS_UINT i = 0; i < n_thr_to_add; i++) {
        this->compute_threads.push_back(
            std::thread(&Scheduler::compute_thread_fn, this));
      }
    } else {
      this->n_compute_thr.store(new_num);
    }
  }
}  // namespace flash
