// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "scheduler/cache.h"
#include "timer.h"

namespace {
  void print_keys_if_not_empty(
      std::unordered_map<flash::Key, flash::Value> &map) {
    for (auto &k_v : map) {
      GLOG_FAIL("Key:", std::string(k_v.first), ", n_refs=", k_v.second.n_refs);
    }
  }

  void assert_and_print(std::unordered_map<flash::Key, flash::Value> &map) {
    print_keys_if_not_empty(map);
    GLOG_ASSERT(map.empty(), "map not empty");
  }
}  // namespace

namespace flash {
  Cache::Cache(IoExecutor &io_exec, const FBLAS_UINT max_size)
      : io_exec(io_exec), max_size(max_size) {
    this->real_size = 0;
    this->commit_size = 0;
    this->single_use_discard = false;
  }

  Cache::~Cache() {
    mutex_locker(this->cache_mut);
    GLOG_DEBUG("checking if active_map is empty");
    assert_and_print(this->active_map);
    GLOG_DEBUG("checking if zero_ref_map is empty");
    assert_and_print(this->zero_ref_map);
    GLOG_DEBUG("checking if io_map is empty");
    assert_and_print(this->io_map);
    /*
    GLOG_DEBUG("checking if alloc_backlog is empty");
    assert_and_print(this->alloc_backlog);
    */

    GLOG_DEBUG("cache destroyed");
  }

  void Cache::flush() {
    GLOG_DEBUG("checking if active_map is empty");
    assert_and_print(this->active_map);
    GLOG_DEBUG("checking if zero_ref_map is empty");
    if (!this->zero_ref_map.empty()) {
      std::unordered_set<Key> evict_keys;
      for (auto &k_v : this->zero_ref_map) {
        evict_keys.insert(k_v.first);
      }
      this->evict(evict_keys);
    }
    while (!this->io_map.empty()) {
      GLOG_DEBUG("waiting for cache to flush to disk");
      service_backlog();
      usleep(100 * 1000);  // 100ms
    }

    assert_and_print(this->zero_ref_map);
    GLOG_DEBUG("checking if io_map is empty");
    assert_and_print(this->io_map);
    /*
    GLOG_DEBUG("checking if alloc_backlog is empty");
    assert_and_print(this->alloc_backlog);
    */
    GLOG_PASS("cache flushed to disk");
  }

  void Cache::evict(const std::unordered_set<Key> &keys) {
    for (auto &k : keys) {
      GLOG_ASSERT(is_zero_ref(k), "attempted to evict non-zero-ref buf");
      Value v = this->zero_ref_map[k];
      GLOG_ASSERT(v.n_refs == 0, "non-zero ref buf in zero-ref-buf map");

      // remove from zero_ref_map
      this->zero_ref_map.erase(k);
      auto sub_size = buf_size(k.sinfo);
      this->commit_size -= sub_size;
      GLOG_DEBUG("EVICT:", sub_size, ", commit_size=", this->commit_size);
      // check if `write_back`
      if (v.write_back) {
        v.evicted = true;

        // NOTE :: v.complete will be freed when completion is reaped
        v.complete = new std::atomic<bool>(false);
        auto completion = v.complete;
        auto buf = v.buf;
        auto real_size_ptr = &(this->real_size);
        auto callback = [completion, buf, real_size_ptr, sub_size]() {
          completion->store(true);
          free(buf);
          real_size_ptr->fetch_sub(sub_size);
          GLOG_DEBUG("DEALLOC:", sub_size,
                     ", real_size=", real_size_ptr->load());
        };

        // add entry to map
        this->io_map[k] = v;

        // construct and issue write
        this->io_exec.add_write(k.fptr, k.sinfo, v.buf, callback);
      } else {
        // R-only buf
        free(v.buf);
        this->real_size.fetch_sub(sub_size);
        GLOG_DEBUG("DEALLOC:", sub_size,
                   ", real_size=", this->real_size.load());
      }
    }
  }

  bool Cache::try_evict(const std::unordered_set<Key> &exclude_keys,
                        const FBLAS_UINT               evict_size) {
    // fast return `false` if zero_ref_map is empty
    if (this->zero_ref_map.empty()) {
      return false;
    }

    // check if operation is possible
    FBLAS_UINT              evicted_size = 0;
    std::unordered_set<Key> evict_keys;
    for (const auto &k_v : zero_ref_map) {
      const Key &key = k_v.first;
      // `key` not asked to be excluded
      if (exclude_keys.find(key) == exclude_keys.end()) {
        evicted_size += buf_size(key.sinfo);
        evict_keys.insert(key);
        if (evicted_size >= evict_size) {
          break;
        }
      }
    }

    if (evicted_size < evict_size) {
      evict_keys.clear();
      return false;
    }

    // evict keys in `evict_keys` set
    this->evict(evict_keys);

    return true;
  }

  void *Cache::get_buf(const flash_ptr<void> &fptr, const StrideInfo &sinfo,
                       bool write_back) {
    mutex_locker lk(this->cache_mut);

    void *result = nullptr;
    Key   k{fptr, sinfo};
    // GLOG_DEBUG("query=", std::string(k));
    bool found = false;
    if (is_active(k)) {
      found = true;
      this->active_map[k].n_refs++;
    } else if (is_in_io(k) && !this->io_map[k].evicted &&
               this->io_map[k].complete->load()) {
      reap_io_completion(k);
      move_io_to_active(k);
      found = true;
    } else if (is_zero_ref(k)) {
      move_zero_to_active(k);
      found = true;
    }
    if (found) {
      Value &v = this->active_map[k];
      result = v.buf;
      v.write_back |= write_back;
      GLOG_ASSERT(result != nullptr, "bad move semantics");
    }

    lk.unlock();

    return result;
  }

  void Cache::add_backlog(const Key &k, bool alloc_only, bool write_back) {
    if (is_queued(k)) {
      return;
    }

    static FBLAS_UINT n_added = 0;
    this->commit_size += buf_size(k.sinfo);
    GLOG_ASSERT(this->commit_size <= this->max_size,
                "got commit_size=", commit_size, ", max_mem=", max_size);
    GLOG_DEBUG("COMMIT:", buf_size(k.sinfo),
               ", commit_size=", this->commit_size);

    // construct null Value
    Value v;
    v.alloc_only = alloc_only;
    v.write_back = write_back;
    // add to backlog
    // this->alloc_backlog.[k] = v;
    this->alloc_backlog.push_back(std::make_pair(k, v));
  }

  void Cache::reap_io_completion(const Key &k) {
    GLOG_ASSERT(is_in_io(k), "bad reap issue");
    Value &v = this->io_map[k];
    GLOG_ASSERT(v.complete->load(), "tried to reap incomplete I/O");

    delete v.complete;
    v.complete = nullptr;
  }

  void Cache::alloc_bufs(BaseTask *tsk) {
    std::unordered_set<Key> read_keys;
    std::unordered_set<Key> write_keys;

    for (auto &fptr_sinfo : tsk->read_list) {
      read_keys.insert({fptr_sinfo.first, fptr_sinfo.second});
    }

    for (auto &fptr_sinfo : tsk->write_list) {
      write_keys.insert({fptr_sinfo.first, fptr_sinfo.second});
    }

    auto       read_only_keys = set_difference(read_keys, write_keys);
    auto       write_only_keys = set_difference(write_keys, read_keys);
    auto       read_write_keys = set_intersection(write_keys, read_keys);
    FBLAS_UINT union_size =
        read_only_keys.size() + write_only_keys.size() + read_write_keys.size();
    GLOG_ASSERT(union_size <= (read_keys.size() + write_keys.size()),
                "bad intersection | difference");

    // (R \ W) set (R-only)
    for (auto &key : read_only_keys) {
      if (is_active(key)) {
        GLOG_DEBUG("HIT:", std::string(key), ":ACTIVE_MAP");
        // FOUND in active
        Value &v = this->active_map[key];
        v.n_refs++;
        tsk->in_mem_ptrs[key.fptr] = v.buf;
      } else if (is_in_io(key)) {
        // FOUND in I/O
        Value &v = this->io_map[key];
        if (v.evicted) {
          GLOG_DEBUG("MISS:", std::string(key), ":EVICTED");
          add_backlog(key, false, false);
        } else {
          if (v.complete->load()) {
            GLOG_DEBUG("HIT:", std::string(key), ":IO_MAP");
            reap_io_completion(key);
            move_io_to_active(key);
            Value &v2 = this->active_map[key];
            v2.n_refs = 1;
            tsk->in_mem_ptrs[key.fptr] = v2.buf;
          }
        }
      } else if (is_zero_ref(key)) {
        GLOG_DEBUG("HIT:", std::string(key), ":ZERO_MAP");
        // FOUND in zero-ref
        move_zero_to_active(key);
        tsk->in_mem_ptrs[key.fptr] = this->active_map[key].buf;
      } else {
        GLOG_DEBUG("MISS:", std::string(key), ":QUEUEING");
        add_backlog(key, false, false);
      }
    }

    // (W \ R) set (W-only)
    for (auto &key : write_only_keys) {
      if (is_active(key)) {
        GLOG_ERROR("write-only-buf in active-map");
      } else if (is_in_io(key)) {
        GLOG_ERROR("write-only-buf in io-map");
      } else if (is_zero_ref(key)) {
        GLOG_ERROR("write-only-buf in zero-ref-map");
      } else {
        GLOG_DEBUG("MISS:", std::string(key), ":QUEUEING");
        add_backlog(key, true, true);
      }
    }

    // (R intersection W) set (R+W)
    for (auto &key : read_write_keys) {
      // skip already already processed keys
      if (is_active(key)) {
        GLOG_DEBUG("HIT:", std::string(key), ":ACTIVE_MAP");
        // FOUND in active
        Value &v = this->active_map[key];
        v.n_refs++;
        v.write_back = true;
        tsk->in_mem_ptrs[key.fptr] = v.buf;
      } else if (is_in_io(key)) {
        // FOUND in I/O
        Value &v = this->io_map[key];
        if (v.evicted) {
          GLOG_DEBUG("MISS:", std::string(key), ":EVICTED");
          if (!is_queued(key)) {
            add_backlog(key, false, true);
          } else {
            /*
            // make buffer write-back if already queued
            Value &v2 = this->alloc_backlog[key];
            v2.write_back = true;
            */
            // wait for read to complete and then mark it as write-back
          }
        } else if (v.complete->load()) {
          GLOG_DEBUG("HIT:", std::string(key), ":IO_MAP");
          reap_io_completion(key);
          move_io_to_active(key);
          Value &v2 = this->active_map[key];
          v2.n_refs = 1;
          tsk->in_mem_ptrs[key.fptr] = v2.buf;
        }
      } else if (is_zero_ref(key)) {
        GLOG_DEBUG("HIT:", std::string(key), ":ZERO_MAP");
        // FOUND in zero-ref
        move_zero_to_active(key);
        tsk->in_mem_ptrs[key.fptr] = this->active_map[key].buf;
        this->active_map[key].write_back = true;
      } else {
        GLOG_DEBUG("MISS:", std::string(key), ":QUEUEING");
        add_backlog(key, false, true);
      }
    }
  }

  void Cache::move_active_to_zero(const Key &k) {
    Value v = this->active_map[k];
    GLOG_ASSERT(v.n_refs == 0, "bad move semantics");
    this->active_map.erase(k);
    this->zero_ref_map[k] = v;
  }

  void Cache::move_zero_to_active(const Key &k) {
    Value v = this->zero_ref_map[k];
    this->zero_ref_map.erase(k);
    this->active_map[k] = v;
    this->active_map[k].n_refs = 1;
  }

  void Cache::move_io_to_active(const Key &k) {
    Value v = this->io_map[k];
    this->io_map.erase(k);
    this->active_map[k] = v;
    this->active_map[k].n_refs = 1;
  }

  bool Cache::allocate(BaseTask *tsk) {
    std::unordered_set<Key> ask_keys;
    for (auto &fptr_sinfo : tsk->read_list) {
      ask_keys.insert({fptr_sinfo.first, fptr_sinfo.second});
    }

    for (auto &fptr_sinfo : tsk->write_list) {
      ask_keys.insert({fptr_sinfo.first, fptr_sinfo.second});
    }

    mutex_locker lk(this->cache_mut);
    // determine amount of extra mem needed
    FBLAS_UINT ask_size = 0;
    for (auto &key : ask_keys) {
      if (is_active(key) || is_zero_ref(key)) {
        continue;
      } else if (is_in_io(key)) {
        if (this->io_map[key].evicted) {
          // to be re-read into mem
          ask_size += buf_size(key.sinfo);
        }
      } else {
        ask_size += buf_size(key.sinfo);
      }
    }
    bool alloc = false;

    // if cache is not full
    // NOTE :: C++ standard mandates short-circuit of `||` operator
    if (has_spare_mem_for(ask_size)) {
      GLOG_DEBUG("alloc-because has spare_mem");
      alloc_bufs(tsk);
      alloc = true;
    } else if (try_evict(ask_keys, ask_size)) {
      GLOG_DEBUG("alloc-because evicted");
      alloc_bufs(tsk);
      alloc = true;
    }

    lk.unlock();

    GLOG_DEBUG("alloc_size=", ask_size, ", alloc=", alloc,
               ", commit_size=", this->commit_size);
    return alloc;
  }

  void Cache::release(const BaseTask *tsk) {
    std::unordered_set<Key> ret_keys;
    for (auto &fptr_sinfo : tsk->read_list) {
      ret_keys.insert({fptr_sinfo.first, fptr_sinfo.second});
    }

    for (auto &fptr_sinfo : tsk->write_list) {
      ret_keys.insert({fptr_sinfo.first, fptr_sinfo.second});
    }

    mutex_locker lk(this->cache_mut);

    for (auto &key : ret_keys) {
      GLOG_ASSERT(is_active(key), "active key not found in active_map");
      Value &v = this->active_map[key];
      if (v.write_back) {
        GLOG_DEBUG("write-back:n_refs=", v.n_refs);
      }
      v.n_refs--;

      // deprecate from active -> zero-ref
      if (v.n_refs == 0) {
        if (this->single_use_discard) {
          void *buf = v.buf;
          this->active_map.erase(key);
          FBLAS_UINT bsize = buf_size(key.sinfo);
          this->commit_size -= bsize;
          GLOG_DEBUG("EVICT:", bsize, ", commit_size=", this->commit_size);
          this->real_size.fetch_sub(bsize);
          free(buf);
          GLOG_DEBUG("DEALLOC:", bsize, ", real_size=", this->real_size.load());
        } else {
          move_active_to_zero(key);
        }
      }
    }

    lk.unlock();
  }

  void Cache::service_backlog() {
    mutex_locker lk(this->cache_mut);
    Timer        timer;
    // cleanup io_map
    // move `k` from `io_map` to `zero_ref_map` if `NOT v.evicted`
    for (auto it = this->io_map.begin(); it != this->io_map.end();) {
      if (it->second.complete->load()) {
        reap_io_completion(it->first);
        const Key &k = it->first;
        Value &    v = it->second;
        if (!v.evicted) {
          v.n_refs = 0;
          // buf goes into active AND NOT zero-ref to avoid being evicted again,
          // even though it's already promised to some tasks
          GLOG_ASSERT(!is_active(k), "trying to replace active buf");
          this->active_map[k] = v;
        } else {
          GLOG_DEBUG("eviction:k=", std::string(k), " complete");
        }
        it = this->io_map.erase(it);
      } else {
        it++;
      }
    }

    if (timer.elapsed() > 0) {
      GLOG_DEBUG("TIME: I/O completion = ", timer.elapsed(), "ms");
    }
    timer.reset();

    static FBLAS_UINT n_serviced = 0;
    FPTYPE            evict_time = 0.0f;
    FPTYPE            alloc_time = 0.0f;

    std::unordered_set<Key> empty_exclude_set;
    for (auto it = this->alloc_backlog.begin();
         it != this->alloc_backlog.end();) {
      timer.reset();
      // extract key to alloc
      const Key &k = it->first;
      // check if alloc possible
      FBLAS_UINT bsize = buf_size(k.sinfo);
      if (!has_spare_real_mem_for(bsize)) {
        // couldn't accommodate this buffer?
        // wait for its evicted write-backs to finish
        break;
        /*
        // try evicting that size
        if (!try_evict(empty_exclude_set, bsize)) {
          break;
        }
        // check if it can be accommodated now?
        if (!has_spare_real_mem_for(bsize)) {
          break;
        }
        */
        // fall through if has spare real mem for `buf`
      }

      evict_time += timer.elapsed();
      timer.reset();

      // if already in io_map [EVICTED], skip for this round
      if (is_in_io(k)) {
        GLOG_WARN("preventing data race (read->write)");
        it++;
        // continue;;
        break;
      }

      GLOG_DEBUG("backlog-PROCESSING:", std::string(k));
      n_serviced++;
      Value &v = it->second;
      // if `alloc_only`, always write-back when evicted
      v.write_back |= v.alloc_only;

      // alloc buf & update size tracker
      this->real_size.fetch_add(bsize);
      // v.buf = malloc(bsize);
      alloc_aligned(&v.buf, ROUND_UP(bsize, SECTOR_LEN));
      GLOG_DEBUG("ALLOC:", ROUND_UP(bsize, SECTOR_LEN),
                 ", real_size=", this->real_size.load());

      if (!v.alloc_only) {
        v.complete = new std::atomic<bool>(false);
        auto completion = v.complete;
        auto callback = [completion]() { completion->store(true); };

        // add to I/O set
        this->io_map[k] = v;

        // read from disk
        this->io_exec.add_read(k.fptr, k.sinfo, v.buf, callback);
      } else {
        // memset(v.buf, 0, bsize); -> PERFORMANCE HIT
        // HACK to not get added buffer evicted
        v.complete = new std::atomic<bool>(true);
        v.evicted = false;
        // this->zero_ref_map[k] = v;
        this->io_map[k] = v;
      }
      alloc_time += timer.elapsed();

      // remove from alloc backlog
      it = this->alloc_backlog.erase(it);
    }
    if (evict_time + alloc_time > 0) {
      GLOG_DEBUG("TIME: Evict Latency = ", evict_time,
                 "ms, Alloc Latency=", alloc_time, "ms");
    }

    timer.reset();
    // GLOG_DEBUG("n_serviced=", n_serviced);

    lk.unlock();
  }

  void Cache::drop_if_in_cache(std::unordered_set<Key> &keys) {
    mutex_locker lk(this->cache_mut);
    auto         it = keys.begin();
    while (it != keys.end()) {
      const Key &k = *it;
      if (is_active(k) || (is_in_io(k) && !this->io_map[k].evicted) ||
          is_zero_ref(k) || is_queued(k)) {
        it = keys.erase(it);
      } else {
        it++;
      }
    }

    lk.unlock();
  }

  void Cache::keep_if_in_cache(std::unordered_set<Key> &keys) {
    mutex_locker lk(this->cache_mut);
    auto         it = keys.begin();
    while (it != keys.end()) {
      const Key &k = *it;
      if (is_active(k) || (is_in_io(k) && !this->io_map[k].evicted) ||
          is_zero_ref(k) || is_queued(k)) {
        it++;
      } else {
        it = keys.erase(it);
      }
    }

    lk.unlock();
  }

}  // namespace flash
