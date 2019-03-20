// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#include "bof_utils.h"

namespace flash {
  template<typename T>
  class ConcurrentVector {
    typedef std::chrono::milliseconds    chrono_ms_t;
    typedef std::unique_lock<std::mutex> mutex_locker;

    std::vector<T>          vec;
    std::mutex              mut;
    std::condition_variable cv;
    std::condition_variable push_cv;
    T                       null_T;

   public:
    ConcurrentVector() {
      this->null_T = static_cast<T>(0);
    }

    ConcurrentVector(T nullT) {
      this->null_T = nullT;
    }

    ~ConcurrentVector() {
      this->cv.notify_all();
      this->push_cv.notify_all();
    }

    // queue stats
    uint64_t size() {
      mutex_locker lk(this->mut);
      uint64_t     sz = vec.size();
      lk.unlock();
      return sz;
    }

    bool empty() {
      return (this->size() == 0);
    }

    // push functions
    void push_back(T& new_val) {
      mutex_locker lk(this->mut);
      this->vec.push_back(new_val);
      lk.unlock();
    }

    template<class Iterator>
    void insert(Iterator iter_begin, Iterator iter_end) {
      mutex_locker lk(this->mut);
      this->vec.insert(this->vec.end(), iter_begin, iter_end);
      lk.unlock();
    }

    // `weak` form of waiting with 100ms timout
    void wait_for_notify(
        std::chrono::milliseconds wait_time = std::chrono::milliseconds(100)) {
      mutex_locker lk(this->mut);
      this->cv.wait_for(lk, wait_time);
      lk.unlock();
    }

    // keeps elements that return keep_fn(el)=true
    // returns remaining elements
    std::vector<T> filter(const std::function<bool(T&)>& keep_fn) {
      std::vector<T> discarded;
      mutex_locker   lk(this->mut);

      for (auto it = this->vec.begin(); it != this->vec.end();) {
        if (!keep_fn(*it)) {
          discarded.push_back(*it);
          // `std::vector<T>::erase` automatically increments `it`
          this->vec.erase(it);
        } else {
          // manually increment it
          ++it;
        }
      }
      lk.unlock();
      return discarded;
    }

    // `update_fn` is expected to overwrite its input
    void update(const std::function<void(T&)>& update_fn) {
      mutex_locker lk(this->mut);
      for (auto& el : this->vec) {
        update_fn(el);
      }
      lk.unlock();
    }

    std::vector<T> update_and_filter(const std::function<void(T&)>  update_fn,
                                     const std::function<bool(T&)>& keep_fn) {
      std::vector<T> discarded;
      mutex_locker   lk(this->mut);

      for (auto it = this->vec.begin(); it != this->vec.end();) {
        // update element
        update_fn(*it);

        // discard if necessary
        if (!keep_fn(*it)) {
          discarded.push_back(*it);
          // `std::vector<T>::erase` automatically increments `it`
          this->vec.erase(it);
        } else {
          // manually increment it
          ++it;
        }
      }
      lk.unlock();
      return discarded;
    }

    // `wake up` a thread sleeping on a notification
    void notify_one() {
      this->cv.notify_one();
    }

    // `wake up` all threads sleeping on a notification
    void notify_all() {
      this->cv.notify_all();
    }
  };
}  // namespace flash
