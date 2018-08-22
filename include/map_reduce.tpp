// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// contains implementation if templates for map and reduce functions

#include <unistd.h>
#include "scheduler/scheduler.h"
#include "tasks/map_reduce_task.h"

namespace flash {
  extern Scheduler sched;
  template<typename InType, typename OutType>
  FBLAS_INT map(flash_ptr<InType> in_fptr, flash_ptr<OutType> out_fptr, FBLAS_UINT len,
          std::function<OutType(const InType &)> &mapper) {
    FBLAS_UINT blk_size = MAP_BLK_SIZE;
    FBLAS_UINT n_blks = ROUND_UP(len, blk_size) / blk_size;

    MapTask<InType, OutType> **map_tasks =
        new MapTask<InType, OutType> *[n_blks];
    for (FBLAS_UINT i = 0; i < n_blks; i++) {
      FBLAS_UINT start_idx = i * blk_size;
      FBLAS_UINT blk_len = std::min(len - i * blk_size, blk_size);
      map_tasks[i] = new MapTask<InType, OutType>(mapper, in_fptr, out_fptr,
                                                  start_idx, blk_len);
    }
    for (FBLAS_UINT i = 0; i < n_blks; i += 2) {
      if (i == n_blks - 1) {
        // if only one task in total
        if (i == 0) {
          continue;
        } else {
          GLOG_DEBUG("adding dependency : ", i, "->", i - 1);
          map_tasks[i - 1]->add_parent(map_tasks[i]->get_id());
        }
      } else {
        if (i != 0) {
          GLOG_DEBUG("adding dependency : ", i, "->", i - 1);
          map_tasks[i - 1]->add_parent(map_tasks[i]->get_id());
        }
        GLOG_DEBUG("adding dependency : ", i, "->", i + 1);
        map_tasks[i + 1]->add_parent(map_tasks[i]->get_id());
      }
    }
    for (FBLAS_UINT i = 0; i < n_blks; i++) {
      sched.add_task(map_tasks[i]);
    }
    sleep_wait_for_complete(map_tasks, n_blks);
    for (FBLAS_UINT i = 0; i < n_blks; i++) {
      delete map_tasks[i];
    }
    delete[] map_tasks;
    return 0;
  }

  template<typename T>
  T reduce(flash_ptr<T> fptr, FBLAS_UINT len, T &id,
           std::function<T(T &, T &)> &reducer) {
    FBLAS_UINT blk_size = REDUCE_BLK_SIZE;
    FBLAS_UINT n_blks = ROUND_UP(len, blk_size) / blk_size;

    ReduceTask<T> **reduce_tasks = new ReduceTask<T> *[n_blks];
    // reducer outputs for each block
    T *blk_results = new T[n_blks];
    for (FBLAS_UINT i = 0; i < n_blks; i++) {
      FBLAS_UINT cur_len = std::min(len - i * blk_size, blk_size);
      blk_results[i] = id;
      reduce_tasks[i] =
          new ReduceTask<T>(reducer, fptr, id, i * blk_size, cur_len);
      sched.add_task(reduce_tasks[i]);
    }

    // wait for all reducer tasks to finish reducing
    sleep_wait_for_complete(reduce_tasks, n_blks);

    // reduce all local results
    T result = id;
    for (FBLAS_UINT i = 0; i < n_blks; i++) {
      result = reducer(result, blk_results[i]);
    }
    // cleanup tasks
    delete[] blk_results;
    for (FBLAS_UINT i = 0; i < n_blks; i++) {
      delete reduce_tasks[i];
    }
    delete[] reduce_tasks;

    // return global result
    return result;
  }
} // namespace flash
