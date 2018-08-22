// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <omp.h>
#include "pointers/pointer.h"
#include "tasks/task.h"

namespace flash {
  template<class InType, class OutType>
  class MapTask : public BaseTask {
    typedef std::function<OutType(const InType &)> MapFnType;

    MapFnType          map_fn;
    flash_ptr<InType>  in_fptr;
    flash_ptr<OutType> out_fptr;
    FBLAS_UINT         len;

   public:
    MapTask(MapFnType &mapfn, flash_ptr<InType> base_in,
            flash_ptr<OutType> base_out, FBLAS_UINT start_idx,
            FBLAS_UINT blk_size)
        : map_fn(mapfn) {
      this->len = blk_size;
      this->in_fptr = base_in + start_idx;
      this->out_fptr = base_out + start_idx;
      StrideInfo sinfo{0, 0, 0};
      sinfo.n_strides = 1;
      sinfo.len_per_stride = blk_size * sizeof(InType);
      this->add_read(in_fptr, sinfo);
      sinfo.len_per_stride = blk_size * sizeof(OutType);
      this->add_write(out_fptr, sinfo);
    }
    void execute() {
      InType * in_ptr = (InType *) this->in_mem_ptrs[this->in_fptr];
      OutType *out_ptr = (OutType *) this->in_mem_ptrs[this->out_fptr];

#pragma omp parallel for
      for (FBLAS_UINT i = 0; i < len; i++) {
        out_ptr[i] = this->map_fn(in_ptr[i]);
      }
    }

    FBLAS_UINT size() {
      return this->len * (sizeof(InType) + sizeof(OutType));
    }
  };
  template<class T>
  class ReduceTask : public BaseTask {
    typedef std::function<T(T &, T &)> OpType;

    OpType       op;
    flash_ptr<T> in_fptr;
    T            id;
    T            result;
    FBLAS_UINT   len;

   public:
    ReduceTask(OpType &op, flash_ptr<T> base_in, T id, FBLAS_UINT start_idx,
               FBLAS_UINT blk_size)
        : op(op), id(id), result(id) {
      this->len = blk_size;
      this->in_fptr = base_in + start_idx;
      StrideInfo sinfo{0, 0, 0};
      sinfo.n_strides = 1;
      sinfo.len_per_stride = blk_size * sizeof(T);
      this->add_read(this->in_fptr, sinfo);
    }

    void execute() {
      T *in_ptr = (T *) this->in_mem_ptrs[this->in_fptr];
      GLOG_ASSERT(in_ptr != nullptr, "null input to ReduceTask");
      FBLAS_UINT n_threads = omp_get_num_threads();
      FBLAS_UINT thread_blk_size = ROUND_UP(this->len, n_threads) / n_threads;
      this->result = id;
#pragma omp parallel for num_threads(n_threads)
      for (FBLAS_UINT i = 0; i < n_threads; i++) {
        T          l_res(id);
        FBLAS_UINT start = i * thread_blk_size;
        FBLAS_UINT end = std::min((i + 1) * thread_blk_size, this->len);
        for (FBLAS_UINT j = start; j < end; j++) {
          l_res = op(l_res, in_ptr[j]);
        }
// add local result
#pragma omp critical
        { this->result = op(this->result, l_res); }
      }
    }

    T get_result() {
      return this->result;
    }

    FBLAS_UINT size() {
      return (this->len + omp_get_num_threads() + 1) * sizeof(T);
    }
  };
}  // namespace flash