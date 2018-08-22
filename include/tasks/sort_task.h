// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <malloc.h>
#include <cmath>
#include <cstdlib>
#include <parallel/algorithm>
#include "pointers/pointer.h"
#include "tasks/task.h"

namespace flash {
  template<class T, class Comparator>
  class SampleSplit : public BaseTask {
    flash_ptr<T> fptr;
    FBLAS_UINT   arr_size;
    Comparator   cmp;
    T*           samples;
    FBLAS_UINT   n_samples;

   public:
    SampleSplit(flash_ptr<T> base_fptr, FBLAS_UINT start_idx,
                FBLAS_UINT arr_size, T* samples, FBLAS_UINT n_samples,
                Comparator cmp)
        : cmp(cmp), samples(samples), n_samples(n_samples) {
      this->fptr = base_fptr + start_idx;
      StrideInfo sinfo{0, 0, 0};
      sinfo.n_strides = 1;
      this->arr_size = arr_size;
      sinfo.len_per_stride = arr_size * sizeof(T);
      this->add_read(this->fptr, sinfo);
      this->add_write(this->fptr, sinfo);
    }

    void execute() {
      T* ptr = (T*) this->in_mem_ptrs[this->fptr];
      __gnu_parallel::sort(ptr, ptr + this->arr_size, this->cmp);
      // sample every size/n_samples element
      for (FBLAS_UINT i = 0; i < n_samples; i++) {
        FBLAS_UINT idx = rand() % arr_size;
        this->samples[i] = ptr[idx];
      }
    }

    FBLAS_UINT size() {
      return this->arr_size * sizeof(T);
    }
  };

  template<class T, class Comparator>
  class SampleSegment : public BaseTask {
    FBLAS_INT*   starts = nullptr;
    FBLAS_INT*   ends = nullptr;
    T*           pivots = nullptr;
    Comparator   cmp;
    flash_ptr<T> blk_fptr;
    FBLAS_UINT   blk_size;
    FBLAS_UINT   n_pivots;

   public:
    SampleSegment(FBLAS_INT* starts, FBLAS_INT* ends, T* pivots,
                  FBLAS_UINT n_pivots, flash_ptr<T> blk_base_fptr,
                  FBLAS_UINT offset, FBLAS_UINT blk_size, Comparator cmp)
        : cmp(cmp) {
      this->blk_fptr = blk_base_fptr + offset;
      StrideInfo sinfo{0, 0, 0};
      sinfo.n_strides = 1;
      sinfo.len_per_stride = blk_size * sizeof(T);
      this->add_read(this->blk_fptr, sinfo);
      this->starts = starts;
      this->ends = ends;
      this->pivots = pivots;
      this->blk_size = blk_size;
      this->n_pivots = n_pivots;
    }

    void execute() {
      // sizes[i] = -1 if no element in the range
      T* ptr = (T*) this->in_mem_ptrs[this->blk_fptr];

      // for each pivot, find the start and end
      // pivot 0 : start = -1, end = some idx(if present)
      FBLAS_UINT pivot_idx = 0;
      // find starting pivot
      FBLAS_UINT start_piv = 0;
      for (start_piv = 0; start_piv < this->n_pivots; start_piv++) {
        if (cmp(ptr[0], pivots[start_piv])) {
          starts[start_piv] = 0;
          ends[start_piv] = 0;
          break;
        } else {
          starts[start_piv] = -1;
          ends[start_piv] = -1;
        }
      }
      if (start_piv == this->n_pivots) {
        GLOG_WARN("unable to segment");
        return;
      }
      pivot_idx = start_piv;
      for (FBLAS_UINT ptr_idx = 1; ptr_idx < this->blk_size; ptr_idx++) {
        // if current element is not as large as bucket end
        if (cmp(ptr[ptr_idx], pivots[pivot_idx])) {
          // increase bucket length of this pivot;
          ends[pivot_idx]++;
        } else {
          // move on to next pivot
          pivot_idx++;
          // find next pivot that's larger than current element
          for (; pivot_idx < this->n_pivots; pivot_idx++) {
            if (cmp(ptr[ptr_idx], pivots[pivot_idx])) {
              // select this pivot
              starts[pivot_idx] = ptr_idx;
              ends[pivot_idx] = ptr_idx;
              break;
            } else {
              // skip this pivot
              starts[pivot_idx] = -1;
              ends[pivot_idx] = -1;
            }
          }
          // if no next pivot was found, then select last bucket and exit
          if (pivot_idx == this->n_pivots) {
            starts[this->n_pivots] = ptr_idx;
            ends[this->n_pivots] = blk_size - 1;
            break;
          }
        }
      }
      // set other buckets to be of 0 length
      pivot_idx++;
      for (; pivot_idx <= this->n_pivots; pivot_idx++) {
        starts[pivot_idx] = -1;
        ends[pivot_idx] = -1;
      }
    }

    FBLAS_UINT size() {
      return this->blk_size * sizeof(T);
    }
  };

  template<class T, class Comparator>
  class SampleMerge : public BaseTask {
    flash_ptr<T>              out_fptr;
    std::vector<flash_ptr<T>> in_fptrs;
    std::vector<FBLAS_UINT>   sizes;
    Comparator                cmp;
    FBLAS_UINT                total_size;

   public:
    SampleMerge(flash_ptr<T> base_infptr, std::vector<FBLAS_UINT> base_inoffs,
                std::vector<FBLAS_UINT> sizes, flash_ptr<T> base_outfptr,
                FBLAS_UINT out_off, FBLAS_UINT out_size, Comparator cmp)
        : cmp(cmp) {
      StrideInfo sinfo{0, 0, 0};
      sinfo.n_strides = 1;
      this->total_size = 0;
      for (FBLAS_UINT i = 0; i < sizes.size(); i++) {
        if (sizes[i] > 0) {
          // flash_ptr for in_blk[i]
          this->in_fptrs.push_back(base_infptr + base_inoffs[i]);
          this->sizes.push_back(sizes[i]);
          sinfo.len_per_stride = this->sizes[i] * sizeof(T);
          this->add_read(this->in_fptrs[i], sinfo);
          this->total_size += this->sizes[i];
        } else {
          GLOG_WARN("0 size");
        }
      }
      // record block sizes
      this->out_fptr = base_outfptr + out_off;
      sinfo.len_per_stride = this->total_size * sizeof(T);
      this->add_write(this->out_fptr, sinfo);
    }

    void execute() {
      T* out_ptr = (T*) this->in_mem_ptrs[this->out_fptr];
      GLOG_ASSERT(out_ptr != nullptr, "null output");

      // memcpy
      FBLAS_UINT n_blks = in_fptrs.size();
      FBLAS_UINT cur_offset = 0;
      for (FBLAS_UINT i = 0; i < n_blks; i++) {
        T* in_ptr = (T*) this->in_mem_ptrs[in_fptrs[i]];
        /*
                GLOG_ASSERT(malloc_usable_size(in_ptr) >= this->sizes[i] *
           sizeof(T),
                            "bad prefetch");
        */
        memcpy(out_ptr + cur_offset, in_ptr, this->sizes[i] * sizeof(T));
        cur_offset += this->sizes[i];
      }

      // std sort on out_ptr
      __gnu_parallel::sort(out_ptr, out_ptr + this->total_size, this->cmp);
    }

    FBLAS_UINT size() {
      return 2 * sizeof(T) * this->total_size;
    }
  };
}  // namespace flash
