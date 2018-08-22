// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstring>
#include "pointers/pointer.h"
#include "tasks/task.h"

namespace flash {
  class BlockCsrCscTask : public BaseTask {
    FBLAS_UINT pdim;
    FBLAS_UINT nnzs;

    SparseBlock A_blk;
    SparseBlock A_tr_blk;

   public:
    BlockCsrCscTask(SparseBlock A_block, SparseBlock A_tr_block)
        : A_blk(A_block), A_tr_blk(A_tr_block) {
      this->pdim = std::max(A_blk.nrows, A_blk.ncols);

      nnzs = A_blk.offs[A_blk.blk_size] - A_blk.offs[0];
      GLOG_INFO("will transpose nnzs=", nnzs,
                ", starting at row=", A_blk.start);

      StrideInfo sinfo;
      sinfo.n_strides = 1;
      sinfo.stride = 0;

      // reads & writes for `column indices`
      sinfo.len_per_stride = nnzs * sizeof(MKL_INT);
      this->add_read(A_blk.idxs_fptr, sinfo);
      this->add_write(A_tr_blk.idxs_fptr, sinfo);

      // reads & writes for `matrix values`
      sinfo.len_per_stride = nnzs * sizeof(FPTYPE);
      this->add_read(A_blk.vals_fptr, sinfo);
      this->add_write(A_tr_blk.vals_fptr, sinfo);
    }

    void execute() {
      // extract matrix dimensions
      FBLAS_UINT A_cols = A_blk.ncols;
      FBLAS_UINT A_rows = A_blk.nrows;

      MKL_INT *input_offs = new MKL_INT[pdim + 1];
      MKL_INT *output_offs = new MKL_INT[pdim + 1];
      for (FBLAS_UINT i = 0; i <= A_blk.blk_size; i++) {
        input_offs[i] = A_blk.offs[i] - A_blk.offs[0];
      }
      // expand csr matrix with zero rows
      for (FBLAS_UINT i = A_blk.blk_size + 1; i <= pdim; i++) {
        input_offs[i] = input_offs[A_blk.blk_size];
      }

      mkl_set_num_threads_local(CSRCSC_MKL_NTHREADS);

      SparseBlock A_pblk(A_blk), A_tr_pblk(A_tr_blk);
      A_pblk.offs = input_offs;
      A_tr_pblk.offs = output_offs;

      // fill in-memory pointers into blocks
      fill_sparse_block_ptrs(this->in_mem_ptrs, A_pblk);
      fill_sparse_block_ptrs(this->in_mem_ptrs, A_tr_pblk);

      // prepare MKL call
      MKL_INT job[6] = {0, 0, 0, -1, -1, 1};
      MKL_INT dim = pdim;
      MKL_INT info = -1;  // not used

      // make MKL call
      mkl_csrcsc(job, &dim, A_pblk.vals_ptr, A_pblk.idxs_ptr, A_pblk.offs,
                 A_tr_pblk.vals_ptr, A_tr_pblk.idxs_ptr, A_tr_pblk.offs, &info);

// add A_blk.start to `A_pblk.idxs_ptr`
#pragma omp parallel for num_threads(CSRCSC_MKL_NTHREADS)
      for (FBLAS_UINT i = 0; i < nnzs; i++) {
        A_tr_pblk.idxs_ptr[i] += A_blk.start;
      }

      memcpy(A_tr_blk.offs, A_tr_pblk.offs, (A_cols + 1) * sizeof(MKL_INT));

      delete[] A_pblk.offs;
      delete[] A_tr_pblk.offs;

      GLOG_ASSERT(A_tr_blk.offs[A_cols] == nnzs,
                  "bad csrcsc params:input nnzs=", nnzs,
                  ", output nnzs=", A_tr_blk.offs[A_cols]);

      GLOG_INFO("transposed:nnzs=", A_tr_blk.offs[A_cols]);
    }

    // DEPRECATED
    FBLAS_UINT size() {
      return (1 << 20);
    }
  };

  // Horizontally merge [column join] CSR matrices into one CSR matrix
  class BlockMergeTask : public BaseTask {
    SparseBlock              A_blk;
    std::vector<SparseBlock> A_blks;

   public:
    BlockMergeTask(SparseBlock A_block, std::vector<SparseBlock> A_blocks)
        : A_blk(A_block) {
      A_blks.reserve(A_blocks.size());
      FBLAS_UINT total_nnzs = A_blk.offs[A_blk.blk_size] - A_blk.offs[0];
      GLOG_INFO("merging nnzs=", total_nnzs);
      StrideInfo sinfo = {1, 1, 1};
      sinfo.len_per_stride = total_nnzs * sizeof(MKL_INT);
      this->add_write(A_blk.idxs_fptr, sinfo);
      sinfo.len_per_stride = total_nnzs * sizeof(FPTYPE);
      this->add_write(A_blk.vals_fptr, sinfo);
      FBLAS_UINT got_nnzs = 0;
      for (auto blk : A_blocks) {
        FBLAS_UINT blk_nnzs = blk.offs[blk.blk_size] - blk.offs[0];
        got_nnzs += blk_nnzs;
        if (blk_nnzs == 0) {
          GLOG_WARN("ignoring 0-block in merge");
          continue;
        }

        this->A_blks.push_back(blk);

        sinfo.len_per_stride = blk_nnzs * sizeof(MKL_INT);
        this->add_read(blk.idxs_fptr, sinfo);
        sinfo.len_per_stride = blk_nnzs * sizeof(FPTYPE);
        this->add_read(blk.vals_fptr, sinfo);
      }
      GLOG_ASSERT(got_nnzs == total_nnzs, " expected nnzs=", total_nnzs,
                  ", got nnzs=", got_nnzs);
    }

    void execute() {
      // fill in sparse blocks
      fill_sparse_block_ptrs(this->in_mem_ptrs, A_blk);
      for (auto &blk : A_blks) {
        fill_sparse_block_ptrs(this->in_mem_ptrs, blk);
      }

#pragma omp parallel for schedule(dynamic, 1) num_threads(CSRCSC_MKL_NTHREADS)
      for (FBLAS_UINT row = 0; row < A_blk.blk_size; row++) {
        FBLAS_UINT fill_offset = (A_blk.offs[row] - A_blk.offs[0]);
        for (auto &blk : A_blks) {
          FBLAS_UINT read_offset = (blk.offs[row] - blk.offs[0]);
          FBLAS_UINT nnzs_in_blk = (blk.offs[row + 1] - blk.offs[row]);
          memcpy(A_blk.idxs_ptr + fill_offset, blk.idxs_ptr + read_offset,
                 nnzs_in_blk * sizeof(MKL_INT));
          memcpy(A_blk.vals_ptr + fill_offset, blk.vals_ptr + read_offset,
                 nnzs_in_blk * sizeof(FPTYPE));
          fill_offset += nnzs_in_blk;
        }
        FBLAS_UINT expected_nnzs_in_row =
            (A_blk.offs[row + 1] - A_blk.offs[row]);
        FBLAS_UINT got_nnzs_in_row =
            fill_offset - (A_blk.offs[row] - A_blk.offs[0]);
        GLOG_ASSERT(expected_nnzs_in_row == got_nnzs_in_row,
                    ", expected to fill ", expected_nnzs_in_row,
                    ", filled only ", got_nnzs_in_row);
      }
    }

    // DEPRECATED
    FBLAS_UINT size() {
      return (1 << 20);
    }
  };
}  // namespace flash
