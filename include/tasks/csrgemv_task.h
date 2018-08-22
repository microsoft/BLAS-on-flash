// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <cstring>
#include <thread>
#include "tasks/task.h"
#include "types.h"
#include "utils.h"

namespace flash {
  class CsrGemvNoTransInMem : public BaseTask {
    // matrix specs
    MKL_INT*           ia;
    flash_ptr<MKL_INT> ja;
    flash_ptr<FPTYPE>  a;
    FBLAS_UINT         dim;
    FBLAS_UINT         a_nrows;
    FBLAS_UINT         nnzs;

    // vector specs
    FPTYPE* in;
    FPTYPE* out;

   public:
    CsrGemvNoTransInMem(FBLAS_UINT start_row, FBLAS_UINT a_rows,
                        FBLAS_UINT a_cols, FBLAS_UINT a_rblk_size, MKL_INT* ia,
                        flash_ptr<MKL_INT> ja, flash_ptr<FPTYPE> a,
                        FPTYPE* v_in, FPTYPE* v_out) {
      // matrix specs
      this->a_nrows = std::min(a_rows - start_row, a_rblk_size);
      this->dim = std::max(a_nrows, a_cols);
      this->ja = ja + ia[start_row];
      this->a = a + ia[start_row];
      // copy over offsets and remove start offset
      this->ia = new MKL_INT[this->dim + 1];
      memcpy(this->ia, ia + start_row, (this->a_nrows + 1) * sizeof(MKL_INT));
      for (FBLAS_UINT i = 1; i <= this->a_nrows; i++) {
        this->ia[i] -= this->ia[0];
      }
      this->ia[0] = 0;
      for (FBLAS_UINT i = this->a_nrows + 1; i <= this->dim; i++) {
        this->ia[i] = this->ia[this->a_nrows];
      }

      this->in = v_in;
      this->out = v_out + start_row;

      // add reads
      this->nnzs = this->ia[this->dim] - this->ia[0];
      StrideInfo sinfo;
      sinfo.stride = 0;
      sinfo.n_strides = 1;
      sinfo.len_per_stride = this->nnzs * sizeof(FPTYPE);
      this->add_read(this->a, sinfo);
      sinfo.len_per_stride = this->nnzs * sizeof(MKL_INT);
      this->add_read(this->ja, sinfo);
    }

    void execute() {
      MKL_INT* ja_ptr = (MKL_INT*) this->in_mem_ptrs[this->ja];
      FPTYPE*  a_ptr = (FPTYPE*) this->in_mem_ptrs[this->a];
      FPTYPE*  v_out = nullptr;
      if (this->dim > this->a_nrows) {
        v_out = new FPTYPE[this->dim];
      } else {
        v_out = this->out;
      }

      // MKL parameters;
      char    transa = 'N';
      MKL_INT m = this->dim;
      // execute MKL call
      mkl_csrgemv(&transa, &m, a_ptr, this->ia, ja_ptr, this->in, v_out);

      if (this->dim > this->a_nrows) {
        memcpy(this->out, v_out, this->a_nrows * sizeof(FPTYPE));
        delete[] v_out;
      }

      // free memory for ia
      delete[] this->ia;
    }

    FBLAS_UINT size() {
      if (this->dim > this->a_nrows) {
        return (this->nnzs * (sizeof(MKL_INT) + sizeof(FPTYPE)) +
                (this->dim + this->a_nrows) * sizeof(FPTYPE));
      } else {
        return (this->nnzs * (sizeof(MKL_INT) + sizeof(FPTYPE)) +
                (this->a_nrows * sizeof(FPTYPE)));
      }
    }
  };

  class CsrGemvTransInMem : public BaseTask {
    // matrix specs
    MKL_INT*           ia;
    flash_ptr<MKL_INT> ja;
    flash_ptr<FPTYPE>  a;
    FBLAS_UINT         blk_size;
    FBLAS_UINT         a_rows;
    FBLAS_UINT         a_cols;
    FBLAS_UINT         dim;
    FBLAS_UINT         nnzs;

    // `atomic` access to output array
    std::mutex& mut;

    // vector specs
    FPTYPE* in;
    FPTYPE* out;

   public:
    CsrGemvTransInMem(FBLAS_UINT start_row, FBLAS_UINT a_rows,
                      FBLAS_UINT a_cols, FBLAS_UINT a_rblk_size, MKL_INT* ia,
                      flash_ptr<MKL_INT> ja, flash_ptr<FPTYPE> a, FPTYPE* v_in,
                      FPTYPE* v_out, std::mutex& sync_mut)
        : mut(std::ref(sync_mut)) {
      // matrix specs
      this->blk_size = std::min(a_rows - start_row, a_rblk_size);
      this->dim = std::max(a_rows, a_cols);
      this->ja = ja + ia[start_row];
      this->a = a + ia[start_row];
      // copy over offsets and remove start offset
      this->ia = new MKL_INT[this->dim + 1];
      memcpy(this->ia, ia + start_row, (this->blk_size + 1) * sizeof(MKL_INT));
      for (FBLAS_UINT i = 1; i <= this->blk_size; i++) {
        this->ia[i] -= this->ia[0];
      }
      this->ia[0] = 0;
      for (FBLAS_UINT i = this->blk_size + 1; i <= this->dim; i++) {
        this->ia[i] = this->ia[this->blk_size];
      }

      this->in = v_in + start_row;
      this->out = v_out;
      this->a_rows = a_rows;
      this->a_cols = a_cols;

      // add reads
      this->nnzs = this->ia[this->dim] - this->ia[0];
      StrideInfo sinfo;
      sinfo.stride = 0;
      sinfo.n_strides = 1;
      sinfo.len_per_stride = this->nnzs * sizeof(FPTYPE);
      this->add_read(this->a, sinfo);
      sinfo.len_per_stride = this->nnzs * sizeof(MKL_INT);
      this->add_read(this->ja, sinfo);
    }

    void execute() {
      MKL_INT* ja_ptr = (MKL_INT*) this->in_mem_ptrs[this->ja];
      FPTYPE*  a_ptr = (FPTYPE*) this->in_mem_ptrs[this->a];
      // prepare MKL parameters;
      char    transa = 'T';
      MKL_INT m = (MKL_INT) this->dim;
      FPTYPE* v_out = new FPTYPE[this->dim];
      memset(v_out, 0, this->dim * sizeof(FPTYPE));
      FPTYPE* v_in = new FPTYPE[this->dim];
      memset(v_in, 0, this->dim * sizeof(FPTYPE));
      memcpy(v_in, this->in, this->blk_size * sizeof(FPTYPE));

      // execute MKL call
      mkl_csrgemv(&transa, &m, a_ptr, this->ia, ja_ptr, v_in, v_out);
      delete[] this->ia;
      delete[] v_in;

      // lock and add to existing result
      {
        std::unique_lock<std::mutex> lk(this->mut);
#pragma omp                          parallel for
        for (FBLAS_UINT i = 0; i < this->a_cols; i++) {
          this->out[i] += v_out[i];
        }
      }

      delete[] v_out;
    }

    FBLAS_UINT size() {
      return (this->nnzs * (sizeof(MKL_INT) + sizeof(FPTYPE))) +
             (this->dim * (sizeof(FPTYPE) + sizeof(MKL_INT))) +
             (this->dim > this->blk_size ? this->dim * sizeof(FPTYPE) : 0);
    }
  };
}  // namespace flash
