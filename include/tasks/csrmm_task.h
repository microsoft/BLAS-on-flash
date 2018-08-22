// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <malloc.h>
#include <cstring>
#include "tasks/task.h"

namespace anon {
  // scatter from src to dest using sinfo
  template<typename T>
  void scatter(T *dest, T *src, flash::StrideInfo sinfo) {
    GLOG_DEBUG("scatter sinfo:lps=", sinfo.len_per_stride,
               ", nstrides=", sinfo.n_strides, ", stride=", sinfo.stride);
    FBLAS_UINT n_scatters = sinfo.n_strides;
    for (FBLAS_UINT i = 0; i < n_scatters; i++) {
      memcpy((char *) dest + (sinfo.stride * i),
             (char *) src + (sinfo.len_per_stride * i), sinfo.len_per_stride);
    }
  }
  // gather from src to dest using sinfo
  template<typename T>
  void gather(T *dest, T *src, flash::StrideInfo sinfo) {
    GLOG_DEBUG("gather sinfo:lps=", sinfo.len_per_stride,
               ", nstrides=", sinfo.n_strides, ", stride=", sinfo.stride);
    FBLAS_UINT n_gathers = sinfo.n_strides;
    for (FBLAS_UINT i = 0; i < n_gathers; i++) {
      memcpy((char *) dest + (sinfo.len_per_stride * i),
             (char *) src + (sinfo.stride * i), sinfo.len_per_stride);
    }
  }
}  // namespace anon

namespace flash {
  class CsrmmRmTask : public BaseTask {
    MKL_INT *          ia;
    flash_ptr<MKL_INT> ja;
    flash_ptr<FPTYPE>  a;
    flash_ptr<FPTYPE>  b;
    flash_ptr<FPTYPE>  c;
    FBLAS_UINT         a_nrows;
    FBLAS_UINT         a_ncols;
    FBLAS_UINT         b_ncols;
    FBLAS_UINT         nnzs;
    FPTYPE             alpha;
    FPTYPE             beta;

   public:
    CsrmmRmTask(const FBLAS_UINT start_row, const FBLAS_UINT start_col,
                const FBLAS_UINT a_blk_size, const FBLAS_UINT b_blk_size,
                const FBLAS_UINT a_rows, const FBLAS_UINT a_cols,
                const FBLAS_UINT b_cols, const MKL_INT *ia,
                flash_ptr<MKL_INT> ja, flash_ptr<FPTYPE> a, flash_ptr<FPTYPE> b,
                flash_ptr<FPTYPE> c, const FPTYPE alpha, const FPTYPE beta)
        : ja(ja), a(a), b(b), c(c) {
      this->alpha = alpha;
      this->beta = beta;
      FBLAS_UINT start_offset = ia[start_row] - ia[0];
      this->ja = ja + start_offset;
      this->a = a + start_offset;
      this->a_nrows = std::min((FBLAS_UINT)(a_rows - start_row), a_blk_size);
      this->ia = new MKL_INT[a_nrows + 1];  // free in this->execute()
#pragma omp parallel for schedule(static, CSRMM_RM_MKL_NTHREADS)
      for (FBLAS_UINT i = 0; i <= a_nrows; i++) {
        this->ia[i] = ia[start_row + i] - ia[start_row];
      }
      this->a_ncols = a_cols;
      this->b = b + start_col;
      this->b_ncols = std::min(b_cols - start_col, b_blk_size);
      this->c = c + ((start_row * b_cols) + start_col);
      StrideInfo sinfo;
      nnzs = (ia[start_row + this->a_nrows] - ia[start_row]);
      sinfo.len_per_stride = nnzs * sizeof(FBLAS_UINT);
      sinfo.stride = 0;
      sinfo.n_strides = 1;
      this->add_read(this->ja, sinfo);
      sinfo.len_per_stride = nnzs * sizeof(FPTYPE);
      this->add_read(this->a, sinfo);
      sinfo.len_per_stride = b_ncols * sizeof(FPTYPE);
      sinfo.n_strides = (this->a_ncols - 1);
      sinfo.stride = b_cols * sizeof(FPTYPE);
      this->add_read(this->b, sinfo);
      sinfo.len_per_stride = b_ncols * sizeof(FPTYPE);
      sinfo.n_strides = (this->a_nrows - 1);
      sinfo.stride = b_cols * sizeof(FPTYPE);

      if (beta != 0.0f) {
        this->add_read(this->c, sinfo);
      }

      this->add_write(this->c, sinfo);
    }

    void execute() {
      mkl_set_num_threads_local(CSRMM_RM_MKL_NTHREADS);
      FPTYPE * a_ptr = (FPTYPE *) this->in_mem_ptrs[this->a];
      FPTYPE * b_ptr = (FPTYPE *) this->in_mem_ptrs[this->b];
      FPTYPE * c_ptr = (FPTYPE *) this->in_mem_ptrs[this->c];
      MKL_INT *ja_ptr = (MKL_INT *) this->in_mem_ptrs[this->ja];
      GLOG_ASSERT(a_ptr != nullptr, "nullptr for a");
      GLOG_ASSERT(ja_ptr != nullptr, "nullptr for ja");
      GLOG_ASSERT(b_ptr != nullptr, "nullptr for b");
      GLOG_ASSERT(c_ptr != nullptr, "nullptr for c");

      // prepare csrmm parameters
      CHAR    trans_a = 'N';
      MKL_INT m = (MKL_INT) this->a_nrows;
      MKL_INT n = (MKL_INT) this->b_ncols;
      MKL_INT k = (MKL_INT) this->a_ncols;
      CHAR    matdescra[5] = {'G', 'X', 'X', 'C', 'X'};
      // execute csrmm
      mkl_csrmm(&trans_a, &m, &n, &k, &this->alpha, &matdescra[0], a_ptr,
                ja_ptr, this->ia, this->ia + 1, b_ptr, &n, &this->beta, c_ptr,
                &n);

      // cleanup
      delete[] this->ia;
    }

    FBLAS_UINT size() {
      FBLAS_UINT a_size = nnzs * (sizeof(FPTYPE) + sizeof(MKL_INT));
      FBLAS_UINT b_size = this->a_ncols * this->b_ncols * sizeof(FPTYPE);
      FBLAS_UINT c_size = this->a_nrows * this->b_ncols * sizeof(FPTYPE);
      return a_size + b_size + c_size;
    }
  };

  class SimpleCsrmmRmTask : public BaseTask {
    SparseBlock       A_blk;
    flash_ptr<FPTYPE> b;
    flash_ptr<FPTYPE> c;
    FBLAS_UINT        b_ncols;
    FBLAS_UINT        nnzs;
    FPTYPE            alpha;
    FPTYPE            beta;
    FBLAS_UINT        idx_delta, val_delta;
    FBLAS_UINT        idx_len, val_len;

   public:
    SimpleCsrmmRmTask(const SparseBlock &A_block, flash_ptr<FPTYPE> b,
                      flash_ptr<FPTYPE> c, FBLAS_UINT b_start_col,
                      FBLAS_UINT b_blk_size, FBLAS_UINT b_cols, FPTYPE alpha,
                      FPTYPE beta) {
      this->A_blk = A_block;
      this->alpha = alpha;
      this->beta = beta;

      this->b = b + b_start_col;
      this->b_ncols = std::min(b_cols - b_start_col, b_blk_size);
      this->c = c + ((A_blk.start * b_cols) + b_start_col);
      this->nnzs = (A_blk.offs[A_blk.blk_size] - A_blk.offs[0]);
      StrideInfo sinfo = {1, 1, 1};

      // round up/down to get aligned access
      FBLAS_UINT idx_start_b = ROUND_DOWN(A_blk.idxs_fptr.foffset, SECTOR_LEN);
      FBLAS_UINT idx_end_b = ROUND_UP(
          A_blk.idxs_fptr.foffset + nnzs * sizeof(MKL_INT), SECTOR_LEN);
      this->idx_delta = A_blk.idxs_fptr.foffset - idx_start_b;
      this->idx_len = idx_end_b - idx_start_b;
      A_blk.idxs_fptr.foffset = idx_start_b;
      sinfo.len_per_stride = this->idx_len;
      this->add_read(A_blk.idxs_fptr, sinfo);

      FBLAS_UINT val_start_b = ROUND_DOWN(A_blk.vals_fptr.foffset, SECTOR_LEN);
      FBLAS_UINT val_end_b =
          ROUND_UP(A_blk.vals_fptr.foffset + nnzs * sizeof(FPTYPE), SECTOR_LEN);
      this->val_delta = A_blk.vals_fptr.foffset - val_start_b;
      this->val_len = val_end_b - val_start_b;
      A_blk.vals_fptr.foffset = val_start_b;
      sinfo.len_per_stride = this->val_len;
      this->add_read(A_blk.vals_fptr, sinfo);

      // if (full B | only a block of B) is to be used
      bool use_full = (b_start_col == 0 && this->b_ncols == b_cols);

      if (use_full) {
        GLOG_INFO("Using complete B matrix");
        sinfo.len_per_stride = A_blk.ncols * this->b_ncols * sizeof(FPTYPE);
        sinfo.n_strides = 1;
        this->add_read(this->b, sinfo);

        // prepare sinfo for `c`
        sinfo.len_per_stride = A_blk.blk_size * this->b_ncols * sizeof(FPTYPE);
      } else {
        sinfo.n_strides = A_blk.ncols;
        sinfo.len_per_stride = b_ncols * sizeof(FPTYPE);
        sinfo.stride = b_cols * sizeof(FPTYPE);
        this->add_read(this->b, sinfo);

        sinfo.n_strides = A_blk.blk_size;
      }

      if (beta != 0.0f) {
        this->add_read(this->c, sinfo);
      }

      this->add_write(this->c, sinfo);
    }

    void execute() {
      mkl_set_num_threads_local(CSRMM_RM_MKL_NTHREADS);
      fill_sparse_block_ptrs(this->in_mem_ptrs, A_blk);

      // recover original array
      A_blk.idxs_ptr = offset_buf(A_blk.idxs_ptr, this->idx_delta);
      A_blk.vals_ptr = offset_buf(A_blk.vals_ptr, this->val_delta);
#ifdef DEBUG
      verify_csr_block(A_blk, false);
#endif
      FPTYPE *b_ptr = (FPTYPE *) this->in_mem_ptrs[this->b];
      FPTYPE *c_ptr = (FPTYPE *) this->in_mem_ptrs[this->c];

      GLOG_ASSERT(A_blk.vals_ptr != nullptr, "nullptr for A_blk.vals");
      GLOG_ASSERT(A_blk.idxs_ptr != nullptr, "nullptr for A_blk.idxs");
      GLOG_ASSERT(b_ptr != nullptr, "nullptr for b");
      GLOG_ASSERT(c_ptr != nullptr, "nullptr for c");

      // prepare csrmm parameters
      CHAR    trans_a = 'N';
      MKL_INT m = (MKL_INT) A_blk.blk_size;
      MKL_INT n = (MKL_INT) this->b_ncols;
      MKL_INT k = (MKL_INT) A_blk.ncols;
      CHAR    matdescra[5] = {'G', 'X', 'X', 'C', 'X'};
      // execute csrmm
      mkl_csrmm(&trans_a, &m, &n, &k, &this->alpha, &matdescra[0],
                A_blk.vals_ptr, A_blk.idxs_ptr, A_blk.offs, A_blk.offs + 1,
                b_ptr, &n, &this->beta, c_ptr, &n);
    }

    // DEPRECATED
    FBLAS_UINT size() {
      return (1 << 20);
    }
  };

  class SimpleCsrmmCmTask : public BaseTask {
    SparseBlock       A_blk;
    flash_ptr<FPTYPE> b;
    flash_ptr<FPTYPE> c;
    FBLAS_UINT        b_ncols;
    FBLAS_UINT        nnzs;
    FPTYPE            alpha;
    FPTYPE            beta;

   public:
    SimpleCsrmmCmTask(const SparseBlock &A_block, flash_ptr<FPTYPE> b,
                      flash_ptr<FPTYPE> c, FBLAS_UINT b_start_col,
                      FBLAS_UINT b_blk_size, FBLAS_UINT b_cols, FPTYPE alpha,
                      FPTYPE beta) {
      this->A_blk = A_block;
      this->alpha = alpha;
      this->beta = beta;

      this->b = b + (b_start_col * A_blk.ncols);
      this->b_ncols = std::min(b_cols - b_start_col, b_blk_size);
      this->c = c + ((A_blk.nrows * b_start_col) + A_blk.start);

      StrideInfo sinfo = {1, 1, 1};
      this->nnzs = (A_blk.offs[A_blk.blk_size] - A_blk.offs[0]);
      sinfo.len_per_stride = nnzs * sizeof(MKL_INT);
      this->add_read(A_blk.idxs_fptr, sinfo);
      sinfo.len_per_stride = nnzs * sizeof(FPTYPE);
      this->add_read(A_blk.vals_fptr, sinfo);

      sinfo.len_per_stride = A_blk.ncols * this->b_ncols * sizeof(FPTYPE);
      sinfo.n_strides = 1;
      this->add_read(this->b, sinfo);

      sinfo.len_per_stride = A_blk.blk_size * sizeof(FPTYPE);
      sinfo.n_strides = this->b_ncols;
      sinfo.stride = A_blk.nrows * sizeof(FPTYPE);
      if (beta != 0.0f) {
        this->add_read(this->c, sinfo);
      }
      this->add_write(this->c, sinfo);
    }

    void execute() {
      mkl_set_num_threads_local(CSRMM_CM_MKL_NTHREADS);
      fill_sparse_block_ptrs(this->in_mem_ptrs, A_blk);
      FPTYPE *b_ptr = (FPTYPE *) this->in_mem_ptrs[this->b];
      FPTYPE *c_ptr = (FPTYPE *) this->in_mem_ptrs[this->c];

      GLOG_ASSERT(A_blk.vals_ptr != nullptr, "nullptr for A_blk.vals");
      GLOG_ASSERT(A_blk.idxs_ptr != nullptr, "nullptr for A_blk.idxs");
      GLOG_ASSERT(b_ptr != nullptr, "nullptr for b");
      GLOG_ASSERT(c_ptr != nullptr, "nullptr for c");

// ja is 0-based indexing => convert to 1-based for easy MKL call
#pragma omp parallel for schedule(static, \
                                  1048576) num_threads(CSRMM_CM_MKL_NTHREADS)
      for (FBLAS_UINT j = 0; j < this->nnzs; j++) {
        A_blk.idxs_ptr[j]++;
      }
#ifdef DEBUG
      verify_csr_block(A_blk, true);
#endif

      // prepare csrmm parameters
      CHAR    trans_a = 'N';
      MKL_INT m = (MKL_INT) A_blk.blk_size;
      MKL_INT n = (MKL_INT) this->b_ncols;
      MKL_INT k = (MKL_INT) A_blk.ncols;

      // NOTE :: matdescra[3] = 'F' => column major storage & 1-based indexing
      CHAR matdescra[5] = {'G', 'X', 'X', 'F', 'X'};

      // execute csrmm
      mkl_csrmm(&trans_a, &m, &n, &k, &this->alpha, &matdescra[0],
                A_blk.vals_ptr, A_blk.idxs_ptr, A_blk.offs, A_blk.offs + 1,
                b_ptr, &k, &this->beta, c_ptr, &m);
    }

    // DEPRECATED
    FBLAS_UINT size() {
      return (1 << 20);
    }
  };

  class CsrmmCmTask : public BaseTask {
    MKL_INT *          ia;
    flash_ptr<MKL_INT> ja;
    flash_ptr<FPTYPE>  a;
    flash_ptr<FPTYPE>  b;
    flash_ptr<FPTYPE>  c;
    FBLAS_UINT         a_nrows;
    FBLAS_UINT         a_ncols;
    FBLAS_UINT         b_ncols;
    FBLAS_UINT         nnzs;
    FPTYPE             alpha;
    FPTYPE             beta;

   public:
    CsrmmCmTask(const FBLAS_UINT start_row, const FBLAS_UINT start_col,
                const FBLAS_UINT a_blk_size, const FBLAS_UINT b_blk_size,
                const FBLAS_UINT a_rows, const FBLAS_UINT a_cols,
                const FBLAS_UINT b_cols, const MKL_INT *ia,
                flash_ptr<MKL_INT> ja, flash_ptr<FPTYPE> a, flash_ptr<FPTYPE> b,
                flash_ptr<FPTYPE> c, const FPTYPE alpha, const FPTYPE beta)
        : ja(ja), a(a), b(b), c(c) {
      this->alpha = alpha;
      this->beta = beta;
      FBLAS_UINT start_offset = ia[start_row];
      this->ja = ja + start_offset;
      this->a = a + start_offset;
      this->a_nrows = std::min((FBLAS_UINT)(a_rows - start_row), a_blk_size);
      this->ia = new MKL_INT[a_nrows + 1];  // `delete` in this->execute()
      this->ia[0] = 1;
      for (FBLAS_UINT i = 1; i <= a_nrows; i++) {
        this->ia[i] = (ia[start_row + i] - ia[start_row]) + this->ia[0];
      }
      this->a_ncols = a_cols;
      this->b = b + (a_cols * start_col);
      this->b_ncols = std::min(b_cols - start_col, b_blk_size);
      this->c = c + ((start_col * a_rows) + start_row);
      StrideInfo sinfo;
      nnzs = (this->ia[this->a_nrows] - this->ia[0]);
      GLOG_DEBUG("start_row=", start_row, ", nnzs=", nnzs);
      sinfo.len_per_stride = nnzs * sizeof(FBLAS_UINT);
      sinfo.stride = 0;
      sinfo.n_strides = 1;
      this->add_read(this->ja, sinfo);
      sinfo.len_per_stride = nnzs * sizeof(FPTYPE);
      this->add_read(this->a, sinfo);
      sinfo.len_per_stride = this->b_ncols * a_cols * sizeof(FPTYPE);
      this->add_read(this->b, sinfo);
      sinfo.len_per_stride = this->a_nrows * sizeof(FPTYPE);
      sinfo.n_strides = (this->b_ncols);
      sinfo.stride = a_rows * sizeof(FPTYPE);
      if (beta != 0.0f) {
        this->add_read(this->c, sinfo);
      }
      this->add_write(this->c, sinfo);
    }

    void execute() {
      mkl_set_num_threads_local(CSRMM_CM_MKL_NTHREADS);
      FPTYPE * a_ptr = (FPTYPE *) this->in_mem_ptrs[this->a];
      FPTYPE * b_ptr = (FPTYPE *) this->in_mem_ptrs[this->b];
      FPTYPE * c_ptr = (FPTYPE *) this->in_mem_ptrs[this->c];
      MKL_INT *ja_ptr = (MKL_INT *) this->in_mem_ptrs[this->ja];

// ja is 0-based indexing => convert to 1-based for easy MKL call
#pragma omp parallel for schedule(static, CSRMM_CM_MKL_NTHREADS)
      for (FBLAS_INT j = 0; j < (FBLAS_INT) this->nnzs; j++) {
        ja_ptr[j]++;
      }
      /*
            GLOG_ASSERT(malloc_usable_size(a_ptr) >= nnzs * sizeof(FPTYPE),
                        "bad malloc for a");
            GLOG_ASSERT(malloc_usable_size(ja_ptr) >= nnzs * sizeof(MKL_INT),
                        "bad malloc for ja");
            GLOG_ASSERT(
                malloc_usable_size(b_ptr) >= a_ncols * b_ncols * sizeof(FPTYPE),
                "bad malloc for b");
            GLOG_ASSERT(
                malloc_usable_size(c_ptr) >= a_nrows * b_ncols * sizeof(FPTYPE),
                "bad malloc for c");
      */
      // prepare csrmm parameters
      CHAR    trans_a = 'N';
      MKL_INT m = (MKL_INT) this->a_nrows;
      MKL_INT n = (MKL_INT) this->b_ncols;
      MKL_INT k = (MKL_INT) this->a_ncols;
      // NOTE :: matdescra[3] = 'F' => column major storage & 1-based indexing
      CHAR matdescra[5] = {'G', 'X', 'X', 'F', 'X'};
      // execute csrmm
      mkl_csrmm(&trans_a, &m, &n, &k, &this->alpha, &matdescra[0], a_ptr,
                ja_ptr, this->ia, this->ia + 1, b_ptr, &k, &this->beta, c_ptr,
                &m);

      // cleanup
      delete[] this->ia;
    }

    FBLAS_UINT size() {
      FBLAS_UINT a_size = nnzs * (sizeof(FPTYPE) + sizeof(MKL_INT));
      FBLAS_UINT b_size = this->a_ncols * this->b_ncols * sizeof(FPTYPE);
      FBLAS_UINT c_size = this->a_nrows * this->b_ncols * sizeof(FPTYPE);
      return a_size + b_size + c_size;
    }
  };

  class CsrmmCmInMemTask : public BaseTask {
    MKL_INT *          ia;
    flash_ptr<MKL_INT> ja;
    flash_ptr<FPTYPE>  a;
    FPTYPE *           b;
    FPTYPE *           c;
    FBLAS_UINT         a_nrows;
    FBLAS_UINT         a_ncols;
    FBLAS_UINT         b_ncols;
    StrideInfo         c_sinfo;
    FBLAS_UINT         nnzs;
    FPTYPE             alpha;
    FPTYPE             beta;

   public:
    CsrmmCmInMemTask(const FBLAS_UINT start_row, const FBLAS_UINT start_col,
                     const FBLAS_UINT a_blk_size, const FBLAS_UINT b_blk_size,
                     const FBLAS_UINT a_rows, const FBLAS_UINT a_cols,
                     const FBLAS_UINT b_cols, const MKL_INT *ia,
                     flash_ptr<MKL_INT> ja, flash_ptr<FPTYPE> a, FPTYPE *b,
                     FPTYPE *c, const FPTYPE alpha, const FPTYPE beta) {
      this->alpha = alpha;
      this->beta = beta;
      FBLAS_UINT start_offset = ia[start_row];
      this->ja = ja + start_offset;
      this->a = a + start_offset;
      this->a_nrows = std::min((FBLAS_UINT)(a_rows - start_row), a_blk_size);
      this->ia = new MKL_INT[a_nrows + 1];
      this->ia[0] = 1;
      for (FBLAS_UINT i = 1; i <= a_nrows; i++) {
        this->ia[i] = (ia[start_row + i] - ia[start_row]) + this->ia[0];
      }
      this->a_ncols = a_cols;
      this->b_ncols = std::min(b_cols - start_col, b_blk_size);
      this->b = b + (a_cols * start_col);
      this->c = c + ((start_col * a_rows) + start_row);
      this->c_sinfo.len_per_stride = this->a_nrows * sizeof(FPTYPE);
      this->c_sinfo.n_strides = this->b_ncols;
      this->c_sinfo.stride = a_rows * sizeof(FPTYPE);

      StrideInfo sinfo;
      nnzs = (this->ia[this->a_nrows] - this->ia[0]);
      GLOG_DEBUG("start_row=", start_row, ", nnzs=", nnzs);
      sinfo.len_per_stride = nnzs * sizeof(FBLAS_UINT);
      sinfo.stride = 0;
      sinfo.n_strides = 1;
      this->add_read(this->ja, sinfo);
      sinfo.len_per_stride = nnzs * sizeof(FPTYPE);
      this->add_read(this->a, sinfo);
    }

    void execute() {
      mkl_set_num_threads_local(CSRMM_CM_MKL_NTHREADS);
      FPTYPE * a_ptr = (FPTYPE *) this->in_mem_ptrs[this->a];
      MKL_INT *ja_ptr = (MKL_INT *) this->in_mem_ptrs[this->ja];
      FPTYPE * b_ptr = this->b;
      FPTYPE * c_ptr = new FPTYPE[this->a_nrows * this->b_ncols];
      if (this->beta != 0.0f) {
        GLOG_DEBUG("exec gather");
        anon::gather<FPTYPE>(c_ptr, this->c, c_sinfo);
      } else {
        memset(c_ptr, 0, this->a_nrows * this->b_ncols * sizeof(FPTYPE));
      }

// ja is 0-based indexing => convert to 1-based for easy MKL call
#pragma omp parallel for schedule(static, CSRMM_CM_MKL_NTHREADS)
      for (FBLAS_INT j = 0; j < (FBLAS_INT) this->nnzs; j++) {
        ja_ptr[j]++;
      }

      GLOG_ASSERT(a_ptr != nullptr, "nullptr for a");
      GLOG_ASSERT(ja_ptr != nullptr, "nullptr for ja");
      GLOG_ASSERT(b_ptr != nullptr, "nullptr for b");
      GLOG_ASSERT(c_ptr != nullptr, "nullptr for c");

      // prepare csrmm parameters
      CHAR    trans_a = 'N';
      MKL_INT m = (MKL_INT) this->a_nrows;
      MKL_INT n = (MKL_INT) this->b_ncols;
      MKL_INT k = (MKL_INT) this->a_ncols;
      // NOTE :: matdescra[3] = 'F' => column major storage & 1-based indexing
      CHAR matdescra[5] = {'G', 'X', 'X', 'F', 'X'};
      // execute csrmm
      mkl_csrmm(&trans_a, &m, &n, &k, &this->alpha, &matdescra[0], a_ptr,
                ja_ptr, this->ia, this->ia + 1, b_ptr, &k, &this->beta, c_ptr,
                &m);
      anon::scatter<FPTYPE>(this->c, c_ptr, c_sinfo);

      // cleanup
      delete[] this->ia;
      delete[] c_ptr;
    }

    FBLAS_UINT size() {
      FBLAS_UINT a_size = nnzs * (sizeof(FPTYPE) + sizeof(MKL_INT)) +
                          (this->a_nrows * sizeof(MKL_INT));
      FBLAS_UINT temp_c_size = this->a_nrows * this->b_ncols * sizeof(FPTYPE);
      return a_size + temp_c_size;
    }
  };
  class CsrmmRmInMemTask : public BaseTask {
    MKL_INT *          ia;
    flash_ptr<MKL_INT> ja;
    flash_ptr<FPTYPE>  a;
    FPTYPE *           b;
    FPTYPE *           c;
    FBLAS_UINT         a_nrows;
    FBLAS_UINT         a_ncols;
    FBLAS_UINT         b_ncols;
    StrideInfo         b_sinfo;
    StrideInfo         c_sinfo;
    FBLAS_UINT         nnzs;
    FPTYPE             alpha;
    FPTYPE             beta;
    bool               use_orig = false;

   public:
    CsrmmRmInMemTask(const FBLAS_UINT start_row, const FBLAS_UINT start_col,
                     const FBLAS_UINT a_blk_size, const FBLAS_UINT b_blk_size,
                     const FBLAS_UINT a_rows, const FBLAS_UINT a_cols,
                     const FBLAS_UINT b_cols, const MKL_INT *ia,
                     flash_ptr<MKL_INT> ja, flash_ptr<FPTYPE> a, FPTYPE *b,
                     FPTYPE *c, const FPTYPE alpha, const FPTYPE beta) {
      GLOG_DEBUG("const params:start_row=", start_row,
                 ", start_col=", start_col, ", a_blk_size=", a_blk_size,
                 ", b_blk_size=", b_blk_size, ", a_rows=", a_rows,
                 ", a_cols=", a_cols, ", b_cols=", b_cols);
      this->alpha = alpha;
      this->beta = beta;
      FBLAS_UINT start_offset = ia[start_row];
      this->ja = ja + start_offset;
      this->a = a + start_offset;
      this->a_nrows = std::min((FBLAS_UINT)(a_rows - start_row), a_blk_size);
      this->ia = new MKL_INT[a_nrows + 1];
      this->ia[0] = 0;
      for (FBLAS_UINT i = 1; i <= a_nrows; i++) {
        this->ia[i] = (ia[start_row + i] - ia[start_row]);
      }
      this->a_ncols = a_cols;
      this->b_ncols = std::min(b_cols - start_col, b_blk_size);
      // use original B if using all of it
      if (start_col == 0 && this->b_ncols == b_cols) {
        GLOG_DEBUG("directly using C for output");
        this->use_orig = true;
      }
      this->b = b + start_col;
      this->c = c + ((start_row * b_cols) + start_col);
      this->c_sinfo.len_per_stride = this->b_ncols * sizeof(FPTYPE);
      this->c_sinfo.n_strides = this->a_nrows;
      this->c_sinfo.stride = b_cols * sizeof(FPTYPE);
      this->b_sinfo.len_per_stride = this->b_ncols * sizeof(FPTYPE);
      this->b_sinfo.n_strides = a_cols;
      this->b_sinfo.stride = b_cols * sizeof(FPTYPE);

      StrideInfo sinfo;
      this->nnzs = (this->ia[this->a_nrows] - this->ia[0]);
      GLOG_DEBUG("start_row=", start_row, ", nnzs=", nnzs);
      sinfo.len_per_stride = nnzs * sizeof(FBLAS_UINT);
      sinfo.stride = 0;
      sinfo.n_strides = 1;
      this->add_read(this->ja, sinfo);
      sinfo.len_per_stride = nnzs * sizeof(FPTYPE);
      this->add_read(this->a, sinfo);
    }

    void execute() {
      // GLOG_WARN("using original B and C as direct input/output arrays");
      mkl_set_num_threads_local(CSRMM_RM_MKL_NTHREADS);
      FPTYPE * a_ptr = (FPTYPE *) this->in_mem_ptrs[this->a];
      MKL_INT *ja_ptr = (MKL_INT *) this->in_mem_ptrs[this->ja];
      FPTYPE * b_ptr = nullptr;
      FPTYPE * c_ptr = nullptr;
      // allocate & gather/scatter only if not working on original B & C
      if (!this->use_orig) {
        b_ptr = new FPTYPE[this->a_ncols * this->b_ncols];
        anon::gather<FPTYPE>(b_ptr, this->b, this->b_sinfo);
        c_ptr = new FPTYPE[this->a_nrows * this->b_ncols];
        if (this->beta != 0.0f) {
          anon::gather<FPTYPE>(c_ptr, this->c, this->c_sinfo);
        } else {
          memset(c_ptr, 0, this->a_nrows * this->b_ncols * sizeof(FPTYPE));
        }
      } else {
        b_ptr = this->b;
        c_ptr = this->c;
      }
      GLOG_ASSERT(malloc_usable_size(a_ptr) >= this->nnzs * sizeof(FPTYPE),
                  "bad alloc for a_ptr");
      GLOG_ASSERT(malloc_usable_size(ja_ptr) >= this->nnzs * sizeof(MKL_INT),
                  "bad alloc for ja_ptr");
      if (!this->use_orig) {
        GLOG_ASSERT(malloc_usable_size(b_ptr) >=
                        this->a_ncols * this->b_ncols * sizeof(FPTYPE),
                    "bad alloc for b_ptr");
        GLOG_ASSERT(malloc_usable_size(c_ptr) >=
                        this->a_nrows * this->b_ncols * sizeof(FPTYPE),
                    "bad alloc for c_ptr, expected=");
      }
      // prepare csrmm parameters
      CHAR    trans_a = 'N';
      MKL_INT m = (MKL_INT) this->a_nrows;
      MKL_INT n = (MKL_INT) this->b_ncols;
      MKL_INT k = (MKL_INT) this->a_ncols;
      CHAR    matdescra[5] = {'G', 'X', 'X', 'C', 'X'};
      GLOG_DEBUG("mkl_in_params:m=", m, ", n=", n, ", k=", k);

      // execute csrmm
      mkl_csrmm(&trans_a, &m, &n, &k, &this->alpha, &matdescra[0], a_ptr,
                ja_ptr, this->ia, this->ia + 1, this->b, &n, &this->beta,
                this->c, &n);

      // write results out and delete temp outputs
      if (!this->use_orig) {
        anon::scatter<FPTYPE>(this->c, c_ptr, c_sinfo);
        // cleanup
        delete[] b_ptr;
        delete[] c_ptr;
      }

      delete[] this->ia;
    }

    FBLAS_UINT size() {
      FBLAS_UINT a_size = this->nnzs * (sizeof(FPTYPE) + sizeof(MKL_INT)) +
                          (this->a_nrows * sizeof(MKL_INT));
      // If not using original B and C matrices, then we need temporary copies
      if (!this->use_orig) {
        return (this->a_ncols * this->b_ncols * sizeof(FPTYPE)) +
               (this->a_nrows * this->b_ncols * sizeof(FPTYPE)) + a_size;
      } else {
        return a_size;
      }
    }
  };
}  // namespace flash
