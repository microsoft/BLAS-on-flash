// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "blas_utils.h"
#include "flash_blas.h"
#include "lib_funcs.h"
#include "tasks/csrmm_task.h"

namespace flash {
  extern Scheduler sched;
}  // namespace flash

namespace {
  using namespace flash;
  void csrmm_no_trans_rm(FBLAS_UINT m, FBLAS_UINT n, FBLAS_UINT k, FPTYPE alpha,
                         FPTYPE beta, flash_ptr<FPTYPE> a,
                         flash_ptr<MKL_INT> ia, flash_ptr<MKL_INT> ja,
                         flash_ptr<FPTYPE> b, flash_ptr<FPTYPE> c) {
    // read `ia` into buffer
    MKL_INT *ia_ptr = new MKL_INT[m + 1];
    ia.fop->read(ia.foffset, (m + 1) * sizeof(MKL_INT), (void *) ia_ptr,
                 flash::dummy_std_func);
    std::vector<FBLAS_UINT> blks;
    std::vector<FBLAS_UINT> offs;
    FBLAS_UINT              cur_start = 0;
    for (; cur_start < m;) {
      FBLAS_UINT cblk_size =
          get_next_blk_size(ia_ptr + cur_start, m - cur_start,
                            SECTOR_LEN / sizeof(FPTYPE), CSRMM_RM_RBLK_SIZE);
      blks.push_back(cblk_size);
      offs.push_back(cur_start);
      cur_start += cblk_size;
      GLOG_DEBUG("choosing blk_size=", cblk_size);
    }
    FBLAS_UINT    col_blk_size = CSRMM_RM_CBLK_SIZE;
    FBLAS_UINT    n_row_blks = blks.size();
    FBLAS_UINT    n_col_blks = ROUND_UP(k, col_blk_size) / col_blk_size;
    CsrmmRmTask **csr_tasks = new CsrmmRmTask *[n_row_blks * n_col_blks];

    // iterate over row blocks
    for (FBLAS_UINT i = 0; i < n_row_blks; i++) {
      FBLAS_UINT start_row = offs[i];
      FBLAS_UINT rblk_size = blks[i];
      for (FBLAS_UINT j = 0; j < n_col_blks; j++) {
        csr_tasks[i * n_col_blks + j] = new CsrmmRmTask(
            start_row, j * col_blk_size, rblk_size, col_blk_size, m, n, k,
            ia_ptr, ja, a, b, c, alpha, beta);
        sched.add_task(csr_tasks[i * n_col_blks + j]);
      }
    }

    // sync and cleanup
    sleep_wait_for_complete(csr_tasks, n_row_blks * n_col_blks);
    for (FBLAS_UINT l = 0; l < (n_row_blks * n_col_blks); l++) {
      delete csr_tasks[l];
    }

    delete[] csr_tasks;

    // flush result to disk
    sched.flush_cache();
  }

  void csrmm_no_trans_rm2(FBLAS_UINT m, FBLAS_UINT n, FBLAS_UINT k,
                          FPTYPE alpha, FPTYPE beta, flash_ptr<FPTYPE> a,
                          flash_ptr<MKL_INT> ia, flash_ptr<MKL_INT> ja,
                          flash_ptr<FPTYPE> b, flash_ptr<FPTYPE> c) {
    // read `ia` into buffer
    MKL_INT *ia_ptr = new MKL_INT[m + 1];
    ia.fop->read(ia.foffset, (m + 1) * sizeof(MKL_INT), (void *) ia_ptr,
                 flash::dummy_std_func);

    std::vector<FBLAS_UINT> blks;
    std::vector<FBLAS_UINT> offs;

    fill_blocks(ia_ptr, m, blks, offs, SECTOR_LEN / sizeof(FPTYPE),
                CSRMM_RM_RBLK_SIZE);

    FBLAS_UINT          col_blk_size = CSRMM_RM_CBLK_SIZE;
    FBLAS_UINT          n_row_blks = blks.size();
    FBLAS_UINT          n_col_blks = ROUND_UP(k, col_blk_size) / col_blk_size;
    SimpleCsrmmRmTask **csr_tasks =
        new SimpleCsrmmRmTask *[n_row_blks * n_col_blks];
    std::vector<SparseBlock> row_blks;

    // iterate over row blocks
    for (FBLAS_UINT i = 0; i < n_row_blks; i++) {
      FBLAS_UINT start_row = offs[i];
      FBLAS_UINT rblk_size = blks[i];

      // construct row-block
      SparseBlock A_blk;
      A_blk.nrows = m;
      A_blk.ncols = n;
      A_blk.start = start_row;
      A_blk.blk_size = rblk_size;
      A_blk.offs = new MKL_INT[A_blk.blk_size + 1];
      for (FBLAS_UINT i = 0; i <= (FBLAS_UINT) A_blk.blk_size; i++) {
        A_blk.offs[i] = ia_ptr[A_blk.start + i] - ia_ptr[A_blk.start];
      }
      A_blk.idxs_fptr = ja + ia_ptr[A_blk.start];
      A_blk.vals_fptr = a + ia_ptr[A_blk.start];
      row_blks.push_back(A_blk);

      // construct one task for each col-block
      for (FBLAS_UINT j = 0; j < n_col_blks; j++) {
        csr_tasks[i * n_col_blks + j] = new SimpleCsrmmRmTask(
            A_blk, b, c, j * col_blk_size, col_blk_size, k, alpha, beta);
        sched.add_task(csr_tasks[i * n_col_blks + j]);
      }
    }

    // sync and cleanup
    sleep_wait_for_complete(csr_tasks, n_row_blks * n_col_blks);
    for (FBLAS_UINT l = 0; l < (n_row_blks * n_col_blks); l++) {
      delete csr_tasks[l];
    }
    delete[] csr_tasks;
    for (auto &row_blk : row_blks) {
      delete[] row_blk.offs;
    }
    delete[] ia_ptr;

    // flush result to disk
    sched.flush_cache();
  }

  void csrmm_no_trans_cm(FBLAS_UINT m, FBLAS_UINT n, FBLAS_UINT k, FPTYPE alpha,
                         FPTYPE beta, flash_ptr<FPTYPE> a,
                         flash_ptr<MKL_INT> ia, flash_ptr<MKL_INT> ja,
                         flash_ptr<FPTYPE> b, flash_ptr<FPTYPE> c) {
    // read `ia` into buffer
    MKL_INT *ia_ptr = new MKL_INT[m + 1];
    ia.fop->read(ia.foffset, (m + 1) * sizeof(MKL_INT), (void *) ia_ptr,
                 flash::dummy_std_func);
    std::vector<FBLAS_UINT> blks;
    std::vector<FBLAS_UINT> offs;
    FBLAS_UINT              cur_start = 0;
    for (; cur_start < m;) {
      FBLAS_UINT cblk_size =
          get_next_blk_size(ia_ptr + cur_start, m - cur_start,
                            SECTOR_LEN / sizeof(FPTYPE), CSRMM_CM_RBLK_SIZE);
      blks.push_back(cblk_size);
      offs.push_back(cur_start);
      cur_start += cblk_size;
      GLOG_DEBUG("choosing blk_size=", cblk_size);
    }
    FBLAS_UINT    n_row_blks = blks.size();
    FBLAS_UINT    col_blk_size = CSRMM_CM_CBLK_SIZE;
    FBLAS_UINT    n_col_blks = ROUND_UP(k, col_blk_size) / col_blk_size;
    CsrmmCmTask **csr_tasks = new CsrmmCmTask *[blks.size() * n_col_blks];

    // iterate over row blocks
    for (FBLAS_UINT i = 0; i < n_row_blks; i++) {
      FBLAS_UINT start_row = offs[i];
      FBLAS_UINT rblk_size = blks[i];
      for (FBLAS_UINT j = 0; j < n_col_blks; j++) {
        csr_tasks[i * n_col_blks + j] = new CsrmmCmTask(
            start_row, j * col_blk_size, rblk_size, col_blk_size, m, n, k,
            ia_ptr, ja, a, b, c, alpha, beta);
        sched.add_task(csr_tasks[i * n_col_blks + j]);
      }
    }
    /*
    // skip if only one monolithic task
    if (n_row_blks * n_col_blks > 1) {
      // set dependencies
      for (FBLAS_UINT l = 1; l < (n_row_blks * n_col_blks) - 1; l += 2) {
        // backward
        csr_tasks[l]->add_parent(csr_tasks[l - 1]->get_id());
        // forward
        csr_tasks[l]->add_parent(csr_tasks[l + 1]->get_id());
      }
      // last element
      if ((n_row_blks * n_col_blks) % 2 != 0) {
        // backward
        csr_tasks[n_row_blks * n_col_blks - 1]->add_parent(
            csr_tasks[n_row_blks * n_col_blks - 2]->get_id());
      }
    }

    // add all ready tasks to scheduler
    for (FBLAS_UINT l = 0; l < (n_row_blks * n_col_blks); l += 2) {
      sched.add_task(csr_tasks[l]);
    }
    for (FBLAS_UINT l = 1; l < (n_row_blks * n_col_blks); l += 2) {
      sched.add_task(csr_tasks[l]);
    }
  */
    // sync and cleanup
    sleep_wait_for_complete(csr_tasks, n_row_blks * n_col_blks);
    for (FBLAS_UINT l = 0; l < (n_row_blks * n_col_blks); l++) {
      delete csr_tasks[l];
    }

    delete[] csr_tasks;
    delete[] ia_ptr;

    // flush result to disk
    sched.flush_cache();
  }

  void csrmm_no_trans_cm2(FBLAS_UINT m, FBLAS_UINT n, FBLAS_UINT k,
                          FPTYPE alpha, FPTYPE beta, flash_ptr<FPTYPE> a,
                          flash_ptr<MKL_INT> ia, flash_ptr<MKL_INT> ja,
                          flash_ptr<FPTYPE> b, flash_ptr<FPTYPE> c) {
    // read `ia` into buffer
    MKL_INT *ia_ptr = new MKL_INT[m + 1];
    ia.fop->read(ia.foffset, (m + 1) * sizeof(MKL_INT), (void *) ia_ptr,
                 flash::dummy_std_func);

    std::vector<FBLAS_UINT> blks;
    std::vector<FBLAS_UINT> offs;

    fill_blocks(ia_ptr, m, blks, offs, SECTOR_LEN / sizeof(FPTYPE),
                CSRMM_CM_RBLK_SIZE);

    FBLAS_UINT          col_blk_size = CSRMM_CM_CBLK_SIZE;
    FBLAS_UINT          n_row_blks = blks.size();
    FBLAS_UINT          n_col_blks = ROUND_UP(k, col_blk_size) / col_blk_size;
    SimpleCsrmmCmTask **csr_tasks =
        new SimpleCsrmmCmTask *[n_row_blks * n_col_blks];
    std::vector<SparseBlock> row_blks;

    // iterate over row blocks
    for (FBLAS_UINT i = 0; i < n_row_blks; i++) {
      FBLAS_UINT start_row = offs[i];
      FBLAS_UINT rblk_size = blks[i];

      // construct row-block
      SparseBlock A_blk;
      A_blk.nrows = m;
      A_blk.ncols = n;
      A_blk.start = start_row;
      A_blk.blk_size = rblk_size;
      A_blk.offs = new MKL_INT[A_blk.blk_size + 1];
      // NOTE :: Use 1-based indexing for col-major CSRMM calls to MKL
      for (FBLAS_UINT i = 0; i <= (FBLAS_UINT) A_blk.blk_size; i++) {
        A_blk.offs[i] = ia_ptr[A_blk.start + i] - ia_ptr[A_blk.start] + 1;
      }
      A_blk.idxs_fptr = ja + ia_ptr[A_blk.start];
      A_blk.vals_fptr = a + ia_ptr[A_blk.start];
      row_blks.push_back(A_blk);

      // construct one task for each col-block
      for (FBLAS_UINT j = 0; j < n_col_blks; j++) {
        csr_tasks[i * n_col_blks + j] = new SimpleCsrmmCmTask(
            A_blk, b, c, j * col_blk_size, col_blk_size, k, alpha, beta);
        sched.add_task(csr_tasks[i * n_col_blks + j]);
      }
    }

    // sync and cleanup
    sleep_wait_for_complete(csr_tasks, n_row_blks * n_col_blks);
    for (FBLAS_UINT l = 0; l < (n_row_blks * n_col_blks); l++) {
      delete csr_tasks[l];
    }
    delete[] csr_tasks;
    for (auto &row_blk : row_blks) {
      delete[] row_blk.offs;
    }
    delete[] ia_ptr;

    // flush result to disk
    sched.flush_cache();
  }

  void csrmm_no_trans_cm_im(FBLAS_UINT m, FBLAS_UINT n, FBLAS_UINT k,
                            FPTYPE alpha, FPTYPE beta, flash_ptr<FPTYPE> a,
                            flash_ptr<MKL_INT> ia, flash_ptr<MKL_INT> ja,
                            FPTYPE *b, FPTYPE *c) {
    // read `ia` into buffer
    MKL_INT *ia_ptr = new MKL_INT[m + 1];
    ia.fop->read(ia.foffset, (m + 1) * sizeof(MKL_INT), (void *) ia_ptr,
                 flash::dummy_std_func);
    std::vector<FBLAS_UINT> blks;
    std::vector<FBLAS_UINT> offs;
    FBLAS_UINT              cur_start = 0;
    for (; cur_start < m;) {
      FBLAS_UINT cblk_size =
          get_next_blk_size(ia_ptr + cur_start, m - cur_start,
                            SECTOR_LEN / sizeof(FPTYPE), CSRMM_RM_RBLK_SIZE);
      blks.push_back(cblk_size);
      offs.push_back(cur_start);
      cur_start += cblk_size;
      GLOG_DEBUG("choosing blk_size=", cblk_size);
    }
    FBLAS_UINT         n_row_blks = blks.size();
    FBLAS_UINT         col_blk_size = CSRMM_CM_CBLK_SIZE;
    FBLAS_UINT         n_col_blks = ROUND_UP(k, col_blk_size) / col_blk_size;
    CsrmmCmInMemTask **csr_tasks =
        new CsrmmCmInMemTask *[n_row_blks * n_col_blks];

    // iterate over row blocks
    for (FBLAS_UINT l = 0; l < n_row_blks * n_col_blks; l++) {
      FBLAS_UINT row_idx = (l % n_row_blks);
      FBLAS_UINT col_idx = (l / n_row_blks);
      FBLAS_UINT start_row = offs[row_idx];
      FBLAS_UINT rblk_size = blks[row_idx];
      csr_tasks[l] = new CsrmmCmInMemTask(start_row, col_idx * col_blk_size,
                                          rblk_size, col_blk_size, m, n, k,
                                          ia_ptr, ja, a, b, c, alpha, beta);
      sched.add_task(csr_tasks[l]);
    }
    // sync and cleanup
    sleep_wait_for_complete(csr_tasks, n_row_blks * n_col_blks);
    for (FBLAS_UINT l = 0; l < (n_row_blks * n_col_blks); l++) {
      delete csr_tasks[l];
    }
    delete[] csr_tasks;
    delete[] ia_ptr;
  }

  void csrmm_no_trans_rm_im(FBLAS_UINT m, FBLAS_UINT n, FBLAS_UINT k,
                            FPTYPE alpha, FPTYPE beta, flash_ptr<FPTYPE> a,
                            flash_ptr<MKL_INT> ia, flash_ptr<MKL_INT> ja,
                            FPTYPE *b, FPTYPE *c) {
    // read `ia` into buffer
    MKL_INT *ia_ptr = new MKL_INT[m + 1];
    ia.fop->read(ia.foffset, (m + 1) * sizeof(MKL_INT), (void *) ia_ptr,
                 flash::dummy_std_func);
    std::vector<FBLAS_UINT> blks;
    std::vector<FBLAS_UINT> offs;
    FBLAS_UINT              cur_start = 0;
    for (; cur_start < m;) {
      FBLAS_UINT cblk_size =
          get_next_blk_size(ia_ptr + cur_start, m - cur_start,
                            SECTOR_LEN / sizeof(FPTYPE), CSRMM_RM_RBLK_SIZE);
      blks.push_back(cblk_size);
      offs.push_back(cur_start);
      cur_start += cblk_size;
      GLOG_DEBUG("choosing blk_size=", cblk_size);
    }
    FBLAS_UINT         n_row_blks = blks.size();
    FBLAS_UINT         col_blk_size = CSRMM_RM_CBLK_SIZE;
    FBLAS_UINT         n_col_blks = ROUND_UP(k, col_blk_size) / col_blk_size;
    CsrmmRmInMemTask **csr_tasks =
        new CsrmmRmInMemTask *[n_row_blks * n_col_blks];

    // iterate over row blocks
    for (FBLAS_UINT i = 0; i < n_row_blks; i++) {
      FBLAS_UINT start_row = offs[i];
      FBLAS_UINT rblk_size = blks[i];
      for (FBLAS_UINT j = 0; j < n_col_blks; j++) {
        csr_tasks[i * n_col_blks + j] = new CsrmmRmInMemTask(
            start_row, j * col_blk_size, rblk_size, col_blk_size, m, n, k,
            ia_ptr, ja, a, b, c, alpha, beta);
        sched.add_task(csr_tasks[i * n_col_blks + j]);
      }
    }
    // sync and cleanup
    sleep_wait_for_complete(csr_tasks, n_row_blks * n_col_blks);
    for (FBLAS_UINT l = 0; l < (n_row_blks * n_col_blks); l++) {
      delete csr_tasks[l];
    }
    delete[] csr_tasks;
    blks.clear();
    offs.clear();
    delete[] ia_ptr;

    // retain cache
  }

  void csrmm_trans_rm(FBLAS_UINT m, FBLAS_UINT n, FBLAS_UINT k, FPTYPE alpha,
                      FPTYPE beta, flash_ptr<FPTYPE> a, flash_ptr<MKL_INT> ia,
                      flash_ptr<MKL_INT> ja, flash_ptr<FPTYPE> b,
                      flash_ptr<FPTYPE> c) {
    // obtain `nnzs`
    MKL_INT *ia_ptr = new MKL_INT[m + 1];
    ia.fop->read(ia.foffset, (m + 1) * sizeof(MKL_INT), (void *) ia_ptr,
                 flash::dummy_std_func);
    FBLAS_UINT nnzs = ia_ptr[m] - ia_ptr[0];
    delete[] ia_ptr;

    flash_ptr<MKL_INT> ia_tr =
        flash_malloc<MKL_INT>((k + 1) * sizeof(MKL_INT), "ia_tr_temp");
    flash_ptr<MKL_INT> ja_tr =
        flash_malloc<MKL_INT>(nnzs * sizeof(MKL_INT), "ja_tr_temp");
    flash_ptr<FPTYPE> a_tr =
        flash_malloc<FPTYPE>((k + 1) * sizeof(MKL_INT), "a_tr_temp");

    // run csrcsc
    csrcsc(m, k, ia, ja, a, ia_tr, ja_tr, a_tr);

    // call csrmm_no_trans_rm
    csrmm_no_trans_rm(m, n, k, alpha, beta, a_tr, ia_tr, ja_tr, b, c);

    flash_free(ia_tr);
    flash_free(ja_tr);
    flash_free(a_tr);
  }

  void csrmm_trans_cm(FBLAS_UINT m, FBLAS_UINT n, FBLAS_UINT k, FPTYPE alpha,
                      FPTYPE beta, flash_ptr<FPTYPE> a, flash_ptr<MKL_INT> ia,
                      flash_ptr<MKL_INT> ja, flash_ptr<FPTYPE> b,
                      flash_ptr<FPTYPE> c) {
    // obtain `nnzs`
    MKL_INT *ia_ptr = new MKL_INT[m + 1];
    ia.fop->read(ia.foffset, (m + 1) * sizeof(MKL_INT), (void *) ia_ptr,
                 flash::dummy_std_func);
    FBLAS_UINT nnzs = ia_ptr[m] - ia_ptr[0];
    delete[] ia_ptr;

    flash_ptr<MKL_INT> ia_tr =
        flash_malloc<MKL_INT>((k + 1) * sizeof(MKL_INT), "ia_tr_temp");
    flash_ptr<MKL_INT> ja_tr =
        flash_malloc<MKL_INT>(nnzs * sizeof(MKL_INT), "ja_tr_temp");
    flash_ptr<FPTYPE> a_tr =
        flash_malloc<FPTYPE>((k + 1) * sizeof(MKL_INT), "a_tr_temp");

    // run csrcsc
    csrcsc(m, k, ia, ja, a, ia_tr, ja_tr, a_tr);

    // call csrmm_no_trans_cm
    csrmm_no_trans_cm(m, n, k, alpha, beta, a_tr, ia_tr, ja_tr, b, c);

    flash_free(ia_tr);
    flash_free(ja_tr);
    flash_free(a_tr);
  }
}  // namespace

namespace flash {
  FBLAS_INT csrmm(CHAR trans_a, FBLAS_UINT m, FBLAS_UINT n, FBLAS_UINT k,
                  FPTYPE alpha, FPTYPE beta, flash_ptr<FPTYPE> a,
                  flash_ptr<MKL_INT> ia, flash_ptr<MKL_INT> ja, CHAR ord_b,
                  flash_ptr<FPTYPE> b, flash_ptr<FPTYPE> c) {
    if (trans_a == 'T') {
      if (ord_b == 'C') {
        csrmm_trans_cm(m, n, k, alpha, beta, a, ia, ja, b, c);
      } else if (ord_b == 'R') {
        csrmm_trans_rm(m, n, k, alpha, beta, a, ia, ja, b, c);
      } else {
        GLOG_ERROR("unrecognized value for param: ord_b = ", ord_b);
        return -1;
      }
    } else if (trans_a == 'N') {
      if (ord_b == 'C') {
        csrmm_no_trans_cm2(m, n, k, alpha, beta, a, ia, ja, b, c);
      } else if (ord_b == 'R') {
        csrmm_no_trans_rm2(m, n, k, alpha, beta, a, ia, ja, b, c);
      } else {
        GLOG_ERROR("unrecognized value for param: ord_b = ", ord_b);
        return -1;
      }
    } else {
      GLOG_ERROR("unrecognized value for param: trans_a = ", trans_a);
      return -1;
    }
    return 0;
  }

  FBLAS_INT csrmm(CHAR trans_a, FBLAS_UINT m, FBLAS_UINT n, FBLAS_UINT k,
                  FPTYPE alpha, FPTYPE beta, flash_ptr<FPTYPE> a,
                  flash_ptr<MKL_INT> ia, flash_ptr<MKL_INT> ja, CHAR ord_b,
                  FPTYPE *b, FPTYPE *c) {
    if (trans_a == 'T') {
      GLOG_ERROR("csrmm in mem transpose not implemented");
      return -1;
    } else if (trans_a == 'N') {
      if (ord_b == 'C') {
        csrmm_no_trans_cm_im(m, n, k, alpha, beta, a, ia, ja, b, c);
      } else {
        csrmm_no_trans_rm_im(m, n, k, alpha, beta, a, ia, ja, b, c);
        return -1;
      }
    } else {
      GLOG_ERROR("unrecognized value for param: trans_a = ", trans_a);
      return -1;
    }
    return 0;
  }
}
