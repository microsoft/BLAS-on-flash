// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <unistd.h>
#include "blas_bof_utils.h"
#include "flash_blas.h"
#include "tasks/csrgemv_task.h"
namespace flash {
  extern Scheduler sched;
}  // namespace flash

namespace {
  using namespace flash;
  void csrgemv_notrans_inmem(FBLAS_UINT m, FBLAS_UINT n, flash_ptr<FPTYPE> a,
                             MKL_INT* ia, flash_ptr<MKL_INT> ja, FPTYPE* b,
                             FPTYPE* c) {
    std::vector<FBLAS_UINT> blks;
    std::vector<FBLAS_UINT> offs;
    FBLAS_UINT              cur_start = 0;
    for (; cur_start < m;) {
      FBLAS_UINT cblk_size =
          get_next_blk_size(ia + cur_start, m - cur_start,
                            SECTOR_LEN / sizeof(FPTYPE), CSRMM_RM_RBLK_SIZE);
      blks.push_back(cblk_size);
      offs.push_back(cur_start);
      cur_start += cblk_size;
      GLOG_DEBUG("choosing blk_size=", cblk_size);
    }
    FBLAS_UINT n_blks = blks.size();
    auto**     tasks = new CsrGemvNoTransInMem*[n_blks];
    for (FBLAS_UINT i = 0; i < n_blks; i++) {
      FBLAS_UINT start_row = offs[i];
      FBLAS_UINT rblk_size = blks[i];
      tasks[i] =
          new CsrGemvNoTransInMem(start_row, m, n, rblk_size, ia, ja, a, b, c);
      sched.add_task(tasks[i]);
    }

    sleep_wait_for_complete(tasks, n_blks);
    for (FBLAS_UINT i = 0; i < n_blks; i++) {
      delete tasks[i];
    }
    delete[] tasks;
  }

  void csrgemv_trans_inmem(FBLAS_UINT m, FBLAS_UINT n, flash_ptr<FPTYPE> a,
                           MKL_INT* ia, flash_ptr<MKL_INT> ja, FPTYPE* b,
                           FPTYPE* c) {
    // mutex to synchronize access to `c` vector
    std::mutex              sync_mut;
    std::vector<FBLAS_UINT> blks;
    std::vector<FBLAS_UINT> offs;
    FBLAS_UINT              cur_start = 0;
    for (; cur_start < m;) {
      FBLAS_UINT cblk_size =
          get_next_blk_size(ia + cur_start, m - cur_start,
                            SECTOR_LEN / sizeof(FPTYPE), CSRMM_RM_RBLK_SIZE);
      blks.push_back(cblk_size);
      offs.push_back(cur_start);
      cur_start += cblk_size;
      GLOG_DEBUG("choosing blk_size=", cblk_size);
    }
    FBLAS_UINT n_blks = blks.size();
    memset(c, 0, n * sizeof(FPTYPE));
    auto** tasks = new CsrGemvTransInMem*[n_blks];
    for (FBLAS_UINT i = 0; i < n_blks; i++) {
      FBLAS_UINT start_row = offs[i];
      FBLAS_UINT rblk_size = blks[i];
      tasks[i] = new CsrGemvTransInMem(start_row, m, n, rblk_size, ia, ja, a, b,
                                       c, sync_mut);
      sched.add_task(tasks[i]);
    }
    sleep_wait_for_complete(tasks, n_blks);
    for (FBLAS_UINT i = 0; i < n_blks; i++) {
      delete tasks[i];
    }
    delete[] tasks;
  }
}  // namespace

namespace flash {
  FBLAS_INT csrgemv(CHAR trans_a, FBLAS_UINT m, FBLAS_UINT n,
                    flash_ptr<FPTYPE> a, flash_ptr<MKL_INT> ia,
                    flash_ptr<MKL_INT> ja, FPTYPE* b, FPTYPE* c) {
    auto* ia_ptr = new MKL_INT[m + 1];
    ia.fop->read(ia.foffset, (m + 1) * sizeof(MKL_INT), ia_ptr,
                 flash::dummy_std_func);
    if (trans_a == 'N') {
      csrgemv_notrans_inmem(m, n, a, ia_ptr, ja, b, c);
    } else if (trans_a == 'T') {
      csrgemv_trans_inmem(m, n, a, ia_ptr, ja, b, c);
    } else {
      GLOG_ERROR("csrgemv trans_a error : expected=N or T, found=", trans_a);
    }
    delete[] ia_ptr;
    return 0;
  }
}  // namespace flash