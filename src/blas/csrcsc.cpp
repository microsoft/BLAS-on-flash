// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <unistd.h>
#include <string>
#include <vector>
#include "blas_bof_utils.h"
#include "lib_funcs.h"
#include "tasks/csrcsc_task.h"

namespace {
  void fill_blocks(MKL_INT *offs, FBLAS_UINT n_rows,
                   std::vector<FBLAS_UINT> &blk_sizes,
                   std::vector<FBLAS_UINT> &offsets, FBLAS_UINT min_blk_size,
                   FBLAS_UINT max_blk_size) {
    FBLAS_UINT cur_start = 0;
    for (; cur_start < n_rows;) {
      FBLAS_UINT cblk_size = flash::get_next_blk_size(
          offs + cur_start, n_rows - cur_start, min_blk_size, max_blk_size);
      blk_sizes.push_back(cblk_size);
      offsets.push_back(cur_start);
      cur_start += cblk_size;
      GLOG_DEBUG("choosing blk_size=", cblk_size);
    }
  }
}  // namespace

namespace flash {
  // expect a scheduler to be defined in some compile unit
  extern Scheduler sched;

  FBLAS_INT csrcsc(FBLAS_UINT m, FBLAS_UINT n, flash_ptr<MKL_INT> ia,
                   flash_ptr<MKL_INT> ja, flash_ptr<FPTYPE> a,
                   flash_ptr<MKL_INT> ia_tr, flash_ptr<MKL_INT> ja_tr,
                   flash_ptr<FPTYPE> a_tr) {
    // first read `ia` into memory
    MKL_INT *ia_ptr = new MKL_INT[m + 1];
    flash::read_sync(ia_ptr, ia, m + 1);

    GLOG_DEBUG("Transposing nnzs=", ia_ptr[m]);
    GLOG_DEBUG("Using CSRCSC_RBLK_SIZE=", CSRCSC_RBLK_SIZE,
               ", CSRCSC_CBLK_SIZE=", CSRCSC_CBLK_SIZE);

    std::vector<FBLAS_UINT> rblk_sizes, rblk_offsets;
    ::fill_blocks(ia_ptr, m, rblk_sizes, rblk_offsets, 10, CSRCSC_RBLK_SIZE);

    FBLAS_UINT n_rblks = rblk_sizes.size();

    std::vector<SparseBlock> A_rblks(n_rblks);
    std::vector<SparseBlock> A_tr_cblks(n_rblks);
    BlockCsrCscTask **       transpose_tasks = new BlockCsrCscTask *[n_rblks];
    for (FBLAS_UINT i = 0; i < n_rblks; i++) {
      FBLAS_UINT rstart = rblk_offsets[i];
      FBLAS_UINT rend = rblk_offsets[i] + rblk_sizes[i];
      FBLAS_UINT blk_nnzs = ia_ptr[rend] - ia_ptr[rstart];
      A_rblks[i].offs = ia_ptr + rstart;
      A_rblks[i].idxs_fptr = ja + *A_rblks[i].offs;
      A_rblks[i].vals_fptr = a + *A_rblks[i].offs;
      A_rblks[i].nrows = m;
      A_rblks[i].ncols = n;
      A_rblks[i].start = rstart;
      A_rblks[i].blk_size = rblk_sizes[i];

      A_tr_cblks[i].offs = new MKL_INT[n + 1]();
      A_tr_cblks[i].idxs_fptr =
          flash_malloc<MKL_INT>(blk_nnzs * sizeof(MKL_INT),
                                std::string("blk_ja-") + std::to_string(i));
      A_tr_cblks[i].vals_fptr = flash_malloc<FPTYPE>(
          blk_nnzs * sizeof(FPTYPE), std::string("blk_a-") + std::to_string(i));
      A_tr_cblks[i].nrows = n;
      A_tr_cblks[i].ncols = m;
      A_tr_cblks[i].start = 0;
      A_tr_cblks[i].blk_size = n;

      transpose_tasks[i] = new BlockCsrCscTask(A_rblks[i], A_tr_cblks[i]);
      sched.add_task(transpose_tasks[i]);
    }

    sleep_wait_for_complete(transpose_tasks, n_rblks);
    sched.flush_cache();

    // clear memory for task objects
    for (FBLAS_UINT i = 0; i < n_rblks; i++) {
      delete transpose_tasks[i];
    }
    delete[] transpose_tasks;

    // compute nnzs in each block
    MKL_INT *ia_tr_ptr = new MKL_INT[n + 1]();
    for (const auto &blk : A_tr_cblks) {
      for (FBLAS_UINT i = 1; i <= n; i++) {
        ia_tr_ptr[i] += (blk.offs[i] - blk.offs[i - 1]);
      }
    }
    for (FBLAS_UINT i = 1; i <= n; i++) {
      ia_tr_ptr[i] += ia_tr_ptr[i - 1];
    }
    GLOG_ASSERT(ia_tr_ptr[n] == ia_ptr[m], "expected nnzs=", ia_ptr[m],
                " got nnzs=", ia_tr_ptr[n]);

    // compute block sizes
    std::vector<FBLAS_UINT> cblk_sizes, cblk_offsets;
    ::fill_blocks(ia_tr_ptr, n, cblk_sizes, cblk_offsets, 10, CSRCSC_CBLK_SIZE);
    FBLAS_UINT n_cblks = cblk_sizes.size();
    GLOG_DEBUG("Using n_cblks=", n_cblks);

    BlockMergeTask **merge_tasks = new BlockMergeTask *[n_cblks];
    // collect block offset information
    for (FBLAS_UINT j = 0; j < n_cblks; j++) {
      FBLAS_UINT cstart = cblk_offsets[j];
      FBLAS_UINT cblk_size = cblk_sizes[j];

      SparseBlock A_tr_rblk;
      A_tr_rblk.offs = ia_tr_ptr + cstart;
      A_tr_rblk.idxs_fptr = ja_tr + *A_tr_rblk.offs;
      A_tr_rblk.vals_fptr = a_tr + *A_tr_rblk.offs;
      A_tr_rblk.start = cstart;
      A_tr_rblk.blk_size = cblk_size;
      A_tr_rblk.nrows = n;
      A_tr_rblk.ncols = m;

      std::vector<SparseBlock> A_tr_rblks;
      for (FBLAS_UINT i = 0; i < n_rblks; i++) {
        SparseBlock &cblk = A_tr_cblks[i];
        SparseBlock  rblk;
        rblk.offs = cblk.offs + cstart;
        rblk.idxs_fptr = cblk.idxs_fptr + *rblk.offs;
        rblk.vals_fptr = cblk.vals_fptr + *rblk.offs;
        rblk.start = cstart;
        rblk.blk_size = cblk_size;
        rblk.nrows = n;
        rblk.ncols = m;
        A_tr_rblks.push_back(rblk);
      }

      merge_tasks[j] = new BlockMergeTask(A_tr_rblk, A_tr_rblks);
      sched.add_task(merge_tasks[j]);
    }

    // wait for SimpleMergeTasks to complete
    sleep_wait_for_complete(merge_tasks, n_cblks);
    sched.flush_cache();

    for (FBLAS_UINT i = 0; i < n_cblks; i++) {
      delete merge_tasks[i];
    }
    delete[] merge_tasks;

    // write ia_tr_ptr to disk
    flash::write_sync(ia_tr, ia_tr_ptr, n + 1);

    // free resources
    delete[] ia_ptr;
    delete[] ia_tr_ptr;
    for (auto &cblk : A_tr_cblks) {
      delete[] cblk.offs;
    }
    return 0;
  }

}  // namespace flash
