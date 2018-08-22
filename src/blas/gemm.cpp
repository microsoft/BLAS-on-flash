// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <cassert>
#include <vector>
#include "flash_blas.h"
#include "scheduler/scheduler.h"
#include "tasks/gemm_task.h"
#include "types.h"
#include "utils.h"

using std::vector;
using std::min;
using std::max;

template<typename T>
using vec2 = vector<vector<T>>;
template<typename T>
using vec3 = vector<vec2<T>>;

namespace flash {
  extern Scheduler sched;

  FBLAS_INT gemm(CHAR mat_ord, CHAR trans_a, CHAR trans_b, FBLAS_UINT m,
                 FBLAS_UINT n, FBLAS_UINT k, FPTYPE alpha, FPTYPE beta,
                 flash_ptr<FPTYPE> a, flash_ptr<FPTYPE> b, flash_ptr<FPTYPE> c,
                 FBLAS_UINT lda_a, FBLAS_UINT lda_b, FBLAS_UINT lda_c) {
    GLOG_DEBUG("parameters: mat_ord=", mat_ord, ", trans_a=", trans_a,
               ", trans_b=", trans_b, ", m=", m, ", n=", n, ", k=", k,
               ", alpha=", alpha, ", beta=", beta);

    GLOG_ASSERT(mat_ord == 'R' || mat_ord == 'C', "mat_ord must be 'C' or 'R'");
    GLOG_ASSERT(trans_a == 'N' || trans_a == 'T', "trans_a must be 'T' or 'N'");
    GLOG_ASSERT(trans_b == 'N' || trans_b == 'T', "trans_b must be 'T' or 'N'");

    bool       transA = trans_a == 'T';
    bool       transB = trans_b == 'T';
    bool       colMajor = mat_ord == 'C';
    bool       swapMat[3] = {false, false, false};
    FBLAS_UINT NUM_B[3], MKN_B[3], ROW[3], COL[3];
    FBLAS_UINT MKN[3] = {m, k, n}, LDA[3] = {lda_a, lda_b, lda_c};

    for (int i = 0; i < 3; i++) {
      ROW[i] = min(i, (i + 1) % 3), COL[i] = max(i, (i + 1) % 3);
      // TODO : own blocking code
      MKN_B[i] = std::min((FBLAS_UINT) GEMM_BLK_SIZE, MKN[i]);
    }

    if (transA ^ colMajor)
      swapMat[0] = true;
    if (transB ^ colMajor)
      swapMat[1] = true;
    if (colMajor)
      swapMat[2] = true;

    for (int i = 0; i < 3; i++)
      if (swapMat[i])
        std::swap(ROW[i], COL[i]);

    for (int i = 0; i < 3; i++) {
      if (LDA[i] == 0)
        LDA[i] = MKN[COL[i]];
      GLOG_ASSERT(LDA[i] >= MKN[COL[i]], "lda specified too small");
    }

    for (int i = 0; i < 3; i++) {
      FBLAS_UINT div = (MKN[i] / MKN_B[i]);
      if (MKN[i] - div * MKN_B[i] < SECTOR_LEN / sizeof(FPTYPE))
        NUM_B[i] = div;
      else
        NUM_B[i] = div + 1;
    }

    GLOG_DEBUG("blocking info: a_nrow_blks=", NUM_B[0],
               ", a_ncol_blks=", NUM_B[1], ", b_ncol_blks=", NUM_B[2]);

    vec3<GemmTask *> tasks(
        NUM_B[1], vec2<GemmTask *>(NUM_B[0], vector<GemmTask *>(NUM_B[2])));

    for (FBLAS_UINT l = 0; l < NUM_B[1]; l++) {
      for (FBLAS_UINT i = 0; i < NUM_B[0]; i++) {
        for (FBLAS_UINT j = 0; j < NUM_B[2]; j++) {
          StrideInfo stride_info[3];
          FBLAS_UINT indices[3] = {i, l, j};
          FBLAS_UINT ptr_offset[3], IKJ_NUM[3];

          for (int d = 0; d < 3; d++) {
            if (indices[d] == NUM_B[d] - 1)
              IKJ_NUM[d] = MKN[d] - indices[d] * MKN_B[d];
            else
              IKJ_NUM[d] = MKN_B[d];
          }

          for (int mat = 0; mat < 3; mat++) {
            FBLAS_UINT blkx = indices[ROW[mat]];
            FBLAS_UINT blky = indices[COL[mat]];

            FBLAS_UINT m_b = blkx * MKN_B[ROW[mat]];
            FBLAS_UINT n_b = blky * MKN_B[COL[mat]];

            FBLAS_UINT m_num = IKJ_NUM[ROW[mat]];
            FBLAS_UINT n_num = IKJ_NUM[COL[mat]];

            stride_info[mat].n_strides = m_num;
            stride_info[mat].len_per_stride = n_num * sizeof(FPTYPE);
            stride_info[mat].stride = LDA[mat] * sizeof(FPTYPE);

            ptr_offset[mat] = m_b * LDA[mat] + n_b;
          }

          if (l > 0)
            beta = 1.0;

          tasks[l][i][j] = new GemmTask(
              a, b, c, IKJ_NUM[0], IKJ_NUM[1], IKJ_NUM[2], ptr_offset,
              IKJ_NUM[COL[0]], IKJ_NUM[COL[1]], IKJ_NUM[COL[2]], stride_info,
              alpha, beta, trans_a, trans_b, mat_ord);

          if (l > 0) {
            tasks[l][i][j]->add_parent(tasks[l - 1][i][j]->get_id());
            GLOG_DEBUG("adding dependency:", tasks[l - 1][i][j]->get_id(), "->",
                       tasks[l][i][j]->get_id());
          }
        }
      }
    }

    // TODO : change once /blas-on-flash/issues/40 is complete
    /*
    for (FBLAS_UINT it = 1; it < NUM_B[0] * NUM_B[2]; it += 2) {
      if (colMajor) {
        FBLAS_UINT i = it % NUM_B[0], j = it / NUM_B[0];
        FBLAS_UINT i_n = (it + 1) % NUM_B[0], j_n = (it + 1) / NUM_B[0];
        FBLAS_UINT i_p = (it - 1) % NUM_B[0], j_p = (it - 1) / NUM_B[0];

        if (it + 1 < NUM_B[0] * NUM_B[2]) {
          tasks[0][i][j]->add_parent(tasks[NUM_B[1] - 1][i_n][j_n]->get_id());
          GLOG_DEBUG("adding dependency:", tasks[0][i_n][j_n]->get_id(), "->",
                     tasks[NUM_B[1] - 1][i][j]->get_id());
        }
        tasks[0][i][j]->add_parent(tasks[NUM_B[1] - 1][i_p][j_p]->get_id());
        GLOG_DEBUG("adding dependency:", tasks[0][i_p][j_p]->get_id(), "->",
                   tasks[NUM_B[1] - 1][i][j]->get_id());

      } else {
        FBLAS_UINT i = it / NUM_B[2], j = it % NUM_B[2];
        FBLAS_UINT i_n = (it + 1) / NUM_B[2], j_n = (it + 1) % NUM_B[2];
        FBLAS_UINT i_p = (it - 1) / NUM_B[2], j_p = (it - 1) % NUM_B[2];

        if (it + 1 < NUM_B[0] * NUM_B[2]) {
          tasks[0][i][j]->add_parent(tasks[NUM_B[1] - 1][i_n][j_n]->get_id());
          GLOG_DEBUG("adding dependency:", tasks[0][i_n][j_n]->get_id(), "->",
                     tasks[NUM_B[1] - 1][i][j]->get_id());
        }
        tasks[0][i][j]->add_parent(tasks[NUM_B[1] - 1][i_p][j_p]->get_id());
        GLOG_DEBUG("adding dependency:", tasks[0][i_p][j_p]->get_id(), "->",
                   tasks[NUM_B[1] - 1][i][j]->get_id());
      }
    }
    if (colMajor) {
      for (FBLAS_UINT c = 0; c < NUM_B[2]; c++)
        if (NUM_B[0] != 1 and (NUM_B[0] & 1) != 0)
          tasks[0][NUM_B[0] - 1][c]->add_parent(
              tasks[NUM_B[1] - 1][0][c]->get_id());
    } else {
      for (FBLAS_UINT r = 0; r < NUM_B[0]; r++)
        if (NUM_B[2] != 1 and (NUM_B[2] & 1) != 0)
          tasks[0][r][NUM_B[2] - 1]->add_parent(
              tasks[NUM_B[1] - 1][r][0]->get_id());
    }

    */
    for (FBLAS_UINT l = 0; l < NUM_B[1]; l++) {
      for (FBLAS_UINT i = 0; i < NUM_B[0]; i++) {
        for (FBLAS_UINT j = 0; j < NUM_B[2]; j++) {
          GLOG_DEBUG("added task[", l, ", ", i, ", ", j,
                     "]: addr=", tasks[l][i][j]);
          sched.add_task(tasks[l][i][j]);
        }
      }
    }

    for (FBLAS_UINT l = 0; l < NUM_B[1]; l++) {
      for (FBLAS_UINT i = 0; i < NUM_B[0]; i++) {
        for (FBLAS_UINT j = 0; j < NUM_B[2]; j++) {
          while (tasks[l][i][j]->get_status() != Complete) {
            ::usleep(100);
          }
          GLOG_PASS("task[", l, ", ", i, ", ", j, "] complete");
          delete tasks[l][i][j];
          GLOG_PASS("task[", i, ", ", j, ", ", l, "] deleted");
        }
      }
    }

    // flush cache
    sched.flush_cache();
    return 0;
  }
}  // namespace flash
