// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include "bof_types.h"
#include "bof_utils.h"
#include "mkl.h"
#include "pointers/pointer.h"
#include "tasks/task.h"

namespace flash {

  class KMeansTask : public BaseTask {
    flash_ptr<FPTYPE>       matA, matB, matC;
    FPTYPE *                c_l2sq, *p_l2sq, *ones;
    MKL_INT                 a_nrows, a_ncols, b_ncols;
    MKL_INT                 lda_a, lda_b, lda_c;
    FPTYPE                  alpha, beta;
    decltype(CblasNoTrans)  trans_a, trans_b;
    decltype(CblasRowMajor) mat_ord;

   public:
    KMeansTask(flash_ptr<FPTYPE> a, flash_ptr<FPTYPE> b, flash_ptr<FPTYPE> c,
               FBLAS_UINT a_nrows, FBLAS_UINT a_ncols, FBLAS_UINT b_ncols,
               FBLAS_UINT ptr_offset[3], FBLAS_UINT lda_a, FBLAS_UINT lda_b,
               FBLAS_UINT lda_c, StrideInfo stride_info[3], FPTYPE alpha,
               FPTYPE beta, CHAR trans_a, CHAR trans_b, CHAR mat_ord,
               FPTYPE* c_l2sq, FPTYPE* p_l2sq, FPTYPE* ones) {
      this->alpha = alpha;
      this->beta = beta;
      this->c_l2sq = c_l2sq;
      this->p_l2sq = p_l2sq;
      this->ones = ones;
      this->trans_a = (trans_a == 'T' ? CblasTrans : CblasNoTrans);
      this->trans_b = (trans_b == 'T' ? CblasTrans : CblasNoTrans);
      this->mat_ord = (mat_ord == 'R' ? CblasRowMajor : CblasColMajor);
      this->matA = a + ptr_offset[0];
      this->matB = b + ptr_offset[1];
      this->matC = c + ptr_offset[2];

      this->a_nrows = a_nrows;
      this->a_ncols = a_ncols;
      this->b_ncols = b_ncols;
      this->lda_a = lda_a;
      this->lda_b = lda_b;
      this->lda_c = lda_c;

      this->add_read(this->matA, stride_info[0]);
      this->add_read(this->matB, stride_info[1]);
      this->add_write(this->matC, stride_info[2]);
    }

    void execute() {
      mkl_set_num_threads_local(GEMM_MKL_NTHREADS);
      FPTYPE* a_ptr = (FPTYPE*) in_mem_ptrs[matA];
      FPTYPE* b_ptr = (FPTYPE*) in_mem_ptrs[matB];
      FPTYPE* c_ptr = (FPTYPE*) in_mem_ptrs[matC];
      GLOG_ASSERT(a_ptr != NULL, "null a_ptr");
      GLOG_ASSERT(b_ptr != NULL, "null b_ptr");
      GLOG_ASSERT(c_ptr != NULL, "null c_ptr");
      GLOG_DEBUG("MKL params : trans_a:", trans_a == CblasTrans ? 'T' : 'N',
                 ", trans_b:", trans_b == CblasTrans ? 'T' : 'N',
                 ", a_nrows:", a_nrows, ", b_ncols:", b_ncols,
                 ", a_ncols:", a_ncols, ", alpha:", alpha, ", beta:", beta,
                 ", lda_a:", lda_a, ", lda_b:", lda_b, ", lda_c:", lda_c);

      // Determine parameters for MKL call
      mkl_gemm(mat_ord, trans_a, trans_b,          // ordering
               a_nrows, b_ncols, a_ncols,          // sizes
               alpha, a_ptr, lda_a, b_ptr, lda_b,  // input
               beta, c_ptr, lda_c);                // output

      mkl_gemm(mat_ord, CblasNoTrans, CblasTrans,    // ordering
               a_nrows, b_ncols, 1,                  // sizes
               1.0, c_l2sq, a_nrows, ones, b_ncols,  // input
               1.0, c_ptr, lda_c);                   // output

      mkl_gemm(mat_ord, CblasNoTrans, CblasTrans,    // ordering
               a_nrows, b_ncols, 1,                  // sizes
               1.0, ones, a_nrows, p_l2sq, b_ncols,  // input
               1.0, c_ptr, lda_c);                   // output
    }

    FBLAS_UINT size() {
      FBLAS_UINT a_mem = a_nrows * a_ncols * sizeof(FPTYPE);
      FBLAS_UINT b_mem = a_ncols * b_ncols * sizeof(FPTYPE);
      FBLAS_UINT c_mem = a_nrows * b_ncols * sizeof(FPTYPE);

      return (a_mem + b_mem + c_mem);
    }
  };
}  // namespace flash
