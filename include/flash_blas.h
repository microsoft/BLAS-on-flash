// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <functional>
#include "bof_types.h"
#include "bof_logger.h"
#include "pointers/allocator.h"
#include "pointers/pointer.h"

namespace flash {

  // C = alpha*A*B + beta*C
  FBLAS_INT gemm(CHAR mat_ord, CHAR trans_a, CHAR trans_b, FBLAS_UINT m,
                 FBLAS_UINT n, FBLAS_UINT k, FPTYPE alpha, FPTYPE beta,
                 flash_ptr<FPTYPE> a, flash_ptr<FPTYPE> b, flash_ptr<FPTYPE> c,
                 FBLAS_UINT lda_a = 0, FBLAS_UINT lda_b = 0,
                 FBLAS_UINT lda_c = 0);

  FBLAS_INT kmeans(CHAR mat_ord, CHAR trans_a, CHAR trans_b, FBLAS_UINT m,
                   FBLAS_UINT n, FBLAS_UINT k, FPTYPE alpha, FPTYPE beta,
                   flash_ptr<FPTYPE> a, flash_ptr<FPTYPE> b,
                   flash_ptr<FPTYPE> c, FBLAS_UINT lda_a, FBLAS_UINT lda_b,
                   FBLAS_UINT lda_c, FPTYPE* c_l2sq, FPTYPE* p_l2sq,
                   FPTYPE* ones);

  // y = alpha*A*x + beta*y
  FBLAS_INT gemv(CHAR mat_ord, CHAR trans_a, FBLAS_UINT m, FBLAS_UINT n,
                 FPTYPE alpha, FPTYPE beta, flash_ptr<FPTYPE> a,
                 flash_ptr<FPTYPE> x, flash_ptr<FPTYPE> y);

  // - C = alpha*A*B + beta*C
  // - C = alpha*A^T*B + beta*C
  // * A : m x n, is in CSR format
  // * B : n x k, is dense [RM|CM]
  // * C : m x k, is dense [RM|CM]
  FBLAS_INT csrmm(CHAR trans_a, FBLAS_UINT m, FBLAS_UINT n, FBLAS_UINT k,
                  FPTYPE alpha, FPTYPE beta, flash_ptr<FPTYPE> a,
                  flash_ptr<MKL_INT> ia, flash_ptr<MKL_INT> ja, CHAR ord_b,
                  flash_ptr<FPTYPE> b, flash_ptr<FPTYPE> c);

  // in-memory variant with `B` and `C` in memory
  FBLAS_INT csrmm(CHAR trans_a, FBLAS_UINT m, FBLAS_UINT n, FBLAS_UINT k,
                  FPTYPE alpha, FPTYPE beta, flash_ptr<FPTYPE> a,
                  flash_ptr<MKL_INT> ia, flash_ptr<MKL_INT> ja, CHAR ord_b,
                  FPTYPE* b, FPTYPE* c);

  // A : CSR(ia, ja, a, m, n) -> A^T : CSR(ia_tr, ja_tr, a_tr, n, m)
  FBLAS_INT csrcsc(FBLAS_UINT m, FBLAS_UINT n, flash_ptr<MKL_INT> ia,
                   flash_ptr<MKL_INT> ja, flash_ptr<FPTYPE> a,
                   flash_ptr<MKL_INT> ia_tr, flash_ptr<MKL_INT> ja_tr,
                   flash_ptr<FPTYPE> a_tr);

  // A : CSR(ia, ja, a, m, n)
  FBLAS_INT csrgemv(CHAR trans_a, FBLAS_UINT m, FBLAS_UINT n,
                    flash_ptr<FPTYPE> a, flash_ptr<MKL_INT> ia,
                    flash_ptr<MKL_INT> ja, FPTYPE* b, FPTYPE* c);

  // parallel external memory sort
  // implements Sample Sort
  template<typename T, typename Comparator = std::less<T>>
  extern FBLAS_INT sort(flash_ptr<T> in_fptr, flash_ptr<T> out_fptr,
                        FBLAS_UINT n_vals, Comparator cmp = std::less<T>());

  // reduce file `fptr` using `reducer`
  template<typename T>
  extern T reduce(flash_ptr<T> fptr, FBLAS_UINT len, T& id,
                  std::function<T(T&, T&)>& reducer);

  // map `InType` to `OutType`
  template<typename InType, typename OutType>
  extern FBLAS_INT map(flash_ptr<InType> in_fptr, flash_ptr<OutType> out_fptr,
                       FBLAS_UINT                             len,
                       std::function<OutType(const InType&)>& mapper);
}  // namespace flash

#include "map_reduce.tpp"
#include "sort.tpp"
