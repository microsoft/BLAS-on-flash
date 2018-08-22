// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <cfloat>
#include <cstdint>
#include "mkl.h"

// By deault, compile Single Precision kernels
#define FP_SINGLE_PRECISION

// USE 64-bit INT and UINT values by default
typedef int64_t  FBLAS_INT;
typedef uint64_t FBLAS_UINT;
typedef char     CHAR;

#ifdef FP_SINGLE_PRECISION
typedef float  FPTYPE;
typedef double LONGFPTYPE;
#define mkl_gemm cblas_sgemm
#define mkl_gemv cblas_sgemv
#define mkl_axpy cblas_saxpy
#define mkl_csrmm mkl_scsrmm
#define mkl_csrcsc mkl_scsrcsc
#define mkl_csrgemv mkl_cspblas_scsrgemv
#define mkl_dot cblas_sdot
#define mkl_imin cblas_isamin
#define FPTYPE_MAX FLT_MAX
#else
typedef double      FPTYPE;
typedef long double LONGFPTYPE;
#define mkl_gemm cblas_dgemm
#define mkl_gemv cblas_dgemv
#define mkl_axpy cblas_daxpy
#define mkl_csrmm mkl_dcsrmm
#define mkl_csrcsc mkl_dcsrcsc
#define mkl_csrgemv mkl_cspblas_dcsrgemv
#define mkl_dot cblas_ddot
#define mkl_imin cblas_idamin
#define FPTYPE_MAX DBL_MAX
#endif  // FP_SINGLE_PRECISION
