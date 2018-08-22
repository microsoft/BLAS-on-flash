// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <parallel/algorithm>
#include "tasks/task.h"
#include "types.h"

namespace flash {
  struct SparseBlock {
    // Offsets (Row/Col)
    MKL_INT *offs = nullptr;

    // Bon-zero indices (on flash)
    flash_ptr<MKL_INT> idxs_fptr;
    MKL_INT *          idxs_ptr = nullptr;

    // Non-zero vals (on flash)
    flash_ptr<FPTYPE> vals_fptr;
    FPTYPE *          vals_ptr = nullptr;

    // BLOCK DESCRIPTORS
    // Block start (Row/Col)
    MKL_INT start;
    // Matrix Dims (CSR/CSC)
    MKL_INT nrows;
    MKL_INT ncols;
    // Block size (Row/Col)
    MKL_INT blk_size;

    SparseBlock() {
      this->offs = nullptr;
      this->idxs_ptr = nullptr;
      this->vals_ptr = nullptr;
      this->start = 0;
      this->nrows = 0;
      this->ncols = 0;
      this->blk_size = 0;
    }

    SparseBlock(const SparseBlock &other) {
      this->offs = other.offs;
      this->idxs_fptr = other.idxs_fptr;
      this->vals_fptr = other.vals_fptr;
      this->idxs_ptr = other.idxs_ptr;
      this->vals_ptr = other.vals_ptr;
      this->start = other.start;
      this->nrows = other.nrows;
      this->ncols = other.ncols;
      this->blk_size = other.blk_size;
    }
  };

  // Given a <flash_ptr, in_mem_ptr> mapping, obtain indices
  // and values pointers for given SparseBlock
  inline void fill_sparse_block_ptrs(
      std::unordered_map<flash::flash_ptr<void>, void *, flash::FlashPtrHasher,
                         flash::FlashPtrEq> &in_mem_ptrs,
      SparseBlock &                          blk) {
    if (in_mem_ptrs.find(blk.idxs_fptr) == in_mem_ptrs.end()) {
      GLOG_FATAL("idxs fptr not found in in_mem_ptrs");
    }
    if (in_mem_ptrs.find(blk.vals_fptr) == in_mem_ptrs.end()) {
      GLOG_FATAL("vals fptr not found in in_mem_ptrs");
    }
    blk.idxs_ptr = (decltype(blk.idxs_ptr)) in_mem_ptrs[blk.idxs_fptr];
    blk.vals_ptr = (decltype(blk.vals_ptr)) in_mem_ptrs[blk.vals_fptr];
  }

  // for sparse matrices in CSR format only
  inline FBLAS_UINT get_next_blk_size(MKL_INT *offs_ptr, MKL_INT nrows,
                                      MKL_INT min_size, MKL_INT max_size) {
    FBLAS_UINT max_nnzs = MAX_NNZS;
    FBLAS_UINT blk_size = min_size;
    while (blk_size < (FBLAS_UINT) nrows &&
           ((FBLAS_UINT)(offs_ptr[blk_size] - offs_ptr[0]) <= max_nnzs)) {
      blk_size++;
    }

    return std::min(blk_size, (FBLAS_UINT) max_size);
  }

  inline void fill_blocks(MKL_INT *offs, FBLAS_UINT n_rows,
                          std::vector<FBLAS_UINT> &blk_sizes,
                          std::vector<FBLAS_UINT> &offsets,
                          FBLAS_UINT min_blk_size, FBLAS_UINT max_blk_size) {
    FBLAS_UINT cur_start = 0;
    while (cur_start < n_rows) {
      FBLAS_UINT cblk_size = flash::get_next_blk_size(
          offs + cur_start, n_rows - cur_start, min_blk_size, max_blk_size);
      blk_sizes.push_back(cblk_size);
      offsets.push_back(cur_start);
      cur_start += cblk_size;
      GLOG_DEBUG("choosing blk_size=", cblk_size);
    }
  }

  // to be run in DEBUG mode only
  inline void verify_csr_block(const SparseBlock &blk,
                               bool               one_based_indexing) {
    GLOG_ASSERT_LE(blk.blk_size, blk.nrows);
    GLOG_ASSERT_LE(blk.start, blk.nrows);
    GLOG_ASSERT_LE(blk.start + blk.blk_size, blk.nrows);
    GLOG_ASSERT_NOT_NULL(blk.offs);
    GLOG_ASSERT_NOT_NULL(blk.idxs_ptr);
    GLOG_ASSERT_NOT_NULL(blk.vals_ptr);

    if (one_based_indexing) {
      GLOG_ASSERT_EQ(blk.offs[0], (MKL_INT) 1);
    } else {
      GLOG_ASSERT_EQ(blk.offs[0], (MKL_INT) 0);
    }

    FBLAS_UINT nnzs_processed = 0;
    for (FBLAS_UINT i = 0; i < blk.blk_size; i++) {
      FBLAS_UINT row_offset = blk.offs[i] - blk.offs[0];
      FBLAS_UINT row_nnzs = blk.offs[i + 1] - blk.offs[i];
      nnzs_processed += row_nnzs;
      MKL_INT *row_idxs_ptr = blk.idxs_ptr + row_offset;

      // assert col indices are sorted
      for (FBLAS_UINT j = 0; j < row_nnzs - 1; j++) {
        if (one_based_indexing) {
          GLOG_ASSERT_LE((FBLAS_UINT) row_idxs_ptr[j], blk.ncols);
        } else {
          GLOG_ASSERT_LT((FBLAS_UINT) row_idxs_ptr[j], blk.ncols);
        }
        GLOG_ASSERT_LE(row_idxs_ptr[j], row_idxs_ptr[j + 1]);
      }
      if (one_based_indexing) {
        GLOG_ASSERT_LE((FBLAS_UINT) row_idxs_ptr[row_nnzs - 1], blk.ncols);
      } else {
        GLOG_ASSERT_LT((FBLAS_UINT) row_idxs_ptr[row_nnzs - 1], blk.ncols);
      }
    }

    GLOG_ASSERT_EQ(nnzs_processed,
                   (FBLAS_UINT)(blk.offs[blk.blk_size] - blk.offs[0]));
    GLOG_DEBUG("CSR Block Verification passed");
  }
}  // namespace flash
