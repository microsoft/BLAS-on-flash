// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "bof_types.h"
#include "flash_blas.h"
#include "lib_funcs.h"

#include <algorithm>
#include <vector>

using namespace flash;
flash::Logger logger("kmeans");

FPTYPE distsq(FPTYPE *p1_coords, FPTYPE *p2_coords, const FBLAS_UINT dim) {
  return mkl_dot(dim, p1_coords, 1, p1_coords, 1) +
         mkl_dot(dim, p2_coords, 1, p2_coords, 1) -
         2 * mkl_dot(dim, p1_coords, 1, p2_coords, 1);
}

void distsq_points_to_centers(
    const FBLAS_UINT dim, FBLAS_UINT ncenters, flash_ptr<FPTYPE> centers,
    FPTYPE *centers_l2sq, FBLAS_UINT npoints, flash_ptr<FPTYPE> points,
    FPTYPE *points_l2sq, flash_ptr<FPTYPE> dist_matrix,
    FPTYPE *ones_vec = nullptr)  // Scratchspace of npoints size and init to 1.0
{
  flash_ptr<FPTYPE> centers_l2sq_fptr(centers_l2sq);
  flash_ptr<FPTYPE> points_l2sq_fptr(points_l2sq);
  bool              ones_vec_alloc = false;
  if (ones_vec == nullptr) {
    ones_vec = new FPTYPE[npoints > ncenters ? npoints : ncenters];
    std::fill_n(ones_vec, npoints > ncenters ? npoints : ncenters,
                (FPTYPE) 1.0);
    ones_vec_alloc = true;
  }

  flash::kmeans('C', 'T', 'N', ncenters, npoints, dim, (FPTYPE) -2.0,
                (FPTYPE) 0.0, centers, points, dist_matrix, dim, dim, ncenters,
                centers_l2sq, points_l2sq, ones_vec);

  if (ones_vec_alloc) {
    delete[] ones_vec;
  }
}

void distsq_to_closest_center(
    const FBLAS_UINT ndims, FBLAS_UINT ncenters, flash_ptr<FPTYPE> centers,
    FPTYPE *centers_l2sq, FBLAS_UINT npoints, flash_ptr<FPTYPE> points,
    FPTYPE *points_l2sq, FPTYPE *const min_dist,
    FPTYPE *ones_vec)  // Scratchspace of npoints size and init to 1.0
{
  flash_ptr<FPTYPE> dist_matrix = flash::flash_malloc<FPTYPE>(
      ncenters * npoints * sizeof(FPTYPE), "dist_mat");
  distsq_points_to_centers(ndims, ncenters, centers, centers_l2sq, npoints,
                           points, points_l2sq, dist_matrix, ones_vec);
#pragma omp parallel for
  for (FBLAS_INT d = 0; d < npoints; ++d) {
    FPTYPE            min = FPTYPE_MAX;
    flash_ptr<FPTYPE> cur_pt_dists_fptr = (dist_matrix + d * ncenters);
    // WARNING :: gets the MMAP'ed pointer; might exceed system memory
    FPTYPE *cur_pt_dists = cur_pt_dists_fptr.get_raw_ptr();
    for (FBLAS_UINT c = 0; c < ncenters; ++c)
      if (cur_pt_dists[c] < min) {
        min = cur_pt_dists[c];
      }
    min_dist[d] = min > (FPTYPE) 0.0 ? min : (FPTYPE) 0.0;
    // TODO: Ugly round about for small distance errors;
  }
  // WARNING :: this might not delete the underlying allocated file
  flash::flash_free<FPTYPE>(dist_matrix);
}

void closest_centers(
    flash_ptr<FPTYPE> points, const FBLAS_UINT ncenters,
    flash_ptr<FPTYPE> centers, FPTYPE *points_l2sq, FBLAS_UINT *center_index,
    const FBLAS_UINT npoints, const FBLAS_UINT ndims,
    flash_ptr<FPTYPE> dist_matrix)  // Scratch init to ncenters*npoints size
{
  FPTYPE *const centers_l2sq = new FPTYPE[ncenters];
#pragma omp     parallel for
  for (FBLAS_UINT c = 0; c < ncenters; ++c) {
    // TIP :: cache centers[c] in cur_center_ptr to avoid mmap business
    FPTYPE *          cur_center_ptr = new FPTYPE[ndims];
    flash_ptr<FPTYPE> cur_center_fptr = (centers + c * ndims);
    // read current center
    flash::read_sync(cur_center_ptr, cur_center_fptr, ndims);

    // compute L2-square norm
    centers_l2sq[c] = mkl_dot(ndims, cur_center_ptr, 1, cur_center_ptr, 1);
    delete[] cur_center_ptr;
  }
  distsq_points_to_centers(ndims, ncenters, centers, centers_l2sq, npoints,
                           points, points_l2sq, dist_matrix);

#pragma omp parallel for
  for (FBLAS_INT d = 0; d < npoints; ++d) {
    flash_ptr<FPTYPE> cur_pt_dists_fptr = (dist_matrix + d * ncenters);
    FPTYPE *          cur_pt_dists_ptr = cur_pt_dists_fptr.get_raw_ptr();
    center_index[d] = (FBLAS_UINT) mkl_imin(ncenters, cur_pt_dists_ptr, 1);
  }
  delete[] centers_l2sq;
}

FPTYPE lloyds_iter(flash_ptr<FPTYPE> points, const FBLAS_UINT ncenters,
                   flash_ptr<FPTYPE> centers, FPTYPE *points_l2sq,
                   std::vector<FBLAS_UINT> *closest_points,
                   const FBLAS_UINT npoints, const FBLAS_UINT ndims,
                   bool weighted,  // If true, supply weights
                   const std::vector<size_t> &weights) {
  if (weighted)
    assert(weights.size() == npoints);

  bool return_point_partition = (closest_points != nullptr);

  flash_ptr<FPTYPE> dist_matrix = flash::flash_malloc<FPTYPE>(
      ncenters * npoints * sizeof(FPTYPE), "dist_mat");
  FBLAS_UINT *const closest_center = new FBLAS_UINT[npoints];
  closest_centers(points, ncenters, centers, points_l2sq, closest_center,
                  npoints, ndims, dist_matrix);

  if (closest_points == nullptr) {
    closest_points = new std::vector<FBLAS_UINT>[ncenters];
  } else {
    for (FBLAS_UINT c = 0; c < ncenters; ++c) {
      closest_points[c].clear();
    }
  }
  for (FBLAS_UINT d = 0; d < npoints; ++d) {
    closest_points[closest_center[d]].push_back(d);
  }
  flash::flash_memset<FPTYPE>(centers, 0, sizeof(FPTYPE) * ncenters * ndims);

  FPTYPE *cur_point_ptr = new FPTYPE[ndims];
  FPTYPE *cur_center_ptr = new FPTYPE[ndims];

  for (FBLAS_INT c = 0; c < ncenters; ++c) {
    // TIP :: cache centers[c] in cur_center_ptr to avoid mmap business
    flash_ptr<FPTYPE> cur_center_fptr = (centers + c * ndims);
    flash::read_sync(cur_center_ptr, cur_center_fptr, ndims);

    for (auto iter = closest_points[c].begin(); iter != closest_points[c].end();
         ++iter) {
      // TIP :: cache cur_point in cur_point_ptr to avoid mmap business
      flash_ptr<FPTYPE> cur_point_fptr = (points + (*iter) * ndims);
      // read current point
      flash::read_sync(cur_point_ptr, cur_point_fptr, ndims);

      if (weighted)
        mkl_axpy(ndims, (FPTYPE)(weights[*iter]) / closest_points[c].size(),
                 cur_point_ptr, 1, cur_center_ptr, 1);
      else
        mkl_axpy(ndims, (FPTYPE)(1.0) / closest_points[c].size(), cur_point_ptr,
                 1, cur_center_ptr, 1);
    }

    flash::write_sync(cur_center_fptr, cur_center_ptr, ndims);
  }

  delete[] cur_center_ptr;
  delete[] cur_point_ptr;

  FBLAS_INT BUF_PAD = 32;
  FBLAS_INT CHUNK_SIZE = 8196;
  FBLAS_INT nchunks =
      npoints / CHUNK_SIZE + (npoints % CHUNK_SIZE == 0 ? 0 : 1);
  std::vector<FPTYPE> residuals(nchunks * BUF_PAD, 0.0);

#pragma omp parallel for
  for (FBLAS_INT chunk = 0; chunk < nchunks; ++chunk) {
    for (FBLAS_UINT d = chunk * CHUNK_SIZE;
         d < npoints && d < (chunk + 1) * CHUNK_SIZE; ++d) {
      // WARNING :: MMAP performance hits
      FPTYPE *cur_pt = (points + d * ndims).get_raw_ptr();
      FPTYPE *cur_center = (centers + closest_center[d] * ndims).get_raw_ptr();
      residuals[chunk * BUF_PAD] += (weighted ? weights[d] : (FPTYPE) 1.0) *
                                    distsq(cur_pt, cur_center, ndims);
    }
  }

  if (!return_point_partition)
    delete[] closest_points;
  delete[] closest_center;
  flash::flash_free<FPTYPE>(dist_matrix);

  FPTYPE residual = 0.0;
  for (FBLAS_INT chunk = 0; chunk < nchunks; ++chunk)
    residual += residuals[chunk * BUF_PAD];

  return residual;
}

int main(int argc, char **argv) {
  LOG_ASSERT(logger, argc == 6,
             "Usage Mode : <exec> <points> <centers>"
             "<npoints> <ndims> <ncenters>");

  std::string points_fname = std::string(argv[1]);
  std::string centers_in_fname = std::string(argv[2]);

  // problem dimension
  FBLAS_UINT npoints = (FBLAS_UINT) std::stol(argv[3]);
  FBLAS_UINT ndims = (FBLAS_UINT) std::stol(argv[4]);
  FBLAS_UINT ncenters = (FBLAS_UINT) std::stol(argv[5]);

  // map files to flash_ptrs
  LOG_INFO(logger, "Mapping files");
  flash_ptr<FPTYPE> points =
      flash::map_file<FPTYPE>(points_fname, flash::Mode::READ);
  flash_ptr<FPTYPE> centers =
      flash::map_file<FPTYPE>(centers_in_fname, flash::Mode::READWRITE);
  FPTYPE *points_ptr = points.get_raw_ptr();
  FPTYPE *centers_ptr = centers.get_raw_ptr();

  FPTYPE *points_l2sq = new FPTYPE[npoints];
  for (FBLAS_INT p = 0; p < npoints; ++p) {
    points_l2sq[p] =
        mkl_dot(ndims, points_ptr + p * ndims, 1, points_ptr + p * ndims, 1);
  }

  for (FBLAS_UINT i = 0; i < 1; i++) {
    lloyds_iter(points, ncenters, centers, points_l2sq, nullptr, npoints, ndims,
                false, std::vector<size_t>());
  }

  // free memory
  delete[] points_l2sq;
  flash::unmap_file(points);
  flash::unmap_file(centers);
}
