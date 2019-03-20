// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "bof_types.h"
#include "flash_blas.h"

#include <algorithm>
#include <vector>

using namespace flash;
flash::Logger logger("in_mem");

FPTYPE distsq(const FPTYPE *const p1_coords, const FPTYPE *const p2_coords,
              const FBLAS_UINT dim) {
  return mkl_dot(dim, p1_coords, 1, p1_coords, 1) +
         mkl_dot(dim, p2_coords, 1, p2_coords, 1) -
         2 * mkl_dot(dim, p1_coords, 1, p2_coords, 1);
}

void distsq_points_to_centers(
    const FBLAS_UINT dim, FBLAS_UINT ncenters, const FPTYPE *const centers,
    const FPTYPE *const centers_l2sq, FBLAS_UINT npoints,
    const FPTYPE *const points, const FPTYPE *const points_l2sq,
    FPTYPE *dist_matrix,
    FPTYPE *ones_vec = nullptr)  // Scratchspace of npoints size and init to 1.0
{
  bool ones_vec_alloc = false;
  if (ones_vec == nullptr) {
    ones_vec = new FPTYPE[npoints > ncenters ? npoints : ncenters];
    std::fill_n(ones_vec, npoints > ncenters ? npoints : ncenters,
                (FPTYPE) 1.0);
    ones_vec_alloc = true;
  }
  mkl_gemm(CblasColMajor, CblasTrans, CblasNoTrans, ncenters, npoints, dim,
           (FPTYPE) -2.0, centers, dim, points, dim, (FPTYPE) 0.0, dist_matrix,
           ncenters);
  mkl_gemm(CblasColMajor, CblasNoTrans, CblasTrans, ncenters, npoints, 1,
           (FPTYPE) 1.0, centers_l2sq, ncenters, ones_vec, npoints,
           (FPTYPE) 1.0, dist_matrix, ncenters);
  mkl_gemm(CblasColMajor, CblasNoTrans, CblasTrans, ncenters, npoints, 1,
           (FPTYPE) 1.0, ones_vec, ncenters, points_l2sq, npoints, (FPTYPE) 1.0,
           dist_matrix, ncenters);
  if (ones_vec_alloc)
    delete[] ones_vec;
}

void distsq_to_closest_center(
    const FBLAS_UINT ndims, FBLAS_UINT ncenters, const FPTYPE *const centers,
    const FPTYPE *const centers_l2sq, FBLAS_UINT npoints,
    const FPTYPE *const points, const FPTYPE *const points_l2sq,
    FPTYPE *const min_dist,
    FPTYPE *      ones_vec)  // Scratchspace of npoints size and init to 1.0
{
  FPTYPE *dist_matrix = new FPTYPE[ncenters * npoints];
  distsq_points_to_centers(ndims, ncenters, centers, centers_l2sq, npoints,
                           points, points_l2sq, dist_matrix, ones_vec);
#pragma omp parallel for
  for (FBLAS_INT d = 0; d < npoints; ++d) {
    FPTYPE min = FPTYPE_MAX;
    for (FBLAS_UINT c = 0; c < ncenters; ++c)
      if (dist_matrix[c + d * ncenters] < min)
        min = dist_matrix[c + d * ncenters];
    min_dist[d] = min > (FPTYPE) 0.0 ? min : (FPTYPE) 0.0;
    // TODO: Ugly round about for small distance errors;
  }
  delete[] dist_matrix;
}

void closest_centers(
    const FPTYPE *const points, const FBLAS_UINT ncenters,
    const FPTYPE *const centers, const FPTYPE *const points_l2sq,
    FBLAS_UINT *center_index, const FBLAS_UINT npoints, const FBLAS_UINT ndims,
    FPTYPE *const dist_matrix)  // Scratch init to ncenters*npoints size
{
  FPTYPE *const centers_l2sq = new FPTYPE[ncenters];
  for (FBLAS_UINT c = 0; c < ncenters; ++c)
    centers_l2sq[c] =
        mkl_dot(ndims, centers + c * ndims, 1, centers + c * ndims, 1);
  distsq_points_to_centers(ndims, ncenters, centers, centers_l2sq, npoints,
                           points, points_l2sq, dist_matrix);

#pragma omp parallel for
  for (FBLAS_INT d = 0; d < npoints; ++d)
    center_index[d] =
        (FBLAS_UINT) mkl_imin(ncenters, dist_matrix + d * ncenters, 1);
  delete[] centers_l2sq;
}

FPTYPE lloyds_iter(const FPTYPE *const points, const FBLAS_UINT ncenters,
                   FPTYPE *centers, const FPTYPE *const points_l2sq,
                   std::vector<FBLAS_UINT> *closest_points,
                   const FBLAS_UINT npoints, const FBLAS_UINT ndims,
                   bool weighted,  // If true, supply weights
                   const std::vector<size_t> &weights) {
  if (weighted)
    assert(weights.size() == npoints);

  bool return_point_partition = (closest_points != nullptr);

  FPTYPE *const     dist_matrix = new FPTYPE[ncenters * npoints];
  FBLAS_UINT *const closest_center = new FBLAS_UINT[npoints];
  closest_centers(points, ncenters, centers, points_l2sq, closest_center,
                  npoints, ndims, dist_matrix);

  if (closest_points == nullptr)
    closest_points = new std::vector<FBLAS_UINT>[ncenters];
  else
    for (FBLAS_UINT c = 0; c < ncenters; ++c)
      closest_points[c].clear();
  for (FBLAS_UINT d = 0; d < npoints; ++d)
    closest_points[closest_center[d]].push_back(d);
  memset(centers, 0, sizeof(FPTYPE) * ncenters * ndims);

#pragma omp parallel for
  for (FBLAS_INT c = 0; c < ncenters; ++c)
    if (weighted)
      for (auto iter = closest_points[c].begin();
           iter != closest_points[c].end(); ++iter)
        mkl_axpy(ndims, (FPTYPE)(weights[*iter]) / closest_points[c].size(),
                 points + (*iter) * ndims, 1, centers + c * ndims, 1);
    else
      for (auto iter = closest_points[c].begin();
           iter != closest_points[c].end(); ++iter)
        mkl_axpy(ndims, (FPTYPE)(1.0) / closest_points[c].size(),
                 points + (*iter) * ndims, 1, centers + c * ndims, 1);

  FBLAS_INT BUF_PAD = 32;
  FBLAS_INT CHUNK_SIZE = 8196;
  FBLAS_INT nchunks =
      npoints / CHUNK_SIZE + (npoints % CHUNK_SIZE == 0 ? 0 : 1);
  std::vector<FPTYPE> residuals(nchunks * BUF_PAD, 0.0);

#pragma omp parallel for
  for (FBLAS_INT chunk = 0; chunk < nchunks; ++chunk)
    for (FBLAS_UINT d = chunk * CHUNK_SIZE;
         d < npoints && d < (chunk + 1) * CHUNK_SIZE; ++d)
      residuals[chunk * BUF_PAD] +=
          (weighted ? weights[d] : (FPTYPE) 1.0) *
          distsq(points + d * ndims, centers + closest_center[d] * ndims,
                 ndims);

  if (!return_point_partition)
    delete[] closest_points;
  delete[] closest_center;
  delete[] dist_matrix;

  FPTYPE residual = 0.0;
  for (FBLAS_INT chunk = 0; chunk < nchunks; ++chunk)
    residual += residuals[chunk * BUF_PAD];

  return residual;
}

int main(int argc, char **argv) {
  LOG_ASSERT(logger, argc == 6,
             "Usage Mode : <exec> <points> <centers_in> "
             "<npoints> <ndims> <ncenters>");

  std::string points_fname = std::string(argv[1]);
  std::string centers_in_fname = std::string(argv[2]);

  // problem dimension
  FBLAS_UINT npoints = (FBLAS_UINT) std::stol(argv[3]);
  FBLAS_UINT ndims = (FBLAS_UINT) std::stol(argv[4]);
  FBLAS_UINT ncenters = (FBLAS_UINT) std::stol(argv[5]);

  float *points = new float[npoints * ndims];
  float *centers = new float[ndims * ncenters];

  LOG_INFO(logger, "Reading matrix A into memory");
  std::ifstream points_file(points_fname, std::ios::binary);
  points_file.read((char *) points, npoints * ndims * sizeof(float));
  points_file.close();
  LOG_INFO(logger, "Reading matrix B into memory");
  std::ifstream centers_in_file(centers_in_fname, std::ios::binary);
  centers_in_file.read((char *) centers, ndims * ncenters * sizeof(float));
  centers_in_file.close();

  FPTYPE *points_l2sq = new FPTYPE[npoints];
  for (FBLAS_INT p = 0; p < npoints; ++p)
    points_l2sq[p] =
        mkl_dot(ndims, points + p * ndims, 1, points + p * ndims, 1);

  for (FBLAS_UINT i = 0; i < 1; i++) {
    lloyds_iter(points, ncenters, centers, points_l2sq, nullptr, npoints, ndims,
                false, std::vector<size_t>());
  }

  LOG_INFO(logger, "Writing C to file");
  std::ofstream cout_file(centers_in_fname, std::ios::binary);
  cout_file.write((char *) centers, ncenters * ndims * sizeof(float));
  cout_file.close();

  // free memory
  delete[] points_l2sq;
  delete[] points;
  delete[] centers;
}
