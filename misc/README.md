# BLAS-on-Flash MISC Files
This folder contains misc scripts for testing purposes.
- `dense_create.cpp` -> `../bin/dense_create <FILE_NAME> <NUM_ROWS> <NUM_COLS> <FILL_MODE>` creates a file named `FILE_NAME` containing a dense FP32 matrix with `NUM_ROWS` rows and `NUM_COLS` columns. 
    - `FILL_MODE == r` => elements generated using `rand_r()`
    - `FILL_MODE == s` -> elements are in integers in `[0, 9]`
    - `FILL_MODE == z` -> elements are zeros

- `sparse_create.cpp` -> `../bin/sparse_create <FILE_NAME> <NUM_ROWS> <NUM_COLS> <SPARSITY>` creates a sparse matrix `A` with `NUM_ROWS` rows and `NUM_COLS` cols containing approximately `NNZS = NUM_ROWS * NUM_COLS * SPARSITY` elements (`SPARSITY < 1.0`). Elements are generated using `rand_r()` and the resulting matrix is stored in a Compressed Sparse Row (CSR) format in 3 files:
    - `FILE_NAME.csr` -> contains all non-zero values in `A`; Size = `NNZS * sizeof(float)` bytes
    - `FILE_NAME.col` -> contains indices of non-zero values in `A`; Size = `NNZS * sizeof(MKL_INT)` bytes
    - `FILE_NAME.off` -> contains the offsets array for CSR form of `A`; Size = `(NUM_ROWS + 1) * sizeof(MKL_INT)` bytes
    - For more information on the CSR format for storing Sparse Matrices, refer to <https://www5.in.tum.de/lehre/vorlesungen/parnum/WS10/PARNUM_6.pdf>

- `flash_file_handle.cpp` -> `../bin/flash_file_handle_test <TMP_FILE> <TMP_FILE_SIZE>` tests the flash file handle according parameters specified in `../CMakeLists.txt`. A temporary file of size `TMP_FILE_SIZE` is created at `TMP_FILE` and filled natural numbers of `FBLAS_UINT` type. The executable tests 4 key functionalities of `flash::FlashFileHandle`:
    - `read()` - Sequential read, 1 request (logically, but library might split into multiple depending on `MAX_CHUNK_SIZE` in `../src/file_handles/flash_file_handle.cpp`). 
    - `write()` - Sequential write, 1 request
    - `sread()` - Strided read, multiple requests
    - `swrite()` - Strided write, multiple requests

- `gemm_run.sh` -> tests correctness of `gemm()` by generating random matrices
# Credits
`dense_create.cpp` and `sparse_create.cpp` were contributed by [Srajan Garg](https://github.com/srajangarg)
