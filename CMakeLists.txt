cmake_minimum_required(VERSION 3.0.0)
project(blas-on-flash)
# Program config flags
## Scheduler Config
# N_READ_THR=[4] 							:	number of read threads
# N_COMPUTE_THR=[7] 					:	number of compute threads
# N_WRITE_THR=[4] 						:	number of write threads
## FlashFileHandle Config
# MAX_SIMUL_REQS=[512]				:	number of outstanding AIO requests
# MAX_STRIDES=[256] 					:	number of strides in one write/read call
# MAX_EVENTS=[512] 						:	number of events to reap from kernel
# SECTOR_LEN=[512]						:	sector size(logical) for disk
# IS_ALIGNED=[IS_512_ALIGNED]	:	alignment function
## _gemm config
# GEMM_BLK_SIZE=[4096]				: block size
# GEMM_MKL_NTHREADS=[4]				: # of MKL threads per _gemm task
## _csrcsc config
# OMP_CHUNK_SIZE=[32768]			: chunking for OpenMP
# CSRCSC_RBLK_SIZE=[65536]		: # of rows per _csrcsc invocation
# CSRCSC_CBLK_SIZE=[1024]			: # of cols per _csrcsc invocation
# CSRCSC_MKL_NTHREADS=[4]			: # of MKL threads per _csrcsc task
## _csrmm config
# MAX_NNZS=[10000000]					: max nnzs per row blk
# CSRMM_RM_RBLK_SIZE=[16384]	: MAX # of rows per _csrmm invocation
# CSRMM_RM_CBLK_SIZE=[2048]		:	MAX # of cols per _csrmm invocation
# CSRMM_RM_MKL_NTHREADS=[4]		:	# of MKL threads per _csrmm task
# CSRMM_CM_RBLK_SIZE=[262144]	:	same as CSRMM_RM_RBLK_SIZE (but for Column-Major)
# CSRMM_CM_CBLK_SIZE=[256]		:	same as CSRMM_RM_CBLK_SIZE (but for Column-Major)
# CSRMM_CM_MKL_NTHREADS=[4]		:	same as CSRMM_RM_MKL_NTHREADS (but for Column-Major)
## _csrgemv config
# CSRGEMV_NT_RBLK_SIZE=[262144]	: MAX # of rows per _csrgemv invocation
# CSRGEMV_T_RBLK_SIZE=[262144]	: MAX # of rows per _csrgemv invocation
## map config
# MAP_BLK_SIZE=[262144]				: # of elements per map task invocation
## reduce config
# REDUCE_BLK_SIZE=[262144]		: # of elements per reduce task invocation

set(N_IO_THR 4 CACHE STRING "")
set(PROGRAM_BUDGET 8589934592 CACHE STRING "")
set(N_COMPUTE_THR 4 CACHE STRING "")
set(MAX_SIMUL_REQS 4096 CACHE STRING "")
set(MAX_STRIDES 4096 CACHE STRING "")
set(MAX_EVENTS 4096 CACHE STRING "")
set(SECTOR_LEN 512 CACHE STRING "")
set(IS_ALIGNED IS_512_ALIGNED CACHE STRING "")
set(GEMM_BLK_SIZE 8192 CACHE STRING "")
set(GEMM_MKL_NTHREADS 4 CACHE STRING "")
set(OMP_CHUNK_SIZE 32768 CACHE STRING "")
set(CSRCSC_CBLK_SIZE 4096 CACHE STRING "")
set(CSRCSC_RBLK_SIZE 1048576 CACHE STRING "")
set(CSRCSC_MKL_NTHREADS 4 CACHE STRING "")
set(MAX_NNZS 10000000 CACHE STRING "")
set(CSRMM_RM_RBLK_SIZE 131072 CACHE STRING "")
set(CSRMM_RM_CBLK_SIZE 1024 CACHE STRING "")
set(CSRMM_RM_MKL_NTHREADS 4 CACHE STRING "")
set(CSRMM_CM_RBLK_SIZE 131072 CACHE STRING "")
set(CSRMM_CM_CBLK_SIZE 1024 CACHE STRING "")
set(CSRMM_CM_MKL_NTHREADS 4 CACHE STRING "")
set(CSRGEMV_NT_RBLK_SIZE 32768 CACHE STRING "")
set(CSRGEMV_T_RBLK_SIZE 262144 CACHE STRING "")
set(MAP_BLK_SIZE 1048576 CACHE STRING "")
set(REDUCE_BLK_SIZE 1048576 CACHE STRING "")
set(OVERLAP_CHECK TRUE CACHE STRING "")

add_definitions(-DN_IO_THR=${N_IO_THR}
                -DN_COMPUTE_THR=${N_COMPUTE_THR}
                -DPROGRAM_BUDGET=${PROGRAM_BUDGET}
                -DMAX_SIMUL_REQS=${MAX_SIMUL_REQS}
                -DMAX_STRIDES=${MAX_STRIDES}
                -DMAX_EVENTS=${MAX_EVENTS}
                -DSECTOR_LEN=${SECTOR_LEN}
                -DIS_ALIGNED=${IS_ALIGNED}
                -DGEMM_BLK_SIZE=${GEMM_BLK_SIZE}
                -DGEMM_MKL_NTHREADS=${GEMM_MKL_NTHREADS}
                -DOMP_CHUNK_SIZE=${OMP_CHUNK_SIZE}
                -DCSRCSC_CBLK_SIZE=${CSRCSC_CBLK_SIZE}
                -DCSRCSC_RBLK_SIZE=${CSRCSC_RBLK_SIZE}
                -DCSRCSC_MKL_NTHREADS=${CSRCSC_MKL_NTHREADS}
                -DMAX_NNZS=${MAX_NNZS}
                -DCSRMM_RM_RBLK_SIZE=${CSRMM_RM_RBLK_SIZE}
                -DCSRMM_RM_CBLK_SIZE=${CSRMM_RM_CBLK_SIZE}
                -DCSRMM_RM_MKL_NTHREADS=${CSRMM_RM_MKL_NTHREADS}
                -DCSRMM_CM_RBLK_SIZE=${CSRMM_CM_RBLK_SIZE}
                -DCSRMM_CM_CBLK_SIZE=${CSRMM_CM_CBLK_SIZE}
                -DCSRMM_CM_MKL_NTHREADS=${CSRMM_CM_MKL_NTHREADS}
                -DCSRGEMV_NT_RBLK_SIZE=${CSRGEMV_NT_RBLK_SIZE}
                -DCSRGEMV_T_RBLK_SIZE=${CSRGEMV_T_RBLK_SIZE}
                -DMAP_BLK_SIZE=${MAP_BLK_SIZE}
                -DREDUCE_BLK_SIZE=${REDUCE_BLK_SIZE}
                -DOVERLAP_CHECK=${OVERLAP_CHECK}
                )

# Compiler options
set(CXX g++)

# CXX Config
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG -O0 -fsanitize=address -fsanitize=leak -fsanitize=undefined")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
add_compile_options(-std=c++14 -Wall ${CONFIG_FLAGS} -fopenmp -Wfatal-errors)

#MKL Config
set(INTEL_ROOT /opt/intel/compilers_and_libraries/linux)
set(MKL_ROOT ${INTEL_ROOT}/mkl)
add_definitions(-DMKL_ILP64)
add_compile_options(-m64 -Wl,--no-as-needed)
include_directories(include ${INTEL_ROOT}/include ${MKL_ROOT}/include)
link_directories(${INTEL_ROOT}/lib/intel64 ${MKL_ROOT}/lib/intel64)
link_libraries(mkl_intel_ilp64 mkl_intel_thread mkl_core iomp5 pthread m dl)

# Other config
add_compile_options(${OPTIM_CXXFLAGS})
link_libraries(aio)

# Generate library
file(GLOB FHANDLES "src/file_handles/*.cpp")
file(GLOB BLAS "src/blas/*.cpp")
file(GLOB SCHED "src/scheduler/*.cpp")
add_library(fblas STATIC ${FHANDLES} ${BLAS} ${SCHED} src/lib_funcs.cpp src/utils.cpp)
link_libraries(fblas)

# Generate drivers
add_executable(in_mem_gemm_driver drivers/in_mem_gemm.cpp)
add_executable(in_mem_kmeans_driver drivers/in_mem_kmeans.cpp)
add_executable(gemm_driver drivers/gemm.cpp)
add_executable(kmeans_driver drivers/kmeans.cpp)
add_executable(in_mem_csrmm_driver drivers/in_mem_csrmm.cpp)
add_executable(csrmm_driver drivers/csrmm.cpp)
add_executable(csrmm_pmem_driver drivers/csrmm_pmem.cpp)
add_executable(in_mem_csrgemv_driver drivers/in_mem_csrgemv.cpp)
add_executable(csrgemv_driver drivers/csrgemv.cpp)
add_executable(in_mem_csrcsc_driver drivers/in_mem_csrcsc.cpp)
add_executable(csrcsc_driver drivers/csrcsc.cpp)
add_executable(in_mem_sort_driver drivers/in_mem_sort.cpp)
add_executable(sort_driver drivers/sort.cpp)
add_executable(map_reduce_driver drivers/map_reduce.cpp)

# Generate tests/misc related stuff
add_executable(flash_file_handle_test misc/flash_file_handle_test.cpp)
add_executable(dense_create misc/dense_create.cpp misc/gen_common.h)
add_executable(sparse_create misc/sparse_create.cpp misc/gen_common.h)

add_custom_command(TARGET fblas PRE_BUILD COMMAND clang-format-4.0 -i ../include/*.h ../include/*/*.h ../src/*.cpp ../src/*/*.cpp ../misc/*.cpp ../drivers/*.cpp)
