// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "file_handles/flash_file_handle.h"
#include <errno.h>
#include <fcntl.h>
#include <malloc.h>
#include <sys/mman.h>
#include <unistd.h>
// #include <xfs/xfs.h>
#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include "queue.h"
#include "types.h"
#include "utils.h"

// max chunk size to fetch/put from/to disk in one request
// NOTE : Some devices might have higher throughput with more requests of
// smaller sizes (IOPS heavy)
#define MAX_CHUNK_SIZE ((FBLAS_UINT) 1 << 25)

namespace {
  void submit_and_reap(io_context_t ctx, struct iocb* cb, FBLAS_UINT n_requests,
                       FBLAS_UINT n_retries = 5) {
    typedef struct io_event io_event_t;
    // using std::vector for automatic mem-management
    struct iocb** cbs = new struct iocb*[n_requests];
    io_event_t*   evts = new io_event_t[n_requests];

    // initialize `cbs` using `cb` array
    for (FBLAS_UINT i = 0; i < n_requests; i++) {
      cbs[i] = (cb + i);
    }

    FBLAS_UINT n_tries = 0;
    while (n_tries < n_retries) {
      // issue reads
      FBLAS_INT ret = io_submit(ctx, (FBLAS_INT) n_requests, cbs);
      // if requests didn't get accepted
      if (ret != (FBLAS_INT) n_requests) {
        GLOG_ERROR("io_submit() failed; returned ", ret,
                   ", expected=", n_requests, ", ernno=", errno, "=",
                   ::strerror(errno), ", try #", n_tries + 1);
        n_tries++;
        // try again
        continue;
      } else {
        // wait on io_getevents
        ret = io_getevents(ctx, (FBLAS_INT) n_requests, (FBLAS_INT) n_requests,
                           evts, nullptr);
        // if requests didn't complete
        if (ret != (FBLAS_INT) n_requests) {
          GLOG_ERROR("io_getevents() failed; returned ", ret,
                     ", expected=", n_requests, ", ernno=", errno, "=",
                     ::strerror(errno), ", try #", n_tries + 1);
          n_tries++;
          // try again
          continue;
        } else {
          break;
        }
      }
    }
    // free resources
    delete[] cbs;
    delete[] evts;

    if (n_tries == n_retries) {
      GLOG_FATAL("unable to complete IO request");
    }
  }

  void execute_io(io_context_t ctx, int fd, std::vector<FBLAS_UINT>& offsets,
                  std::vector<FBLAS_UINT>& sizes, std::vector<void*>& bufs,
                  bool is_write, FBLAS_UINT max_ops = MAX_SIMUL_REQS) {
    FBLAS_UINT   n_ops = offsets.size();
    struct iocb* cb = new struct iocb[max_ops];
    auto         prep_func = (is_write ? io_prep_pwrite : io_prep_pread);
    FBLAS_UINT   n_iters = ROUND_UP(n_ops, max_ops) / max_ops;

    // submit and reap atmost `max_ops` in each iter
    for (FBLAS_UINT i = 0; i < n_iters; i++) {
      FBLAS_UINT start_idx = (max_ops * i);
      FBLAS_UINT cur_nops = std::min(n_ops - start_idx, max_ops);

      for (FBLAS_UINT j = 0; j < cur_nops; j++) {
        prep_func(cb + j, fd, bufs[start_idx + j], sizes[start_idx + j],
                  offsets[start_idx + j]);
      }

      submit_and_reap(ctx, cb, cur_nops);

      memset(cb, 0, max_ops * sizeof(struct iocb));
    }

    delete[] cb;
  }

  // return `buf` offset by `offset`
  template<typename T>
  T* offset_buf(T* buf, FBLAS_UINT offset) {
    return (T*) ((char*) buf + offset);
  }
}  // namespace anonymous

namespace flash {

  FlashFileHandle::FlashFileHandle() {
    GLOG_DEBUG("MAX_SIMUL_REQS : ", MAX_SIMUL_REQS);
    this->file_desc = -1;
  }

  FlashFileHandle::~FlashFileHandle() {
    FBLAS_INT ret;
    // check to make sure file_desc is closed
    ret = ::fcntl(this->file_desc, F_GETFD);
    if (ret == -1) {
      if (errno != EBADF) {
        GLOG_WARN("close() not called");
        // close file desc
        ret = ::close(this->file_desc);
        // error checks
        if (ret == -1) {
          GLOG_ERROR("close() failed; returned ", ret, ", errno=", errno, ":",
                     ::strerror(errno));
        }
      }
    }
  }

  // defining because C++ complains otherwise
  std::unordered_map<std::thread::id, io_context_t> FlashFileHandle::ctx_map;
  std::mutex FlashFileHandle::ctx_mut;

  io_context_t FlashFileHandle::get_ctx() {
#ifdef DEBUG
    // perform checks only in DEBUG mode
    if (FlashFileHandle::ctx_map.find(std::this_thread::get_id()) ==
        FlashFileHandle::ctx_map.end()) {
      GLOG_ASSERT(false, "bad ctx find");
    }
#endif
    // access map in a lock-free manner
    return FlashFileHandle::ctx_map[std::this_thread::get_id()];
  }

  void FlashFileHandle::register_thread() {
    auto                         my_id = std::this_thread::get_id();
    std::unique_lock<std::mutex> lk(FlashFileHandle::ctx_mut);
    if (FlashFileHandle::ctx_map.find(my_id) !=
        FlashFileHandle::ctx_map.end()) {
      GLOG_FATAL("double registration");
    } else {
      io_context_t ctx = 0;
      int          ret = io_setup(MAX_EVENTS, &ctx);
      if (ret != 0) {
        lk.unlock();
        GLOG_ASSERT(errno != EAGAIN, "MAX_EVENTS too large");
        GLOG_ASSERT(errno != ENOMEM, "insufficient kernel resources");
        GLOG_FATAL("io_setup() failed; returned ", ret, ", errno=", errno, ":",
                   ::strerror(errno));
      }
      GLOG_DEBUG("thread_id=", my_id, ", ctx=", ctx);
      FlashFileHandle::ctx_map[my_id] = ctx;
    }

    lk.unlock();
  }

  void FlashFileHandle::deregister_thread() {
    auto                         my_id = std::this_thread::get_id();
    std::unique_lock<std::mutex> lk(FlashFileHandle::ctx_mut);
    if (FlashFileHandle::ctx_map.find(my_id) !=
        FlashFileHandle::ctx_map.end()) {
      GLOG_DEBUG("returning ctx");
      io_context_t ctx = FlashFileHandle::get_ctx();
      int          ret = io_destroy(ctx);
      GLOG_ASSERT(ret == 0, "io_detroy() failed; returned ", ret,
                  ", errno=", errno, ":", ::strerror(errno));
      FlashFileHandle::ctx_map.erase(my_id);
    } else {
      GLOG_FATAL("attempting to return un-registered ctx");
    }
    lk.unlock();
  }

  FBLAS_INT FlashFileHandle::open(std::string& fname, Mode fmode,
                                  FBLAS_UINT size) {
    int flags = O_DIRECT;
    if (fmode == Mode::READ)
      flags |= O_RDONLY;
    else if (fmode == Mode::WRITE)
      flags |= O_WRONLY;
    else if (fmode == Mode::READWRITE)
      flags |= O_RDWR;
    else
      GLOG_FATAL("bad file flags");

    this->file_desc = ::open(fname.c_str(), flags);

    /*
    // xfs info
    struct dioattr xfs_dio;
    int rt = xfsctl(fname.c_str(), this->file_desc, XFS_IOC_DIOINFO, &xfs_dio);
    GLOG_DEBUG("d_mem=", xfs_dio.d_mem, ", d_miniosz=", xfs_dio.d_miniosz,
               ", d_maxiosz", xfs_dio.d_maxiosz);
*/

    this->filename = fname;
    GLOG_DEBUG("opening : ", this->filename);

    // error checks
    if (this->file_desc == -1) {
      GLOG_FATAL("open() failed; returned ", this->file_desc, ", errno=", errno,
                 ":", ::strerror(errno));
    } else {
      std::ifstream in(filename.c_str(),
                       std::ifstream::ate | std::ifstream::binary);
      this->file_sz = in.tellg();
      in.close();
      return 0;
    }
    // adding this so g++ doesn't complain about not returning a value
    return -1;
  }

  FBLAS_INT FlashFileHandle::close() {
    FBLAS_INT ret;

    // check to make sure file_desc is closed
    ret = ::fcntl(this->file_desc, F_GETFD);
    GLOG_ASSERT(ret != -1, "fcntl() failed; returned ", ret, ", errno=", errno,
                ":", ::strerror(errno));

    ret = ::close(this->file_desc);
    GLOG_ASSERT(ret != -1, "close() failed; returned ", ret, ", errno=", errno,
                ":", ::strerror(errno));

    return 0;
  }

  FBLAS_INT FlashFileHandle::read(FBLAS_UINT offset, FBLAS_UINT len, void* buf,
                                  const std::function<void(void)>& callback) {
    if (len == 0) {
      GLOG_WARN("0 len read");
      return 0;
    }
    GLOG_ASSERT(len > 0, "can't read 0 len");
    GLOG_ASSERT(buf != nullptr, "nullptr buf not allowed");

    io_context_t ctx = FlashFileHandle::get_ctx();
    GLOG_DEBUG("ctx=", ctx);

    // check buf alignment
    // if not aligned, align to SECTOR_LEN-byte boundary
    // NOTE :: required for libaio to work with O_DIRECT flags
    void*      read_buf;
    bool       alloc = false;
    FBLAS_UINT start_offset = ROUND_DOWN(offset, SECTOR_LEN);
    FBLAS_UINT read_len = ROUND_UP(offset + len, SECTOR_LEN) - start_offset;
    std::vector<FBLAS_UINT> offsets;
    std::vector<FBLAS_UINT> sizes;
    std::vector<void*>      bufs;

    // check if buf is aligned
    if (!IS_ALIGNED(buf) || start_offset != offset || read_len != len) {
      // store old_buf
      alloc_aligned(&read_buf, read_len, SECTOR_LEN);
      alloc = true;
    } else {
      read_buf = buf;
    }
    // if only one request
    if (read_len <= MAX_CHUNK_SIZE) {
      // push params
      offsets.push_back(start_offset);
      sizes.push_back(read_len);
      bufs.push_back(read_buf);
    } else {
      // break down request into multiple requests
      FBLAS_UINT n_requests =
          ROUND_UP(read_len, MAX_CHUNK_SIZE) / MAX_CHUNK_SIZE;
      offsets.resize(n_requests);
      sizes.resize(n_requests);
      bufs.resize(n_requests);
      for (FBLAS_UINT i = 0; i < n_requests; i++) {
        // calculate parameters for this read
        bufs[i] = offset_buf(read_buf, i * MAX_CHUNK_SIZE);
        offsets[i] = start_offset + i * MAX_CHUNK_SIZE;
        sizes[i] = std::min(MAX_CHUNK_SIZE, read_len - (i * MAX_CHUNK_SIZE));
      }
    }

    // execute io
    execute_io(ctx, this->file_desc, offsets, sizes, bufs, false);

    // copy out from read_buf if any of the parameters were not aligned
    if (alloc) {
      memcpy(buf, offset_buf(read_buf, (offset - start_offset)), len);
      free(read_buf);
    }

    // execute callback
    callback();

    return 0;
  }

  FBLAS_INT FlashFileHandle::write(FBLAS_UINT offset, FBLAS_UINT len, void* buf,
                                   const std::function<void(void)>& callback) {
    if (len == 0) {
      GLOG_WARN("0 len write");
      return 0;
    }

    GLOG_ASSERT(buf != nullptr, "nullptr buf not allowed");

    io_context_t ctx = FlashFileHandle::get_ctx();

    // check buf alignment
    // if not aligned, align to SECTOR_LEN-byte boundary
    // NOTE :: required for libaio to work with O_DIRECT flags
    void*      write_buf = nullptr;
    FBLAS_UINT start_offset = ROUND_DOWN(offset, SECTOR_LEN);
    FBLAS_UINT end_offset = ROUND_UP(offset + len, SECTOR_LEN);
    FBLAS_UINT write_len = end_offset - start_offset;
    FBLAS_UINT n_requests =
        ROUND_UP(write_len, MAX_CHUNK_SIZE) / MAX_CHUNK_SIZE;
    bool alloc = false;

    std::vector<FBLAS_UINT> offsets;
    std::vector<FBLAS_UINT> sizes;
    std::vector<void*>      bufs;

    // buf alloc if UNALIGNED
    if (IS_ALIGNED(buf) && IS_ALIGNED(offset) && IS_ALIGNED(len)) {
      write_buf = buf;
    } else {
      alloc = true;
      alloc_aligned(&write_buf, write_len, SECTOR_LEN);
    }

    // fetch first sector if needed
    if (!IS_ALIGNED(offset)) {
      offsets.push_back(start_offset);
      sizes.push_back(SECTOR_LEN);
      bufs.push_back(write_buf);
    }
    // fetch last sector if needed
    if (write_len > SECTOR_LEN && !IS_ALIGNED(offset + len) && alloc) {
      offsets.push_back(end_offset - SECTOR_LEN);
      sizes.push_back(SECTOR_LEN);
      bufs.push_back(offset_buf(write_buf, write_len - SECTOR_LEN));
    }
    // fetch  first/last sectors if required
    if (!offsets.empty()) {
      // execute io
      execute_io(ctx, this->file_desc, offsets, sizes, bufs, false);

      // clear vectors for re-use
      offsets.clear();
      sizes.clear();
      bufs.clear();
    }

    // dirty buf if alloc'ed
    if (alloc) {
      memcpy(offset_buf(write_buf, (offset - start_offset)), buf, len);
    }

    // prepare write request(s)
    offsets.resize(n_requests);
    sizes.resize(n_requests);
    bufs.resize(n_requests);
    for (FBLAS_UINT i = 0; i < n_requests; i++) {
      // calculate parameters for this read
      bufs[i] = offset_buf(write_buf, i * MAX_CHUNK_SIZE);
      offsets[i] = start_offset + i * MAX_CHUNK_SIZE;
      sizes[i] = std::min(MAX_CHUNK_SIZE, write_len - (i * MAX_CHUNK_SIZE));
    }

    // execute io
    execute_io(ctx, this->file_desc, offsets, sizes, bufs, true);

    // free any resources
    if (alloc) {
      free(write_buf);
    }
#ifdef DEBUG
    // extra verification step
    void* test_buf = malloc(len);
    this->read(offset, len, test_buf);
    int rt = memcmp(test_buf, buf, len);
    free(test_buf);
    if (rt != 0)
      GLOG_FAIL("write failed; memcmp returned ", rt, ", expected 0");
#endif
    // execute callback
    callback();

    return 0;
  }

  FBLAS_INT FlashFileHandle::copy(FBLAS_UINT self_offset, BaseFileHandle& dest,
                                  FBLAS_UINT dest_offset, FBLAS_UINT len,
                                  const std::function<void(void)>& callback) {
    // Create buf to copy from src
    void* buf = malloc(len);

    // src_file -> DRAM
    this->read(self_offset, len, buf);
    // DRAM -> dest_file
    dest.write(dest_offset, len, buf);

    // delete buffer
    free(buf);

    // execute callback
    callback();

    return 0;
  }

  FBLAS_INT FlashFileHandle::sread(FBLAS_UINT offset, StrideInfo sinfo,
                                   void*                            buf,
                                   const std::function<void(void)>& callback) {
    GLOG_ASSERT(sinfo.n_strides != 0, "n_strides = 0; update to n_strides = 1");
    GLOG_ASSERT(sinfo.len_per_stride <= sinfo.stride, "bad sinfo:");

    if (sinfo.len_per_stride == 0) {
      GLOG_WARN("0 len sread");
      return 0;
    }

    FBLAS_UINT   n_reads = sinfo.n_strides;
    void*        read_buf = nullptr;
    io_context_t ctx = FlashFileHandle::get_ctx();

    // if all params are aligned
    if (IS_ALIGNED(buf) && IS_ALIGNED(sinfo.len_per_stride) &&
        IS_ALIGNED(offset) && IS_ALIGNED(sinfo.stride)) {
      FBLAS_UINT              n_reads = sinfo.n_strides;
      std::vector<FBLAS_UINT> starts(n_reads, 0);
      std::vector<FBLAS_UINT> sizes(n_reads, sinfo.len_per_stride);
      std::vector<void*>      bufs(n_reads, nullptr);
      for (FBLAS_UINT idx = 0; idx < n_reads; idx++) {
        starts[idx] = offset + (sinfo.stride * idx);
        bufs[idx] = offset_buf(buf, idx * sinfo.len_per_stride);
      }

      // execute reads
      execute_io(ctx, this->file_desc, starts, sizes, bufs, false);

      return 0;
    }

    // fill reads
    std::vector<FBLAS_UINT> starts(n_reads, 0);       // read offset on disk
    std::vector<FBLAS_UINT> sizes(n_reads, 0);        // size of read on disk
    std::vector<FBLAS_UINT> buf_offsets(n_reads, 0);  // offset in read buf
    std::vector<FBLAS_UINT> buf_deltas(n_reads, 0);   // offset in bufs[i]
    FBLAS_UINT              buf_size = 0;
    const FBLAS_UINT        stride = sinfo.stride;
    const FBLAS_UINT        lps = sinfo.len_per_stride;

    for (FBLAS_UINT idx = 0; idx < n_reads; idx++) {
      buf_offsets[idx] = buf_size;
      starts[idx] = ROUND_DOWN(offset + (stride * idx), SECTOR_LEN);
      buf_deltas[idx] = offset + (stride * idx) - starts[idx];
      FBLAS_UINT end = ROUND_UP(offset + (stride * idx) + lps, SECTOR_LEN);
      sizes[idx] = end - starts[idx];
      buf_size += sizes[idx];
    }

    alloc_aligned(&read_buf, buf_size, SECTOR_LEN);

    // buffer assignment
    std::vector<void*> bufs(n_reads, nullptr);
    for (FBLAS_UINT i = 0; i < n_reads; i++) {
      bufs[i] = offset_buf(read_buf, buf_offsets[i]);
    }

    // execute reads
    execute_io(ctx, this->file_desc, starts, sizes, bufs, false);

    // copy from `bufs` into `buf`
    for (FBLAS_UINT i = 0; i < n_reads; i++) {
      void* src_buf = offset_buf(bufs[i], buf_deltas[i]);
      void* dest_buf = offset_buf(buf, lps * i);
      memcpy(dest_buf, src_buf, lps);
    }

    // free mem
    free(read_buf);

    // execute callback
    callback();

    // return success
    return 0;
  }

  FBLAS_INT FlashFileHandle::swrite(FBLAS_UINT offset, StrideInfo sinfo,
                                    void*                            buf,
                                    const std::function<void(void)>& callback) {
    GLOG_ASSERT(sinfo.n_strides != 0, "n_strides = 0; update to n_strides = 1");
    GLOG_ASSERT(sinfo.len_per_stride <= sinfo.stride, "bad sinfo");
    if (sinfo.len_per_stride == 0) {
      GLOG_WARN("0 len swrite");
      return 0;
    }

    const FBLAS_UINT stride = sinfo.stride;
    const FBLAS_UINT lps = sinfo.len_per_stride;
    const FBLAS_UINT n_strides = sinfo.n_strides;
    io_context_t     ctx = FlashFileHandle::get_ctx();

    // if all params are aligned
    if (IS_ALIGNED(buf) && IS_ALIGNED(lps) && IS_ALIGNED(offset) &&
        IS_ALIGNED(stride)) {
      std::vector<FBLAS_UINT> starts(n_strides, 0);
      std::vector<FBLAS_UINT> sizes(n_strides, lps);
      std::vector<void*>      bufs(n_strides, nullptr);
      for (FBLAS_UINT idx = 0; idx < n_strides; idx++) {
        starts[idx] = offset + (stride * idx);
        bufs[idx] = offset_buf(buf, idx * lps);
      }

      // execute writes
      execute_io(ctx, this->file_desc, starts, sizes, bufs, true);
      return 0;
    }

    // get stride limits
    std::vector<FBLAS_UINT> starts(n_strides, 0);
    std::vector<FBLAS_UINT> ends(n_strides, 0);
    for (FBLAS_UINT idx = 0; idx < n_strides; idx++) {
      starts[idx] = offset + (stride * idx);
      ends[idx] = offset + (stride * idx) + lps;
      starts[idx] = ROUND_DOWN(starts[idx], SECTOR_LEN);
      ends[idx] = ROUND_UP(ends[idx], SECTOR_LEN);
    }

    // check if merging is required
    bool merge_required = false;
    for (FBLAS_UINT i = 0; i < n_strides - 1; i++) {
      if (ends[i] > starts[i + 1]) {
        merge_required = true;
        break;
      }
    }

    if (!merge_required) {
      FBLAS_UINT              buf_size = 0;
      std::vector<FBLAS_UINT> sizes(n_strides, 0);
      std::vector<FBLAS_UINT> buf_offsets(n_strides, 0);
      std::vector<FBLAS_UINT> buf_deltas(n_strides, 0);
      for (FBLAS_UINT i = 0; i < n_strides; i++) {
        buf_offsets[i] = buf_size;
        buf_deltas[i] = offset + (stride * i) - starts[i];
        sizes[i] = (ends[i] - starts[i]);
        buf_size += sizes[i];
      }
      // alloc buf
      void* write_buf = nullptr;
      alloc_aligned(&write_buf, buf_size, SECTOR_LEN * 8);

      std::vector<void*> bufs(n_strides, nullptr);
      for (FBLAS_UINT i = 0; i < n_strides; i++) {
        bufs[i] = offset_buf(write_buf, buf_offsets[i]);
      }

      if (lps >= 3 * SECTOR_LEN) {
        std::vector<FBLAS_UINT> sector_size(n_strides, SECTOR_LEN);
        std::vector<void*>      last_sectors(n_strides, nullptr);
        std::vector<FBLAS_UINT> ends_minus_sector(n_strides, 0);

        for (FBLAS_UINT i = 0; i < n_strides; i++) {
          last_sectors[i] = offset_buf(bufs[i], sizes[i] - SECTOR_LEN);
          ends_minus_sector[i] = ends[i] - SECTOR_LEN;
        }

#ifdef DEBUG
        execute_io(ctx, this->file_desc, starts, sector_size, bufs, false);
        execute_io(ctx, this->file_desc, ends_minus_sector, sector_size,
                   last_sectors, false);
#else
        sector_size.resize(2 * n_strides, SECTOR_LEN);
        ends_minus_sector.insert(ends_minus_sector.end(), starts.begin(),
                                 starts.end());
        last_sectors.insert(last_sectors.end(), bufs.begin(), bufs.end());
        execute_io(ctx, this->file_desc, ends_minus_sector, sector_size,
                   last_sectors, false);
#endif
      } else {
        // read existing data
        execute_io(ctx, this->file_desc, starts, sizes, bufs, false);
      }

      // dirty bufs
      for (FBLAS_UINT i = 0; i < n_strides; i++) {
        void* src_buf = offset_buf(buf, lps * i);
        void* dest_buf = offset_buf(bufs[i], buf_deltas[i]);
        memcpy(dest_buf, src_buf, lps);
      }

      // write back bufs
      execute_io(ctx, this->file_desc, starts, sizes, bufs, true);

      // free write buf
      free(write_buf);

#ifdef DEBUG
      // extra verification step
      void* test_buf = malloc(n_strides * lps);
      this->sread(offset, sinfo, test_buf);
      int rt = memcmp(test_buf, buf, n_strides * lps);
      if (rt != 0) {
        GLOG_FAIL("swrite failed");
      }
      free(test_buf);
#endif

      return 0;
    }

    // by default `0` gets its own block
    std::vector<FBLAS_UINT> merges(1, 0);
    if (n_strides > 0) {
      merges.push_back(1);
    }
    std::vector<FBLAS_UINT> m_starts(1, starts[0]);
    std::vector<FBLAS_UINT> m_ends(1, ends[0]);
    // pre-allocate vector
    merges.reserve(n_strides);
    m_starts.reserve(n_strides);
    m_ends.reserve(n_strides);
    for (FBLAS_UINT i = 1; i < n_strides; i++) {
      FBLAS_UINT& next_blk_start = merges.back();
      FBLAS_UINT& c_end = m_ends.back();
      if (starts[i] < c_end) {
        // add to current block
        next_blk_start++;
        // update end of current block
        c_end = ends[i];
      } else {
        // start a new block
        merges.emplace_back(i + 1);
        m_starts.push_back(starts[i]);
        m_ends.push_back(ends[i]);
      }
    }

    FBLAS_UINT m_nblks = m_starts.size();
    // buffer occupancy
    std::vector<FBLAS_UINT> m_offs(m_starts.size(), 0);
    std::vector<FBLAS_UINT> m_sizes(m_starts.size(), 0);
    FBLAS_UINT              cur_off = 0;
    for (FBLAS_UINT i = 0; i < m_starts.size(); i++) {
      m_offs[i] = cur_off;
      m_sizes[i] = m_ends[i] - m_starts[i];
      cur_off += m_sizes[i];
    }

    // alloc buf
    void* write_buf = nullptr;
    alloc_aligned(&write_buf, cur_off, SECTOR_LEN);

    std::vector<void*> m_bufs(m_nblks, nullptr);
    for (FBLAS_UINT i = 0; i < m_nblks; i++) {
      m_bufs[i] = offset_buf(write_buf, m_offs[i]);
    }

    // execute reads
    execute_io(ctx, this->file_desc, m_starts, m_sizes, m_bufs, false);

    // dirty in-mem buf
    for (FBLAS_UINT m = 0; m < m_nblks; m++) {
      FBLAS_UINT m_idx_start = merges[m];
      FBLAS_UINT m_idx_end = merges[m + 1];
      for (FBLAS_UINT i = m_idx_start; i < m_idx_end; i++) {
        FBLAS_UINT stride_idx = i;
        FBLAS_UINT buf_offset = (offset + stride_idx * stride) - m_starts[m];
        void*      src_buf = offset_buf(buf, (stride_idx * lps));
        void*      dest_buf = offset_buf(m_bufs[m], buf_offset);
        memcpy(dest_buf, src_buf, lps);
      }
    }

    // write back dirty bufs
    execute_io(ctx, this->file_desc, m_starts, m_sizes, m_bufs, true);

    // free buf
    free(write_buf);

#ifdef DEBUG
    // extra verification step
    void* test_buf = malloc(n_strides * lps);
    this->sread(offset, sinfo, test_buf);
    int rt = memcmp(test_buf, buf, n_strides * lps);
    free(test_buf);
    if (rt != 0)
      GLOG_FAIL("swrite failed");
#endif

    // execute callback
    callback();

    // return success
    return 0;
  }

  FBLAS_INT FlashFileHandle::scopy(FBLAS_UINT self_offset, BaseFileHandle& dest,
                                   FBLAS_UINT dest_offset, StrideInfo sinfo,
                                   const std::function<void(void)>& callback) {
    char* buf = new char[(sinfo.n_strides) * sinfo.len_per_stride];
    this->sread(self_offset, sinfo, buf);
    dest.swrite(dest_offset, sinfo, buf);
    delete[] buf;

    return 0;
  }

}  // namespace flash
