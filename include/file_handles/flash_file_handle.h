// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <libaio.h>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <queue>
#include <string>
#include <unordered_map>

#include "file_handles/file_handle.h"
#include "queue.h"
#include "types.h"
#include "utils.h"

namespace flash {
  class FlashFileHandle : public BaseFileHandle {
    // file descriptor
    std::string filename;

    static std::unordered_map<std::thread::id, io_context_t> ctx_map;
    static std::mutex ctx_mut;

    // returns a unique context for each thread that calls it
    // NOTE :: must call FlashFileHandle::setup_contexts() &
    //                   FlashFileHandle::register_thread() from the thread
    //                   before asking for a `ctx`
    // WARN :: `MT-unsafe` if thread not registered
    static io_context_t get_ctx();

   public:
    FBLAS_UINT file_sz;
    int        file_desc;

    FlashFileHandle();
    ~FlashFileHandle();

    // register thread-id for a context
    static void register_thread();

    // de-register thread-id for a context
    static void deregister_thread();

    std::string get_filename() {
      return this->filename;
    }

    // Open & close ops
    // Blocking calls
    FBLAS_INT open(std::string &fname, Mode fmode, FBLAS_UINT size = 0);
    FBLAS_INT close();

    // Contiguous read & write ops
    FBLAS_INT read(FBLAS_UINT offset, FBLAS_UINT len, void *buf,
                   const std::function<void(void)> &callback = dummy_std_func);
    FBLAS_INT write(FBLAS_UINT offset, FBLAS_UINT len, void *buf,
                    const std::function<void(void)> &callback = dummy_std_func);
    FBLAS_INT copy(FBLAS_UINT self_offset, BaseFileHandle &dest,
                   FBLAS_UINT dest_offset, FBLAS_UINT len,
                   const std::function<void(void)> &callback = dummy_std_func);

    // Non contiguous | strided read & write ops
    // NOTE :: stride >= len
    FBLAS_INT sread(FBLAS_UINT offset, StrideInfo sinfo, void *buf,
                    const std::function<void(void)> &callback = dummy_std_func);
    FBLAS_INT swrite(
        FBLAS_UINT offset, StrideInfo sinfo, void *buf,
        const std::function<void(void)> &callback = dummy_std_func);
    FBLAS_INT scopy(FBLAS_UINT self_offset, BaseFileHandle &dest,
                    FBLAS_UINT dest_offset, StrideInfo sinfo,
                    const std::function<void(void)> &callback = dummy_std_func);
  };
}  // namespace flash
