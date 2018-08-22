// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "lib_funcs.h"
namespace flash {
  // NOTE :: logger must be initialized first
  Logger __global_logger("global");
  // Scheduler                 sched((FBLAS_UINT) 16 * 1024 * 1024 * 1024);
  Scheduler   sched(N_IO_THR, N_COMPUTE_THR, (FBLAS_UINT) PROGRAM_BUDGET);
  std::string mnt_dir = "./";
  std::function<void(void)> dummy_std_func = [](void) {
    // GLOG_DEBUG("default callback()");
  };

  // used to assign IDs to tasks
  std::atomic<FBLAS_UINT> global_task_counter(0);

  void flash_setup(std::string mntdir) {
    // register main program thread for I/O
    FlashFileHandle::register_thread();
    GLOG_DEBUG("setting mnt_dir = ", mntdir);
    mnt_dir = mntdir;
  }

  void flash_destroy() {
    // de-register main program thread
    FlashFileHandle::deregister_thread();
    // std::string fname = mnt_dir + std::string("/tmp_file") + "*";
    // std::string command = "rm -f " + fname;
    // GLOG_DEBUG("system(", command, ")");
    // int ret = system(command.c_str());
    // GLOG_ASSERT(ret != 0, "system() exited with return=", ret);
  }
}  // namespace flash
