// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// Library logging macros and loggers

#pragma once

#include <cstdlib>
#include <iostream>
#include <mutex>
#include <thread>

#define LOG_INFO(_1, ...) _1.info(__func__, __LINE__, __VA_ARGS__)
#define LOG_ERROR(_1, ...) _1.error(__func__, __LINE__, __VA_ARGS__)
#define LOG_WARN(_1, ...) _1.warn(__func__, __LINE__, __VA_ARGS__)
#define LOG_FATAL(_1, ...) _1.fatal(__func__, __LINE__, __VA_ARGS__)
#define LOG_PASS(_1, ...) _1.pass(__func__, __LINE__, __VA_ARGS__)
#define LOG_FAIL(_1, ...) _1.fail(__func__, __LINE__, __VA_ARGS__)

// activate debug assert and debug macros only if compiled in debug mode
#ifdef DEBUG
#define LOG_DEBUG(_1, ...) _1.debug(__func__, __LINE__, __VA_ARGS__)
#define LOG_ASSERT(_1, _2, ...) \
  if (!(_2))                    \
    _1.fatal(__func__, __LINE__, "assert:(", #_2, ") failed: ", __VA_ARGS__);
#define LOG_ASSERT_LE(_1, _2, _3)                                            \
  if (_2 > _3)                                                               \
    LOG_ASSERT(_1, _2 <= _3, "expected ", #_2, "<=", _3, ", got ", #_2, "=", \
               _2);
#define LOG_ASSERT_LT(_1, _2, _3) \
  if (_2 >= _3)                   \
    LOG_ASSERT(_1, _2 < _3, "expected ", #_2, "<", _3, ", got ", #_2, "=", _2);
#define LOG_ASSERT_EQ(_1, _2, _3) \
  LOG_ASSERT(_1, _2 == _3, "expected ", #_2, "=", _3, ", got ", #_2, "=", _2);
#define LOG_ASSERT_NOT_NULL(_1, _2) \
  LOG_ASSERT(_1, _2 != nullptr, " expected non-nullptr, got nullptr");
#else
#define LOG_DEBUG(_1, ...)
#define LOG_ASSERT(_1, _2, ...)
#define LOG_ASSERT_LE(_1, _2, _3)
#define LOG_ASSERT_LT(_1, _2, _3)
#define LOG_ASSERT_EQ(_1, _2, _3)
#define LOG_ASSERT_NOT_NULL(_1, _2)
#endif

// GLOBAL logger
#define GLOG_INFO(...) LOG_INFO(flash::__global_logger, __VA_ARGS__)
#define GLOG_DEBUG(...) LOG_DEBUG(flash::__global_logger, __VA_ARGS__)
#define GLOG_ERROR(...) LOG_ERROR(flash::__global_logger, __VA_ARGS__)
#define GLOG_WARN(...) LOG_WARN(flash::__global_logger, __VA_ARGS__)
#define GLOG_FATAL(...) LOG_FATAL(flash::__global_logger, __VA_ARGS__)
#define GLOG_PASS(...) LOG_PASS(flash::__global_logger, __VA_ARGS__)
#define GLOG_FAIL(...) LOG_FAIL(flash::__global_logger, __VA_ARGS__)
#define GLOG_ASSERT(...) LOG_ASSERT(flash::__global_logger, __VA_ARGS__)
#define GLOG_ASSERT_LE(...) LOG_ASSERT_LE(flash::__global_logger, __VA_ARGS__)
#define GLOG_ASSERT_LT(...) LOG_ASSERT_LT(flash::__global_logger, __VA_ARGS__)
#define GLOG_ASSERT_EQ(...) LOG_ASSERT_EQ(flash::__global_logger, __VA_ARGS__)
#define GLOG_ASSERT_NOT_NULL(...) \
  LOG_ASSERT_NOT_NULL(flash::__global_logger, __VA_ARGS__)

namespace flash {
  // Thread-safe logger
  // Also produces color-coded output on some terminals
  class Logger {
    std::ostream &fstr = std::cout;
    std::string   name;
    std::mutex    mut;
    char          time_buf[100];
    std::time_t   cur_time;
    struct tm *   loc_time;

    // recursive template unpacking
    template<typename T, typename... Args>
    void write_out(T t, Args... args) {
      fstr << t;
      write_out(args...);
    }

    // printing last arg
    template<typename T>
    void write_out(T t) {
      fstr << t;
    }

   public:
    Logger(std::string name) {
      this->name = name;
    }

    template<typename fname, typename line_no, typename... Args>
    void info(fname func_name, line_no line, Args... args) {
      std::lock_guard<std::mutex> l(this->mut);
      this->cur_time = ::time(nullptr);
      this->loc_time = localtime(&this->cur_time);
      strftime(this->time_buf, 100, "%d/%m/%Y|%H:%M:%S", this->loc_time);
      fstr << "\033[1;37;40m[info][" << this->time_buf << "][" << this->name
           << "][thread:" << std::this_thread::get_id() << "]:" << func_name
           << ":" << line << ":"
           << "\033[0;37;40m";
      write_out(args...);
      fstr << "\033[0m" << std::endl << std::flush;
    }

    template<typename fname, typename line_no, typename... Args>
    void debug(fname func_name, line_no line, Args... args) {
      std::lock_guard<std::mutex> l(this->mut);
      this->cur_time = ::time(nullptr);
      this->loc_time = localtime(&this->cur_time);
      strftime(this->time_buf, 100, "%d/%m/%Y|%H:%M:%S", this->loc_time);
      fstr << "\033[1;36;40m[dbg][" << this->time_buf << "][" << this->name
           << "][thread:" << std::this_thread::get_id() << "]:" << func_name
           << ":" << line << ":"
           << "\033[0;36;40m";
      write_out(args...);
      fstr << "\033[0m" << std::endl << std::flush;
    }

    template<typename fname, typename line_no, typename... Args>
    void error(fname func_name, line_no line, Args... args) {
      std::lock_guard<std::mutex> l(this->mut);
      this->cur_time = ::time(nullptr);
      this->loc_time = localtime(&this->cur_time);
      strftime(this->time_buf, 100, "%d/%m/%Y|%H:%M:%S", this->loc_time);
      fstr << "\033[1;31;40m[err][" << this->time_buf << "][" << this->name
           << "][thread:" << std::this_thread::get_id() << "]:" << func_name
           << ":" << line << ":"
           << "\033[0;31;40m";
      write_out(args...);
      fstr << "\033[0m" << std::endl << std::flush;
    }

    template<typename fname, typename line_no, typename... Args>
    void fail(fname func_name, line_no line, Args... args) {
      std::lock_guard<std::mutex> l(this->mut);
      this->cur_time = ::time(nullptr);
      this->loc_time = localtime(&this->cur_time);
      strftime(this->time_buf, 100, "%d/%m/%Y|%H:%M:%S", this->loc_time);
      fstr << "\033[1;31;40m[fail][" << this->time_buf << "][" << this->name
           << "][thread:" << std::this_thread::get_id() << "]:" << func_name
           << ":" << line << ":"
           << "\033[0;31;40m";
      write_out(args...);
      fstr << "\033[0m" << std::endl << std::flush;
    }

    template<typename fname, typename line_no, typename... Args>
    void pass(fname func_name, line_no line, Args... args) {
      std::lock_guard<std::mutex> l(this->mut);
      this->cur_time = ::time(nullptr);
      this->loc_time = localtime(&this->cur_time);
      strftime(this->time_buf, 100, "%d/%m/%Y|%H:%M:%S", this->loc_time);
      fstr << "\033[1;32;40m[pass][" << this->time_buf << "][" << this->name
           << "][thread:" << std::this_thread::get_id() << "]:" << func_name
           << ":" << line << ":"
           << "\033[0;32;40m";
      write_out(args...);
      fstr << "\033[0m" << std::endl << std::flush;
    }

    template<typename fname, typename line_no, typename... Args>
    void warn(fname func_name, line_no line, Args... args) {
      std::lock_guard<std::mutex> l(this->mut);
      this->cur_time = ::time(nullptr);
      this->loc_time = localtime(&this->cur_time);
      strftime(this->time_buf, 100, "%d/%m/%Y|%H:%M:%S", this->loc_time);
      fstr << "\033[1;33;40m[warn][" << this->time_buf << "][" << this->name
           << "][thread:" << std::this_thread::get_id() << "]:" << func_name
           << ":" << line << ":"
           << "\033[0;33;40m";
      write_out(args...);
      fstr << "\033[0m" << std::endl << std::flush;
    }

    // NOTE :: Most of the time, you'll want to use `error` and not `fatal`
    template<typename fname, typename line_no, typename... Args>
    void fatal(fname func_name, line_no line, Args... args) {
      std::lock_guard<std::mutex> l(this->mut);
      this->cur_time = ::time(nullptr);
      this->loc_time = localtime(&this->cur_time);
      strftime(this->time_buf, 100, "%d/%m/%Y|%H:%M:%S", this->loc_time);
      fstr << "\033[1;37;41m[fatal][" << this->time_buf << "][" << this->name
           << "][thread:" << std::this_thread::get_id() << "]:" << func_name
           << ":" << line << ":"
           << "\033[0;37;41m";
      write_out(args...);
      fstr << "\033[0m" << std::endl << std::flush;
      exit(-1);
    }
  };

  // expect a __global_logger to be instantiated in some compile unit
  extern Logger __global_logger;

}  // namespace flash
