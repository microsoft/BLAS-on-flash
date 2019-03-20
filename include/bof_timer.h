// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// borrowed from https://gist.github.com/tzutalin/fd0340a93bb8d998abb9
#include <chrono>

namespace flash {
  class Timer {
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1>> second_;
    std::chrono::time_point<clock_> m_beg;

   public:
    Timer() : m_beg(clock_::now()) {
    }

    void reset() {
      m_beg = clock_::now();
    }

    // returns elapsed time in `ms`
    float elapsed() const {
      return std::chrono::duration_cast<std::chrono::milliseconds>(
                 clock_::now() - m_beg)
          .count();
    }
  };
}