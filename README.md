# BLAS-on-Flash Project Files
BLAS-on-Flash implements a small subset of BLAS routines (BLAS-2/3) that work with matrices/vectors stored on flash storage.
In addition to BLAS routines, BlasonFlash also provides other routines like kmeans, sort, map, reduce for large-scale processing on disk-resident data.
Currently, only the following routines are supported.
- `_gemm`
- `_csrmm`
- `_csrgemv`
- `_csrcsc`
- `sort`
- `kmeans`
- `map`
- `reduce`

# Requirements
- Ubuntu 16.04 or newer running Linux Kernel v4.13 or newer (Older kernels have issues setting `nr_requests` parameter)
- `cat /sys/block/<dev>/queue/nr_requests` is at least `32768`
- Intel MKL 2017+ is installed (usually in `/opt/intel`) and added to LD_PRELOAD path; See `CMakeLists.txt` for finer control over MKL paths
- `libaio-dev` package installed
- `make` and `cmake` installed with `cmake > 3.0.0`

# Build instructions
- `git clone ssh://git@github.com/Microsoft/BLAS-on-flash blas-on-flash`
- `cd blas-on-flash`
- `mkdir bin && cd bin`
- `cmake -DCMAKE_BUILD_TYPE=X ..` where `X=Debug` or `X=Release`
- `make -Bj`

# Contributing
This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
