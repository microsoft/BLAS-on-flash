// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "../bof_types.h"
#include "../pointers/pointer.h"
#include "../tasks/task.h"
#include "../utils.h"
#include "io_executor.h"

namespace flash {
  struct Key {
    // main fields
    flash_ptr<void> fptr;
    StrideInfo      sinfo = {0, 0, 0};
    // hash shortcut
    uint64_t hash_value = 0;

    Key(flash_ptr<void> fptr, StrideInfo sinfo) : fptr(fptr), sinfo(sinfo) {
      this->hash_value = flash::fnv64a((const char *) this,
                                       sizeof(Key) - sizeof(this->hash_value));
    }

    Key(const Key &copy_from) {
      this->fptr = copy_from.fptr;
      this->sinfo = copy_from.sinfo;
      this->hash_value = copy_from.hash_value;
    }

    bool operator==(const Key &other) const {
      bool hash_eq = (this->hash_value == other.hash_value);
#ifdef DEBUG
      bool classic_eq =
          (this->fptr == other.fptr) && (this->sinfo == other.sinfo);
      GLOG_ASSERT(classic_eq == hash_eq, "welp");
#endif
      return hash_eq;
    }

    operator std::string() const {
      return std::string(fptr) + "-" + std::string(sinfo);
    }
  };

  struct Value {
    // main fields
    void *     buf = nullptr;
    FBLAS_UINT n_refs = 0;
    bool       write_back = false;

    /* used by other functions */
    bool               evicted = false;     // used for eviction
    bool               alloc_only = false;  // used when buf init
    std::atomic<bool> *complete = nullptr;  // used for I/O tracking
  };
}  // namespace flash

// specialization of hash function for cache keys
namespace std {
  template<>
  struct hash<flash::Key> {
    inline std::size_t operator()(const flash::Key &k) const {
      if (k.hash_value) {
        return k.hash_value;
      } else {
        GLOG_FAIL("bad copy");
        return 0;
      }
    }
  };
}  // namespace std

namespace flash {
  class Cache {
    // <key-buffer> maps
    // Value.n_refs >= 0,  no I/O in progress, in-use|promised
    std::unordered_map<Key, Value> active_map;

    // Value.n_refs >= 0, I/O in progress, unused|not-promised
    std::unordered_map<Key, Value> io_map;

    // Value.n_refs == 0, no I/O in progress, unused|not-promised
    std::unordered_map<Key, Value> zero_ref_map;

    // backlogs
    // each entry in `alloc_backlog` is waiting for some evicted key
    // to get flushed before its budget is re-allocated to this key
    // `alloc_backlog[key].alloc_only = false` => need to read from disk
    std::deque<std::pair<Key, Value>> alloc_backlog;

    // If `true`, discards ecah buffer after a single use
    // Default : `false`
    bool single_use_discard;

    // access serialization for cache primitives
    typedef std::unique_lock<std::mutex> mutex_locker;
    std::mutex                           cache_mut;

    // actual memory footprint
    std::atomic<FBLAS_UINT> real_size;

    // `promised`/`commit`-ed footprint
    FBLAS_UINT commit_size;

    // max `actual` footprint
    const FBLAS_UINT max_size;

    // IO executor
    IoExecutor &io_exec;

    /*  helper functions  */
    inline bool is_active(const Key &key) const {
      return this->active_map.find(key) != this->active_map.end();
    }

    inline bool is_in_io(const Key &key) const {
      return this->io_map.find(key) != this->io_map.end();
    }

    inline bool is_zero_ref(const Key &key) const {
      return this->zero_ref_map.find(key) != this->zero_ref_map.end();
    }

    inline bool is_queued(const Key &key) const {
      // return this->alloc_backlog.find(key) != this->alloc_backlog.end();
      for (auto &k_v : this->alloc_backlog) {
        if (k_v.first == key) {
          return true;
        }
      }

      return false;
    }

    inline bool has_spare_mem_for(const FBLAS_UINT req_size) const {
      return (this->commit_size + req_size) <= this->max_size;
    }

    inline bool has_spare_real_mem_for(const FBLAS_UINT req_size) const {
      return (this->real_size.load() + req_size <= this->max_size);
    }

    // evicts `evict_keys` from `zero_ref_map`
    // issues writes if `zero_ref_map[k].write_back == true`
    void evict(const std::unordered_set<Key> &evict_keys);

    // reduce `commit_size` by at least `evict_size`, but don't drop
    // `exclude_keys`
    // returns `true` if successful, `false` otherwise
    // if returns `true`, function also issues eviction orders
    // if returns `false`, no eviction orders are issued
    bool try_evict(const std::unordered_set<Key> &exclude_keys,
                   const FBLAS_UINT               evict_size);

    // adds `k` to a backlog
    // when `has_spare_mem_for(buf_size(k.sinfo)) == true`,
    //    `v.buf` is malloc'ed and reads issued (if required)
    //    * NOTE :: malloc'ing happens in `service_backlog`
    void add_backlog(const Key &k, bool alloc_only, bool write_back);

    // deletes `io_map[k].complete`
    // calling function must explicitly move from `io_map` to `zero_ref_map` or
    // `active_map` (if `!io_map[k].evicted`)
    // calling function must explicitly remove from `io_map` if
    // `io_map[k].evicted`
    void reap_io_completion(const Key &k);

    // state transition functions
    void move_active_to_zero(const Key &k);  // after becoming 0-ref
    // NOTE :: sets `active_map[k].n_refs = 1` to maintain invariance
    void move_zero_to_active(const Key &k);  // after cache-hit
    // NOTE :: sets `active_map[k].n_refs = 1` to maintain invariance
    void move_io_to_active(const Key &k);  // after I/O completion

    // `claims` buffers for `tsk` and issues I/O requests for bufs not in cache
    void alloc_bufs(BaseTask *tsk);

   public:
    Cache(IoExecutor &io_exec, const FBLAS_UINT max_size);

    ~Cache();

    // returns `non-nullptr` if the query is already cached
    // * if access in `active_map`, returns `buf`
    // * if access in `zero_ref_map`, moves to `active_map`, and returns `buf`
    // * if access in `io_map` and `NOT evicted` and `complete`, reaps
    // completion, moves to `active_map`, and returns `buf`
    // * else : returns `nullptr`
    void *get_buf(const flash_ptr<void> &fptr, const StrideInfo &sinfo,
                  bool write_back);

    // returns `true` if buffers successfully alloc'ed
    // if returns `false`, cache-state may still be changed by trying to evict
    bool allocate(BaseTask *tsk);

    // reduces reference count in `active_map`
    // moves a key `k` to `zero_ref_map` if `active_map[k].n_refs == 0` to
    // maintain invariance
    void release(const BaseTask *tsk);

    // cleans up `io_map` be reaping completed I/O requests
    // if `has_spare_mem_for(buf_size(k.sinfo)) == true` for `k` in
    // `alloc_backlog`,
    //    mallocs `v.buf` and issues I/O request if required
    void service_backlog();

    // flushes all write-back entries in cache
    // drops all ready-only entries
    // WARNING : program exits FATALLY if entries are active
    void flush();

    // drops keys if in cache
    void drop_if_in_cache(std::unordered_set<Key> &keys);
    // keeps keys if in cache
    void keep_if_in_cache(std::unordered_set<Key> &keys);

    // Allow `Scheduler` to access private variables
    friend class Scheduler;
  };
}  // namespace flash
