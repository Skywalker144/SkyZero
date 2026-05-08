#ifndef SKYZERO_VCT_VCF_HASH_TABLE_H
#define SKYZERO_VCT_VCF_HASH_TABLE_H

// Lifted from KataGomo cpp/vcfsolver/VCFHashTable.{h,cpp}. Only changes:
//  * include path / namespace
//  * remove the commented-out duplicate MutexPool struct
//  * thin Entry has trivial ctor/dtor inline

#include <cstdint>
#include <mutex>

#include "hash128.h"
#include "mutex_pool.h"

namespace skyzero {
namespace vct {

class VCFHashTable {
public:
    VCFHashTable(int sizePowerOfTwo, int mutexPoolSizePowerOfTwo);
    ~VCFHashTable();

    VCFHashTable(const VCFHashTable&) = delete;
    VCFHashTable& operator=(const VCFHashTable&) = delete;

    // Thread-safe lookup. Returns 0 on miss.
    int64_t get(Hash128 hash);
    // Thread-safe insert / overwrite.
    void set(Hash128 hash, int64_t result);

private:
    struct Entry {
        Hash128 hash;
        int64_t result;
        Entry() : hash(), result(0) {}
    };

    Entry* entries_;
    MutexPool* mutexPool_;
    uint64_t tableSize_;
    uint64_t tableMask_;
    uint32_t mutexPoolMask_;
};

}  // namespace vct
}  // namespace skyzero

#endif
