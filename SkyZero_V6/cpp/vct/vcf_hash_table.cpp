#include "vcf_hash_table.h"

namespace skyzero {
namespace vct {

VCFHashTable::VCFHashTable(int sizePowerOfTwo, int mutexPoolSizePowerOfTwo) {
    tableSize_ = ((uint64_t)1) << sizePowerOfTwo;
    tableMask_ = tableSize_ - 1;
    entries_ = new Entry[tableSize_];
    uint32_t mutexPoolSize = ((uint32_t)1) << mutexPoolSizePowerOfTwo;
    mutexPoolMask_ = mutexPoolSize - 1;
    mutexPool_ = new MutexPool(mutexPoolSize);
}

VCFHashTable::~VCFHashTable() {
    delete[] entries_;
    delete mutexPool_;
}

int64_t VCFHashTable::get(Hash128 hash) {
    uint64_t idx = hash.hash0 & tableMask_;
    uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask_;
    Entry& entry = entries_[idx];
    std::mutex& mtx = mutexPool_->getMutex(mutexIdx);
    std::lock_guard<std::mutex> lock(mtx);
    if (entry.hash == hash) {
        return entry.result;
    }
    return 0;
}

void VCFHashTable::set(Hash128 hash, int64_t result) {
    uint64_t idx = hash.hash0 & tableMask_;
    uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask_;
    Entry& entry = entries_[idx];
    std::mutex& mtx = mutexPool_->getMutex(mutexIdx);
    std::lock_guard<std::mutex> lock(mtx);
    entry.hash = hash;
    entry.result = result;
}

}  // namespace vct
}  // namespace skyzero
