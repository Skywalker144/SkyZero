#ifndef SKYZERO_VCT_MUTEX_POOL_H
#define SKYZERO_VCT_MUTEX_POOL_H

// Lifted from KataGomo cpp/search/mutexpool.{h,cpp}. Multithread.h wrapper
// dropped — V6 unconditionally has std::mutex available.

#include <cstdint>
#include <mutex>

namespace skyzero {
namespace vct {

class MutexPool {
public:
    explicit MutexPool(uint32_t n) : numMutexes_(n), mutexes_(new std::mutex[n]) {}
    ~MutexPool() { delete[] mutexes_; }
    MutexPool(const MutexPool&) = delete;
    MutexPool& operator=(const MutexPool&) = delete;

    uint32_t getNumMutexes() const { return numMutexes_; }
    std::mutex& getMutex(uint32_t idx) { return mutexes_[idx]; }

private:
    uint32_t numMutexes_;
    std::mutex* mutexes_;
};

}  // namespace vct
}  // namespace skyzero

#endif
