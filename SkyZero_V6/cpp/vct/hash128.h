#ifndef SKYZERO_VCT_HASH128_H
#define SKYZERO_VCT_HASH128_H

// Lifted from KataGomo cpp/core/hash.h. Slimmed to only the inline operators
// VCFsolver actually uses (==, ^, ^=, hash0/hash1 access). Stream / toString
// / mixInt / Hash:: namespace utilities dropped. Header-only.

#include <cstdint>

namespace skyzero {
namespace vct {

struct Hash128 {
    uint64_t hash0;
    uint64_t hash1;

    constexpr Hash128() : hash0(0), hash1(0) {}
    constexpr Hash128(uint64_t h0, uint64_t h1) : hash0(h0), hash1(h1) {}

    bool operator==(const Hash128 other) const {
        return hash0 == other.hash0 && hash1 == other.hash1;
    }
    bool operator!=(const Hash128 other) const {
        return hash0 != other.hash0 || hash1 != other.hash1;
    }
    Hash128 operator^(const Hash128 other) const {
        return Hash128(hash0 ^ other.hash0, hash1 ^ other.hash1);
    }
    Hash128& operator^=(const Hash128 other) {
        hash0 ^= other.hash0;
        hash1 ^= other.hash1;
        return *this;
    }
};

}  // namespace vct
}  // namespace skyzero

#endif
