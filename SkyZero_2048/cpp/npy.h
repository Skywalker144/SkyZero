#ifndef SKYZERO_NPY_H
#define SKYZERO_NPY_H

// Minimal NumPy .npy v1.0 buffer builder (little-endian hosts). Just enough for
// the 2048 self-play npz writer. Extracted from the inherited Gomoku
// npz_writer.h so the 2048 build has no Gomoku dependency.

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace skyzero {

inline std::vector<uint8_t> make_npy_buffer(
    const std::string& descr,                 // e.g. "|i1" or "<f4"
    const std::vector<int64_t>& shape,
    const void* data,
    size_t nbytes
) {
    std::ostringstream header;
    header << "{'descr': '" << descr << "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        header << shape[i];
        if (shape.size() == 1 || i + 1 < shape.size()) {
            header << ", ";
        }
    }
    header << "), }";
    std::string header_str = header.str();

    // Pad so (10 + header_len) is a multiple of 64 and ends with '\n'.
    const size_t unpadded = 10 + header_str.size() + 1;
    const size_t padded = ((unpadded + 63) / 64) * 64;
    header_str.append(padded - unpadded, ' ');
    header_str.push_back('\n');

    std::vector<uint8_t> buf;
    buf.reserve(10 + header_str.size() + nbytes);
    const uint8_t magic[] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    buf.insert(buf.end(), std::begin(magic), std::end(magic));
    buf.push_back(0x01);
    buf.push_back(0x00);
    const uint16_t hlen = static_cast<uint16_t>(header_str.size());
    buf.push_back(static_cast<uint8_t>(hlen & 0xFF));
    buf.push_back(static_cast<uint8_t>((hlen >> 8) & 0xFF));
    buf.insert(buf.end(), header_str.begin(), header_str.end());
    const auto* bytes = reinterpret_cast<const uint8_t*>(data);
    buf.insert(buf.end(), bytes, bytes + nbytes);
    return buf;
}

}  // namespace skyzero

#endif
