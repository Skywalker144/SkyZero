#ifndef SKYZERO_NUMPYWRITE_H
#define SKYZERO_NUMPYWRITE_H

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

/*
  NumpyBuffer: Creates an in-memory .npy array with a header.
  ZipFile: Writes multiple .npy buffers into a .npz (zip) file.
  Adapted from KataGo's numpywrite.h/.cpp for SkyZero_V4.
*/

template <typename T>
struct NumpyBuffer {
    T* dataIncludingHeader;
    T* data;
    int64_t headerLen;
    int64_t dataLen;
    std::vector<int64_t> shape;
    std::string dtype;
    size_t shapeStartByte;

    static const int TOTAL_HEADER_BYTES = 256;

    NumpyBuffer(const std::vector<int64_t>& shp, const char* dt);
    ~NumpyBuffer();

    NumpyBuffer(const NumpyBuffer&) = delete;
    NumpyBuffer& operator=(const NumpyBuffer&) = delete;

    uint64_t prepareHeaderWithNumRows(int64_t numWriteableRows);
};

class ZipFile {
public:
    ZipFile(const std::string& fileName);
    ~ZipFile();

    ZipFile(const ZipFile&) = delete;
    ZipFile& operator=(const ZipFile&) = delete;

    void writeBuffer(const char* nameWithinZip, void* data, uint64_t numBytes);
    void close();

private:
    std::string fileName;
    void* file;
};

// ---- Template implementations ----

template <typename T>
NumpyBuffer<T>::NumpyBuffer(const std::vector<int64_t>& shp, const char* dt)
    : shape(shp), dtype(dt)
{
    dataLen = 1;
    assert(shape.size() > 0);
    for (size_t i = 0; i < shape.size(); i++) {
        assert(shape[i] >= 0);
        dataLen *= shape[i];
    }

    int sizeOfT = sizeof(T);
    assert(sizeOfT > 0 && sizeOfT <= TOTAL_HEADER_BYTES);

    headerLen = TOTAL_HEADER_BYTES / sizeOfT;
    assert(headerLen * sizeOfT == TOTAL_HEADER_BYTES);

    dataIncludingHeader = new T[headerLen + dataLen];
    data = dataIncludingHeader + headerLen;

    char* s = (char*)dataIncludingHeader;
    s[0] = (char)0x93;
    s[1] = 'N'; s[2] = 'U'; s[3] = 'M'; s[4] = 'P'; s[5] = 'Y';
    s[6] = 0x1; s[7] = 0x0;
    s[8] = (char)((TOTAL_HEADER_BYTES - 10) & 0xFF);
    s[9] = (char)((TOTAL_HEADER_BYTES - 10) >> 8);

    char dictBuf[128];
    snprintf(dictBuf, sizeof(dictBuf), "{'descr':'%s','fortran_order':False,'shape':(", dt);
    std::string dictStr(dictBuf);

    if (dictStr.size() > TOTAL_HEADER_BYTES - 40)
        throw std::runtime_error("Numpy header dict too long");
    strcpy(s + 10, dictStr.c_str());
    shapeStartByte = dictStr.size() + 10;
}

template <typename T>
NumpyBuffer<T>::~NumpyBuffer() {
    delete[] dataIncludingHeader;
}

template <typename T>
uint64_t NumpyBuffer<T>::prepareHeaderWithNumRows(int64_t numWriteableRows) {
    size_t idx = shapeStartByte;
    char* s = (char*)dataIncludingHeader;

    int64_t actualDataLen = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        int64_t x = (i == 0) ? numWriteableRows : shape[i];
        actualDataLen *= x;

        int numDigits = 0;
        char digitsRev[32];
        if (x == 0) { digitsRev[0] = '0'; numDigits = 1; }
        else { while (x > 0) { digitsRev[numDigits++] = '0' + (x % 10); x /= 10; } }

        for (int j = numDigits - 1; j >= 0; j--) {
            s[idx] = digitsRev[j]; idx += 1;
        }
        s[idx] = ','; idx += 1;
    }
    s[idx] = ')'; idx += 1;
    s[idx] = '}'; idx += 1;

    while (idx < TOTAL_HEADER_BYTES - 1) { s[idx] = ' '; idx += 1; }
    s[idx] = '\n'; idx += 1;

    return (uint64_t)(TOTAL_HEADER_BYTES + actualDataLen * sizeof(T));
}

// ---- Convenience factory for common types ----

inline NumpyBuffer<float>* makeNumpyBufferFloat(const std::vector<int64_t>& shape) {
    return new NumpyBuffer<float>(shape, "<f4");
}
inline NumpyBuffer<int8_t>* makeNumpyBufferInt8(const std::vector<int64_t>& shape) {
    return new NumpyBuffer<int8_t>(shape, "|i1");
}

#endif // SKYZERO_NUMPYWRITE_H
