#include "numpywrite.h"
#include <zip.h>

// ---- ZipFile implementation using libzip ----

struct ZipError {
    zip_error_t value;
    ZipError() { zip_error_init(&value); }
    ~ZipError() { zip_error_fini(&value); }
    ZipError(const ZipError&) = delete;
    ZipError& operator=(const ZipError&) = delete;
};

ZipFile::ZipFile(const std::string& fName)
    : fileName(fName), file(NULL)
{
    ZipError zipError;
    zip_source_t* zipFileSource = zip_source_file_create(fileName.c_str(), 0, -1, &(zipError.value));
    if (zipFileSource == NULL)
        throw std::runtime_error("Could not open zip file " + fileName + ": " + zip_error_strerror(&(zipError.value)));
    zip_t* fileHandle = zip_open_from_source(zipFileSource, ZIP_CREATE | ZIP_TRUNCATE, &(zipError.value));
    file = fileHandle;
    if (file == NULL) {
        zip_source_free(zipFileSource);
        throw std::runtime_error("Could not open zip file " + fileName + ": " + zip_error_strerror(&(zipError.value)));
    }
}

ZipFile::~ZipFile() {
    if (file != NULL)
        zip_discard((zip_t*)file);
}

void ZipFile::writeBuffer(const char* nameWithinZip, void* data, uint64_t numBytes) {
    ZipError zipError;
    zip_source_t* dataSource = zip_source_buffer((zip_t*)file, data, numBytes, 0);
    if (dataSource == NULL)
        throw std::runtime_error(
            std::string("Could not create zip buffer for ") + nameWithinZip +
            " in " + fileName + ": " + zip_error_strerror(&(zipError.value)));

    zip_int64_t idx = zip_file_add((zip_t*)file, nameWithinZip, dataSource, ZIP_FL_OVERWRITE);
    if (idx < 0) {
        zip_source_free(dataSource);
        throw std::runtime_error(
            std::string("Could not write ") + nameWithinZip +
            " to zip " + fileName + ": " + zip_strerror((zip_t*)file));
    }
}

void ZipFile::close() {
    int result = zip_close((zip_t*)file);
    if (result < 0)
        throw std::runtime_error("Could not close zip file " + fileName + ": " + zip_strerror((zip_t*)file));
    else
        file = NULL;
}
