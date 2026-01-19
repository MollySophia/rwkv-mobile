#ifndef RMPACK_H
#define RMPACK_H

#include "json.hpp"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <iostream>

using json = nlohmann::json;

class RMPackReader {
public:
    struct FileInfo {
        std::string filename;
        uint64_t size;
        uint64_t offset;
    };

    struct MMapInfo {
        void* addr;
        size_t size;
    };

    struct MemoryInfo {
        char* data;
        size_t size;
    };

    explicit RMPackReader(const std::string& file_path);
    ~RMPackReader();

    RMPackReader(const RMPackReader&) = delete;
    RMPackReader& operator=(const RMPackReader&) = delete;

    const json& getConfig() const;
    const std::vector<FileInfo>& getFiles() const;
    const FileInfo* getFileInfo(const std::string& filename) const;
    size_t getFileSize(const std::string& filename) const;
    bool hasFile(const std::string& filename) const;

    void* mmapFile(const std::string& filename);
    void unmapFile(const std::string& filename);

    void* readFileToMemory(const std::string& filename);
    void freeFileMemory(const std::string& filename);

private:
    static const char* MAGIC_HEADER;
    static const size_t MAGIC_HEADER_SIZE;

    std::string file_path_;
    int fd_;
    std::ifstream* file_;
    size_t file_size_;
    json config_;
    std::vector<FileInfo> files_;

    std::map<std::string, MMapInfo> mmap_mappings_;

    std::map<std::string, MemoryInfo> memory_data_;

    void loadFile();
};

#endif // RMPACK_H