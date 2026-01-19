#include "rmpack.h"
#include <fstream>
#include <stdexcept>
#include <cstring>
#ifndef _WIN32
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif
#include "json.hpp"

using json = nlohmann::json;

const char* RMPackReader::MAGIC_HEADER = "RWKVMBLE";
const size_t RMPackReader::MAGIC_HEADER_SIZE = 8;

RMPackReader::RMPackReader(const std::string& file_path) : file_path_(file_path), fd_(-1) {
    loadFile();
}

RMPackReader::~RMPackReader() {
#ifndef _WIN32
    for (auto& mapping : mmap_mappings_) {
        if (mapping.second.addr != nullptr) {
            munmap(mapping.second.addr, mapping.second.size);
        }
    }
#endif
    for (auto& memory : memory_data_) {
        if (memory.second.data != nullptr) {
            delete[] memory.second.data;
        }
    }
#ifndef _WIN32
    if (fd_ != -1) {
        close(fd_);
    }
#endif
    if (file_ != nullptr) {
        file_->close();
        delete file_;
    }
}

void RMPackReader::loadFile() {
#ifndef _WIN32
    fd_ = open(file_path_.c_str(), O_RDONLY);
    if (fd_ == -1) {
        throw std::runtime_error("failed to open file: " + file_path_);
    }
#endif
    file_ = new std::ifstream(file_path_, std::ios::binary);
    if (!file_->is_open()) {
        throw std::runtime_error("failed to open file: " + file_path_);
    }

    file_->seekg(0, std::ios::end);
    file_size_ = file_->tellg();
    file_->seekg(0, std::ios::beg);

    char header[MAGIC_HEADER_SIZE];
    file_->read(header, MAGIC_HEADER_SIZE);
    if (file_->fail()) {
        throw std::runtime_error("failed to read file: " + file_path_);
    }

    if (std::memcmp(header, MAGIC_HEADER, MAGIC_HEADER_SIZE) != 0) {
        throw std::runtime_error("invalid rwkv model file format: " + file_path_);
    }

    uint32_t config_len;
    file_->read(reinterpret_cast<char*>(&config_len), sizeof(config_len));
    if (file_->fail()) {
        throw std::runtime_error("failed to read config length");
    }

    std::vector<char> config_buffer(config_len);
    file_->read(config_buffer.data(), config_len);
    if (file_->fail()) {
        throw std::runtime_error("failed to read config");
    }

    try {
        std::string config_str(config_buffer.begin(), config_buffer.end());
        config_ = json::parse(config_str);
    } catch (const json::exception& e) {
        throw std::runtime_error("failed to parse config json: " + std::string(e.what()));
    }

    uint32_t file_count;
    file_->read(reinterpret_cast<char*>(&file_count), sizeof(file_count));
    if (file_->fail()) {
        throw std::runtime_error("failed to read file count");
    }

    for (uint32_t i = 0; i < file_count; ++i) {
        FileInfo file_info;

        uint32_t filename_len;
        file_->read(reinterpret_cast<char*>(&filename_len), sizeof(filename_len));
        if (file_->fail()) {
            throw std::runtime_error("failed to read file name length");
        }

        std::vector<char> filename_buffer(filename_len);
        file_->read(filename_buffer.data(), filename_len);
        if (file_->fail()) {
            throw std::runtime_error("failed to read file name");
        }
        file_info.filename = std::string(filename_buffer.begin(), filename_buffer.end());

        file_->read(reinterpret_cast<char*>(&file_info.size), sizeof(file_info.size));
        if (file_->fail()) {
            throw std::runtime_error("failed to read file size");
        }

        file_->read(reinterpret_cast<char*>(&file_info.offset), sizeof(file_info.offset));
        if (file_->fail()) {
            throw std::runtime_error("failed to read file offset");
        }

        files_.push_back(file_info);
    }
}

const json& RMPackReader::getConfig() const {
    return config_;
}

const std::vector<RMPackReader::FileInfo>& RMPackReader::getFiles() const {
    return files_;
}

const RMPackReader::FileInfo* RMPackReader::getFileInfo(const std::string& filename) const {
    for (const auto& file : files_) {
        if (file.filename == filename) {
            return &file;
        }
    }
    return nullptr;
}

void* RMPackReader::mmapFile(const std::string& filename) {
#ifndef _WIN32
    auto it = mmap_mappings_.find(filename);
    if (it != mmap_mappings_.end()) {
        return it->second.addr;
    }

    const FileInfo* file_info = getFileInfo(filename);
    if (!file_info) {
        throw std::runtime_error("file not found: " + filename);
    }

    void* addr = mmap(nullptr, file_info->size, PROT_READ, MAP_PRIVATE, fd_, file_info->offset);
    if (addr == MAP_FAILED) {
        throw std::runtime_error("failed to mmap file: " + filename);
    }

    MMapInfo mmap_info;
    mmap_info.addr = addr;
    mmap_info.size = file_info->size;
    mmap_mappings_[filename] = mmap_info;

    return addr;
#else
    return nullptr;
#endif
}

void RMPackReader::unmapFile(const std::string& filename) {
#ifndef _WIN32
    auto it = mmap_mappings_.find(filename);
    if (it == mmap_mappings_.end()) {
        return;
    }

    if (it->second.addr != nullptr) {
        munmap(it->second.addr, it->second.size);
    }

    mmap_mappings_.erase(it);
#endif
}

void* RMPackReader::readFileToMemory(const std::string& filename) {
    auto it = memory_data_.find(filename);
    if (it != memory_data_.end()) {
        return it->second.data;
    }

    const FileInfo* file_info = getFileInfo(filename);
    if (!file_info) {
        throw std::runtime_error("file not found: " + filename);
    }

    char* data = new char[file_info->size];

    auto current_pos = file_->tellg();
    file_->seekg(file_info->offset, std::ios::beg);

    if (file_->fail()) {
        delete[] data;
        throw std::runtime_error("failed to seek file: " + filename);
    }

    file_->read(data, file_info->size);
    if (file_->fail() || file_->gcount() != static_cast<std::streamsize>(file_info->size)) {
        delete[] data;
        throw std::runtime_error("failed to read file: " + filename);
    }

    file_->seekg(current_pos);

    MemoryInfo memory_info;
    memory_info.data = data;
    memory_info.size = file_info->size;
    memory_data_[filename] = memory_info;

    return data;
}

void RMPackReader::freeFileMemory(const std::string& filename) {
    auto it = memory_data_.find(filename);
    if (it == memory_data_.end()) {
        return;
    }

    if (it->second.data != nullptr) {
        delete[] it->second.data;
    }

    memory_data_.erase(it);
}

size_t RMPackReader::getFileSize(const std::string& filename) const {
    const FileInfo* file_info = getFileInfo(filename);
    return file_info ? file_info->size : 0;
}

bool RMPackReader::hasFile(const std::string& filename) const {
    return getFileInfo(filename) != nullptr;
}
