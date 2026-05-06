#ifndef PTH_LOADER_H
#define PTH_LOADER_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "half.hpp"

namespace rwkvmobile {

enum class PthDType {
    Float32,
    Float16,
    BFloat16,
};

struct PthTensorInfo {
    std::string name;
    PthDType dtype;
    std::vector<int64_t> shape;
    std::vector<int64_t> stride;
    uint64_t offset = 0;
    uint64_t storage_size = 0;
};

class PthFile {
public:
    explicit PthFile(const std::string &path);
    ~PthFile();

    PthFile(const PthFile &) = delete;
    PthFile &operator=(const PthFile &) = delete;

    bool has_tensor(const std::string &name) const;
    const PthTensorInfo *get_tensor_info(const std::string &name) const;
    std::vector<std::string> tensor_names() const;

    std::vector<float> read_tensor_float32(const std::string &name) const;
    std::vector<half_float::half> read_tensor_float16(const std::string &name) const;

private:
    struct Impl;
    Impl *impl_;
};

std::vector<std::vector<half_float::half>> load_pth_time_states(
    const std::string &path,
    int expected_layers,
    int expected_hidden_size);

} // namespace rwkvmobile

#endif
