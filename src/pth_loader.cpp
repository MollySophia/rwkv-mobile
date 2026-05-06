#include "pth_loader.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <utility>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace rwkvmobile {
namespace {

uint16_t rd16(const uint8_t *p) {
    return uint16_t(p[0]) | (uint16_t(p[1]) << 8);
}

uint32_t rd32(const uint8_t *p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
           (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

uint64_t rd64(const uint8_t *p) {
    return uint64_t(rd32(p)) | (uint64_t(rd32(p + 4)) << 32);
}

template <typename T>
T load_scalar(const uint8_t *p) {
    T v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

float bf16_to_float(uint16_t x) {
    uint32_t u = uint32_t(x) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

float fp16_to_float(uint16_t x) {
    half_float::half h;
    std::memcpy(&h, &x, sizeof(x));
    return static_cast<float>(h);
}

half_float::half half_from_bits(uint16_t x) {
    half_float::half h;
    std::memcpy(&h, &x, sizeof(x));
    return h;
}

uint64_t numel(const std::vector<int64_t> &shape) {
    uint64_t n = 1;
    for (int64_t d : shape) {
        if (d < 0) throw std::runtime_error("negative tensor dimension");
        n *= uint64_t(d);
    }
    return n;
}

bool contiguous(const std::vector<int64_t> &shape, const std::vector<int64_t> &stride) {
    int64_t expect = 1;
    for (size_t i = shape.size(); i > 0; --i) {
        const size_t j = i - 1;
        if (shape[j] != 1 && stride[j] != expect) return false;
        expect *= shape[j];
    }
    return true;
}

size_t dtype_size(PthDType dtype) {
    return dtype == PthDType::Float32 ? 4 : 2;
}

struct MappedFile {
    const uint8_t *data = nullptr;
    size_t size = 0;
#ifndef _WIN32
    int fd = -1;
#else
    std::vector<uint8_t> owned;
#endif

    explicit MappedFile(const std::string &path) {
#ifndef _WIN32
        fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("failed to open pth file: " + path);
        struct stat st {};
        if (fstat(fd, &st) != 0 || st.st_size <= 0) {
            throw std::runtime_error("failed to stat pth file: " + path);
        }
        size = static_cast<size_t>(st.st_size);
        void *p = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (p == MAP_FAILED) throw std::runtime_error("failed to mmap pth file: " + path);
        data = static_cast<const uint8_t *>(p);
#else
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("failed to open pth file: " + path);
        f.seekg(0, std::ios::end);
        size = static_cast<size_t>(f.tellg());
        f.seekg(0, std::ios::beg);
        owned.resize(size);
        f.read(reinterpret_cast<char *>(owned.data()), static_cast<std::streamsize>(size));
        if (!f) throw std::runtime_error("failed to read pth file: " + path);
        data = owned.data();
#endif
    }

    ~MappedFile() {
#ifndef _WIN32
        if (data) munmap(const_cast<uint8_t *>(data), size);
        if (fd >= 0) close(fd);
#endif
    }
};

struct ZipEntry {
    std::string name;
    uint16_t method = 0;
    uint64_t size = 0;
    uint64_t local = 0;
    uint64_t data = 0;
};

struct Zip {
    MappedFile file;
    std::vector<ZipEntry> entries;

    explicit Zip(const std::string &path) : file(path) {
        const uint8_t *bytes = file.data;
        const size_t nbytes = file.size;
        if (nbytes < 22) throw std::runtime_error("bad pth zip");
        const size_t min_pos = nbytes > 65557 ? nbytes - 65557 : 0;
        size_t eocd = std::string::npos;
        for (size_t p = nbytes - 22;; --p) {
            if (rd32(bytes + p) == 0x06054b50u) {
                eocd = p;
                break;
            }
            if (p == min_pos) break;
        }
        if (eocd == std::string::npos) throw std::runtime_error("pth zip eocd not found");

        uint64_t count = rd16(bytes + eocd + 10);
        uint64_t cd_offset = rd32(bytes + eocd + 16);
        if ((count == 0xffffu || cd_offset == 0xffffffffu) && eocd >= 20 &&
            rd32(bytes + eocd - 20) == 0x07064b50u) {
            const uint64_t eocd64 = rd64(bytes + eocd - 12);
            if (eocd64 + 56 > nbytes || rd32(bytes + eocd64) != 0x06064b50u) {
                throw std::runtime_error("bad pth zip64 eocd");
            }
            count = rd64(bytes + eocd64 + 32);
            cd_offset = rd64(bytes + eocd64 + 48);
        }

        size_t p = static_cast<size_t>(cd_offset);
        for (uint64_t i = 0; i < count; ++i) {
            if (p + 46 > nbytes || rd32(bytes + p) != 0x02014b50u) {
                throw std::runtime_error("bad pth zip central dir");
            }
            ZipEntry e;
            e.method = rd16(bytes + p + 10);
            e.size = rd32(bytes + p + 24);
            uint64_t compressed_size = rd32(bytes + p + 20);
            e.local = rd32(bytes + p + 42);
            const uint16_t name_len = rd16(bytes + p + 28);
            const uint16_t extra_len = rd16(bytes + p + 30);
            const uint16_t comment_len = rd16(bytes + p + 32);
            if (p + 46ull + name_len + extra_len + comment_len > nbytes) {
                throw std::runtime_error("truncated pth zip central dir");
            }
            e.name.assign(reinterpret_cast<const char *>(bytes + p + 46), name_len);

            const uint8_t *extra = bytes + p + 46 + name_len;
            size_t ep = 0;
            if (e.size == 0xffffffffull || compressed_size == 0xffffffffull || e.local == 0xffffffffull) {
                while (ep + 4 <= extra_len) {
                    const uint16_t tag = rd16(extra + ep);
                    const uint16_t len = rd16(extra + ep + 2);
                    ep += 4;
                    if (ep + len > extra_len) break;
                    if (tag == 0x0001) {
                        size_t zp = ep;
                        if (e.size == 0xffffffffull && zp + 8 <= ep + len) {
                            e.size = rd64(extra + zp);
                            zp += 8;
                        }
                        if (compressed_size == 0xffffffffull && zp + 8 <= ep + len) {
                            compressed_size = rd64(extra + zp);
                            zp += 8;
                        }
                        if (e.local == 0xffffffffull && zp + 8 <= ep + len) {
                            e.local = rd64(extra + zp);
                        }
                        break;
                    }
                    ep += len;
                }
            }

            if (e.local + 30 > nbytes || rd32(bytes + static_cast<size_t>(e.local)) != 0x04034b50u) {
                throw std::runtime_error("bad pth zip local header");
            }
            e.data = e.local + 30 +
                     rd16(bytes + static_cast<size_t>(e.local + 26)) +
                     rd16(bytes + static_cast<size_t>(e.local + 28));
            if (e.data + e.size > nbytes) throw std::runtime_error("pth zip entry out of range");
            entries.push_back(std::move(e));
            p += 46 + name_len + extra_len + comment_len;
        }
    }

    const ZipEntry *find(const std::string &name) const {
        for (const auto &e : entries) {
            if (e.name == name) return &e;
        }
        return nullptr;
    }

    std::vector<uint8_t> read(const ZipEntry &e) const {
        if (e.method != 0) throw std::runtime_error("compressed pth zip entry is not supported: " + e.name);
        return std::vector<uint8_t>(file.data + e.data, file.data + e.data + e.size);
    }

    const uint8_t *data(const ZipEntry &e) const {
        if (e.method != 0) throw std::runtime_error("compressed pth zip entry is not supported: " + e.name);
        return file.data + e.data;
    }
};

struct TensorRecord {
    std::string name;
    std::string storage_key;
    PthDType dtype = PthDType::BFloat16;
    uint64_t storage_size = 0;
    uint64_t offset = 0;
    std::vector<int64_t> shape;
    std::vector<int64_t> stride;
};

struct GlobalRef { std::string module, name; };
struct StorageRef { std::string key; PthDType dtype = PthDType::BFloat16; uint64_t size = 0; };

struct Value {
    enum Kind { None, Mark, Int, Bool, String, Global, Tuple, Dict, List, Storage, Tensor, OrderedDict } kind = None;
    int64_t i = 0;
    bool b = false;
    std::string s;
    GlobalRef global;
    StorageRef storage;
    TensorRecord tensor;
    std::vector<Value> items;
};

PthDType dtype_from_storage_name(const std::string &name) {
    if (name == "FloatStorage") return PthDType::Float32;
    if (name == "HalfStorage") return PthDType::Float16;
    if (name == "BFloat16Storage") return PthDType::BFloat16;
    throw std::runtime_error("unsupported pth storage dtype: " + name);
}

struct PickleParser {
    std::vector<uint8_t> b;
    size_t p = 0;
    bool stopped = false;
    std::vector<Value> stack;
    std::unordered_map<uint32_t, Value> memo;
    std::vector<TensorRecord> records;

    explicit PickleParser(std::vector<uint8_t> bytes) : b(std::move(bytes)) {}

    uint8_t byte() {
        if (p >= b.size()) throw std::runtime_error("pth pickle eof");
        return b[p++];
    }
    uint16_t u16() {
        if (p + 2 > b.size()) throw std::runtime_error("pth pickle eof u16");
        const uint16_t x = rd16(b.data() + p);
        p += 2;
        return x;
    }
    uint32_t u32() {
        if (p + 4 > b.size()) throw std::runtime_error("pth pickle eof u32");
        const uint32_t x = rd32(b.data() + p);
        p += 4;
        return x;
    }
    uint64_t u64() {
        if (p + 8 > b.size()) throw std::runtime_error("pth pickle eof u64");
        const uint64_t x = rd64(b.data() + p);
        p += 8;
        return x;
    }
    int32_t i32() {
        uint32_t u = u32();
        int32_t x;
        std::memcpy(&x, &u, sizeof(x));
        return x;
    }
    std::string sized(uint64_t n) {
        if (p + n > b.size()) throw std::runtime_error("pth pickle string eof");
        std::string s(reinterpret_cast<const char *>(b.data() + p), static_cast<size_t>(n));
        p += static_cast<size_t>(n);
        return s;
    }
    std::string line() {
        const size_t s = p;
        while (p < b.size() && b[p] != '\n') ++p;
        if (p >= b.size()) throw std::runtime_error("pth pickle line eof");
        std::string out(reinterpret_cast<const char *>(b.data() + s), p - s);
        ++p;
        return out;
    }
    void push(Value v) { stack.push_back(std::move(v)); }
    void push_kind(Value::Kind k) {
        Value v;
        v.kind = k;
        push(std::move(v));
    }
    void push_int(int64_t x) {
        Value v;
        v.kind = Value::Int;
        v.i = x;
        push(std::move(v));
    }
    void push_string(std::string s) {
        Value v;
        v.kind = Value::String;
        v.s = std::move(s);
        push(std::move(v));
    }
    size_t mark() const {
        for (size_t i = stack.size(); i > 0; --i) {
            if (stack[i - 1].kind == Value::Mark) return i - 1;
        }
        throw std::runtime_error("pth pickle mark not found");
    }
    std::vector<int64_t> tuple_ints(const Value &v) {
        std::vector<int64_t> out;
        if (v.kind != Value::Tuple && v.kind != Value::List) return out;
        for (const auto &x : v.items) {
            if (x.kind != Value::Int) return {};
            out.push_back(x.i);
        }
        return out;
    }
    void make_tuple(size_t n) {
        if (stack.size() < n) throw std::runtime_error("pth pickle tuple underflow");
        Value v;
        v.kind = Value::Tuple;
        v.items.assign(stack.end() - static_cast<ptrdiff_t>(n), stack.end());
        stack.resize(stack.size() - n);
        push(std::move(v));
    }
    void tuple_mark(Value::Kind kind = Value::Tuple) {
        const size_t m = mark();
        Value v;
        v.kind = kind;
        v.items.assign(stack.begin() + static_cast<ptrdiff_t>(m + 1), stack.end());
        stack.resize(m);
        push(std::move(v));
    }
    void persistent_id() {
        Value pid = std::move(stack.back());
        stack.pop_back();
        if (pid.kind != Value::Tuple || pid.items.size() < 5 || pid.items[0].s != "storage") {
            throw std::runtime_error("bad pth persistent id");
        }
        if (pid.items[1].kind != Value::Global) {
            throw std::runtime_error("bad pth storage dtype ref");
        }
        Value v;
        v.kind = Value::Storage;
        v.storage.dtype = dtype_from_storage_name(pid.items[1].global.name);
        v.storage.key = pid.items[2].s;
        v.storage.size = static_cast<uint64_t>(pid.items[4].i);
        push(std::move(v));
    }
    void reduce() {
        Value args = std::move(stack.back());
        stack.pop_back();
        Value fn = std::move(stack.back());
        stack.pop_back();
        if (fn.global.module == "collections" && fn.global.name == "OrderedDict") {
            push_kind(Value::OrderedDict);
            return;
        }
        if (fn.global.module == "torch._utils" &&
            (fn.global.name == "_rebuild_tensor_v2" || fn.global.name == "_rebuild_tensor")) {
            if (args.items.size() < 4 || args.items[0].kind != Value::Storage) {
                throw std::runtime_error("bad pth tensor rebuild args");
            }
            Value v;
            v.kind = Value::Tensor;
            v.tensor.storage_key = args.items[0].storage.key;
            v.tensor.dtype = args.items[0].storage.dtype;
            v.tensor.storage_size = args.items[0].storage.size;
            v.tensor.offset = static_cast<uint64_t>(args.items[1].i);
            v.tensor.shape = tuple_ints(args.items[2]);
            v.tensor.stride = tuple_ints(args.items[3]);
            push(std::move(v));
            return;
        }
        throw std::runtime_error("unsupported pth pickle reduce: " + fn.global.module + "." + fn.global.name);
    }
    void stack_global() {
        Value name = std::move(stack.back());
        stack.pop_back();
        Value module = std::move(stack.back());
        stack.pop_back();
        if (module.kind != Value::String || name.kind != Value::String) {
            throw std::runtime_error("bad pth STACK_GLOBAL");
        }
        Value v;
        v.kind = Value::Global;
        v.global = {module.s, name.s};
        push(std::move(v));
    }
    void setitems() {
        const size_t m = mark();
        for (size_t i = m + 1; i + 1 < stack.size(); i += 2) {
            if ((stack[i].kind == Value::String || stack[i].kind == Value::Int) &&
                stack[i + 1].kind == Value::Tensor) {
                TensorRecord r = std::move(stack[i + 1].tensor);
                r.name = stack[i].kind == Value::String ? stack[i].s : std::to_string(stack[i].i);
                records.push_back(std::move(r));
            }
        }
        Value container = stack[m - 1];
        stack.resize(m - 1);
        push(std::move(container));
    }
    std::vector<TensorRecord> parse() {
        while (p < b.size() && !stopped) {
            const uint8_t op = byte();
            switch (op) {
                case 0x80: byte(); break;              // PROTO
                case 0x95: p += 8; break;              // FRAME
                case 0x94: memo[uint32_t(memo.size())] = stack.back(); break; // MEMOIZE
                case '}': push_kind(Value::Dict); break;
                case ']': push_kind(Value::List); break;
                case '(': push_kind(Value::Mark); break;
                case 'q': memo[byte()] = stack.back(); break;
                case 'r': memo[u32()] = stack.back(); break;
                case 'h': push(memo.at(byte())); break;
                case 'j': push(memo.at(u32())); break;
                case 'X': push_string(sized(u32())); break;
                case 0x8c: push_string(sized(byte())); break; // SHORT_BINUNICODE
                case 0x8d: push_string(sized(u64())); break;  // BINUNICODE8
                case 'U': push_string(sized(byte())); break;  // SHORT_BINSTRING
                case 'c': { Value v; v.kind = Value::Global; v.global = {line(), line()}; push(std::move(v)); break; }
                case 0x93: stack_global(); break;
                case 'K': push_int(byte()); break;
                case 'M': push_int(u16()); break;
                case 'J': push_int(i32()); break;
                case 't': tuple_mark(Value::Tuple); break;
                case 'l': tuple_mark(Value::List); break;
                case ')': push_kind(Value::Tuple); break;
                case 0x85: make_tuple(1); break;
                case 0x86: make_tuple(2); break;
                case 0x87: make_tuple(3); break;
                case 0x88: { Value v; v.kind = Value::Bool; v.b = true; push(std::move(v)); break; }
                case 0x89: { Value v; v.kind = Value::Bool; v.b = false; push(std::move(v)); break; }
                case 'Q': persistent_id(); break;
                case 'R': reduce(); break;
                case 's': {
                    if (stack.size() < 2) throw std::runtime_error("pth pickle setitem underflow");
                    stack.pop_back();
                    stack.pop_back();
                    break;
                }
                case 'u': setitems(); break;
                case 'b': if (!stack.empty()) stack.pop_back(); break; // BUILD metadata is ignored.
                case '.': stopped = true; break;
                default: throw std::runtime_error("unsupported pth pickle opcode: " + std::to_string(op));
            }
            if (p > b.size()) throw std::runtime_error("pth pickle frame out of range");
        }
        return records;
    }
};

float read_element_as_float(const uint8_t *storage, PthDType dtype, uint64_t index) {
    const uint8_t *p = storage + index * dtype_size(dtype);
    switch (dtype) {
        case PthDType::Float32: return load_scalar<float>(p);
        case PthDType::Float16: return fp16_to_float(load_scalar<uint16_t>(p));
        case PthDType::BFloat16: return bf16_to_float(load_scalar<uint16_t>(p));
    }
    return 0.0f;
}

half_float::half read_element_as_half(const uint8_t *storage, PthDType dtype, uint64_t index) {
    const uint8_t *p = storage + index * dtype_size(dtype);
    switch (dtype) {
        case PthDType::Float32: return half_float::half(load_scalar<float>(p));
        case PthDType::Float16: return half_from_bits(load_scalar<uint16_t>(p));
        case PthDType::BFloat16: return half_float::half(bf16_to_float(load_scalar<uint16_t>(p)));
    }
    return half_float::half(0.0f);
}

template <typename Fn>
void for_each_tensor_offset(
    const std::vector<int64_t> &shape,
    const std::vector<int64_t> &stride,
    size_t dim,
    uint64_t off,
    Fn &&fn) {
    if (dim == shape.size()) {
        fn(off);
        return;
    }
    for (int64_t i = 0; i < shape[dim]; ++i) {
        for_each_tensor_offset(shape, stride, dim + 1, off + uint64_t(i * stride[dim]), fn);
    }
}

bool tensor_name_less(const std::string &a, const std::string &b) {
    return a < b;
}

} // namespace

struct PthFile::Impl {
    Zip zip;
    std::string prefix;
    std::unordered_map<std::string, TensorRecord> records;
    std::unordered_map<std::string, PthTensorInfo> infos;

    explicit Impl(const std::string &path) : zip(path) {
        for (const auto &e : zip.entries) {
            const std::string suffix = "/data.pkl";
            if (e.name.size() >= suffix.size() &&
                e.name.compare(e.name.size() - suffix.size(), suffix.size(), suffix) == 0) {
                prefix = e.name.substr(0, e.name.size() - suffix.size());
                break;
            }
        }
        if (prefix.empty()) throw std::runtime_error("pth data.pkl not found");
        const ZipEntry *pkl = zip.find(prefix + "/data.pkl");
        if (!pkl) throw std::runtime_error("pth data.pkl missing");
        auto parsed = PickleParser(zip.read(*pkl)).parse();
        records.reserve(parsed.size());
        infos.reserve(parsed.size());
        for (auto &r : parsed) {
            PthTensorInfo info;
            info.name = r.name;
            info.dtype = r.dtype;
            info.shape = r.shape;
            info.stride = r.stride;
            info.offset = r.offset;
            info.storage_size = r.storage_size;
            infos.emplace(info.name, std::move(info));
            records.emplace(r.name, std::move(r));
        }
    }

    const TensorRecord *record(const std::string &name) const {
        auto it = records.find(name);
        if (it == records.end()) return nullptr;
        return &it->second;
    }

    const uint8_t *storage(const TensorRecord &r) const {
        const ZipEntry *e = zip.find(prefix + "/data/" + r.storage_key);
        if (!e) throw std::runtime_error("pth storage missing for tensor: " + r.name);
        return zip.data(*e);
    }
};

PthFile::PthFile(const std::string &path) : impl_(new Impl(path)) {}

PthFile::~PthFile() {
    delete impl_;
}

bool PthFile::has_tensor(const std::string &name) const {
    return impl_->record(name) != nullptr;
}

const PthTensorInfo *PthFile::get_tensor_info(const std::string &name) const {
    auto it = impl_->infos.find(name);
    if (it == impl_->infos.end()) return nullptr;
    return &it->second;
}

std::vector<std::string> PthFile::tensor_names() const {
    std::vector<std::string> out;
    out.reserve(impl_->records.size());
    for (const auto &kv : impl_->records) out.push_back(kv.first);
    std::sort(out.begin(), out.end(), tensor_name_less);
    return out;
}

std::vector<float> PthFile::read_tensor_float32(const std::string &name) const {
    const TensorRecord *r = impl_->record(name);
    if (!r) throw std::runtime_error("pth tensor not found: " + name);
    const uint8_t *base = impl_->storage(*r);
    std::vector<float> out;
    out.reserve(static_cast<size_t>(numel(r->shape)));
    if (contiguous(r->shape, r->stride)) {
        const uint64_t n = numel(r->shape);
        for (uint64_t i = 0; i < n; ++i) {
            out.push_back(read_element_as_float(base, r->dtype, r->offset + i));
        }
    } else {
        for_each_tensor_offset(r->shape, r->stride, 0, r->offset, [&](uint64_t off) {
            out.push_back(read_element_as_float(base, r->dtype, off));
        });
    }
    return out;
}

std::vector<half_float::half> PthFile::read_tensor_float16(const std::string &name) const {
    const TensorRecord *r = impl_->record(name);
    if (!r) throw std::runtime_error("pth tensor not found: " + name);
    const uint8_t *base = impl_->storage(*r);
    std::vector<half_float::half> out;
    out.reserve(static_cast<size_t>(numel(r->shape)));
    if (contiguous(r->shape, r->stride)) {
        const uint64_t n = numel(r->shape);
        for (uint64_t i = 0; i < n; ++i) {
            out.push_back(read_element_as_half(base, r->dtype, r->offset + i));
        }
    } else {
        for_each_tensor_offset(r->shape, r->stride, 0, r->offset, [&](uint64_t off) {
            out.push_back(read_element_as_half(base, r->dtype, off));
        });
    }
    return out;
}

std::vector<std::vector<half_float::half>> load_pth_time_states(
    const std::string &path,
    int expected_layers,
    int expected_hidden_size) {
    PthFile pth(path);
    std::vector<std::vector<half_float::half>> states(static_cast<size_t>(expected_layers));
    for (int i = 0; i < expected_layers; ++i) {
        const std::string name = "blocks." + std::to_string(i) + ".att.time_state";
        const PthTensorInfo *info = pth.get_tensor_info(name);
        if (!info) throw std::runtime_error("pth state tensor missing: " + name);
        if (info->shape.size() != 3) {
            throw std::runtime_error("pth state tensor must be rank 3: " + name);
        }
        const int64_t hidden = info->shape[0] * info->shape[1];
        if (hidden != expected_hidden_size) {
            throw std::runtime_error(
                "pth state hidden size mismatch for " + name +
                ": got " + std::to_string(hidden) +
                ", expected " + std::to_string(expected_hidden_size));
        }
        states[static_cast<size_t>(i)] = pth.read_tensor_float16(name);
    }
    return states;
}

} // namespace rwkvmobile
