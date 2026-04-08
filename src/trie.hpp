#ifndef TRIE_OPTIMIZED_HPP
#define TRIE_OPTIMIZED_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <memory>
#include <string>
#include <unordered_set>
#include <sstream>
#include <cctype>
#include <cstdint>
#include <filesystem>

static inline std::ifstream openInputFile(const std::string& file_name) {
#ifdef _WIN32
    std::ifstream file(std::filesystem::u8path(file_name));
    if (file.is_open()) {
        return file;
    }
#endif
    return std::ifstream(file_name);
}

std::string processVocabFormat(const std::string &input) {
    std::string final;

    // remove starting quotes and end quotes (if string starts with "b" remove that too)
    if (input.length() > 0 && (input[0] == '\'' || input[0] == '\"')) {
        final = input.substr(1, input.length() - 3);
    } else if (input.length() > 0 && input[0] == 'b' && (input[1] == '\'' || input[1] == '\"')) {
        final = input.substr(2, input.length() - 4);
    } else {
        final = input;
    }

    return final;
}

std::vector<uint8_t> processEscapes(const std::string &input, bool utf8_string = false, int utf8_byte_length = -1, bool debug = false) {
    if (utf8_string && utf8_byte_length > 0 && input.length() > 0 && input[0] == '\\' && (input[1] == 'u' || input[1] == 'x')){
        std::vector<uint8_t> result;
        std::istringstream stream(input);
        char ch;

        while (stream.get(ch)) {
            if (ch == '\\' && (stream.peek() == 'u' || stream.peek() == 'x')) {
                std::string hexCode;
                stream.get(ch);
                for (int i = 0; i < 4 && stream.get(ch); ++i) {
                    hexCode += ch;
                }
                std::istringstream hexStream(hexCode);
                uint32_t codePoint;
                hexStream >> std::hex >> codePoint;

                if (codePoint <= 0x7F) {
                    result.push_back(static_cast<uint8_t>(codePoint));
                } else if (codePoint <= 0x7FF) {
                    result.push_back(static_cast<uint8_t>(192 + (codePoint >> 6)));
                    result.push_back(static_cast<uint8_t>(128 + (codePoint & 0x3F)));
                } else if (codePoint <= 0xFFFF) {
                    result.push_back(static_cast<uint8_t>(224 + (codePoint >> 12)));
                    result.push_back(static_cast<uint8_t>(128 + ((codePoint >> 6) & 0x3F)));
                    result.push_back(static_cast<uint8_t>(128 + (codePoint & 0x3F)));
                } else if (codePoint <= 0x10FFFF) {
                    result.push_back(static_cast<uint8_t>(240 + (codePoint >> 18)));
                    result.push_back(static_cast<uint8_t>(128 + ((codePoint >> 12) & 0x3F)));
                    result.push_back(static_cast<uint8_t>(128 + ((codePoint >> 6) & 0x3F)));
                    result.push_back(static_cast<uint8_t>(128 + (codePoint & 0x3F)));
                }
            } else {
                result.push_back(static_cast<uint8_t>(ch));
            }
        }

        while (result.size() < utf8_byte_length) {
            result.insert(result.begin(), 0);
        }
        return result;
    }

    std::vector<uint8_t> result;
    bool escape = false;

    for (size_t i = 0; i < input.length(); ++i) {
        char c = input[i];
        if (escape) {
            switch (c) {
                case 'x': {
                    std::string hexDigits;
                    for (int j = 0; j < 2 && i + 1 < input.length(); ++j) {
                        hexDigits += input[++i];
                    }
                    if (hexDigits.length() == 2) {
                        result.push_back(static_cast<uint8_t>(std::stoul(hexDigits, nullptr, 16)));
                    }
                    break;
                }
                case 'n': result.push_back('\n'); break;
                case 't': result.push_back('\t'); break;
                case 'r': result.push_back('\r'); break;
                case '\\': result.push_back('\\'); break;
                case '\'': result.push_back('\''); break;
                case '\"': result.push_back('\"'); break;
                default:
                    result.push_back('\\');
                    result.push_back(static_cast<uint8_t>(c));
                    break;
            }
            escape = false;
        } else {
            if (c == '\\') {
                escape = true;
            } else {
                result.push_back(static_cast<uint8_t>(c));
            }
        }
    }

    return result;
}

static inline std::string trimWhitespace(const std::string& input) {
    size_t start = input.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }
    size_t end = input.find_last_not_of(" \t\r\n");
    return input.substr(start, end - start + 1);
}

static inline int hexValue(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
    if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
    return -1;
}

static inline void appendUtf8Bytes(std::vector<uint8_t>& out, uint32_t codePoint) {
    if (codePoint <= 0x7F) {
        out.push_back(static_cast<uint8_t>(codePoint));
    } else if (codePoint <= 0x7FF) {
        out.push_back(static_cast<uint8_t>(192 + (codePoint >> 6)));
        out.push_back(static_cast<uint8_t>(128 + (codePoint & 0x3F)));
    } else if (codePoint <= 0xFFFF) {
        out.push_back(static_cast<uint8_t>(224 + (codePoint >> 12)));
        out.push_back(static_cast<uint8_t>(128 + ((codePoint >> 6) & 0x3F)));
        out.push_back(static_cast<uint8_t>(128 + (codePoint & 0x3F)));
    } else if (codePoint <= 0x10FFFF) {
        out.push_back(static_cast<uint8_t>(240 + (codePoint >> 18)));
        out.push_back(static_cast<uint8_t>(128 + ((codePoint >> 12) & 0x3F)));
        out.push_back(static_cast<uint8_t>(128 + ((codePoint >> 6) & 0x3F)));
        out.push_back(static_cast<uint8_t>(128 + (codePoint & 0x3F)));
    }
}

static inline std::vector<uint8_t> parsePythonLiteralBytes(const std::string& literal) {
    std::string input = trimWhitespace(literal);
    if (input.empty()) {
        return {};
    }

    bool is_bytes = false;
    size_t i = 0;
    char quote = 0;
    if (input.size() >= 2 && input[0] == 'b' && (input[1] == '\'' || input[1] == '"')) {
        is_bytes = true;
        quote = input[1];
        i = 2;
    } else if (input[0] == '\'' || input[0] == '"') {
        quote = input[0];
        i = 1;
    } else {
        return std::vector<uint8_t>(input.begin(), input.end());
    }

    size_t end = input.size();
    if (end > 0 && input[end - 1] == quote) {
        --end;
    }

    std::vector<uint8_t> out;
    for (; i < end; ++i) {
        unsigned char c = static_cast<unsigned char>(input[i]);
        if (c != '\\') {
            // Literal UTF-8 bytes are already present in the source string.
            out.push_back(static_cast<uint8_t>(c));
            continue;
        }

        if (i + 1 >= end) {
            out.push_back(static_cast<uint8_t>('\\'));
            break;
        }

        char esc = input[++i];
        switch (esc) {
            case 'n':
                if (is_bytes) out.push_back('\n'); else appendUtf8Bytes(out, '\n');
                break;
            case 't':
                if (is_bytes) out.push_back('\t'); else appendUtf8Bytes(out, '\t');
                break;
            case 'r':
                if (is_bytes) out.push_back('\r'); else appendUtf8Bytes(out, '\r');
                break;
            case '\\':
                if (is_bytes) out.push_back('\\'); else appendUtf8Bytes(out, '\\');
                break;
            case '\'':
                if (is_bytes) out.push_back('\''); else appendUtf8Bytes(out, '\'');
                break;
            case '"':
                if (is_bytes) out.push_back('"'); else appendUtf8Bytes(out, '"');
                break;
            case 'a':
                if (is_bytes) out.push_back('\a'); else appendUtf8Bytes(out, '\a');
                break;
            case 'b':
                if (is_bytes) out.push_back('\b'); else appendUtf8Bytes(out, '\b');
                break;
            case 'f':
                if (is_bytes) out.push_back('\f'); else appendUtf8Bytes(out, '\f');
                break;
            case 'v':
                if (is_bytes) out.push_back('\v'); else appendUtf8Bytes(out, '\v');
                break;
            case 'x': {
                if (i + 2 <= end - 1) {
                    int hi = hexValue(input[i + 1]);
                    int lo = hexValue(input[i + 2]);
                    if (hi >= 0 && lo >= 0) {
                        uint8_t byte = static_cast<uint8_t>((hi << 4) | lo);
                        if (is_bytes) {
                            out.push_back(byte);
                        } else {
                            appendUtf8Bytes(out, byte);
                        }
                        i += 2;
                        break;
                    }
                }
                if (is_bytes) out.push_back('x'); else appendUtf8Bytes(out, 'x');
                break;
            }
            case 'u':
            case 'U': {
                int hex_count = (esc == 'u') ? 4 : 8;
                uint32_t codePoint = 0;
                bool ok = true;
                if (i + hex_count <= end - 1) {
                    for (int j = 0; j < hex_count; ++j) {
                        int hv = hexValue(input[i + 1 + j]);
                        if (hv < 0) {
                            ok = false;
                            break;
                        }
                        codePoint = (codePoint << 4) | static_cast<uint32_t>(hv);
                    }
                } else {
                    ok = false;
                }

                if (ok) {
                    if (is_bytes) {
                        // In bytes literals, \u/\U are not valid; keep raw.
                        out.push_back(static_cast<uint8_t>(esc));
                    } else {
                        appendUtf8Bytes(out, codePoint);
                        i += hex_count;
                    }
                } else {
                    if (is_bytes) out.push_back(static_cast<uint8_t>(esc));
                    else appendUtf8Bytes(out, static_cast<uint8_t>(esc));
                }
                break;
            }
            default: {
                if (esc >= '0' && esc <= '7') {
                    uint32_t value = static_cast<uint32_t>(esc - '0');
                    size_t j = i;
                    for (int digits = 1; digits < 3 && j + 1 < end; ++digits) {
                        char next = input[j + 1];
                        if (next < '0' || next > '7') {
                            break;
                        }
                        value = (value << 3) | static_cast<uint32_t>(next - '0');
                        ++j;
                    }
                    i = j;
                    if (is_bytes) {
                        out.push_back(static_cast<uint8_t>(value & 0xFF));
                    } else {
                        appendUtf8Bytes(out, value & 0xFF);
                    }
                } else {
                    if (is_bytes) out.push_back(static_cast<uint8_t>(esc));
                    else appendUtf8Bytes(out, static_cast<uint8_t>(esc));
                }
                break;
            }
        }
    }

    return out;
}

struct VectorEqual {
    bool operator()(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) const noexcept {
        return a == b;
    }
};

struct VectorHash {
    size_t operator()(const std::vector<uint8_t>& vec) const noexcept {
        size_t hash = 0;
        for (uint8_t byte : vec) {
            hash = hash * 31 + byte;
        }
        return hash;
    }
};

class OptimizedTrie {
private:
    uint8_t ch;
    std::unordered_map<uint8_t, std::unique_ptr<OptimizedTrie>> children;
    std::unordered_set<int> values;
    OptimizedTrie* parent;

public:
    OptimizedTrie(OptimizedTrie* parent = nullptr, uint8_t ch = 0)
        : parent(parent), ch(ch) {}

    OptimizedTrie* add(const std::vector<uint8_t>& key, size_t idx = 0, int val = -1) {
        if (idx == key.size()) {
            if (val != -1) {
                values.insert(val);
            }
            return this;
        }

        uint8_t uchar = key[idx];
        auto& child = children[uchar];
        if (!child) {
            child = std::make_unique<OptimizedTrie>(this, uchar);
        }
        return child->add(key, idx + 1, val);
    }

    std::tuple<size_t, int> find_longest_fast(const std::vector<uint8_t>& key, size_t idx = 0) {
        auto* u = this;
        std::tuple<size_t, int> ret;

        while (idx < key.size()) {
            uint8_t uchar = key[idx];
            auto it = u->children.find(uchar);
            if (it == u->children.end()) {
                break;
            }

            u = it->second.get();
            ++idx;

            if (!u->values.empty()) {
                ret = std::make_tuple(idx, *u->values.begin());
            }
        }

        return ret;
    }

    size_t get_memory_usage() const {
        size_t size = sizeof(OptimizedTrie);
        size += children.size() * (sizeof(uint8_t) + sizeof(std::unique_ptr<OptimizedTrie>));
        size += values.size() * sizeof(int);

        for (const auto& child : children) {
            if (child.second) {
                size += child.second->get_memory_usage();
            }
        }
        return size;
    }
};

class TokenMapping {
private:
    std::vector<std::vector<uint8_t>> token_data;
    std::unordered_map<int, size_t> idx_to_data_idx;
    std::unordered_map<std::vector<uint8_t>, int, VectorHash, VectorEqual> data_to_idx;

public:
    void add_token(int token_id, const std::vector<uint8_t>& data) {
        auto it = data_to_idx.find(data);
        if (it != data_to_idx.end()) {
            idx_to_data_idx[token_id] = it->second;
        } else {
            size_t data_idx = token_data.size();
            token_data.push_back(data);
            idx_to_data_idx[token_id] = data_idx;
            data_to_idx[data] = token_id;
        }
    }

    const std::vector<uint8_t>& get_token_data(int token_id) const {
        auto it = idx_to_data_idx.find(token_id);
        if (it != idx_to_data_idx.end()) {
            return token_data[it->second];
        }
        static const std::vector<uint8_t> empty;
        return empty;
    }

    int get_token_id(const std::vector<uint8_t>& data) const {
        auto it = data_to_idx.find(data);
        return it != data_to_idx.end() ? it->second : -1;
    }

    size_t get_memory_usage() const {
        size_t size = sizeof(TokenMapping);

        for (const auto& data : token_data) {
            size += data.size();
        }

        size += idx_to_data_idx.size() * (sizeof(int) + sizeof(size_t));
        size += data_to_idx.size() * (sizeof(std::vector<uint8_t>) + sizeof(int));

        return size;
    }
};

class OptimizedTrieTokenizer {
private:
    TokenMapping token_mapping;
    std::unique_ptr<OptimizedTrie> root;
    bool _inited = false;

    std::vector<uint8_t> stringToBytes(const std::string& str) {
        return std::vector<uint8_t>(str.begin(), str.end());
    }

    std::string bytesToString(const std::vector<uint8_t>& bytes) {
        return std::string(bytes.begin(), bytes.end());
    }

public:
    OptimizedTrieTokenizer(const std::string& file_name) {
        root = std::make_unique<OptimizedTrie>();
        std::ifstream file = openInputFile(file_name);
        if (!file.is_open()) {
            return;
        }

        std::string base_name = file_name;
        size_t slash_pos = base_name.find_last_of("/\\");
        if (slash_pos != std::string::npos) {
            base_name = base_name.substr(slash_pos + 1);
        }
        bool is_converted = base_name.rfind("b_", 0) == 0;

        std::string line;
        while (getline(file, line)) {
            size_t firstSpace = line.find(' ');
            size_t lastSpace = line.rfind(' ');
            int idx = std::stoi(line.substr(0, firstSpace));
            int utf8_byte_length = std::stoi(line.substr(lastSpace + 1));
            std::string token_literal = line.substr(firstSpace + 1, lastSpace - firstSpace);
            std::vector<uint8_t> x;
            if (is_converted) {
                bool utf8_string = line[firstSpace + 1] != 'b';
                x = processEscapes(
                    processVocabFormat(token_literal),
                    utf8_string,
                    utf8_byte_length
                );
            } else {
                x = parsePythonLiteralBytes(token_literal);
                while (utf8_byte_length > 0 && x.size() < static_cast<size_t>(utf8_byte_length)) {
                    x.insert(x.begin(), 0);
                }
            }

            token_mapping.add_token(idx, x);
            root->add(x, 0, idx);
        }

        auto eod_data = stringToBytes("<EOD>");
        token_mapping.add_token(0, eod_data);
        root->add(eod_data, 0, 0);

        _inited = true;
    }

    std::vector<uint8_t> decodeBytes(const std::vector<int>& tokens) {
        std::vector<uint8_t> resultBytes;
        for (int token : tokens) {
            const auto& bytes = token_mapping.get_token_data(token);
            resultBytes.insert(resultBytes.end(), bytes.begin(), bytes.end());
        }
        return resultBytes;
    }

    std::vector<int> encodeBytes(const std::vector<uint8_t>& src) {
        std::vector<int> tokens;
        tokens.reserve(src.size());
        size_t idx = 0;

        while (idx < src.size()) {
            int token;
            size_t old_idx = idx;

            std::tie(idx, token) = root->find_longest_fast(src, idx);

            if (idx > old_idx && token != -1) {
                tokens.push_back(token);
            } else {
                break;
            }
        }

        return tokens;
    }

    std::vector<int> encode(const std::string& src) {
        return encodeBytes(stringToBytes(src));
    }

    std::string decode(const std::vector<int>& tokens) {
        return bytesToString(decodeBytes(tokens));
    }

    bool inited() {
        return _inited;
    }
};

#endif // TRIE_OPTIMIZED_HPP
