#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "trie.hpp"

namespace {

struct VocabMeta {
    std::unordered_set<int> ids;
    std::unordered_map<int, int> lengths;
};

VocabMeta read_vocab_meta(const std::string& path) {
    VocabMeta meta;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open vocab file: " << path << std::endl;
        return meta;
    }

    std::string line;
    while (std::getline(file, line)) {
        size_t first_space = line.find(' ');
        size_t last_space = line.rfind(' ');
        if (first_space == std::string::npos || last_space == std::string::npos || first_space == last_space) {
            continue;
        }
        int id = std::stoi(line.substr(0, first_space));
        int len = std::stoi(line.substr(last_space + 1));
        meta.ids.insert(id);
        meta.lengths[id] = len;
    }

    return meta;
}

std::string bytes_to_hex(const std::vector<uint8_t>& bytes, size_t max_len = 16) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    size_t limit = std::min(bytes.size(), max_len);
    for (size_t i = 0; i < limit; ++i) {
        oss << std::setw(2) << static_cast<int>(bytes[i]);
        if (i + 1 < limit) {
            oss << ' ';
        }
    }
    if (bytes.size() > max_len) {
        oss << " ...";
    }
    return oss.str();
}

}  // namespace

std::vector<uint8_t> read_binary_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open text file: " << path << std::endl;
        return {};
    }
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

std::string tokens_to_string(const std::vector<int>& tokens, size_t max_len = 16) {
    std::ostringstream oss;
    size_t limit = std::min(tokens.size(), max_len);
    for (size_t i = 0; i < limit; ++i) {
        oss << tokens[i];
        if (i + 1 < limit) {
            oss << ' ';
        }
    }
    if (tokens.size() > max_len) {
        oss << " ...";
    }
    return oss.str();
}

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <vocab_unconverted> <vocab_converted> [text_file]" << std::endl;
        return 1;
    }

    const std::string vocab_unconverted = argv[1];
    const std::string vocab_converted = argv[2];
    const std::string text_file = (argc == 4) ? argv[3] : "";

    OptimizedTrieTokenizer tokenizer_unconverted(vocab_unconverted);
    OptimizedTrieTokenizer tokenizer_converted(vocab_converted);
    if (!tokenizer_unconverted.inited() || !tokenizer_converted.inited()) {
        std::cerr << "Failed to initialize tokenizers." << std::endl;
        return 1;
    }

    VocabMeta meta_unconverted = read_vocab_meta(vocab_unconverted);
    VocabMeta meta_converted = read_vocab_meta(vocab_converted);
    std::unordered_set<int> all_ids = meta_unconverted.ids;
    all_ids.insert(meta_converted.ids.begin(), meta_converted.ids.end());
    all_ids.insert(0);  // EOD token is added internally.

    size_t mismatch_count = 0;
    size_t length_mismatch_count = 0;

    for (int id : all_ids) {
        std::vector<uint8_t> bytes_unconverted = tokenizer_unconverted.decodeBytes({id});
        std::vector<uint8_t> bytes_converted = tokenizer_converted.decodeBytes({id});

        if (bytes_unconverted != bytes_converted) {
            if (mismatch_count < 10) {
                std::cerr << "Mismatch token id " << id << ": "
                          << "unconverted(" << bytes_unconverted.size() << " bytes, "
                          << bytes_to_hex(bytes_unconverted) << ") vs converted("
                          << bytes_converted.size() << " bytes, "
                          << bytes_to_hex(bytes_converted) << ")" << std::endl;
            }
            ++mismatch_count;
        }

        auto len_unconverted = meta_unconverted.lengths.find(id);
        if (len_unconverted != meta_unconverted.lengths.end() &&
            bytes_unconverted.size() != static_cast<size_t>(len_unconverted->second)) {
            if (length_mismatch_count < 10) {
                std::cerr << "Length mismatch (unconverted) id " << id << ": expected "
                          << len_unconverted->second << ", got " << bytes_unconverted.size()
                          << std::endl;
            }
            ++length_mismatch_count;
        }

        auto len_converted = meta_converted.lengths.find(id);
        if (len_converted != meta_converted.lengths.end() &&
            bytes_converted.size() != static_cast<size_t>(len_converted->second)) {
            if (length_mismatch_count < 10) {
                std::cerr << "Length mismatch (converted) id " << id << ": expected "
                          << len_converted->second << ", got " << bytes_converted.size()
                          << std::endl;
            }
            ++length_mismatch_count;
        }
    }

    if (mismatch_count == 0 && length_mismatch_count == 0) {
        std::cout << "Vocab load match: all tokens identical across both files." << std::endl;
    } else {
        std::cerr << "Found " << mismatch_count << " token mismatches and "
                  << length_mismatch_count << " length mismatches." << std::endl;
    }

    if (text_file.empty()) {
        return (mismatch_count == 0 && length_mismatch_count == 0) ? 0 : 1;
    }

    std::vector<uint8_t> text_bytes = read_binary_file(text_file);
    if (text_bytes.empty()) {
        return (mismatch_count == 0 && length_mismatch_count == 0) ? 0 : 1;
    }

    const size_t chunk_size = 4096;
    size_t chunk_mismatch_count = 0;
    size_t token_mismatch_count = 0;
    size_t decode_mismatch_count = 0;

    for (size_t offset = 0; offset < text_bytes.size(); offset += chunk_size) {
        std::cout << "Processing chunk at offset " << offset << std::endl;
        size_t len = std::min(chunk_size, text_bytes.size() - offset);
        std::vector<uint8_t> chunk(text_bytes.begin() + offset, text_bytes.begin() + offset + len);

        std::vector<int> tokens_unconverted = tokenizer_unconverted.encodeBytes(chunk);
        std::vector<int> tokens_converted = tokenizer_converted.encodeBytes(chunk);

        std::vector<uint8_t> decoded_unconverted = tokenizer_unconverted.decodeBytes(tokens_unconverted);
        std::vector<uint8_t> decoded_converted = tokenizer_converted.decodeBytes(tokens_converted);

        if (decoded_unconverted != chunk || decoded_converted != chunk) {
            if (chunk_mismatch_count < 5) {
                std::cerr << "Decode mismatch at offset " << offset << " length " << len
                          << " (unconverted=" << decoded_unconverted.size()
                          << ", converted=" << decoded_converted.size() << ")" << std::endl;
            }
            ++chunk_mismatch_count;
        }

        if (tokens_unconverted != tokens_converted) {
            if (token_mismatch_count < 5) {
                std::cerr << "Token mismatch at offset " << offset
                          << " (unconverted=" << tokens_unconverted.size() << ": "
                          << tokens_to_string(tokens_unconverted)
                          << ", converted=" << tokens_converted.size() << ": "
                          << tokens_to_string(tokens_converted) << ")" << std::endl;
            }
            ++token_mismatch_count;
        }

        if (decoded_unconverted != decoded_converted) {
            if (decode_mismatch_count < 5) {
                std::cerr << "Decoded bytes mismatch at offset " << offset
                          << " (unconverted=" << bytes_to_hex(decoded_unconverted)
                          << ", converted=" << bytes_to_hex(decoded_converted) << ")" << std::endl;
            }
            ++decode_mismatch_count;
        }
    }

    if (chunk_mismatch_count == 0 && token_mismatch_count == 0 && decode_mismatch_count == 0) {
        std::cout << "Text chunk encode/decode match across both vocab files." << std::endl;
    } else {
        std::cerr << "Text chunk mismatches: decode=" << chunk_mismatch_count
                  << ", tokens=" << token_mismatch_count
                  << ", decoded_bytes=" << decode_mismatch_count << std::endl;
    }

    return (mismatch_count == 0 && length_mismatch_count == 0 &&
            chunk_mismatch_count == 0 && token_mismatch_count == 0 &&
            decode_mismatch_count == 0)
        ? 0
        : 1;
}
