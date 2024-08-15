// From https://github.com/m8than/RWKV-World-Tokenizer-CPP
#ifndef TRIE_HPP
#define TRIE_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <unordered_map>
#include <cassert>
#include <memory>
#include <string>
#include <functional>
#include <unordered_set>
#include <string>
#include <sstream>
#include <iomanip>
#include <codecvt>
#include <locale>
#include <future>
#include <execution>

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
std::vector<uint8_t> processUTF8Escapes(const std::string &input, const int utf8_byte_length = 0) {
    std::vector<uint8_t> result;
    std::istringstream stream(input);
    char ch;

    while (stream.get(ch)) {
        if (ch == '\\' && (stream.peek() == 'u' || stream.peek() == 'x')) { // Unicode escape sequence or unicode byte escape sequence
            std::string hexCode;
            stream.get(ch); // consume 'u' or 'x'
            for (int i = 0; i < 4 && stream.get(ch); ++i) { // Get next 4 hex digits
                hexCode += ch;
            }
            std::istringstream hexStream(hexCode);
            uint32_t codePoint;
            hexStream >> std::hex >> codePoint; // Convert hex to decimal

            // Convert codePoint to UTF-8 and append to result
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

    // Ensure the result meets the specified byte length
    if (utf8_byte_length > 0 && result.size() != utf8_byte_length) {
        std::cout << "UTF8 byte length mismatch: " << result.size() << " vs " << utf8_byte_length << std::endl;
        std::cout << "Left padding with 0's" << std::endl;
        while (result.size() < utf8_byte_length) {
            // Add padding at the start
            result.insert(result.begin(), 0);
        }
    }

    return result;
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
    
    // if input contains " print it
    // if (input.find('\\') != std::string::npos) {
    //     std::cout << "input: " << input << std::endl;
    //     std::cout << "output: " << final << std::endl;
    // }
    return final;
}

// Function to check if a string consists only of valid hexadecimal digits
bool isValidHex(const std::string &str) {
    return str.find_first_not_of("0123456789abcdefABCDEF") == std::string::npos;
}

bool isHexadecimal(char c) {
    return (c >= '0' && c <= '9') || 
           (c >= 'a' && c <= 'f') || 
           (c >= 'A' && c <= 'F');
}

std::vector<uint8_t> processEscapes(const std::string &input, bool utf8_string = false, int utf8_byte_length = -1, bool debug = false) {
    if (utf8_string && utf8_byte_length > 0 && input.length() > 0 && input[0] == '\\' && (input[1] == 'u' or input[1] == 'x')){
        return processUTF8Escapes(input, utf8_byte_length);
    }

    std::vector<uint8_t> result;
    bool escape = false;
    bool unicode = false;
    bool hex = false;
    std::string hexDigits;
    std::string unicodeDigits;
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> convert;


    if (debug) {
        std::cout << "Processing string: " << input << std::endl << std::endl;
    }

    for (size_t i = 0; i < input.length(); ++i) {
        char c = input[i];
        if (escape) {
            if (unicode) {
                if (unicodeDigits.length() < 4) { // Collecting 4 hex digits for Unicode
                    // Ensure that the character is a valid hexadecimal digit
                    if (isHexadecimal(c)) {
                        unicodeDigits += c;
                    } else {
                        std::cout << "Invalid Unicode escape sequence: " << unicodeDigits << c << std::endl;
                        throw std::invalid_argument("Invalid Unicode escape sequence");
                    }
                    if (unicodeDigits.length() == 4) { // Got 4 digits, convert to character
                        try {
                            char32_t unicodeChar = static_cast<char32_t>(std::stoul(unicodeDigits, nullptr, 16));
                            if (debug) {
                                std::string utf8string = convert.to_bytes(unicodeChar);
                                std::cout << "Unicode: " << utf8string << std::endl;
                            }

                            for (auto b : convert.to_bytes(unicodeChar)) {
                                result.push_back(static_cast<uint8_t>(b)); // Append each byte to the vector
                            }
                        } catch (const std::invalid_argument& e) {
                            // Handle the exception, possibly log or fix the sequence
                            std::cout << "Exception caught: " << e.what() << std::endl;
                            // Add handling code here, such as skipping this character or adding a placeholder
                        }
                        unicodeDigits.clear(); // Reset unicode digits
                        unicode = false; // Reset unicode processing
                        escape = false; // Reset escape processing
                    }
                }
            } else if (hex) {
                if (hexDigits.length() < 2) {
                    hexDigits += c;
                    if (hexDigits.length() == 2) {
                        // if valid hex and a byte literal
                        if (isValidHex(hexDigits) && utf8_string == false) {
                            result.push_back(static_cast<uint8_t>(std::stoul(hexDigits, nullptr, 16)));
                        } else {
                            std::cout << input << " Invalid hex sequence: " << hexDigits << " - Adding raw bytes" << std::endl;

                            // if invalid hex and a utf8 string
                            if (utf8_string) {
                                std::cout << "Raw hex: " << hexDigits << std::endl;
                                int intValue = std::stoi(hexDigits, nullptr, 16);

                                std::cout << "Int value: " << intValue << std::endl;

                                result.push_back(static_cast<uint8_t>(intValue));
                            // if invalid hex and not a utf8 string
                            } else {
                                result.push_back('\\');
                                result.push_back('x');
                                for (char h : hexDigits) {
                                    result.push_back(h);
                                }
                            }
                        }
                        hex = false;
                        escape = false;
                        hexDigits.clear();
                        continue;
                    }
                }
            } else {
                switch (c) {
                    case 'u': unicode = true; unicodeDigits.clear(); break;
                    case 'x': hex = true; hexDigits.clear(); break;
                    case 'n': result.push_back('\n'); escape = false; break;
                    case 't': result.push_back('\t'); escape = false; break;
                    case 'r': result.push_back('\r'); escape = false; break;
                    case '\\': result.push_back('\\'); escape = false; break;
                    case '\'': result.push_back('\''); escape = false; break;
                    case '\"': result.push_back('\"'); escape = false; break;
                    case 'a': result.push_back('\a'); escape = false; break;
                    case 'b': result.push_back('\b'); escape = false; break;
                    case 'f': result.push_back('\f'); escape = false; break;
                    case 'v': result.push_back('\v'); escape = false; break;
                    default: 
                        result.push_back('\\');
                        result.push_back(static_cast<uint8_t>(c)); // Add both as they are part of escape sequence
                        break;
                }
            }
        } else {
            if (c == '\\') {
                escape = true; // Next character is escaped
            } else {
                result.push_back(static_cast<uint8_t>(c));
            }
        }

        if (debug) {
            std::cout << "Processed: " << input[i] << std::endl;
            // Convert current result to string for debugging
            std::string currentResult(result.begin(), result.end());
            std::cout << "Current result: " << currentResult << std::endl;
        }
    }

    // Handle case where string ends with incomplete escape sequences
    if (unicode && unicodeDigits.length() < 4) {
        result.push_back('\\');
        result.push_back('u');
        for (char u : unicodeDigits) {
            result.push_back(u);
        }
    } else if (hex && hexDigits.length() < 2) {
        result.push_back('\\');
        result.push_back('x');
        for (char h : hexDigits) {
            result.push_back(h);
        }
    }

    return result;
}

class TRIE : public std::enable_shared_from_this<TRIE> {
    private:
        uint8_t ch;
        std::vector<std::shared_ptr<TRIE>> to;
        std::unordered_set<int> values; // Store integer values
        std::shared_ptr<TRIE> front;

    public:
        // Constructor
        TRIE(std::shared_ptr<TRIE> front = nullptr, uint8_t ch = 0) : front(front), ch(ch), to(256, nullptr) {}

        // Add a string to the trie
        std::shared_ptr<TRIE> add(const std::vector<uint8_t>& key, size_t idx = 0, int val = -1) {
            if (idx == key.size()) {
                if (val == -1) {
                    // Convert the key bytes to an integer representation if needed
                    // Or handle the case where no value is provided differently
                    // For now, we'll just not add a value if -1 is still the placeholder
                } else {
                    values.insert(val); // Insert the integer value
                }
                return shared_from_this();
            }
            uint8_t uchar = key[idx];
            if (!to[uchar]) {
                to[uchar] = std::make_shared<TRIE>(shared_from_this(), uchar);
            }
            return to[uchar]->add(key, idx + 1, val);
        }

        std::tuple<size_t, int> find_longest_fast(const std::vector<uint8_t>& key, size_t idx = 0) {
            //std::shared_ptr<TRIE> u = shared_from_this();
            auto u = this;
            std::tuple<size_t, int> ret;

            if (idx < key.size()) {  // Changed from <= to <
                uint8_t uchar = key[idx];
                while (u->to[uchar] != nullptr) {
                    u = u->to[uchar].get();
                    ++idx;
                    if (!u->values.empty()) {
                        ret = std::make_tuple(idx, *u->values.begin());
                    }
                    if (idx >= key.size()) {  // Changed from == to >= for safety, though it's essentially similar in this context
                        break;
                    }
                    uchar = key.size() > idx ? key[idx] : 0; // Safe check for next character
                }
            }

            return ret;
        }

        std::tuple<size_t, std::shared_ptr<TRIE>, std::unordered_set<int>> find_longest(const std::vector<uint8_t>& key, size_t idx = 0) {
            std::shared_ptr<TRIE> u = shared_from_this();
            std::tuple<size_t, std::shared_ptr<TRIE>, std::unordered_set<int>> ret;

            if (idx < key.size()) {  // Changed from <= to <
                uint8_t uchar = key[idx];
                while (u->to[uchar] != nullptr) {
                    u = u->to[uchar];
                    ++idx;
                    if (!u->values.empty()) {
                        ret = std::make_tuple(idx, u, u->values);
                    }
                    if (idx >= key.size()) {  // Changed from == to >= for safety, though it's essentially similar in this context
                        break;
                    }
                    uchar = key.size() > idx ? key[idx] : 0; // Safe check for next character
                }
            }

            return ret;
        }


        // Represent the TRIE as a string (for debugging purposes)
        std::string to_string() const {
            std::string result;
            for (auto fr = shared_from_this(); fr != nullptr; fr = fr->front) { // Directly use shared_ptr for parent
                if (fr->ch != 0) {
                    result = std::string(1, static_cast<char>(fr->ch)) + result;
                }
            }
            // Convert values set to string
            std::string values_str = "{";
            for (const auto& val : values) {
                if (values_str.length() > 1) {
                    values_str += ", ";
                }
                values_str += std::to_string(val);
            }
            values_str += "}";
            return "<TRIE " + result + " " + values_str + ">";
        }
};

class TRIE_TOKENIZER {
    private:
        std::unordered_map<int, std::vector<uint8_t>> idx2token;
        std::unordered_map<std::vector<uint8_t>, int, VectorHash, VectorEqual> token2idx;
        std::shared_ptr<TRIE> root;

        std::vector<uint8_t> stringToBytes(const std::string& str) {
            return std::vector<uint8_t>(str.begin(), str.end());
        }

        std::string bytesToString(const std::vector<uint8_t>& bytes) {
            return std::string(bytes.begin(), bytes.end());
        }

        bool _inited = false;

    public:
        TRIE_TOKENIZER(const std::string& file_name) {
            root = std::make_shared<TRIE>();
            std::ifstream file(file_name);
            if (!file.is_open()) {
                return;
            }
            std::string line;
            while (getline(file, line)) {
                size_t firstSpace = line.find(' ');
                size_t lastSpace = line.rfind(' ');
                int idx = std::stoi(line.substr(0, firstSpace));
                int utf8_byte_length = std::stoi(line.substr(lastSpace + 1));
                bool utf8_string = line[firstSpace+1] != 'b';
                std::vector<uint8_t> x;
                x = processEscapes(processVocabFormat(line.substr(firstSpace + 1, lastSpace - firstSpace)), utf8_string, utf8_byte_length);
                idx2token[idx] = x;
                token2idx[x] = idx;
                root->add(x, 0, idx);
            }
            _inited = true;
        }

        void testStringToBytes(const std::string& str) {
            auto bytes = stringToBytes(str);
            std::cout << "String: " << str << std::endl;
            std::cout << "Bytes: ";
            for (auto i : bytes) {
                std::cout << (int)i << " ";
            }
            std::cout << std::endl;
            std::cout << "String: " << bytesToString(bytes) << std::endl;
        }

        std::vector<u_int8_t> decodeBytes(const std::vector<int>& tokens) {
            std::vector<uint8_t> resultBytes;
            for (int token : tokens) {
                auto it = idx2token.find(token);
                if (it != idx2token.end()) {
                    const auto& bytes = it->second;
                    resultBytes.insert(resultBytes.end(), bytes.begin(), bytes.end());
                }
            }
            return resultBytes; // Convert the byte vector back to a string
        }
        
        std::vector<int> encodeBytes(const std::vector<uint8_t>& src) {
            std::vector<int> tokens;
            tokens.reserve(src.size());
            size_t idx = 0;

            while (idx < src.size()) {
                int token;
                size_t old_idx = idx; // Store the old index to check for progress

                // Perform the longest match search from the current index
                std::tie(idx, token) = root->find_longest_fast(src, idx);

                // Check if the index has advanced, and if any values were found
                if (idx > old_idx && token != -1) {
                    tokens.push_back(token);
                } else {
                    // No progress was made or no values were found; either way, stop the loop
                    break;
                }
            }

            return tokens;
        }


        std::vector<int> encode(const std::string& src) {
            return encodeBytes(stringToBytes(src));
        }

#if 0
        std::vector<std::vector<int>> encodeBatch(const std::vector<std::string>& src) {
            std::vector<std::vector<int>> result(src.size());
            result.reserve(src.size());

            // Parallel execution using std::for_each with par execution policy
            std::for_each(std::execution::par, src.begin(), src.end(),
                [&result, this](const std::string& s) {
                    result.emplace_back(encode(s));
                }
            );

            return result;
        }
#endif

        std::string decode(const std::vector<int>& tokens) {
            return bytesToString(decodeBytes(tokens));
        }

        void printTokens(const std::vector<int>& tokens) {
            for (auto i : tokens) {
                auto s = idx2token[i];
                std::cout << bytesToString(s) << " ";
            }
            std::cout << std::endl;
        }

        bool inited() {
            return _inited;
        }
    };

#endif // TRIE_HPP
