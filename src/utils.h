#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <algorithm>

inline std::string escape_special_chars(const std::string &text) {
    std::string escaped_text;
    // replace '\n' char to "\\n"
    for (size_t i = 0; i < text.length(); i++) {
        if (text[i] == '\n') {
            escaped_text += "\\n";
        } else if (text[i] == '\r') {
            escaped_text += "\\r";
        } else if (text[i] == '\t') {
            escaped_text += "\\t";
        } else if (text[i] == '\b') {
            escaped_text += "\\b";
        } else {
            escaped_text += text[i];
        }
    }

    return escaped_text;
}

inline std::string remove_endl(const std::string &msg) {
    std::string result = msg;
    while (result.size() > 0 && result[result.size() - 1] == '\n') {
        result = result.substr(0, result.size() - 1);
    }
    return result;
}

#endif
