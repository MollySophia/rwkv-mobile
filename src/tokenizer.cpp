#include "tokenizer.h"
#include "trie.hpp"

namespace rwkvmobile {

int trie_tokenizer::load(const std::string vocab_file) {
    _tokenizer = new TRIE_TOKENIZER(vocab_file);
    if (!_tokenizer->inited())
        return RWKV_ERROR_TOKENIZER;
    return RWKV_SUCCESS;
}

std::vector<int> trie_tokenizer::encode(std::string_view str) const {
    auto ids = _tokenizer->encode(std::string(str));
    return ids;
}

std::string trie_tokenizer::decode(int id) const {
    return _tokenizer->decode(std::vector<int>{id});
}

std::string trie_tokenizer::decode(const std::vector<int> &ids) const {
    return _tokenizer->decode(ids);
}

std::vector<int> abc_tokenizer::encode(std::string_view str) const {
  std::vector<int> ids;
  for (int i = 0; i < str.size(); ++i) {
    ids.push_back(str[i]);
  }
  return ids;
}

std::string abc_tokenizer::decode(int id) const {
  if (id <= eos_token_id) {
    return "";
  } else {
    return std::string(1, static_cast<char>(id));
  }
}

std::string abc_tokenizer::decode(const std::vector<int> &ids) const {
  std::string str;
  for (auto id : ids) {
    str += decode(id);
  }
  return str;
}

}