#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include "commondef.h"

class TRIE_TOKENIZER;

namespace rwkvmobile {

class tokenizer_base {
public:
  tokenizer_base(int pad_token_id, int bos_token_id, int eos_token_id)
      : pad_token_id(pad_token_id), bos_token_id(bos_token_id),
        eos_token_id(eos_token_id) {}
  virtual ~tokenizer_base() = default;
  virtual int load(const std::string vocab_file) = 0;
  virtual std::vector<int> encode(std::string_view str) const = 0;
  virtual std::string decode(const std::vector<int> &ids) const = 0;
  virtual std::string decode(int id) const = 0;
  const int pad_token_id;
  const int bos_token_id;
  const int eos_token_id;
};

class trie_tokenizer : public tokenizer_base {
public:
    trie_tokenizer() : tokenizer_base(0, 0, 0) {};
    int load(const std::string vocab_file);
    std::vector<int> encode(std::string_view str) const;
    std::string decode(const std::vector<int> &ids) const;
    std::string decode(int id) const;
private:
    TRIE_TOKENIZER * _tokenizer;
};

class abc_tokenizer : public tokenizer_base {
public:
    abc_tokenizer() : tokenizer_base(0, 2, 3) {};
    int load(const std::string) const {
        return RWKV_SUCCESS;
    };
    std::vector<int> encode(std::string_view str) const;
    std::string decode(const std::vector<int> &ids) const;
    std::string decode(int id) const;
};



}

#endif