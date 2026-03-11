#ifndef BACKEND_H
#define BACKEND_H

// avoid macro conflicts on Windows
#ifdef max
#undef max
#endif

#ifdef min
#undef min
#endif

#include <string>
#include <vector>
#include <memory>
#include <any>
#include "half.hpp"
#include "tensor.h"

#include "commondef.h"

namespace rwkvmobile {

class state_node {
public:
    std::any state;
    std::vector<int> ids;
    std::vector<float> logits;
    std::vector<std::unique_ptr<state_node>> children;
    bool is_constant = false;
    int activation_count;

    state_node() {
        this->activation_count = 10;
    };

    state_node(const std::any state, const std::vector<int> ids, const std::vector<float> logits, bool is_constant = false) {
        this->state = state;
        this->ids = ids;
        this->logits = logits;
        this->is_constant = is_constant;
        this->activation_count = 10;
    }

    ~state_node() {
        if (state.has_value()) {
            state.reset();
        }
        children.clear();
    }
};

class execution_provider {
public:
    virtual int init(void * extra) { return 0; }
    virtual int init(std::string model_path, void * extra) { return 0; }
    virtual int load_model(std::string model_path, void * extra) { return RWKV_ERROR_MODEL; }
    virtual float get_load_progress() const { return -1.f; }
    virtual int eval(int id, Tensor1D & logits) { (void)id; logits = {}; return 0; };
    virtual int eval(std::vector<int> ids, Tensor1D & logits) { (void)ids; logits = {}; return 0; };
    virtual int eval_batch(std::vector<std::vector<int>> ids, Tensor1D & logits) { (void)ids; logits = {}; return RWKV_ERROR_UNSUPPORTED; };
    virtual int eval_with_embeddings(const float *embeddings, int n_tokens, Tensor1D & logits) { (void)embeddings; (void)n_tokens; logits = {}; return RWKV_ERROR_UNSUPPORTED; };
    virtual int get_state(std::any &state) { return 0; }
    virtual int set_state(std::any state) { return 0; }
    virtual int free_state(std::any state) { return 0; }
    virtual int zero_state() { return 0; }
    virtual int release_model() { return 0; };
    virtual int release() { return 0; };
    virtual bool is_available() { return false; };

    virtual int get_state_on_batch_slot(int slot, std::any &state) { return 0; }
    virtual int set_state_on_batch_slot(int slot, std::any state) { return 0; }
    virtual int zero_state_on_batch_slot(int slot) { return 0; }

    virtual double get_prefill_speed() { return -1; }
    virtual double get_decode_speed() { return -1; }

    virtual int load_raw_states(std::vector<std::vector<half_float::half>> states) { return RWKV_ERROR_UNSUPPORTED; };
    virtual int serialize_runtime_state(std::any state, std::vector<uint8_t> &states) { return RWKV_ERROR_UNSUPPORTED; };
    virtual int deserialize_runtime_state(std::vector<uint8_t> &states, std::any &state) { return RWKV_ERROR_UNSUPPORTED; };

    int get_head_count() { return num_heads; }
    int get_hidden_size() { return hidden_size; }
    int get_num_vocab() { return vocab_size; }
    int get_version() { return version; }

    virtual bool embedding_input_force_no_ln0() { return false; }

    int n_layers;
    int num_heads;
    int hidden_size;
    int version;
    int vocab_size;

    std::vector<int> supported_batch_sizes = {1};

    std::string extra_str;

    std::unique_ptr<state_node> state_root;

    state_node* find_deepest_matching_node(const std::vector<int> &ids, bool increment_activation_count = true);
    state_node* match_and_load_state(const std::vector<int> &ids, std::vector<int> &new_ids_to_prefill);
    int register_state_checkpoint(state_node* &node, const std::vector<int> ids, const Tensor1D &logits);
    int register_state_checkpoint_with_state(state_node* &node, const std::vector<int> ids, const Tensor1D &logits, std::any &state);
    int register_batch_state_checkpoint(std::vector<state_node*> &nodes, std::vector<std::any> &states, const std::vector<std::vector<int>> ids, const Tensor1D &logits);

    void cleanup_state_tree();
};

enum {
    RWKV_BACKEND_WEBRWKV = 0,
    RWKV_BACKEND_NCNN,
    RWKV_BACKEND_LLAMACPP,
    RWKV_BACKEND_QNN,
    RWKV_BACKEND_MNN,
    RWKV_BACKEND_MTK_NP7,
    RWKV_BACKEND_COREML,
    RWKV_BACKEND_MLX,
    RWKV_BACKEND_COUNT,
};

std::string backend_enum_to_str(int backend);
int backend_str_to_enum(std::string backend);

}

#endif