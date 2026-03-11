#include "backend.h"
#include "logger.h"
#include "commondef.h"
#include <cstring>
#include <functional>
#include <utility>

namespace rwkvmobile {

static const std::vector<std::pair<std::vector<int>, std::vector<int>>> kStateCacheIdReplacements = {
    {{10080, 261}, {28329, 11}}, // "。" + "\n\n" -> "。\n" + "\n"
    {{9830, 261}, {28324, 11}}, // "…" + "\n\n" -> "…\n" + "\n"
    {{19137, 261}, {28331, 11}}, // "，" + "\n\n" -> "，\n" + "\n"
};

static void apply_state_cache_id_replacement(std::vector<int> &ids) {
    if (ids.size() < 2) return;
    for (const auto &p : kStateCacheIdReplacements) {
        const auto &from = p.first;
        const auto &to = p.second;
        if (from.size() != 2 || to.size() != 2) continue;
        if (ids[ids.size() - 2] == from[0] && ids[ids.size() - 1] == from[1]) {
            ids.resize(ids.size() - 2);
            ids.insert(ids.end(), to.begin(), to.end());
            return;
        }
    }
}

state_node* execution_provider::find_deepest_matching_node(const std::vector<int> &ids, bool increment_activation_count) {
    auto node = state_root.get();
    // find the deepest node that matches the input text
    while (node->children.size() > 0) {
        bool matched = false;
        for (auto &child : node->children) {
            if (child->ids.size() <= ids.size() && std::equal(ids.begin(), ids.begin() + child->ids.size(), child->ids.begin())) {
                node = child.get();
                if (increment_activation_count) {
                    node->activation_count++; // Increment matched child count
                }
                matched = true;
                break;
            }
        }
        if (!matched) {
            break;
        }
    }
    return node;
}

state_node* execution_provider::match_and_load_state(const std::vector<int> &ids, std::vector<int> &new_ids_to_prefill) {
    auto node = find_deepest_matching_node(ids, true);

    set_state(node->state);

    new_ids_to_prefill = std::vector<int>(ids.begin() + node->ids.size(), ids.end());
    return node;
}

int execution_provider::register_state_checkpoint(state_node* &node, const std::vector<int> ids, const Tensor1D &logits) {
    std::any new_state;
    get_state(new_state);
    return register_state_checkpoint_with_state(node, ids, logits, new_state);
}

int execution_provider::register_state_checkpoint_with_state(state_node* &node, const std::vector<int> ids, const Tensor1D &logits, std::any &state) {
    if (logits.data_ptr == nullptr || logits.count < (size_t)vocab_size) {
        LOGE("register_state_checkpoint_with_state: invalid logits tensor");
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    auto new_ids = node->ids;
    new_ids.insert(new_ids.end(), ids.begin(), ids.end());
    apply_state_cache_id_replacement(new_ids);
    auto tmp_node = find_deepest_matching_node(new_ids, false);
    if (tmp_node->ids.size() == new_ids.size() && std::equal(tmp_node->ids.begin(), tmp_node->ids.end(), new_ids.begin())) {
        // avoid duplicate node
        node = tmp_node;
        node->activation_count++;
        return RWKV_SUCCESS;
    }

    auto new_node = std::make_unique<state_node>();
    if (new_node == nullptr) {
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_ALLOC;
    }
    // new_node->ids = std::vector<int>(ids);
    // each node cumulates the ids from the root to the node (after 2-token replacement)
    new_node->ids = new_ids;
    new_node->logits.resize(vocab_size);
    if (logits.dtype == TensorDType::F32) {
        memcpy(new_node->logits.data(), logits.data_ptr, (size_t)vocab_size * sizeof(float));
    } else if (logits.dtype == TensorDType::F16) {
        const half_float::half* h = reinterpret_cast<const half_float::half*>(logits.data_ptr);
        for (int i = 0; i < vocab_size; ++i) new_node->logits[i] = (float)h[i];
    } else {
        LOGE("register_state_checkpoint_with_state: unsupported logits dtype");
        return RWKV_ERROR_UNSUPPORTED;
    }
    new_node->state = std::move(state);
    new_node->activation_count = node->activation_count;

    std::string debug_msg = "register_state_checkpoint: new node ids: ";
    for (auto id : new_node->ids) {
        debug_msg += std::to_string(id) + " ";
    }
    LOGD("%s\n", debug_msg.c_str());

    node->children.push_back(std::move(new_node));
    node = node->children.back().get();
    return RWKV_SUCCESS;
}

int execution_provider::register_batch_state_checkpoint(std::vector<state_node*> &nodes, std::vector<std::any> &states, const std::vector<std::vector<int>> ids, const Tensor1D &logits) {
    auto batch_size = states.size();
    if (ids.size() != batch_size) {
        LOGE("register_batch_state_checkpoint: ids size %d != batch size %d\n", ids.size(), batch_size);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }
    if (logits.data_ptr == nullptr || logits.count < (size_t)(vocab_size * (int)batch_size)) {
        LOGE("register_batch_state_checkpoint: invalid logits tensor, count: %d, expected: %d", logits.count, vocab_size * (int)batch_size);
        return RWKV_ERROR_RUNTIME | RWKV_ERROR_INVALID_PARAMETERS;
    }

    for (size_t i = 0; i < batch_size; i++) {
        std::vector<int> full_ids = nodes[i]->ids;
        full_ids.insert(full_ids.end(), ids[i].begin(), ids[i].end());
        apply_state_cache_id_replacement(full_ids);

        bool duplicate = false;
        for (auto &child : nodes[i]->children) {
            if (child->ids.size() == full_ids.size() && std::equal(child->ids.begin(), child->ids.end(), full_ids.begin())) {
                child->activation_count++;
                duplicate = true;
                break;
            }
        }
        if (duplicate) continue;

        auto new_node = std::make_unique<state_node>();
        new_node->ids = std::move(full_ids);
        new_node->state = std::move(states[i]);
        new_node->logits.resize(vocab_size);
        if (logits.dtype == TensorDType::F32) {
            const float* base = reinterpret_cast<const float*>(logits.data_ptr);
            memcpy(new_node->logits.data(), base + i * vocab_size, (size_t)vocab_size * sizeof(float));
        } else if (logits.dtype == TensorDType::F16) {
            const half_float::half* base = reinterpret_cast<const half_float::half*>(logits.data_ptr);
            for (int j = 0; j < vocab_size; ++j) {
                new_node->logits[j] = (float)base[i * (size_t)vocab_size + (size_t)j];
            }
        } else {
            LOGE("register_batch_state_checkpoint: unsupported logits dtype");
            return RWKV_ERROR_UNSUPPORTED;
        }
        new_node->activation_count = nodes[i]->activation_count;

        nodes[i]->children.push_back(std::move(new_node));
        nodes[i] = nodes[i]->children.back().get();
    }
    return RWKV_SUCCESS;
}

void execution_provider::cleanup_state_tree() {
    if (!state_root) {
        return;
    }

    std::function<void(state_node*)> cleanup_node = [&](state_node* node) {
        if (!node) {
            return;
        }

        for (auto it = node->children.begin(); it != node->children.end();) {
            cleanup_node(it->get());

            if (!(*it)->is_constant && (*it)->activation_count <= 0) {
                it = node->children.erase(it);
            } else {
                if (!(*it)->is_constant) {
                    (*it)->activation_count--;
                }
                ++it;
            }
        }
    };

    cleanup_node(state_root.get());
}

}
