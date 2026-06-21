#include "mtk_rwkv_dlopen.h"

#include "commondef.h"
#include "logger.h"

#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <unistd.h>
#include <vector>

namespace rwkvmobile {

namespace {

static std::string dirname_of(const std::string& path) {
    const auto pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return "";
    }
    return path.substr(0, pos);
}

static std::string join_path(const std::string& dir, const std::string& name) {
    if (dir.empty()) {
        return name;
    }
    if (dir.back() == '/') {
        return dir + name;
    }
    return dir + "/" + name;
}

static std::string get_cwd_path(const char* name) {
    char cwd[4096];
    if (getcwd(cwd, sizeof(cwd)) == nullptr) {
        return "";
    }
    return join_path(cwd, name);
}

static void prepend_library_path(const std::string& dir) {
    if (dir.empty()) {
        return;
    }

    const char* old_value = std::getenv("LD_LIBRARY_PATH");
    if (old_value == nullptr || old_value[0] == '\0') {
        setenv("LD_LIBRARY_PATH", dir.c_str(), 1);
        return;
    }

    const std::string current(old_value);
    if (current == dir ||
        current.rfind(dir + ":", 0) == 0 ||
        current.find(":" + dir + ":") != std::string::npos ||
        (current.size() > dir.size() &&
         current.compare(current.size() - dir.size(), dir.size(), dir) == 0 &&
         current[current.size() - dir.size() - 1] == ':')) {
        return;
    }

    setenv("LD_LIBRARY_PATH", (dir + ":" + current).c_str(), 1);
}

template <typename Fn>
static bool load_symbol(void* handle, Fn& fn, const char* symbol, const char* pretty, const char* tag) {
    dlerror();
    void* ptr = dlsym(handle, symbol);
    const char* err = dlerror();
    if (ptr == nullptr || err != nullptr) {
        LOGE("[%s] dlsym failed for %s (%s): %s\n", tag, pretty, symbol, err ? err : "symbol is null");
        return false;
    }
    fn = ptr;
    return true;
}

} // namespace

MtkRwkvDlopen::~MtkRwkvDlopen() {
    close();
}

int MtkRwkvDlopen::open(const char* tag, const char* env_var, const char* default_library_name, void* extra) {
    if (_handle != nullptr) {
        return RWKV_SUCCESS;
    }

    _tag = tag ? tag : "mtk";
    const std::string default_name = default_library_name ? default_library_name : "";

    std::vector<std::string> candidates;
    if (extra != nullptr && std::strlen(reinterpret_cast<const char*>(extra)) > 0) {
        candidates.emplace_back(reinterpret_cast<const char*>(extra));
    }

    const char* env_path = env_var ? std::getenv(env_var) : nullptr;
    if (env_path != nullptr && env_path[0] != '\0') {
        candidates.emplace_back(env_path);
    }

    const char* generic_env_path = std::getenv("RWKV_MTK_LIB");
    if (generic_env_path != nullptr && generic_env_path[0] != '\0') {
        candidates.emplace_back(generic_env_path);
    }

    if (!default_name.empty()) {
        candidates.emplace_back(default_name);
        const std::string cwd_path = get_cwd_path(default_name.c_str());
        if (!cwd_path.empty()) {
            candidates.emplace_back(cwd_path);
        }
    }

    if (candidates.empty()) {
        LOGE("[%s] no MTK runtime library candidate\n", _tag.c_str());
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
    }

    std::string last_error;
    for (const auto& candidate : candidates) {
        dlerror();
        _handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
        const char* err = dlerror();
        if (_handle != nullptr) {
            _path = candidate;
            LOGI("[%s] loaded MTK runtime library: %s\n", _tag.c_str(), _path.c_str());
            break;
        }
        if (err != nullptr) {
            last_error = err;
        }
        LOGW("[%s] dlopen failed for %s: %s\n", _tag.c_str(), candidate.c_str(), err ? err : "unknown error");
    }

    if (_handle == nullptr) {
        LOGE("[%s] failed to load MTK runtime library: %s\n", _tag.c_str(), last_error.c_str());
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
    }

    bool ok = true;
    ok &= load_symbol(_handle, _api.set_log_callback, "_Z28neuron_rwkv_set_log_callbackPFvPviPKcS1_ES_", "neuron_rwkv_set_log_callback", _tag.c_str());
    ok &= load_symbol(_handle, _api.init, "_Z16neuron_rwkv_initPPvRK16RWKVModelOptionsRK18RWKVRuntimeOptions", "neuron_rwkv_init", _tag.c_str());
    ok &= load_symbol(_handle, _api.release, "_Z19neuron_rwkv_releasePv", "neuron_rwkv_release", _tag.c_str());
    ok &= load_symbol(_handle, _api.inference_once, "_Z26neuron_rwkv_inference_oncePvi", "neuron_rwkv_inference_once", _tag.c_str());
    ok &= load_symbol(_handle, _api.prefill, "_Z19neuron_rwkv_prefillPvPKim", "neuron_rwkv_prefill", _tag.c_str());
    ok &= load_symbol(_handle, _api.eval_with_embeddings, "_Z32neuron_rwkv_eval_with_embeddingsPvPKfm", "neuron_rwkv_eval_with_embeddings", _tag.c_str());
    ok &= load_symbol(_handle, _api.reset, "_Z17neuron_rwkv_resetPv", "neuron_rwkv_reset", _tag.c_str());
    ok &= load_symbol(_handle, _api.get_att_state_size, "_Z30neuron_rwkv_get_att_state_sizePvi", "neuron_rwkv_get_att_state_size", _tag.c_str());
    ok &= load_symbol(_handle, _api.get_wkv_state_size, "_Z30neuron_rwkv_get_wkv_state_sizePvi", "neuron_rwkv_get_wkv_state_size", _tag.c_str());
    ok &= load_symbol(_handle, _api.get_ffn_state_size, "_Z30neuron_rwkv_get_ffn_state_sizePvi", "neuron_rwkv_get_ffn_state_size", _tag.c_str());
    ok &= load_symbol(_handle, _api.get_att_state, "_Z25neuron_rwkv_get_att_statePviS_m", "neuron_rwkv_get_att_state", _tag.c_str());
    ok &= load_symbol(_handle, _api.get_wkv_state, "_Z25neuron_rwkv_get_wkv_statePviS_m", "neuron_rwkv_get_wkv_state", _tag.c_str());
    ok &= load_symbol(_handle, _api.get_ffn_state, "_Z25neuron_rwkv_get_ffn_statePviS_m", "neuron_rwkv_get_ffn_state", _tag.c_str());
    ok &= load_symbol(_handle, _api.set_att_state, "_Z25neuron_rwkv_set_att_statePviPKvm", "neuron_rwkv_set_att_state", _tag.c_str());
    ok &= load_symbol(_handle, _api.set_wkv_state, "_Z25neuron_rwkv_set_wkv_statePviPKvm", "neuron_rwkv_set_wkv_state", _tag.c_str());
    ok &= load_symbol(_handle, _api.set_ffn_state, "_Z25neuron_rwkv_set_ffn_statePviPKvm", "neuron_rwkv_set_ffn_state", _tag.c_str());

    if (!ok) {
        close();
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
    }

    const std::string lib_dir = dirname_of(_path);
    prepend_library_path(lib_dir);

    return RWKV_SUCCESS;
}

void MtkRwkvDlopen::close() {
    if (_handle != nullptr) {
        dlclose(_handle);
    }
    _handle = nullptr;
    _api = {};
    _path.clear();
}

} // namespace rwkvmobile
