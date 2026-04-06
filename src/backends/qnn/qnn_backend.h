#ifndef QNN_BACKEND_H
#define QNN_BACKEND_H

#include "backend.h"
#include "rwkv-qualcomm/Interfaces.hpp"
#include "rwkv-qualcomm/Utils/IOTensor.hpp"
#include "rmpack.h"

#include <atomic>
#include <mutex>

#ifdef ENABLE_MNN
#include <MNN/Interpreter.hpp>
#endif

namespace rwkvmobile {

class qnn_backend_context {
public:
    qnn_backend_context(std::string qnnBackendPath);

    ~qnn_backend_context();

    int qnn_create_power_config_id();
    int qnn_destory_power_config_id();
    int qnn_set_power_config();
    int qnn_register_op_package(std::string package_path, std::string interface_provider);
    int qnn_set_rpc_latency_and_polling();

    uint32_t powerConfigId = 0;
    uint32_t deviceId = 0;
    uint32_t coreId = 0;

    std::string qnnBackendPath;
    std::string qnnBackendBasePath;
    void *qnnBackendLibraryHandle = nullptr;

    qnn::tools::rwkv_app::QnnFunctionPointers qnnFunctionPointers;

    Qnn_LogHandle_t qnnLogHandle = nullptr;
    Qnn_BackendHandle_t qnnBackendHandle = nullptr;
    Qnn_DeviceHandle_t qnnDeviceHandle = nullptr;

    std::mutex qnnMutex;
    int ref_count = 0;
};

class qnn_backend : public execution_provider {
public:
    ~qnn_backend() {
        release_model();
        release();
    }

    int init(void * extra) override;
    int load_model(std::string model_path, void * extra = nullptr) override;
    float get_load_progress() const override;
    int eval(int id, Tensor1D & logits) override;
    int eval(std::vector<int> ids, Tensor1D & logits) override;
    int eval_with_embeddings(const float *embeddings, int n_tokens, Tensor1D & logits) override;
    int eval_batch(std::vector<std::vector<int>> ids, Tensor1D & logits) override;

    bool is_available() override;
    int zero_state() override;
    int get_state(std::any &state) override;
    int set_state(std::any state) override;
    int free_state(std::any state) override;
    int release_model() override;
    int release() override;
    double get_prefill_speed() override {
        return prefill_speed;
    }

    int debug_dump_state();

    int get_state_on_batch_slot(int slot, std::any &state) override;
    int set_state_on_batch_slot(int slot, std::any state) override;
    int zero_state_on_batch_slot(int slot) override;

    int load_raw_states(std::vector<std::vector<half_float::half>> states) override;
    int serialize_runtime_state(std::any state, std::vector<uint8_t> &data) override;
    int deserialize_runtime_state(std::vector<uint8_t> &data, std::any &state) override;

    int copy_float_to_qnn_tensor(Qnn_Tensor_t *qnn_tensor, const float *buffer, size_t element_count);

    int copy_qnn_tensor_to_float(Qnn_Tensor_t *qnn_tensor, float *buffer, size_t element_count);

    int post_graph_execute(Tensor1D & logits);

    int deep_embedding_size = 0;
    bool has_deep_embedding = false;

private:
    double prefill_speed = -1;
    void *qnnModelHandle = nullptr;

    bool isTensorInitialized = false;

    int prefillSequenceLength = 0;
    int embdPrefillSequenceLength = 0;

    std::vector<Qnn_ContextHandle_t> qnnContextHandles;

    uint32_t qnnPrefillGraphsCount = 0;
    GraphInfo_t **qnnPrefillGraphsInfo = nullptr;

    uint32_t qnnEmbdGraphsCount = 0;
    GraphInfo_t **qnnEmbdGraphsInfo = nullptr;

    uint32_t qnnEmbdPrefillGraphsCount = 0;
    GraphInfo_t **qnnEmbdPrefillGraphsInfo = nullptr;

    // Dynamic batch decode graphs - maps batch size to graph info
    std::unordered_map<int, uint32_t> qnnBatchDecodeGraphsCount;
    std::unordered_map<int, GraphInfo_t**> qnnBatchDecodeGraphsInfo;

    uint32_t graphConfigsInfoCount = 0;
    GraphConfigInfo_t **graphConfigsInfo = nullptr;

    Qnn_Tensor_t *inputTensorsPrefill[8] = {nullptr};
    Qnn_Tensor_t *outputTensorsPrefill[8] = {nullptr};

    Qnn_Tensor_t *inputTensorsEmbd[8] = {nullptr};
    Qnn_Tensor_t *outputTensorsEmbd[8] = {nullptr};

    Qnn_Tensor_t *inputTensorsEmbdPrefill[8] = {nullptr};
    Qnn_Tensor_t *outputTensorsEmbdPrefill[8] = {nullptr};

    // Dynamic batch decode tensors - maps batch size to tensor arrays
    std::unordered_map<int, Qnn_Tensor_t**> inputTensorsBatchDecode;
    std::unordered_map<int, Qnn_Tensor_t**> outputTensorsBatchDecode;

    Qnn_Tensor_t *logitsOutputTensor = nullptr;

    Qnn_Tensor_t *vFirstTensor = nullptr;
    Qnn_Tensor_t *vFirstTensorPrefill = nullptr;
    Qnn_Tensor_t *hiddenStateTensor = nullptr;
    Qnn_Tensor_t *hiddenStateTensorPrefill = nullptr;

    // input tensors
    Qnn_Tensor_t *tokenInputTensorPrefill = nullptr;
    Qnn_Tensor_t *tokenInputTensorEmbd = nullptr;
    Qnn_Tensor_t *tokenInputTensorEmbdPrefill = nullptr;

    std::unordered_map<int, Qnn_Tensor_t*> tokenInputTensorBatchDecode;
    std::unordered_map<int, Qnn_Tensor_t*> deepEmbeddingTensors;
    std::unordered_map<int, Qnn_Tensor_t*> deepEmbeddingPrefillTensors;

    IOTensor* qnnIOTensorUtils = nullptr;

    size_t logitsOutputTensorSize = 0;

    // TODO: simplify this
    std::vector<std::unordered_map<std::string, void*>> prefillGraphsTensorNameToTensorPointer;
    std::vector<std::unordered_map<std::string, size_t>> prefillGraphsTensorNameToSize;
    std::vector<std::unordered_map<std::string, void*>> embdGraphsTensorNameToTensorPointer;
    std::vector<std::unordered_map<std::string, size_t>> embdGraphsTensorNameToSize;
    std::vector<std::unordered_map<std::string, void*>> embdPrefillGraphsTensorNameToTensorPointer;
    std::vector<std::unordered_map<std::string, size_t>> embdPrefillGraphsTensorNameToSize;
    // Dynamic batch decode tensor maps - maps batch size to tensor name maps
    std::unordered_map<int, std::vector<std::unordered_map<std::string, void*>>> batchDecodeGraphsTensorNameToTensorPointer;
    std::unordered_map<int, std::vector<std::unordered_map<std::string, size_t>>> batchDecodeGraphsTensorNameToSize;

    std::unordered_map<std::string, void*> stateTensorsNameToTensorPointer;

    int qnn_initialize_tensors();

    void fill_quantized_tensor(float value, Qnn_Tensor_t *tensor);

    int populate_tensor_name_to_size_map(const GraphInfo_t& graphInfo, 
                                        std::unordered_map<std::string, size_t>& tensorNameToSize,
                                        bool isInputTensors);

    // Helper functions for tensor initialization
    int setup_output_tensors_for_graph(int graph_id, int total_graphs_count, const GraphInfo_t& graphInfo,
                                       Qnn_Tensor_t** outputTensors, 
                                       std::unordered_map<std::string, void*>& tensorNameToTensorPointer,
                                       std::unordered_map<std::string, size_t>& tensorNameToSize,
                                       Qnn_ContextHandle_t contextHandle,
                                       Qnn_Tensor_t* vFirstTensorRef, Qnn_Tensor_t* hiddenStateTensorRef,
                                       bool isPrefill);

    void populate_input_shared_tensor_map(const GraphInfo_t& graphInfo, int graph_id,
                                          std::unordered_map<std::string, Qnn_Tensor_t*>& sharedTensorMap,
                                          Qnn_Tensor_t* vFirstTensorRef, Qnn_Tensor_t* hiddenStateTensorRef,
                                          bool isPrefill);

    void map_deep_embedding_tensors(const GraphInfo_t& graphInfo, int graph_id,
                                    std::unordered_map<std::string, void*>& tensorNameToTensorPointer,
                                    std::unordered_map<int, Qnn_Tensor_t*>& deepEmbeddingTensorsRef,
                                    bool isPrefill);

    int execute_graph(GraphInfo_t** graphInfo, int graphsCount, Qnn_Tensor_t** inputTensors, Qnn_Tensor_t** outputTensors);
    int execute_prefill_graph();
    int execute_emb_decode_graph();
    int execute_emb_prefill_graph();
    int execute_batch_decode_graph(int bsz);

    int copy_deep_embedding_to_qnn_tensor_decode(int idx);
    int copy_deep_embedding_to_qnn_tensor_prefill(int idx, int token_offset);

    std::vector<float> logits_buffer;

    std::shared_ptr<uint8_t> external_embeddings = nullptr;
    std::shared_ptr<uint8_t> external_deep_embeddings = nullptr;
    int deep_embeddings_elembytes = 2;
    std::string external_lmhead_filetype = "None";

#ifdef ENABLE_MNN
    MNN::Interpreter *external_lmhead_interpretor = nullptr;
    MNN::Session *external_lmhead_mnn_session = nullptr;
    MNN::Tensor *external_lmhead_input_tensor = nullptr;
#endif

    RMPackReader *rmpack = nullptr;

    // load progress (chunk-based, smoothed on each query)
    std::atomic<int> _load_total_chunks{0};
    std::atomic<int> _load_done_chunks{0};
    mutable std::mutex _load_progress_mutex;
    mutable float _load_progress_reported = 0.f;

    // Helpers for dynamic batch decode processing
    static int parse_bsz_from_graph_name(const std::string &graphName);
    int initialize_batch_decode_graphs(
        uint32_t graphsCount,
        GraphInfo_t **graphsInfo,
        std::vector<std::unordered_map<std::string, void*>> &tensorNameToTensorPointer,
        std::vector<std::unordered_map<std::string, size_t>> &tensorNameToSize,
        Qnn_Tensor_t **inputTensorsArr,
        Qnn_Tensor_t **outputTensorsArr,
        const char *inTensorName,
        int batchSize);

    void cleanup_batch_graphs();
};

}

#endif
