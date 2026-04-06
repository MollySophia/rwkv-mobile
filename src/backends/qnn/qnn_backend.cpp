#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <any>
#include <mutex>

#include "qnn_backend.h"
#include "commondef.h"
#include "soc_detect.h"
#include "PAL/DynamicLoading.hpp"
#include "DynamicLoadUtil.hpp"
#include "DataUtil.hpp"
#include "Utils.hpp"
#include "QnnTypeMacros.hpp"
#include "rmpack.h"
#include <HTP/QnnHtpPerfInfrastructure.h>
#include <HTP/QnnHtpDevice.h>
#include <HTP/QnnHtpGraph.h>
#include <HTP/QnnHtpContext.h>
#include <QnnContext.h>
#include <QnnSdkBuildId.h>

#include "logger.h"
#include "half.hpp"
#include "soc_detect.h"

#ifdef _WIN32
#define USE_MMAP 0
#include <cstdlib>
#else
#define USE_MMAP 1
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#define DEFAULT_QNN_LOGLEVEL QNN_LOG_LEVEL_ERROR

namespace rwkvmobile {

using namespace qnn::tools;

// global qnn backend context pointer
std::shared_ptr<qnn_backend_context> g_qnn_backend_context_ptr = nullptr;

static void logCallback(const char* fmt,
    QnnLog_Level_t level,
    uint64_t timestamp,
    va_list argp) {
    char buffer[1024];

    // TODO
    try {
        vsnprintf(buffer, sizeof(buffer), fmt, argp);
        LOGI("[QNN] %s", buffer);
    } catch (const std::exception& e) {
    }

    return;
}

static void getTensorDims(std::vector<size_t>& dims,
    uint32_t* inDimensions,
    uint32_t rank) {
    if (nullptr == inDimensions) {
        LOGE("input dimensions is nullptr");
        return;
    }
    for (size_t r = 0; r < rank; r++) {
        dims.push_back(inDimensions[r]);
    }
}

static size_t getQnnDatatypeSize(Qnn_DataType_t dataType) {
    switch (dataType) {
        case QNN_DATATYPE_FLOAT_16:
        case QNN_DATATYPE_UFIXED_POINT_16:
        case QNN_DATATYPE_UINT_16:
        case QNN_DATATYPE_INT_16:
            return sizeof(uint16_t);
        case QNN_DATATYPE_FLOAT_32:
        case QNN_DATATYPE_INT_32:
        case QNN_DATATYPE_UINT_32:
            return sizeof(uint32_t);
        case QNN_DATATYPE_UFIXED_POINT_8:
        case QNN_DATATYPE_UINT_8:
        case QNN_DATATYPE_INT_8:
        case QNN_DATATYPE_BOOL_8:
            return sizeof(uint8_t);
        case QNN_DATATYPE_FLOAT_64:
        case QNN_DATATYPE_INT_64:
        case QNN_DATATYPE_UINT_64:
            return sizeof(uint64_t);
        default:
            LOGE("Unsupported data type");
            return 0;
    }
}

int qnn_backend::parse_bsz_from_graph_name(const std::string &graphName) {
    auto pos = graphName.find("bsz");
    if (pos == std::string::npos) return 1;
    pos += 3;
    int val = 0;
    while (pos < graphName.size() && isdigit(graphName[pos])) {
        val = val * 10 + (graphName[pos] - '0');
        pos++;
    }
    return val;
}

int qnn_backend::initialize_batch_decode_graphs(
        uint32_t graphsCount,
        GraphInfo_t **graphsInfo,
        std::vector<std::unordered_map<std::string, void*>> &tensorNameToTensorPointer,
        std::vector<std::unordered_map<std::string, size_t>> &tensorNameToSize,
        Qnn_Tensor_t **inputTensorsArr,
        Qnn_Tensor_t **outputTensorsArr,
        const char *inTensorName,
        int batchSize) {
    for (int graph_id = 0; graph_id < (int)graphsCount; graph_id++) {
        std::unordered_map<std::string, Qnn_Tensor_t*> sharedTensorMap;
        auto graphInfo = (*graphsInfo)[graph_id];
        LOGI("Batch%d Decode Graph %d : %s", batchSize, graph_id, graphInfo.graphName);

        // output sizes
        auto result = populate_tensor_name_to_size_map(graphInfo, tensorNameToSize[graph_id], false);
        if (result != RWKV_SUCCESS) {
            return result;
        }

        // output tensors
        result = setup_output_tensors_for_graph(graph_id, graphsCount, graphInfo,
                                               outputTensorsArr, tensorNameToTensorPointer[graph_id],
                                               tensorNameToSize[graph_id], qnnContextHandles[graph_id],
                                               vFirstTensor, hiddenStateTensor, false);
        if (result != RWKV_SUCCESS) {
            return result;
        }

        // input sizes
        result = populate_tensor_name_to_size_map(graphInfo, tensorNameToSize[graph_id], true);
        if (result != RWKV_SUCCESS) {
            return result;
        }

        // shared inputs
        populate_input_shared_tensor_map(graphInfo, graph_id, sharedTensorMap, vFirstTensor, hiddenStateTensor, false);

        if (!qnnIOTensorUtils->setupInputWithSharedTensors(&inputTensorsArr[graph_id], tensorNameToTensorPointer[graph_id], graphInfo,
                                        tensorNameToSize[graph_id], qnnContextHandles[graph_id], sharedTensorMap)) {
            LOGE("Error in setting up Input Tensors for Batch%d Decode", batchSize);
            return RWKV_ERROR_IO;
        }

        // deep embedding mapping
        map_deep_embedding_tensors(graphInfo, graph_id, tensorNameToTensorPointer[graph_id], 
                                   deepEmbeddingTensors, false);
    }

    // find input tensor name
    if (tensorNameToTensorPointer.size() > 0) {
        if (tensorNameToTensorPointer[0].find(inTensorName) != tensorNameToTensorPointer[0].end()) {
            tokenInputTensorBatchDecode[batchSize] = (Qnn_Tensor_t*)tensorNameToTensorPointer[0][inTensorName];
        } else {
            std::string chunk1Name = std::string(inTensorName) + "_chunk1";
            if (tensorNameToTensorPointer[0].find(chunk1Name) != tensorNameToTensorPointer[0].end()) {
                tokenInputTensorBatchDecode[batchSize] = (Qnn_Tensor_t*)tensorNameToTensorPointer[0][chunk1Name];
            }
        }
    }

    return RWKV_SUCCESS;
}
qnn_backend_context::qnn_backend_context(std::string qnnBackendPath) : qnnBackendPath(qnnBackendPath) {
    LOGI("QNN Backend Path: %s", qnnBackendPath.c_str());
    if (qnnBackendPath.empty()) {
        throw std::invalid_argument("QNN backend path is empty");
    }

    if (!std::filesystem::exists(qnnBackendPath)) {
        throw std::runtime_error("QNN backend path does not exist");
    }

    auto qnnStatusCode = dynamicloadutil::getQnnFunctionPointers(
        qnnBackendPath, &qnnFunctionPointers, &qnnBackendLibraryHandle);

    if (dynamicloadutil::StatusCode::SUCCESS != qnnStatusCode) {
        if (dynamicloadutil::StatusCode::FAIL_LOAD_BACKEND == qnnStatusCode) {
            LOGE("Error initializing QNN Function Pointers: could not load backend: %s", qnnBackendPath.c_str());
            throw std::runtime_error("Failed to get QNN function pointers");
        // } else if (dynamicloadutil::StatusCode::FAIL_LOAD_MODEL == qnnStatusCode) {
        //     LOGE("Error initializing QNN Function Pointers: could not load model:%s ", model_path.c_str());
        //     throw std::runtime_error("Failed to get QNN function pointers, Failed to load model");
        } else {
            LOGE("Error initializing QNN Function Pointers");
            throw std::runtime_error("Failed to get QNN function pointers");
        }
    }

    std::string qnnSystemLibPath;
    qnnBackendBasePath =
#ifdef _WIN32
        qnnBackendPath.substr(0, qnnBackendPath.find("QnnHtp.dll"));// + "QnnSystem.dll";
        qnnSystemLibPath = qnnBackendBasePath + "QnnSystem.dll";
#else
        qnnBackendPath.substr(0, qnnBackendPath.find("libQnnHtp.so"));// + "libQnnSystem.so";
        qnnSystemLibPath = qnnBackendBasePath + "libQnnSystem.so";
#endif

    auto qnnSystemLibStatus = dynamicloadutil::getQnnSystemFunctionPointers(qnnSystemLibPath, &qnnFunctionPointers);
    if (dynamicloadutil::StatusCode::SUCCESS != qnnSystemLibStatus) {
        throw std::runtime_error("Failed to get QNN system function pointers");
    }

    // initialize QNN logging
    auto logLevel = DEFAULT_QNN_LOGLEVEL;
    if (QNN_SUCCESS !=
        qnnFunctionPointers.qnnInterface.logCreate(logCallback, logLevel, &qnnLogHandle)) {
        LOGW("Unable to initialize logging in the backend.");
    }

    // initialize QNN backend
    auto qnnBackendStatus = qnnFunctionPointers.qnnInterface.backendCreate(
        qnnLogHandle, nullptr, &qnnBackendHandle);
    if (QNN_BACKEND_NO_ERROR != qnnBackendStatus) {
      throw std::runtime_error("Failed to initialize backend due to error = " + std::to_string(qnnBackendStatus));
    }
    LOGI("Initialize Backend Returned Status = %lu", qnnBackendStatus);

    if (nullptr != qnnFunctionPointers.qnnInterface.propertyHasCapability) {
        auto qnnDevicePropertyStatus = qnnFunctionPointers.qnnInterface.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
        if (QNN_PROPERTY_NOT_SUPPORTED == qnnDevicePropertyStatus) {
            LOGW("Device property is not supported");
        }
        if (QNN_PROPERTY_ERROR_UNKNOWN_KEY == qnnDevicePropertyStatus) {
            throw std::runtime_error("Device property is not known to backend");
        }

        auto qnnCreateDeviceStatus = qnnFunctionPointers.qnnInterface.deviceCreate(qnnLogHandle, nullptr, &qnnDeviceHandle);
        if (QNN_SUCCESS != qnnCreateDeviceStatus && QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != qnnCreateDeviceStatus) {
            LOGE("Failed to create device");
            throw std::runtime_error("Failed to create device");
        }
    }

    // power config apis
    if (RWKV_SUCCESS != qnn_create_power_config_id()) {
        LOGE("Could not create HTP power config id");
    } else {
        if (RWKV_SUCCESS != qnn_set_rpc_latency_and_polling()) {
            LOGE("Could not set HTP rpc latency and polling");
        } else if (RWKV_SUCCESS != qnn_set_power_config()) {
            LOGE("Could not set HTP power config");
        }
    }

    // load custom op package
    soc_detect soc_detect;
    soc_detect.detect_platform();
    std::string htp_arch = soc_detect.get_htp_arch();
    if (htp_arch.empty() || htp_arch == "Unknown") {
        LOGE("HTP architecture is unknown");
        throw std::runtime_error("HTP architecture is unknown");
    }
    if (htp_arch[0] == 'v') {
        htp_arch[0] = 'V';
    }
    std::string custom_op_name = "libQnnRwkvWkvOpPackage" + htp_arch + ".so";

    std::vector<std::string> paths;
    if (!qnnBackendBasePath.empty()) {
        paths.push_back(qnnBackendBasePath);
    } else {
#ifndef _WIN32
        const char* ldLibraryPath = getenv("LD_LIBRARY_PATH");
        if (ldLibraryPath) {
            std::string pathStr(ldLibraryPath);
            std::stringstream ss(pathStr);
            std::string dir;
            while (std::getline(ss, dir, ':')) {
                paths.push_back(dir);
            }
        }
#endif
    }

    // Don't load custom op package on Windows for now
#ifndef _WIN32
    for (auto dir : paths) {
        std::string fullPath = dir + "/" + custom_op_name;
        std::ifstream file(fullPath);
        if (file.good()) {
            LOGI("Found %s in path: %s", custom_op_name.c_str(), fullPath.c_str());
            if (RWKV_SUCCESS != qnn_register_op_package(fullPath, "RwkvWkvOpPackageInterfaceProvider")) {
                LOGE("Op package registration failed");
            }
            break;
        }
    }
#endif
}

qnn_backend_context::~qnn_backend_context() {
    qnn_destory_power_config_id();

    if (nullptr != qnnFunctionPointers.qnnInterface.propertyHasCapability) {
        auto qnnDevicePropertyStatus = qnnFunctionPointers.qnnInterface.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
        if (QNN_PROPERTY_NOT_SUPPORTED == qnnDevicePropertyStatus) {
            LOGW("Device property is not supported");
        }

        auto qnnStatus = qnnFunctionPointers.qnnInterface.deviceFree(qnnDeviceHandle);
        if (QNN_SUCCESS != qnnStatus && QNN_DEVICE_ERROR_UNSUPPORTED_FEATURE != qnnStatus) {
            LOGE("Failed to free device");
        }
    }

    if ((nullptr != qnnBackendHandle && nullptr != qnnFunctionPointers.qnnInterface.backendFree) &&
        QNN_BACKEND_NO_ERROR != qnnFunctionPointers.qnnInterface.backendFree(qnnBackendHandle)) {
        LOGE("Could not terminate QNN backend");
    }
    qnnBackendHandle = nullptr;

    if (nullptr != qnnFunctionPointers.qnnInterface.logFree && nullptr != qnnLogHandle) {
        if (QNN_SUCCESS != qnnFunctionPointers.qnnInterface.logFree(qnnLogHandle)) {
            LOGW("Unable to terminate logging in the backend.");
        }
    }

    if (qnnBackendLibraryHandle)
        pal::dynamicloading::dlClose(qnnBackendLibraryHandle);
}

int qnn_backend_context::qnn_register_op_package(std::string package_path, std::string interface_provider) {
    if (nullptr == qnnFunctionPointers.qnnInterface.backendRegisterOpPackage) {
        LOGE("backendRegisterOpPackageFnHandle is nullptr.");
        return RWKV_ERROR_UNSUPPORTED;
    }
    if (QNN_BACKEND_NO_ERROR != qnnFunctionPointers.qnnInterface.backendRegisterOpPackage(
                qnnBackendHandle,
                package_path.c_str(),
                interface_provider.c_str(),
                nullptr)) {
        LOGE("Could not register Op Package: %s and interface provider: %s",
            package_path.c_str(),
            interface_provider.c_str());
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
    }
    LOGI("Registered Op Package: %s and interface provider: %s",
        package_path.c_str(),
        interface_provider.c_str()
    );
    return RWKV_SUCCESS;
}

int qnn_backend_context::qnn_create_power_config_id() {
    QnnDevice_Infrastructure_t deviceInfra = nullptr;
    Qnn_ErrorHandle_t devErr = qnnFunctionPointers.qnnInterface.deviceGetInfrastructure(&deviceInfra);
    if (devErr != QNN_SUCCESS) {
        // LOGE("deviceGetInfrastructure error");
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
    }
    QnnHtpDevice_Infrastructure_t *htpInfra = static_cast<QnnHtpDevice_Infrastructure_t *>(deviceInfra);
    QnnHtpDevice_PerfInfrastructure_t perfInfra = htpInfra->perfInfra;
    Qnn_ErrorHandle_t perfInfraErr = perfInfra.createPowerConfigId(deviceId, coreId, &powerConfigId);
    if (perfInfraErr != QNN_SUCCESS) {
      LOGE("createPowerConfigId failed");
      return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
    }
    return RWKV_SUCCESS;
}

int qnn_backend_context::qnn_destory_power_config_id() {
    QnnDevice_Infrastructure_t deviceInfra = nullptr;
    Qnn_ErrorHandle_t devErr = qnnFunctionPointers.qnnInterface.deviceGetInfrastructure(&deviceInfra);
    if (devErr != QNN_SUCCESS) {
        // LOGE("deviceGetInfrastructure error");
        return RWKV_ERROR_BACKEND | RWKV_ERROR_RELEASE;
    }
    QnnHtpDevice_Infrastructure_t *htpInfra = static_cast<QnnHtpDevice_Infrastructure_t *>(deviceInfra);
    QnnHtpDevice_PerfInfrastructure_t perfInfra = htpInfra->perfInfra;

    Qnn_ErrorHandle_t perfInfraErr = perfInfra.destroyPowerConfigId(powerConfigId);
    if (perfInfraErr != QNN_SUCCESS) {
        LOGE("destroyPowerConfigId failed");
        return RWKV_ERROR_BACKEND | RWKV_ERROR_RELEASE;
    }
    return RWKV_SUCCESS;
}

int qnn_backend_context::qnn_set_power_config() {
    QnnDevice_Infrastructure_t deviceInfra = nullptr;
    Qnn_ErrorHandle_t devErr = qnnFunctionPointers.qnnInterface.deviceGetInfrastructure(&deviceInfra);
    if (devErr != QNN_SUCCESS) {
        LOGE("device error");
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
    }
    QnnHtpDevice_Infrastructure_t *htpInfra = static_cast<QnnHtpDevice_Infrastructure_t *>(deviceInfra);
    QnnHtpDevice_PerfInfrastructure_t perfInfra = htpInfra->perfInfra;

    QnnHtpPerfInfrastructure_PowerConfig_t powerConfig;
    memset(&powerConfig, 0, sizeof(powerConfig));
    powerConfig.option                     = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    powerConfig.dcvsV3Config.dcvsEnable    = 0; //True to enable Dcvs, False to disbale
    powerConfig.dcvsV3Config.setDcvsEnable = 1;
    powerConfig.dcvsV3Config.contextId     = powerConfigId;  //use the power config id created

    // refer QnnHtpPerfInfrastructure.h
    powerConfig.dcvsV3Config.powerMode       = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    powerConfig.dcvsV3Config.setSleepLatency = 1; //True to consider Latency parameter otherwise False
    powerConfig.dcvsV3Config.setBusParams    = 1; //True to consider Bus parameter otherwise False
    powerConfig.dcvsV3Config.setCoreParams   = 1; //True to consider Core parameter otherwise False
    powerConfig.dcvsV3Config.sleepDisable    = 1; //True to disable sleep, False to re-enable sleep
    powerConfig.dcvsV3Config.setSleepDisable = 1; //True to consider sleep disable/enable parameter otherwise False

    //Set Sleep latency parameter
    powerConfig.dcvsV3Config.sleepLatency    =  40; // set dsp sleep latency ranges 10-65535 micro sec, refer hexagon sdk

    //set Bus Clock Parameters (refer QnnHtpPerfInfrastructure.h)
    powerConfig.dcvsV3Config.busVoltageCornerMin     = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.busVoltageCornerTarget  = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.busVoltageCornerMax     = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

    //set Core Clock Parameters (refer QnnHtpPerfInfrastructure.h)
    powerConfig.dcvsV3Config.coreVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.coreVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;

    QnnHtpPerfInfrastructure_PowerConfig_t powerConfigHMX;
    memset(&powerConfigHMX, 0, sizeof(powerConfigHMX));
    powerConfigHMX.option                     = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_HMX_V2;
    powerConfigHMX.hmxV2Config.hmxPickDefault = 0;
    powerConfigHMX.hmxV2Config.hmxPerfMode    = QNN_HTP_PERF_INFRASTRUCTURE_CLK_PERF_HIGH;

    powerConfigHMX.hmxV2Config.hmxVoltageCornerMin    = DCVS_EXP_VCORNER_TUR;
    powerConfigHMX.hmxV2Config.hmxVoltageCornerTarget = DCVS_EXP_VCORNER_TUR;
    powerConfigHMX.hmxV2Config.hmxVoltageCornerMax    = DCVS_EXP_VCORNER_TUR;

    // Set power config with different performance parameters
    const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs[] = {&powerConfig, &powerConfigHMX, NULL};

    Qnn_ErrorHandle_t perfInfraErr = perfInfra.setPowerConfig(powerConfigId, powerConfigs);
    if (perfInfraErr != QNN_SUCCESS) {
        const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigsWithoutHMX[] = {&powerConfig, NULL};
        perfInfraErr = perfInfra.setPowerConfig(powerConfigId, powerConfigsWithoutHMX);
    }
    return RWKV_SUCCESS;
}

int qnn_backend_context::qnn_set_rpc_latency_and_polling() {
    QnnDevice_Infrastructure_t deviceInfra = nullptr;
    Qnn_ErrorHandle_t devErr = qnnFunctionPointers.qnnInterface.deviceGetInfrastructure(&deviceInfra);
    if (devErr != QNN_SUCCESS) {
      LOGE("device error");
      return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
    }
    QnnHtpDevice_Infrastructure_t *htpInfra = static_cast<QnnHtpDevice_Infrastructure_t *>(deviceInfra);
    QnnHtpDevice_PerfInfrastructure_t perfInfra = htpInfra->perfInfra;

    // set RPC Control Latency
    QnnHtpPerfInfrastructure_PowerConfig_t rpcControlLatency;            // refer QnnHtpPerfInfrastructure.h
    memset(&rpcControlLatency, 0, sizeof(rpcControlLatency));
    rpcControlLatency.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
    rpcControlLatency.rpcControlLatencyConfig = 100;         // use rpc control latency recommended 100 us, refer hexagon sdk
    const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs1[] = {&rpcControlLatency, NULL};

    Qnn_ErrorHandle_t perfInfraErr = perfInfra.setPowerConfig(powerConfigId, powerConfigs1);  // set RPC latency config on power config id created
    if (perfInfraErr != QNN_SUCCESS) {
        LOGE("setPowerConfig failed");
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
    }

    // set RPC Polling
    QnnHtpPerfInfrastructure_PowerConfig_t rpcPollingTime;   // refer QnnHtpPerfInfrastructure.h
    memset(&rpcPollingTime, 0, sizeof(rpcPollingTime));
    rpcPollingTime.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
    rpcPollingTime.rpcPollingTimeConfig = 9999;     // use rpc polling time recommended 0-10000 us
    const QnnHtpPerfInfrastructure_PowerConfig_t* powerConfigs2[] = {&rpcPollingTime, NULL};

    perfInfraErr = perfInfra.setPowerConfig(powerConfigId, powerConfigs2); // set RPC polling config on power config id created
    if (perfInfraErr != QNN_SUCCESS) {
        LOGE("setPowerConfig failed");
        return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
    }
    return RWKV_SUCCESS;
}

int qnn_backend::init(void * extra) {
    LOGI("QNN_SDK_BUILD_ID: %s", QNN_SDK_BUILD_ID);
    std::string path;
    if (extra != nullptr) {
        path = std::string((char *)extra);
        LOGI("Using QNN Backend Path: %s\n", path.c_str());
    } else {
#ifdef _WIN32
        path = "QnnHtp.dll";
#else
        path = "libQnnHtp.so";
#endif
        LOGI("Using default QNN Backend Path: %s\n", path.c_str());
    }

#ifndef _WIN32
    std::string path_to_set;
    if (path != "libQnnHtp.so") {
        path_to_set = path.substr(0, path.find_last_of('/'));
        LOGI("Setting LD_LIBRARY_PATH and ADSP_LIBRARY_PATH to %s\n", path_to_set.c_str());
        setenv("LD_LIBRARY_PATH", path_to_set.c_str(), 1);
        setenv("ADSP_LIBRARY_PATH", path_to_set.c_str(), 1);
    }
#else
    std::string path_to_set;
    if (path != "QnnHtp.dll") {
        if (path.find('/') != std::string::npos)
            path_to_set = path.substr(0, path.find_last_of('/'));
        else
            path_to_set = path.substr(0, path.find_last_of('\\'));
        LOGI("Setting ADSP_LIBRARY_PATH to %s\n", path_to_set.c_str());
        // setenv("LD_LIBRARY_PATH", path_to_set.c_str(), 1);
        _putenv_s("ADSP_LIBRARY_PATH", path_to_set.c_str());
    }
#endif

    if (g_qnn_backend_context_ptr == nullptr) {
        try {
            g_qnn_backend_context_ptr = std::make_shared<qnn_backend_context>(path);
        } catch (const std::exception& e) {
            LOGE("Error creating QNN backend context: %s", e.what());
            g_qnn_backend_context_ptr = nullptr;
            return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
        }
    }
    {
        std::lock_guard<std::mutex> lock(g_qnn_backend_context_ptr->qnnMutex);
        g_qnn_backend_context_ptr->ref_count++;
        LOGI("[QNN] qnn_backend ref_count: %d", g_qnn_backend_context_ptr->ref_count);
    }

    return RWKV_SUCCESS;
}

int qnn_backend::load_model(std::string model_path, void * extra) {
    _load_total_chunks = 0;
    _load_done_chunks = 0;
    {
        std::lock_guard<std::mutex> lock(_load_progress_mutex);
        _load_progress_reported = 0.f;
    }
    if (!std::filesystem::exists(model_path)) {
        return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
    }

    bool is_context_cache = model_path.find(".dll") == std::string::npos && model_path.find(".so") == std::string::npos;

    bool is_rmpack = model_path.find(".rmpack") != std::string::npos;
    if (is_rmpack) {
        try {
            rmpack = new RMPackReader(model_path);
        } catch (const std::exception& e) {
            LOGE("Error loading rmpack: %s", e.what());
            return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
        }
    }

    qnnIOTensorUtils = new IOTensor(BufferAlloc::SHARED_BUFFER, &g_qnn_backend_context_ptr->qnnFunctionPointers.qnnInterface);

    if (is_context_cache || is_rmpack) {
        if (nullptr == g_qnn_backend_context_ptr->qnnFunctionPointers.qnnSystemInterface.systemContextCreate ||
            nullptr == g_qnn_backend_context_ptr->qnnFunctionPointers.qnnSystemInterface.systemContextGetBinaryInfo ||
            nullptr == g_qnn_backend_context_ptr->qnnFunctionPointers.qnnSystemInterface.systemContextFree) {
            LOGE("QNN System function pointers are not populated.");
            return RWKV_ERROR_BACKEND | RWKV_ERROR_INIT;
        }

        Qnn_ContextHandle_t first_contextHandle{nullptr};
        QnnHtpContext_CustomConfig_t customConfigSF;
        customConfigSF.option = QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS;

        std::vector<std::shared_ptr<uint8_t>> buffer;
        std::vector<uint64_t> bufferSizes;

        int n_chunks = 1;
        size_t pos = 0;
        int spill_fill_buffer_size = 0;
        if (is_rmpack) {
            n_chunks = rmpack->getConfig()["n_chunks"];
            spill_fill_buffer_size = rmpack->getConfig()["spill_fill_buffer_size"];
        } else {
            pos = model_path.find("_chunk");
            if (pos != std::string::npos) {
                n_chunks = std::stoi(model_path.substr(model_path.find("of") + 2));
                LOGI("Number of chunks: %d", n_chunks);
                if (n_chunks == 4) {
                    spill_fill_buffer_size = 320000000;
                }
            }
        }

        buffer.resize(n_chunks);
        bufferSizes.resize(n_chunks);
        qnnContextHandles.resize(n_chunks);
        _load_total_chunks = n_chunks;
        _load_done_chunks = 0;

        int returnStatus = RWKV_SUCCESS;
        std::vector<GraphInfo_t **> graphInfos(n_chunks);
        std::vector<uint32_t> graphCounts(n_chunks);

        // read model binaries
        datautil::StatusCode binaryReadingStatus{datautil::StatusCode::SUCCESS};
        for (int i = 0; i < n_chunks; i++) {
            // get file size and read file to memory / mmap file
            if (is_rmpack) {
                bufferSizes[i] = rmpack->getFileSize("model_" + std::to_string(i));
#if USE_MMAP
                try {
                    buffer[i] = std::shared_ptr<uint8_t>(
                        (uint8_t*)rmpack->mmapFile("model_" + std::to_string(i)), [this, i](uint8_t* p) {
                            if (p) {
                                rmpack->unmapFile("model_" + std::to_string(i));
                            }
                        }
                    );
                } catch (const std::exception& e) {
                    LOGE("Failed to mmap model chunk %d: %s", i, e.what());
                    return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
                }
#else
                try {
                    buffer[i] = std::shared_ptr<uint8_t>(
                        (uint8_t*)rmpack->readFileToMemory("model_" + std::to_string(i)), [this, i](uint8_t* p) {
                            if (p) {
                                rmpack->freeFileMemory("model_" + std::to_string(i));
                            }
                        }
                    );
                } catch (const std::exception& e) {
                    LOGE("Failed to read model chunk %d: %s", i, e.what());
                    return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
                }
#endif
            } else {
                if (n_chunks > 1) {
                    model_path = model_path.substr(0, pos) + "_chunk" + std::to_string(i+1) + "of" + std::to_string(n_chunks) + ".bin";
                    std::cout << "Reading chunk: " << model_path << std::endl;
                }
                std::tie(binaryReadingStatus, bufferSizes[i]) = datautil::getFileSize(model_path);
                if (0 == bufferSizes[i]) {
                    LOGE("Received path to an empty file. Nothing to deserialize.");
                    return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
                }
                std::cout << "Buffer size: " << bufferSizes[i] << std::endl;

#if USE_MMAP
                int fd = open(model_path.c_str(), O_RDONLY);
                if (fd < 0) {
                    LOGE("Failed to open file %s", model_path.c_str());
                    return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
                }

                buffer[i] = std::shared_ptr<uint8_t>(
                    (uint8_t*)mmap(NULL, bufferSizes[i], PROT_READ, MAP_SHARED, fd, 0), [bufferSizes, i](uint8_t* p) {
                        if (p) {
                            munmap(p, bufferSizes[i]);
                        }
                    }
                );

                if (buffer[i].get() == MAP_FAILED) {
                    LOGE("Failed to mmap file %s", model_path.c_str());
                    close(fd);
                    return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
                }
#else
                buffer[i] = std::shared_ptr<uint8_t>(new uint8_t[bufferSizes[i]], std::default_delete<uint8_t[]>());
                if (!buffer[i]) {
                    LOGE("Failed to allocate memory.");
                    return RWKV_ERROR_MODEL | RWKV_ERROR_ALLOC;
                }

                binaryReadingStatus = datautil::readBinaryFromFile(
                    model_path, reinterpret_cast<uint8_t*>(buffer[i].get()), bufferSizes[i]);
                if (binaryReadingStatus != datautil::StatusCode::SUCCESS) {
                    LOGE("Failed to read binary data.");
                    return RWKV_ERROR_MODEL | RWKV_ERROR_IO;
                }
#endif
            }
            // inspect binary info
            QnnSystemContext_Handle_t sysCtxHandle{nullptr};
            if (QNN_SUCCESS != g_qnn_backend_context_ptr->qnnFunctionPointers.qnnSystemInterface.systemContextCreate(&sysCtxHandle)) {
                LOGE("Could not create system handle.");
                returnStatus = RWKV_ERROR_MODEL | RWKV_ERROR_IO;
            }

            const QnnSystemContext_BinaryInfo_t* binaryInfo{nullptr};
            Qnn_ContextBinarySize_t binaryInfoSize{0};
            if (RWKV_SUCCESS == returnStatus &&
                QNN_SUCCESS != g_qnn_backend_context_ptr->qnnFunctionPointers.qnnSystemInterface.systemContextGetBinaryInfo(
                                    sysCtxHandle,
                                    static_cast<void*>(buffer[i].get()),
                                    bufferSizes[i],
                                    &binaryInfo,
                                    &binaryInfoSize)) {
                LOGE("Failed to get context binary info");
                returnStatus = RWKV_ERROR_MODEL | RWKV_ERROR_IO;
            }

            // fill GraphInfo_t based on binary info
            if (RWKV_SUCCESS == returnStatus &&
                !qnn::tools::rwkv_app::copyMetadataToGraphsInfo(binaryInfo, graphInfos[i], graphCounts[i])) {
                LOGE("Failed to copy metadata.");
                returnStatus = RWKV_ERROR_MODEL;
            }
            g_qnn_backend_context_ptr->qnnFunctionPointers.qnnSystemInterface.systemContextFree(sysCtxHandle);
            sysCtxHandle = nullptr;

            if (RWKV_SUCCESS == returnStatus &&
                nullptr == g_qnn_backend_context_ptr->qnnFunctionPointers.qnnInterface.contextCreateFromBinary) {
                LOGE("contextCreateFromBinaryFnHandle is nullptr.");
                returnStatus = RWKV_ERROR_MODEL;
            }

            // make custom configs
            QnnHtpContext_CustomConfig_t ioMemEstimation;
            ioMemEstimation.option          = QNN_HTP_CONTEXT_CONFIG_OPTION_IO_MEM_ESTIMATION;
            ioMemEstimation.ioMemEstimation = true;

            QnnContext_Config_t** cfgs{nullptr};

            int cfgs_count = 1;
            if (spill_fill_buffer_size > 0) {
                cfgs_count++;
            }

            cfgs                  = (QnnContext_Config_t**)malloc((cfgs_count + 1) * sizeof(QnnContext_Config_t*));
            cfgs[0]               = (QnnContext_Config_t*)malloc(sizeof(QnnContext_Config_t));
            cfgs[0]->option       = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
            cfgs[0]->customConfig = reinterpret_cast<QnnContext_CustomConfig_t>(&ioMemEstimation);

            if (spill_fill_buffer_size > 0) {
                QnnHtpContext_GroupRegistration_t groupInfo{nullptr};
                if (i == 0) {
                    groupInfo.firstGroupHandle = 0x0;
                } else {
                    groupInfo.firstGroupHandle = first_contextHandle;
                }
                groupInfo.maxSpillFillBuffer = spill_fill_buffer_size;
                customConfigSF.groupRegistration = groupInfo;

                cfgs[cfgs_count - 1]               = (QnnContext_Config_t*)malloc(sizeof(QnnContext_Config_t));
                cfgs[cfgs_count - 1]->option       = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
                cfgs[cfgs_count - 1]->customConfig = reinterpret_cast<QnnContext_CustomConfig_t>(&customConfigSF);
            }

            cfgs[cfgs_count] = nullptr;


            if (RWKV_SUCCESS == returnStatus &&
                g_qnn_backend_context_ptr->qnnFunctionPointers.qnnInterface.contextCreateFromBinary(
                    g_qnn_backend_context_ptr->qnnBackendHandle,
                    g_qnn_backend_context_ptr->qnnDeviceHandle,
                    (const QnnContext_Config_t**)cfgs,
                    static_cast<void*>(buffer[i].get()),
                    bufferSizes[i],
                    &qnnContextHandles[i],
                    nullptr)) {
                LOGE("Could not create context from binary.");
                returnStatus = RWKV_ERROR_MODEL;
            }

            for (int j = 0; j < cfgs_count; j++) {
                free(cfgs[j]);
                cfgs[j] = nullptr;
            }
            free(cfgs);
            cfgs = nullptr;

            if (RWKV_SUCCESS == returnStatus) {
                for (size_t graphIdx = 0; graphIdx < graphCounts[i]; graphIdx++) {
                    if (nullptr == g_qnn_backend_context_ptr->qnnFunctionPointers.qnnInterface.graphRetrieve) {
                        LOGE("graphRetrieveFnHandle is nullptr.");
                        returnStatus = RWKV_ERROR_MODEL;
                        break;
                    }
                    if (QNN_SUCCESS !=
                        g_qnn_backend_context_ptr->qnnFunctionPointers.qnnInterface.graphRetrieve(
                            qnnContextHandles[i], (*graphInfos[i])[graphIdx].graphName, &((*graphInfos[i])[graphIdx].graph))) {
                        LOGE("Unable to retrieve graph handle for graph Idx: %zu", graphIdx);
                        returnStatus = RWKV_ERROR_MODEL;
                    }
                }
            }
            if (RWKV_SUCCESS != returnStatus) {
                LOGD("Cleaning up graph Info structures.");
                for (int j = 0; j <= i; j++) {
                    freeGraphsInfo(&graphInfos[j], graphCounts[j]);
                }
                return returnStatus;
            }

            if (RWKV_SUCCESS == returnStatus && i == 0) {
                first_contextHandle = qnnContextHandles[i];
            }
            _load_done_chunks = i + 1;
        }

        buffer.clear();

        qnnPrefillGraphsCount = 0;
        qnnEmbdGraphsCount = 0;
        qnnEmbdPrefillGraphsCount = 0;
        qnnBatchDecodeGraphsCount.clear();
        qnnBatchDecodeGraphsInfo.clear();
        supported_batch_sizes.clear();
        for (int i = 0; i < n_chunks; i++) {
            for (int j = 0; j < graphCounts[i]; j++) {
                auto graphName = std::string((*graphInfos[i])[j].graphName);
                if (graphName.find("embedding_prefill") != std::string::npos) {
                    qnnEmbdPrefillGraphsCount++;
                } else if (graphName.find("embedding") != std::string::npos) {
                    qnnEmbdGraphsCount++;
                } else if (graphName.find("prefill") != std::string::npos) {
                    qnnPrefillGraphsCount++;
                } else {
                    int parsedBsz = parse_bsz_from_graph_name(graphName);
                    int effectiveBsz = parsedBsz >= 1 ? parsedBsz : 1;
                    if (i == 0) {
                        if (effectiveBsz == 2) {
                            supported_batch_sizes.push_back(2);
                        } else if (effectiveBsz > 2) {
                            supported_batch_sizes.push_back(effectiveBsz - 1);
                            supported_batch_sizes.push_back(effectiveBsz);
                        } else {
                            supported_batch_sizes.push_back(1);
                        }
                    }
                    qnnBatchDecodeGraphsCount[effectiveBsz]++;
                }
            }
        }


        GraphInfo_t *graphInfoArrPrefill = nullptr;
        GraphInfo_t *graphInfoArrEmbd = nullptr;
        GraphInfo_t *graphInfoArrEmbdPrefill = nullptr;
        std::unordered_map<int, GraphInfo_t*> graphInfoArrBatchDecode;

        if (qnnPrefillGraphsCount > 0) {
            qnnPrefillGraphsInfo = (GraphInfo_t **)calloc(qnnPrefillGraphsCount, sizeof(GraphInfo_t *));
            graphInfoArrPrefill =
                (GraphInfo_t *)calloc(qnnPrefillGraphsCount, sizeof(GraphInfo_t));
        }
        if (qnnEmbdGraphsCount > 0) {
            qnnEmbdGraphsInfo = (GraphInfo_t **)calloc(qnnEmbdGraphsCount, sizeof(GraphInfo_t *));
            graphInfoArrEmbd =
                (GraphInfo_t *)calloc(qnnEmbdGraphsCount, sizeof(GraphInfo_t));
        }
        if (qnnEmbdPrefillGraphsCount > 0) {
            qnnEmbdPrefillGraphsInfo = (GraphInfo_t **)calloc(qnnEmbdPrefillGraphsCount, sizeof(GraphInfo_t *));
            graphInfoArrEmbdPrefill =
                (GraphInfo_t *)calloc(qnnEmbdPrefillGraphsCount, sizeof(GraphInfo_t));
        }
        for (auto& [batchSize, count] : qnnBatchDecodeGraphsCount) {
            if (count > 0) {
                qnnBatchDecodeGraphsInfo[batchSize] = (GraphInfo_t **)calloc(count, sizeof(GraphInfo_t *));
                graphInfoArrBatchDecode[batchSize] = (GraphInfo_t *)calloc(count, sizeof(GraphInfo_t));
            }
        }

        bool allocationError = ((qnnPrefillGraphsCount > 0 && (nullptr == qnnPrefillGraphsInfo || nullptr == graphInfoArrPrefill)) ||
            (qnnEmbdGraphsCount > 0 && (nullptr == qnnEmbdGraphsInfo || nullptr == graphInfoArrEmbd)) ||
            (qnnEmbdPrefillGraphsCount > 0 && (nullptr == qnnEmbdPrefillGraphsInfo || nullptr == graphInfoArrEmbdPrefill)));

        for (auto& [batchSize, count] : qnnBatchDecodeGraphsCount) {
            if (count > 0 && (qnnBatchDecodeGraphsInfo[batchSize] == nullptr || graphInfoArrBatchDecode[batchSize] == nullptr)) {
                allocationError = true;
                break;
            }
        }

        if (allocationError) {
            LOGE("Failed to allocate memory for *graphInfo");
            if (nullptr != qnnPrefillGraphsInfo) {
                free(qnnPrefillGraphsInfo);
            }
            if (nullptr != qnnEmbdGraphsInfo) {
                free(qnnEmbdGraphsInfo);
            }
            if (nullptr != qnnEmbdPrefillGraphsInfo) {
                free(qnnEmbdPrefillGraphsInfo);
            }
            if (nullptr != graphInfoArrPrefill) {
                free(graphInfoArrPrefill);
            }
            if (nullptr != graphInfoArrEmbd) {
                free(graphInfoArrEmbd);
            }
            if (nullptr != graphInfoArrEmbdPrefill) {
                free(graphInfoArrEmbdPrefill);
            }

            for (auto& [batchSize, count] : qnnBatchDecodeGraphsCount) {
                if (qnnBatchDecodeGraphsInfo[batchSize] != nullptr) {
                    free(qnnBatchDecodeGraphsInfo[batchSize]);
                }
                if (graphInfoArrBatchDecode[batchSize] != nullptr) {
                    free(graphInfoArrBatchDecode[batchSize]);
                }
            }
            returnStatus = RWKV_ERROR_MODEL;
        }
        LOGI("qnnPrefillGraphsCount: %d, qnnEmbdGraphsCount: %d, qnnEmbdPrefillGraphsCount: %d", qnnPrefillGraphsCount, qnnEmbdGraphsCount, qnnEmbdPrefillGraphsCount);
        std::string debug_message = "supported_batch_sizes: ";
        std::sort(supported_batch_sizes.begin(), supported_batch_sizes.end());
        for (auto bsz : supported_batch_sizes) {
            debug_message += std::to_string(bsz) + " ";
        }
        LOGI("%s", debug_message.c_str());

        if (RWKV_SUCCESS == returnStatus) {
            int prefill_gidx = 0, embd_gidx = 0, embd_prefill_gidx = 0;
            std::unordered_map<int, int> batch_decode_gidx;
            for (int i = 0; i < n_chunks; i++) {
                for (int j = 0; j < graphCounts[i]; j++) {
                    auto graphName = std::string((*graphInfos[i])[j].graphName);
                    LOGI("Graph %d : %s", j, graphName.c_str());
                    if (graphName.find("embedding_prefill") != std::string::npos) {
                        qnnEmbdPrefillGraphsInfo[embd_prefill_gidx] = graphInfoArrEmbdPrefill + embd_prefill_gidx;
                        qnnEmbdPrefillGraphsInfo[embd_prefill_gidx]->graph = (*graphInfos[i])[j].graph;
                        qnnEmbdPrefillGraphsInfo[embd_prefill_gidx]->graphName = strdup((*graphInfos[i])[j].graphName);
                        qnnEmbdPrefillGraphsInfo[embd_prefill_gidx]->inputTensors = (*graphInfos[i])[j].inputTensors;
                        qnnEmbdPrefillGraphsInfo[embd_prefill_gidx]->numInputTensors = (*graphInfos[i])[j].numInputTensors;
                        qnnEmbdPrefillGraphsInfo[embd_prefill_gidx]->outputTensors = (*graphInfos[i])[j].outputTensors;
                        qnnEmbdPrefillGraphsInfo[embd_prefill_gidx]->numOutputTensors = (*graphInfos[i])[j].numOutputTensors;
                        embd_prefill_gidx++;
                    } else if (graphName.find("embedding") != std::string::npos) {
                        qnnEmbdGraphsInfo[embd_gidx] = graphInfoArrEmbd + embd_gidx;
                        qnnEmbdGraphsInfo[embd_gidx]->graph = (*graphInfos[i])[j].graph;
                        qnnEmbdGraphsInfo[embd_gidx]->graphName = strdup((*graphInfos[i])[j].graphName);
                        qnnEmbdGraphsInfo[embd_gidx]->inputTensors = (*graphInfos[i])[j].inputTensors;
                        qnnEmbdGraphsInfo[embd_gidx]->numInputTensors = (*graphInfos[i])[j].numInputTensors;
                        qnnEmbdGraphsInfo[embd_gidx]->outputTensors = (*graphInfos[i])[j].outputTensors;
                        qnnEmbdGraphsInfo[embd_gidx]->numOutputTensors = (*graphInfos[i])[j].numOutputTensors;
                        embd_gidx++;
                    } else if (graphName.find("prefill") != std::string::npos) {
                        qnnPrefillGraphsInfo[prefill_gidx] = graphInfoArrPrefill + prefill_gidx;
                        qnnPrefillGraphsInfo[prefill_gidx]->graph = (*graphInfos[i])[j].graph;
                        qnnPrefillGraphsInfo[prefill_gidx]->graphName = strdup((*graphInfos[i])[j].graphName);
                        qnnPrefillGraphsInfo[prefill_gidx]->inputTensors = (*graphInfos[i])[j].inputTensors;
                        qnnPrefillGraphsInfo[prefill_gidx]->numInputTensors = (*graphInfos[i])[j].numInputTensors;
                        qnnPrefillGraphsInfo[prefill_gidx]->outputTensors = (*graphInfos[i])[j].outputTensors;
                        qnnPrefillGraphsInfo[prefill_gidx]->numOutputTensors = (*graphInfos[i])[j].numOutputTensors;
                        prefill_gidx++;
                    } else {
                        int parsedBsz = parse_bsz_from_graph_name(graphName);
                        if (batch_decode_gidx.find(parsedBsz) == batch_decode_gidx.end()) {
                            batch_decode_gidx[parsedBsz] = 0;
                        }

                        int current_idx = batch_decode_gidx[parsedBsz];
                        qnnBatchDecodeGraphsInfo[parsedBsz][current_idx] = graphInfoArrBatchDecode[parsedBsz] + current_idx;
                        qnnBatchDecodeGraphsInfo[parsedBsz][current_idx]->graph = (*graphInfos[i])[j].graph;
                        qnnBatchDecodeGraphsInfo[parsedBsz][current_idx]->graphName = strdup((*graphInfos[i])[j].graphName);
                        qnnBatchDecodeGraphsInfo[parsedBsz][current_idx]->inputTensors = (*graphInfos[i])[j].inputTensors;
                        qnnBatchDecodeGraphsInfo[parsedBsz][current_idx]->numInputTensors = (*graphInfos[i])[j].numInputTensors;
                        qnnBatchDecodeGraphsInfo[parsedBsz][current_idx]->outputTensors = (*graphInfos[i])[j].outputTensors;
                        qnnBatchDecodeGraphsInfo[parsedBsz][current_idx]->numOutputTensors = (*graphInfos[i])[j].numOutputTensors;
                        batch_decode_gidx[parsedBsz]++;
                    }
                }
            }
        }
    } else {
        // create context
        qnnContextHandles.resize(1);
        if (QNN_CONTEXT_NO_ERROR != g_qnn_backend_context_ptr->qnnFunctionPointers.qnnInterface.contextCreate(
                    g_qnn_backend_context_ptr->qnnBackendHandle,
                    g_qnn_backend_context_ptr->qnnDeviceHandle,
                    nullptr, // const QnnContext_Config_t**
                    &qnnContextHandles[0])) {
            LOGE("Could not create context");
            return RWKV_ERROR_BACKEND;
        }

        // conpose graphs
        if (graphConfigsInfo == nullptr) {
            graphConfigsInfoCount = 2;

            graphConfigsInfo = new GraphConfigInfo_t*[graphConfigsInfoCount];
            graphConfigsInfo[0] = new GraphConfigInfo_t();
            graphConfigsInfo[0]->graphName = (char*)"model";
            graphConfigsInfo[0]->graphConfigs = (const QnnGraph_Config_t**)new QnnGraph_Config_t*[2];
            graphConfigsInfo[1] = new GraphConfigInfo_t();
            graphConfigsInfo[1]->graphName = (char*)"model_fp16";
            graphConfigsInfo[1]->graphConfigs = (const QnnGraph_Config_t**)new QnnGraph_Config_t*[2];

            static QnnHtpGraph_CustomConfig_t customConfig;
            customConfig.option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
            customConfig.precision = QNN_PRECISION_FLOAT16;
            static QnnGraph_Config_t graphConfig;
            graphConfig.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
            graphConfig.customConfig = &customConfig;
            for (int i = 0; i < graphConfigsInfoCount; i++) {
                graphConfigsInfo[i]->graphConfigs[0] = &graphConfig;
                graphConfigsInfo[i]->graphConfigs[1] = nullptr;
            }
        }

        GraphInfo_t **tmpGraphsInfo = nullptr;
        uint32_t tmpGraphsCount = 0;
        if (ModelError_t::MODEL_NO_ERROR !=
            g_qnn_backend_context_ptr->qnnFunctionPointers.composeGraphsFnHandle(
                g_qnn_backend_context_ptr->qnnBackendHandle,
                g_qnn_backend_context_ptr->qnnFunctionPointers.qnnInterface,
                qnnContextHandles[0],
                (const GraphConfigInfo_t**)graphConfigsInfo,
                graphConfigsInfoCount,
                &tmpGraphsInfo,
                &tmpGraphsCount,
                false,
                logCallback,
                DEFAULT_QNN_LOGLEVEL)) {
          LOGE("Failed in composeGraphs()");
          return RWKV_ERROR_MODEL;
        }

        // finalize graphs
        for (size_t graphIdx = 0; graphIdx < tmpGraphsCount; graphIdx++) {
            if (QNN_GRAPH_NO_ERROR !=
                g_qnn_backend_context_ptr->qnnFunctionPointers.qnnInterface.graphFinalize(
                    (*tmpGraphsInfo)[graphIdx].graph, nullptr, nullptr)) {
                return RWKV_ERROR_MODEL;
            }
        }

        qnnBatchDecodeGraphsCount[1] = tmpGraphsCount;
        qnnBatchDecodeGraphsInfo[1] = tmpGraphsInfo;

        // save context cache
// #if WIN32
//         qnn_save_context_cache(model_path.substr(0, model_path.find('.dll')) + "_cache.bin");
// #else
//         qnn_save_context_cache(model_path.substr(0, model_path.find('.so')) + "_cache.bin");
// #endif

    }

    if (rmpack != nullptr) {
        int use_external_deep_embedding = rmpack->getConfig()["use_external_deep_embedding"];
        has_deep_embedding = use_external_deep_embedding != 0;
    }

    if (RWKV_SUCCESS != qnn_initialize_tensors()) {
        LOGE("Could not initialize tensors");
        return RWKV_ERROR_MODEL;
    }

    std::vector<size_t> dims_state;

    n_layers = stateTensorsNameToTensorPointer.size() / 3;

    getTensorDims(dims_state, QNN_TENSOR_GET_DIMENSIONS((Qnn_Tensor_t*)stateTensorsNameToTensorPointer["state1_out"]), QNN_TENSOR_GET_RANK((Qnn_Tensor_t*)stateTensorsNameToTensorPointer["state1_out"]));
    for (int i = 0; i < dims_state.size(); i++) {
        LOGI("dims_state[%d]: %zu", i, dims_state[i]);
    }
    if (dims_state.size() == 3) {
        num_heads = dims_state[0];
        hidden_size = num_heads * dims_state[1];
    } else {
        num_heads = dims_state[1];
        hidden_size = num_heads * dims_state[2];
    }
    

    // if (rmpack != nullptr) {
    //     vocab_size = rmpack->getConfig()["vocab_size"];
    // } else {
    {
        std::vector<size_t> dims;
        getTensorDims(dims, QNN_TENSOR_GET_DIMENSIONS(logitsOutputTensor), QNN_TENSOR_GET_RANK(logitsOutputTensor));
        for (int i = 0; i < dims.size(); i++) {
            LOGI("logits dims[%d]: %zu", i, dims[i]);
        }
        vocab_size = dims[2];
    }

    if (rmpack != nullptr) {
        int use_external_embedding = rmpack->getConfig()["use_external_embedding"];
        LOGI("use_external_embedding: %d", use_external_embedding);
        if (use_external_embedding) {
            // let's assert that the embedding dtype is the same as the input tensor dtype
            try {
#if USE_MMAP
                external_embeddings = std::shared_ptr<uint8_t>(
                    (uint8_t*)rmpack->mmapFile("embedding"),
                    [this](uint8_t* p) {
                        rmpack->unmapFile("embedding");
                    }
                );
#else
                external_embeddings = std::shared_ptr<uint8_t>(
                    (uint8_t*)rmpack->readFileToMemory("embedding"),
                    [this](uint8_t* p) {
                        rmpack->freeFileMemory("embedding");
                    }
                );
#endif
            } catch (const std::exception& e) {
                LOGE("Failed to load external embedding: %s", e.what());
                return RWKV_ERROR_MODEL;
            }
            if (external_embeddings != nullptr) {
                LOGI("External embeddings loaded");
            } else {
                LOGE("Failed to load external embedding");
                return RWKV_ERROR_MODEL;
            }
        }

        if (has_deep_embedding) {
            // let's assert that the embedding dtype is the same as the input tensor dtype
            deep_embedding_size = rmpack->getConfig()["deep_embedding_size"];
            std::string deep_embedding_dtype = rmpack->getConfig()["external_deep_embedding_dtype"];
            if (deep_embedding_dtype == "fp16" || deep_embedding_dtype == "uint16") {
                deep_embeddings_elembytes = 2;
            } else if (deep_embedding_dtype == "fp32" || deep_embedding_dtype == "uint32") {
                deep_embeddings_elembytes = 4;
            } else if (deep_embedding_dtype == "uint8") {
                deep_embeddings_elembytes = 1;
            } else {
                LOGE("Unsupported deep embedding dtype: %s", deep_embedding_dtype.c_str());
                return RWKV_ERROR_MODEL;
            }
            try {
#if USE_MMAP
                external_deep_embeddings = std::shared_ptr<uint8_t>(
                    (uint8_t*)rmpack->mmapFile("deep_embedding"),
                    [this](uint8_t* p) {
                        rmpack->unmapFile("deep_embedding");
                    }
                );
#else
                external_deep_embeddings = std::shared_ptr<uint8_t>(
                    (uint8_t*)rmpack->readFileToMemory("deep_embedding"),
                    [this](uint8_t* p) {
                        rmpack->freeFileMemory("deep_embedding");
                    }
                );
#endif
            } catch (const std::exception& e) {
                LOGE("Failed to load external deep embedding: %s", e.what());
                return RWKV_ERROR_MODEL;
            }
            if (external_deep_embeddings != nullptr) {
                LOGI("External deep embeddings loaded");
            } else {
                LOGE("Failed to load external deep embedding");
                return RWKV_ERROR_MODEL;
            }
        }

        int use_external_lmhead = rmpack->getConfig()["use_external_lmhead"];
        if (use_external_lmhead) {
            external_lmhead_filetype = rmpack->getConfig()["external_lmhead_filetype"];
#ifdef ENABLE_MNN
            if (external_lmhead_filetype == "mnn") {
                try {
#if USE_MMAP
                    void* buffer = rmpack->mmapFile("lmhead");
#else
                    void* buffer = rmpack->readFileToMemory("lmhead");
#endif // USE_MMAP
                    external_lmhead_interpretor = MNN::Interpreter::createFromBuffer(buffer, rmpack->getFileInfo("lmhead")->size);
                    MNN::ScheduleConfig conf;
                    conf.type = MNN_FORWARD_CPU;
#ifdef __ANDROID__
                    auto cpu_groups = get_cpu_groups();
                    // use second group
                    conf.numThread = cpu_groups[1].ids.size();
#else
                    conf.numThread = 4;
#endif // __ANDROID__
                    MNN::BackendConfig backendConfig;
                    backendConfig.memory = MNN::BackendConfig::Memory_Low;
                    backendConfig.power = MNN::BackendConfig::Power_High;
                    backendConfig.precision = MNN::BackendConfig::Precision_Low;
                    conf.backendConfig = &backendConfig;
                    external_lmhead_mnn_session = external_lmhead_interpretor->createSession(conf);
#if USE_MMAP
                    rmpack->unmapFile("lmhead");
#else
                    rmpack->freeFileMemory("lmhead");
#endif
                } catch (const std::exception& e) {
                    LOGE("Failed to load external lmhead: %s", e.what());
                    return RWKV_ERROR_MODEL;
                }
            } else {
#else
            {
#endif // ENABLE_MNN                
                LOGE("Unsupported external lmhead filetype: %s", external_lmhead_filetype.c_str());
                return RWKV_ERROR_MODEL;
            }
        }
    }
    return RWKV_SUCCESS;
}

float qnn_backend::get_load_progress() const {
    int total = _load_total_chunks.load();
    if (total <= 0) return -1.f;
    int done = _load_done_chunks.load();
    float real = static_cast<float>(done) / static_cast<float>(total);
    float ceiling = (done + 1 <= total) ? static_cast<float>(done + 1) / static_cast<float>(total) : 1.f;
    const float step = 0.02f;
    std::lock_guard<std::mutex> lock(_load_progress_mutex);
    if (_load_progress_reported < real)
        _load_progress_reported = real;
    _load_progress_reported = std::min(ceiling, _load_progress_reported + step);
    return std::max(0.f, std::min(1.f, _load_progress_reported));
}

void qnn_backend::fill_quantized_tensor(float value, Qnn_Tensor_t *tensor) {
    std::vector<size_t> dims;
    for (int j = 0; j < QNN_TENSOR_GET_RANK(*tensor); j++) {
        dims.push_back(*(QNN_TENSOR_GET_DIMENSIONS(*tensor) + j));
    }
    void *buffer = qnnIOTensorUtils->getBuffer(tensor);
    float fpzero = 0.0;
    auto dtype = QNN_TENSOR_GET_DATA_TYPE(*tensor);
    if (dtype == QNN_DATATYPE_UFIXED_POINT_8) {
        uint8_t qtzero = 0;
        datautil::floatToTfN<uint8_t>(&qtzero, &fpzero,
            QNN_TENSOR_GET_QUANT_PARAMS(*tensor).scaleOffsetEncoding.offset,
            QNN_TENSOR_GET_QUANT_PARAMS(*tensor).scaleOffsetEncoding.scale,
            1);
        for (int j = 0; j < datautil::calculateElementCount(dims); j++) {
            ((uint8_t*)buffer)[j] = qtzero;
        }
    } else if (dtype == QNN_DATATYPE_UFIXED_POINT_16) {
        uint16_t qtzero = 0;
        datautil::floatToTfN<uint16_t>(&qtzero, &fpzero,
            QNN_TENSOR_GET_QUANT_PARAMS(*tensor).scaleOffsetEncoding.offset,
            QNN_TENSOR_GET_QUANT_PARAMS(*tensor).scaleOffsetEncoding.scale,
            1);
        for (int j = 0; j < datautil::calculateElementCount(dims); j++) {
            ((uint16_t*)buffer)[j] = qtzero;
        }
    }
}

int qnn_backend::populate_tensor_name_to_size_map(const GraphInfo_t& graphInfo, 
                                                  std::unordered_map<std::string, size_t>& tensorNameToSize,
                                                  bool isInputTensors) {
    const Qnn_Tensor_t* tensors = isInputTensors ? graphInfo.inputTensors : graphInfo.outputTensors;
    const size_t numTensors = isInputTensors ? graphInfo.numInputTensors : graphInfo.numOutputTensors;

    for (size_t i = 0; i < numTensors; i++) {
        size_t tensorDataSize = 1;
        // Calculate tensor data size by multiplying all dimensions
        for (int j = 0; j < QNN_TENSOR_GET_RANK(tensors[i]); j++) {
            tensorDataSize *= *(QNN_TENSOR_GET_DIMENSIONS(tensors[i]) + j);
        }
        
        auto tensorName = std::string(QNN_TENSOR_GET_NAME(tensors[i]));
        size_t typeSize = getQnnDatatypeSize(QNN_TENSOR_GET_DATA_TYPE(tensors[i]));
        if (typeSize == 0) {
            return RWKV_ERROR_IO;
        }
        
        tensorDataSize *= typeSize;
        tensorNameToSize[tensorName] = tensorDataSize;
    }
    
    return RWKV_SUCCESS;
}

int qnn_backend::setup_output_tensors_for_graph(int graph_id, int total_graphs_count, const GraphInfo_t& graphInfo,
                                               Qnn_Tensor_t** outputTensors,
                                               std::unordered_map<std::string, void*>& tensorNameToTensorPointer,
                                               std::unordered_map<std::string, size_t>& tensorNameToSize,
                                               Qnn_ContextHandle_t contextHandle,
                                               Qnn_Tensor_t* vFirstTensorRef, Qnn_Tensor_t* hiddenStateTensorRef,
                                               bool isPrefill) {
    std::unordered_map<std::string, Qnn_Tensor_t*> sharedTensorMap;
    Qnn_Tensor_t* vFirstTensorToUse = isPrefill ? vFirstTensorPrefill : vFirstTensor;
    Qnn_Tensor_t* hiddenStateTensorToUse = isPrefill ? hiddenStateTensorPrefill : hiddenStateTensor;

    if ((logitsOutputTensor != nullptr && graph_id == total_graphs_count - 1) || (hiddenStateTensorToUse != nullptr && graph_id != total_graphs_count - 1)) {
        // tensors initialized previously; set up with shared tensors
        for (size_t i = 0; i < graphInfo.numOutputTensors; i++) {
            auto tensorName = std::string(QNN_TENSOR_GET_NAME(graphInfo.outputTensors[i]));
            if (tensorName.find("v_first") != std::string::npos) {
                if (vFirstTensorToUse != nullptr) {
                    sharedTensorMap[tensorName] = vFirstTensorToUse;
                }
            } else if (tensorName.find("state") != std::string::npos) {
                if (stateTensorsNameToTensorPointer.find(tensorName) != stateTensorsNameToTensorPointer.end()) {
                    sharedTensorMap[tensorName] = (Qnn_Tensor_t*)stateTensorsNameToTensorPointer[tensorName];
                }
            } else if (tensorName.find("out") != std::string::npos) {
                if (graph_id == total_graphs_count - 1) {
                    sharedTensorMap[tensorName] = logitsOutputTensor;
                } else if (hiddenStateTensorToUse != nullptr) {
                    sharedTensorMap[tensorName] = hiddenStateTensorToUse;
                }
            }
        }

        if (!qnnIOTensorUtils->setupOutputWithSharedTensors(&outputTensors[graph_id], tensorNameToTensorPointer, graphInfo,
                                            tensorNameToSize, contextHandle, sharedTensorMap)) {
            LOGE("Error in setting up shared Output Tensors for graph %d", graph_id);
            return RWKV_ERROR_IO;
        }

        for (size_t i = 0; i < graphInfo.numOutputTensors; i++) {
            auto tensorName = std::string(QNN_TENSOR_GET_NAME(graphInfo.outputTensors[i]));
            if (tensorName.find("state") != std::string::npos && stateTensorsNameToTensorPointer.find(tensorName) == stateTensorsNameToTensorPointer.end()) {
                stateTensorsNameToTensorPointer[tensorName] = (Qnn_Tensor_t*)tensorNameToTensorPointer[tensorName];
            }
        }
    } else {
        // allocate output tensors
        if (!qnnIOTensorUtils->setupOutputTensors(&outputTensors[graph_id], tensorNameToTensorPointer, graphInfo,
                                            tensorNameToSize, contextHandle, false)) {
            LOGE("Error in setting up Output Tensors for graph %d", graph_id);
            return RWKV_ERROR_IO;
        }

        for (size_t i = 0; i < graphInfo.numOutputTensors; i++) {
            auto tensorName = std::string(QNN_TENSOR_GET_NAME(graphInfo.outputTensors[i]));
            if (tensorName.find("v_first") != std::string::npos) {
                if (isPrefill && vFirstTensorPrefill == nullptr) {
                    vFirstTensorPrefill = (Qnn_Tensor_t*)tensorNameToTensorPointer[tensorName];
                } else if (!isPrefill && vFirstTensor == nullptr) {
                    vFirstTensor = (Qnn_Tensor_t*)tensorNameToTensorPointer[tensorName];
                }
            } else if (tensorName.find("state") != std::string::npos) {
                stateTensorsNameToTensorPointer[tensorName] = (Qnn_Tensor_t*)tensorNameToTensorPointer[tensorName];
            } else if (tensorName.find("out") != std::string::npos) {
                if (graph_id == total_graphs_count - 1) {
                    logitsOutputTensor = (Qnn_Tensor_t*)tensorNameToTensorPointer[tensorName];
                } else {
                    if (isPrefill && hiddenStateTensorPrefill == nullptr) {
                        hiddenStateTensorPrefill = (Qnn_Tensor_t*)tensorNameToTensorPointer[tensorName];
                    } else if (!isPrefill && hiddenStateTensor == nullptr) {
                        hiddenStateTensor = (Qnn_Tensor_t*)tensorNameToTensorPointer[tensorName];
                    }
                }
            }
        }
    }

    return RWKV_SUCCESS;
}

void qnn_backend::populate_input_shared_tensor_map(const GraphInfo_t& graphInfo, int graph_id,
                                                   std::unordered_map<std::string, Qnn_Tensor_t*>& sharedTensorMap,
                                                   Qnn_Tensor_t* vFirstTensorRef, Qnn_Tensor_t* hiddenStateTensorRef,
                                                   bool isPrefill) {
    for (size_t i = 0; i < graphInfo.numInputTensors; i++) {
        auto tensorName = std::string(QNN_TENSOR_GET_NAME(graphInfo.inputTensors[i]));

        if (tensorName.find("v_first") != std::string::npos) {
            if (vFirstTensorRef != nullptr) {
                sharedTensorMap[tensorName] = vFirstTensorRef;
            }
        } else if (tensorName.find("state") != std::string::npos) {
            std::string outputTensorName = tensorName.substr(0, tensorName.find("_in")) + "_out";
            if (stateTensorsNameToTensorPointer.find(outputTensorName) != stateTensorsNameToTensorPointer.end()) {
                sharedTensorMap[tensorName] = (Qnn_Tensor_t*)stateTensorsNameToTensorPointer[outputTensorName];
            }
        } else if (tensorName.find("in") != std::string::npos && graph_id != 0) {
            if (hiddenStateTensorRef != nullptr) {
                sharedTensorMap[tensorName] = hiddenStateTensorRef;
            }
        }
    }
}

void qnn_backend::map_deep_embedding_tensors(const GraphInfo_t& graphInfo, int graph_id,
                                             std::unordered_map<std::string, void*>& tensorNameToTensorPointer,
                                             std::unordered_map<int, Qnn_Tensor_t*>& deepEmbeddingTensorsRef,
                                             bool isPrefill) {
    if (!has_deep_embedding) {
        return;
    }

    for (size_t i = 0; i < graphInfo.numInputTensors; i++) {
        auto tensorName = std::string(QNN_TENSOR_GET_NAME(graphInfo.inputTensors[i]));
        if (tensorName.find("s_emb") != std::string::npos) {
            int layer_id = std::stoi(tensorName.substr(5, tensorName.find("_in") - 5));
            std::string logMessage = "Found Deep embedding tensor " + std::string("s_emb") + std::to_string(layer_id) + "_in";
            if (isPrefill) {
                logMessage += "_prefill";
            }
            LOGD("%s", logMessage.c_str());
            deepEmbeddingTensorsRef[layer_id] = (Qnn_Tensor_t*)tensorNameToTensorPointer[tensorName];
        }
    }
}

int qnn_backend::qnn_initialize_tensors() {
    if (!isTensorInitialized) {
        qnnIOTensorUtils->initialize(qnnContextHandles[0]);
        if (qnnPrefillGraphsCount > 0) {
            prefillGraphsTensorNameToTensorPointer.resize(qnnPrefillGraphsCount);
            prefillGraphsTensorNameToSize.resize(qnnPrefillGraphsCount);
        }
        if (qnnEmbdGraphsCount > 0) {
            embdGraphsTensorNameToTensorPointer.resize(qnnEmbdGraphsCount);
            embdGraphsTensorNameToSize.resize(qnnEmbdGraphsCount);
        }
        if (qnnEmbdPrefillGraphsCount > 0) {
            embdPrefillGraphsTensorNameToTensorPointer.resize(qnnEmbdPrefillGraphsCount);
            embdPrefillGraphsTensorNameToSize.resize(qnnEmbdPrefillGraphsCount);
        }
        for (auto& [batchSize, count] : qnnBatchDecodeGraphsCount) {
            if (count > 0) {
                batchDecodeGraphsTensorNameToTensorPointer[batchSize].resize(count);
                batchDecodeGraphsTensorNameToSize[batchSize].resize(count);
                // Allocate tensor arrays
                inputTensorsBatchDecode[batchSize] = new Qnn_Tensor_t*[count];
                outputTensorsBatchDecode[batchSize] = new Qnn_Tensor_t*[count];
            }
        }

        // Collect batch sizes and sort from large to small before initializing
        std::vector<int> batchSizes;
        for (const auto& [batchSize, count] : qnnBatchDecodeGraphsCount) {
            batchSizes.push_back(batchSize);
        }
        std::sort(batchSizes.begin(), batchSizes.end(), std::greater<int>());

        for (int batchSize : batchSizes) {
            int count = qnnBatchDecodeGraphsCount[batchSize];
            if (count > 0) {
                std::string inTensorName = (batchSize == 1) ? std::string("in") : (std::string("in_bsz") + std::to_string(batchSize));
                int initStatus = initialize_batch_decode_graphs(
                    count,
                    qnnBatchDecodeGraphsInfo[batchSize],
                    batchDecodeGraphsTensorNameToTensorPointer[batchSize],
                    batchDecodeGraphsTensorNameToSize[batchSize],
                    inputTensorsBatchDecode[batchSize],
                    outputTensorsBatchDecode[batchSize],
                    inTensorName.c_str(),
                    batchSize);
                if (initStatus != RWKV_SUCCESS) return initStatus;
            }
        }


        if (qnnPrefillGraphsCount > 0) {
            for (int graph_id = 0; graph_id < qnnPrefillGraphsCount; graph_id++) {
                std::unordered_map<std::string, Qnn_Tensor_t*> sharedTensorMap;
                auto graphInfo     = (*qnnPrefillGraphsInfo)[graph_id];
                LOGI("Graph %d : %s", graph_id, graphInfo.graphName);

                // Populate output tensor name to size map for prefill graphs
                auto result = populate_tensor_name_to_size_map(graphInfo, prefillGraphsTensorNameToSize[graph_id], false);
                if (result != RWKV_SUCCESS) {
                    return result;
                }

                // Setup output tensors using helper function (special handling for prefill)
                if (logitsOutputTensor != nullptr) {
                    // For prefill graphs, we need special handling of shared tensors
                    for (size_t i = 0; i < graphInfo.numOutputTensors; i++) {
                        auto tensorName = std::string(QNN_TENSOR_GET_NAME(graphInfo.outputTensors[i]));

                        if (tensorName.find("v_first") != std::string::npos && vFirstTensorPrefill != nullptr) {
                            sharedTensorMap[tensorName] = vFirstTensorPrefill;
                        } else if (tensorName.find("state") != std::string::npos) {
                            sharedTensorMap[tensorName] = (Qnn_Tensor_t*)stateTensorsNameToTensorPointer[tensorName];
                        } else if (tensorName.find("out") != std::string::npos) {
                            if (graph_id == qnnPrefillGraphsCount - 1) {
                                sharedTensorMap[tensorName] = logitsOutputTensor;
                            } else if (hiddenStateTensorPrefill != nullptr) {
                                sharedTensorMap[tensorName] = hiddenStateTensorPrefill;
                            }
                        }
                    }

                    if (!qnnIOTensorUtils->setupOutputWithSharedTensors(&outputTensorsPrefill[graph_id], prefillGraphsTensorNameToTensorPointer[graph_id], graphInfo,
                            prefillGraphsTensorNameToSize[graph_id], qnnContextHandles[graph_id], sharedTensorMap)) {
                        LOGE("Error in setting up Output Tensors");
                        return RWKV_ERROR_IO;
                    }
                }

                // Handle tensor assignments for prefill graphs
                for (size_t i = 0; i < graphInfo.numOutputTensors; i++) {
                    auto tensorName = std::string(QNN_TENSOR_GET_NAME(graphInfo.outputTensors[i]));
                    if (tensorName.find("v_first") != std::string::npos && vFirstTensorPrefill == nullptr) {
                        vFirstTensorPrefill = (Qnn_Tensor_t*)prefillGraphsTensorNameToTensorPointer[graph_id][tensorName];
                    } else if (tensorName.find("state") == std::string::npos && tensorName.find("out") != std::string::npos) {
                        if (graph_id != qnnPrefillGraphsCount - 1 && hiddenStateTensorPrefill == nullptr) {
                            hiddenStateTensorPrefill = (Qnn_Tensor_t*)prefillGraphsTensorNameToTensorPointer[graph_id][tensorName];
                        }
                    }
                }

                // Populate input tensor name to size map for prefill graphs
                result = populate_tensor_name_to_size_map(graphInfo, prefillGraphsTensorNameToSize[graph_id], true);
                if (result != RWKV_SUCCESS) {
                    return result;
                }

                // Populate input shared tensor map using helper function
                populate_input_shared_tensor_map(graphInfo, graph_id, sharedTensorMap, vFirstTensorPrefill, hiddenStateTensorPrefill, true);

                if (!qnnIOTensorUtils->setupInputWithSharedTensors(&inputTensorsPrefill[graph_id], prefillGraphsTensorNameToTensorPointer[graph_id], graphInfo,
                                                prefillGraphsTensorNameToSize[graph_id], qnnContextHandles[graph_id], sharedTensorMap)) {
                    LOGE("Error in setting up Input Tensors");
                    return RWKV_ERROR_IO;
                }

                // Map deep embedding tensors using helper function
                map_deep_embedding_tensors(graphInfo, graph_id, prefillGraphsTensorNameToTensorPointer[graph_id], 
                                         deepEmbeddingPrefillTensors, true);
            }

            if (prefillGraphsTensorNameToTensorPointer[0].find("in_prefill") != prefillGraphsTensorNameToTensorPointer[0].end()) {
                tokenInputTensorPrefill = (Qnn_Tensor_t*)prefillGraphsTensorNameToTensorPointer[0]["in_prefill"];
            } else if (prefillGraphsTensorNameToTensorPointer[0].find("in_prefill_chunk1") != prefillGraphsTensorNameToTensorPointer[0].end()) {
                tokenInputTensorPrefill = (Qnn_Tensor_t*)prefillGraphsTensorNameToTensorPointer[0]["in_prefill_chunk1"];
            }
            if (tokenInputTensorPrefill == nullptr) {
                LOGE("Failed to get tokenInputTensorPrefill");
                return RWKV_ERROR_IO;
            }

            // prefillSequenceLength *= *(QNN_TENSOR_GET_DIMENSIONS(tokenInputTensorPrefill) + (QNN_TENSOR_GET_RANK(tokenInputTensorPrefill) - 1));
            std::vector<size_t> dims;
            getTensorDims(dims, QNN_TENSOR_GET_DIMENSIONS(tokenInputTensorPrefill), QNN_TENSOR_GET_RANK(tokenInputTensorPrefill));
            prefillSequenceLength = dims[dims.size() - 1];
            LOGI("Prefill sequence length: %d", prefillSequenceLength);
        }

        if (qnnEmbdGraphsCount > 0) {
            for (int graph_id = 0; graph_id < qnnEmbdGraphsCount; graph_id++) {
                std::unordered_map<std::string, Qnn_Tensor_t*> sharedTensorMap;
                auto graphInfo     = (*qnnEmbdGraphsInfo)[graph_id];
                LOGI("Graph %d : %s", graph_id, graphInfo.graphName);
                
                // Populate output tensor name to size map for embedding graphs  
                auto result = populate_tensor_name_to_size_map(graphInfo, embdGraphsTensorNameToSize[graph_id], false);
                if (result != RWKV_SUCCESS) {
                    return result;
                }

                // Setup output tensors using helper function
                result = setup_output_tensors_for_graph(graph_id, qnnEmbdGraphsCount, graphInfo,
                                                       outputTensorsEmbd, embdGraphsTensorNameToTensorPointer[graph_id],
                                                       embdGraphsTensorNameToSize[graph_id], qnnContextHandles[graph_id],
                                                       vFirstTensor, hiddenStateTensor, false);
                if (result != RWKV_SUCCESS) {
                    return result;
                }

                // Populate input tensor name to size map for embedding graphs
                result = populate_tensor_name_to_size_map(graphInfo, embdGraphsTensorNameToSize[graph_id], true);
                if (result != RWKV_SUCCESS) {
                    return result;
                }

                // Populate input shared tensor map using helper function
                populate_input_shared_tensor_map(graphInfo, graph_id, sharedTensorMap, vFirstTensor, hiddenStateTensor, false);

                if (!qnnIOTensorUtils->setupInputWithSharedTensors(&inputTensorsEmbd[graph_id], embdGraphsTensorNameToTensorPointer[graph_id], graphInfo,
                                                embdGraphsTensorNameToSize[graph_id], qnnContextHandles[graph_id], sharedTensorMap)) {
                    LOGE("Error in setting up Input Tensors");
                    return RWKV_ERROR_IO;
                }

                // Map deep embedding tensors using helper function
                map_deep_embedding_tensors(graphInfo, graph_id, embdGraphsTensorNameToTensorPointer[graph_id], 
                                         deepEmbeddingTensors, false);
            }

            if (embdGraphsTensorNameToTensorPointer[0].find("in_embedding") != embdGraphsTensorNameToTensorPointer[0].end()) {
                tokenInputTensorEmbd = (Qnn_Tensor_t*)embdGraphsTensorNameToTensorPointer[0]["in_embedding"];
            } else if (embdGraphsTensorNameToTensorPointer[0].find("in_embedding_chunk1") != embdGraphsTensorNameToTensorPointer[0].end()) {
                tokenInputTensorEmbd = (Qnn_Tensor_t*)embdGraphsTensorNameToTensorPointer[0]["in_embedding_chunk1"];
            }
        }

        if (qnnEmbdPrefillGraphsCount > 0) {
            for (int graph_id = 0; graph_id < qnnEmbdPrefillGraphsCount; graph_id++) {
                std::unordered_map<std::string, Qnn_Tensor_t*> sharedTensorMap;
                auto graphInfo     = (*qnnEmbdPrefillGraphsInfo)[graph_id];
                LOGI("Graph %d : %s", graph_id, graphInfo.graphName);

                // Populate output tensor name to size map for embedding prefill graphs
                auto result = populate_tensor_name_to_size_map(graphInfo, embdPrefillGraphsTensorNameToSize[graph_id], false);
                if (result != RWKV_SUCCESS) {
                    return result;
                }

                // Special handling for embedding prefill graphs - log tensor info
                for (size_t i = 0; i < graphInfo.numOutputTensors; i++) {
                    auto tensorName = std::string(QNN_TENSOR_GET_NAME(graphInfo.outputTensors[i]));

                    if (tensorName.find("v_first") != std::string::npos && vFirstTensorPrefill != nullptr) {
                        sharedTensorMap[tensorName] = vFirstTensorPrefill;
                    } else if (tensorName.find("state") != std::string::npos) {
                        sharedTensorMap[tensorName] = (Qnn_Tensor_t*)stateTensorsNameToTensorPointer[tensorName];
                    } else if (tensorName.find("out") != std::string::npos) {
                        if (graph_id == qnnEmbdPrefillGraphsCount - 1) {
                            sharedTensorMap[tensorName] = logitsOutputTensor;
                        } else if (hiddenStateTensorPrefill != nullptr) {
                            sharedTensorMap[tensorName] = hiddenStateTensorPrefill;
                        }
                    }
                }

                if (!qnnIOTensorUtils->setupOutputWithSharedTensors(&outputTensorsEmbdPrefill[graph_id], embdPrefillGraphsTensorNameToTensorPointer[graph_id], graphInfo,
                        embdPrefillGraphsTensorNameToSize[graph_id], qnnContextHandles[graph_id], sharedTensorMap)) {
                    LOGE("Error in setting up Output Tensors");
                    return RWKV_ERROR_IO;
                }

                // Handle tensor assignments for embedding prefill graphs
                for (size_t i = 0; i < graphInfo.numOutputTensors; i++) {
                    auto tensorName = std::string(QNN_TENSOR_GET_NAME(graphInfo.outputTensors[i]));
                    if (tensorName.find("v_first") != std::string::npos && vFirstTensorPrefill == nullptr) {
                        vFirstTensorPrefill = (Qnn_Tensor_t*)embdPrefillGraphsTensorNameToTensorPointer[graph_id][tensorName];
                    } else if (tensorName.find("state") == std::string::npos && tensorName.find("out") != std::string::npos) {
                        if (graph_id != qnnEmbdPrefillGraphsCount - 1 && hiddenStateTensorPrefill == nullptr) {
                            hiddenStateTensorPrefill = (Qnn_Tensor_t*)embdPrefillGraphsTensorNameToTensorPointer[graph_id][tensorName];
                        }
                    }
                }

                // Populate input tensor name to size map for embedding prefill graphs
                result = populate_tensor_name_to_size_map(graphInfo, embdPrefillGraphsTensorNameToSize[graph_id], true);
                if (result != RWKV_SUCCESS) {
                    return result;
                }

                // Populate input shared tensor map using helper function
                populate_input_shared_tensor_map(graphInfo, graph_id, sharedTensorMap, vFirstTensorPrefill, hiddenStateTensorPrefill, true);

                if (!qnnIOTensorUtils->setupInputWithSharedTensors(&inputTensorsEmbdPrefill[graph_id], embdPrefillGraphsTensorNameToTensorPointer[graph_id], graphInfo,
                                                embdPrefillGraphsTensorNameToSize[graph_id], qnnContextHandles[graph_id], sharedTensorMap)) {
                    LOGE("Error in setting up Input Tensors");
                    return RWKV_ERROR_IO;
                }

                // Map deep embedding tensors using helper function
                map_deep_embedding_tensors(graphInfo, graph_id, embdPrefillGraphsTensorNameToTensorPointer[graph_id], 
                                         deepEmbeddingPrefillTensors, true);
            }

            if (embdPrefillGraphsTensorNameToTensorPointer[0].find("in_embedding_prefill") != embdPrefillGraphsTensorNameToTensorPointer[0].end()) {
                tokenInputTensorEmbdPrefill = (Qnn_Tensor_t*)embdPrefillGraphsTensorNameToTensorPointer[0]["in_embedding_prefill"];
            } else if (embdPrefillGraphsTensorNameToTensorPointer[0].find("in_embedding_prefill_chunk1") != embdPrefillGraphsTensorNameToTensorPointer[0].end()) {
                tokenInputTensorEmbdPrefill = (Qnn_Tensor_t*)embdPrefillGraphsTensorNameToTensorPointer[0]["in_embedding_prefill_chunk1"];
            }

            if (tokenInputTensorEmbdPrefill == nullptr) {
                LOGE("Failed to get tokenInputTensorEmbdPrefill");
                return RWKV_ERROR_IO;
            }
            embdPrefillSequenceLength = *(QNN_TENSOR_GET_DIMENSIONS(tokenInputTensorEmbdPrefill) + (QNN_TENSOR_GET_RANK(tokenInputTensorEmbdPrefill) - 2));
            LOGI("Embedding Prefill sequence length: %d", embdPrefillSequenceLength);;
        }

        isTensorInitialized = true;

        zero_state();
    }

    return RWKV_SUCCESS;
}

int qnn_backend::copy_float_to_qnn_tensor(Qnn_Tensor_t *qnn_tensor, const float *buffer, size_t element_count) {
    if (qnn_tensor == nullptr) {
        LOGE("%s: qnn_tensor is nullptr", __func__);
        return RWKV_ERROR_IO;
    }

    std::vector<size_t> dims;
    getTensorDims(dims, QNN_TENSOR_GET_DIMENSIONS(qnn_tensor), QNN_TENSOR_GET_RANK(qnn_tensor));
    if (datautil::calculateElementCount(dims) < element_count) {
        LOGE("%s: The element count is larger than the tensor size, tensor = %s", __func__, QNN_TENSOR_GET_NAME(qnn_tensor));
        return RWKV_ERROR_IO;
    }

    void *qnn_buffer = qnnIOTensorUtils->getBuffer(qnn_tensor);
    if (QNN_TENSOR_GET_DATA_TYPE(qnn_tensor) == QNN_DATATYPE_FLOAT_32) {
        memcpy(qnn_buffer, buffer, element_count * sizeof(float));
    } else if (QNN_TENSOR_GET_DATA_TYPE(qnn_tensor) == QNN_DATATYPE_FLOAT_16) {
        half_float::half *ptr = (half_float::half*)qnn_buffer;
        for (int j = 0; j < element_count; j++) {
            ptr[j] = (half_float::half)(buffer[j]);
        }
    } else if (QNN_TENSOR_GET_DATA_TYPE(qnn_tensor) == QNN_DATATYPE_UFIXED_POINT_16) {
        datautil::floatToTfN<uint16_t>(reinterpret_cast<uint16_t*>(qnn_buffer), (float*)(buffer),
                QNN_TENSOR_GET_QUANT_PARAMS(tokenInputTensorEmbdPrefill).scaleOffsetEncoding.offset,
                QNN_TENSOR_GET_QUANT_PARAMS(tokenInputTensorEmbdPrefill).scaleOffsetEncoding.scale,
                element_count);
    } else {
        LOGE("%s: Unsupported data type", __func__);
        return RWKV_ERROR_IO;
    }
    return RWKV_SUCCESS;
}

int qnn_backend::copy_qnn_tensor_to_float(Qnn_Tensor_t *qnn_tensor, float *buffer, size_t element_count) {
    if (qnn_tensor == nullptr) {
        LOGE("%s: qnn_tensor is nullptr", __func__);
        return RWKV_ERROR_IO;
    }

    std::vector<size_t> dims;
    getTensorDims(dims, QNN_TENSOR_GET_DIMENSIONS(qnn_tensor), QNN_TENSOR_GET_RANK(qnn_tensor));
    if (datautil::calculateElementCount(dims) < element_count) {
        LOGE("%s: The element count is larger than the tensor size, tensor = %s", __func__, QNN_TENSOR_GET_NAME(qnn_tensor));
        return RWKV_ERROR_IO;
    }

    void *qnn_buffer = qnnIOTensorUtils->getBuffer(qnn_tensor);
    if (qnn_buffer == nullptr) {
        LOGE("%s: Failed to get buffer for tensor %s", __func__, QNN_TENSOR_GET_NAME(qnn_tensor));
        return RWKV_ERROR_IO;
    }

    if (QNN_TENSOR_GET_DATA_TYPE(qnn_tensor) == QNN_DATATYPE_FLOAT_32) {
        memcpy(buffer, qnn_buffer, element_count * sizeof(float));
    } else if (QNN_TENSOR_GET_DATA_TYPE(qnn_tensor) == QNN_DATATYPE_FLOAT_16) {
        half_float::half *ptr = (half_float::half*)qnn_buffer;
        for (int i = 0; i < element_count; i++) {
            buffer[i] = ptr[i];
        }
    } else if (QNN_TENSOR_GET_DATA_TYPE(qnn_tensor) == QNN_DATATYPE_UFIXED_POINT_16) {
        datautil::tfNToFloat<uint16_t>(buffer, reinterpret_cast<uint16_t*>(qnn_buffer),
            QNN_TENSOR_GET_QUANT_PARAMS(qnn_tensor).scaleOffsetEncoding.offset,
            QNN_TENSOR_GET_QUANT_PARAMS(qnn_tensor).scaleOffsetEncoding.scale,
            element_count);
    } else {
        LOGE("%s: Unsupported data type", __func__);
        return RWKV_ERROR_IO;
    }
    return RWKV_SUCCESS;
}

int qnn_backend::execute_graph(GraphInfo_t** graphsInfo, int graphsCount, Qnn_Tensor_t** inputTensors, Qnn_Tensor_t** outputTensors) {
    for (int graph_id = 0; graph_id < graphsCount; graph_id++) {
        auto graphInfo     = (*graphsInfo)[graph_id];
        auto executeStatus =
            g_qnn_backend_context_ptr->qnnFunctionPointers.qnnInterface.graphExecute(graphInfo.graph,
                                                            inputTensors[graph_id],
                                                            graphInfo.numInputTensors,
                                                            outputTensors[graph_id],
                                                            graphInfo.numOutputTensors,
                                                            nullptr, nullptr);
        if (QNN_GRAPH_NO_ERROR != executeStatus) {
            return RWKV_ERROR_EVAL;
        }
    }
    return RWKV_SUCCESS;
}

int qnn_backend::execute_prefill_graph() {
    return execute_graph(qnnPrefillGraphsInfo, qnnPrefillGraphsCount, inputTensorsPrefill, outputTensorsPrefill);
}

int qnn_backend::execute_emb_decode_graph() {
    return execute_graph(qnnEmbdGraphsInfo, qnnEmbdGraphsCount, inputTensorsEmbd, outputTensorsEmbd);
}

int qnn_backend::execute_emb_prefill_graph() {
    return execute_graph(qnnEmbdPrefillGraphsInfo, qnnEmbdPrefillGraphsCount, inputTensorsEmbdPrefill, outputTensorsEmbdPrefill);
}

int qnn_backend::execute_batch_decode_graph(int bsz) {
    int needed_bsz = (bsz == 1) ? 1 : ((bsz + 1) & ~1);
    if (qnnBatchDecodeGraphsCount.find(needed_bsz) == qnnBatchDecodeGraphsCount.end() ||
        qnnBatchDecodeGraphsCount[needed_bsz] == 0) {
        LOGE("QNN: no graphs available for batch size: %d", needed_bsz);
        return RWKV_ERROR_EVAL;
    }

    return execute_graph(qnnBatchDecodeGraphsInfo[needed_bsz], 
                        qnnBatchDecodeGraphsCount[needed_bsz], 
                        inputTensorsBatchDecode[needed_bsz], 
                        outputTensorsBatchDecode[needed_bsz]);
}

int qnn_backend::post_graph_execute(Tensor1D & logits) {
    if (logits_buffer.empty()) {
        logits_buffer.resize(vocab_size);
    }

    if (logitsOutputTensorSize == 0) {
        std::vector<size_t> dims;
        getTensorDims(dims, QNN_TENSOR_GET_DIMENSIONS(logitsOutputTensor), QNN_TENSOR_GET_RANK(logitsOutputTensor));
        logitsOutputTensorSize = dims[2];
    }

    if (logitsOutputTensorSize != vocab_size) {
#ifdef ENABLE_MNN
        if (external_lmhead_filetype != "mnn" || external_lmhead_interpretor == nullptr || external_lmhead_mnn_session == nullptr) {
            LOGE("The model requires external lmhead, but external lmhead is not loaded");
            return RWKV_ERROR_IO;
        }

        if (external_lmhead_filetype == "mnn") {
            auto input = external_lmhead_interpretor->getSessionInput(external_lmhead_mnn_session, "in");
            if (external_lmhead_input_tensor == nullptr) {
                external_lmhead_input_tensor = new MNN::Tensor(input, MNN::Tensor::CAFFE);
            }
            if (RWKV_SUCCESS != copy_qnn_tensor_to_float(logitsOutputTensor, external_lmhead_input_tensor->host<float>(), hidden_size)) {
                return RWKV_ERROR_IO;
            }
            input->copyFromHostTensor(external_lmhead_input_tensor);

            external_lmhead_interpretor->runSession(external_lmhead_mnn_session);

            auto output = external_lmhead_interpretor->getSessionOutput(external_lmhead_mnn_session, "out");
            void *output_ptr = output->map(MNN::Tensor::MAP_TENSOR_READ, output->getDimensionType());

            memcpy(logits_buffer.data(), output_ptr, vocab_size * sizeof(float));
            output->unmap(MNN::Tensor::MAP_TENSOR_READ, output->getDimensionType(), output_ptr);
        } else {
#else
        {
#endif
            LOGE("Unsupported external lmhead filetype: %s", external_lmhead_filetype.c_str());
            return RWKV_ERROR_IO;
        }
    } else {
        if (RWKV_SUCCESS != copy_qnn_tensor_to_float(logitsOutputTensor, logits_buffer.data(), vocab_size)) {
            return RWKV_ERROR_IO;
        }
    }
    logits = Tensor1D::make(logits_buffer.data(), TensorDType::F32, (size_t)vocab_size);
    return RWKV_SUCCESS;
}

int qnn_backend::copy_deep_embedding_to_qnn_tensor_decode(int idx) {
    // [vocab_size, n_layers * deep_embedding_size]
    if (external_deep_embeddings == nullptr) {
        LOGE("external_deep_embeddings is not loaded");
        return RWKV_ERROR_IO;
    }

    size_t token_offset = idx * n_layers * deep_embedding_size;
    for (int i = 0; i < n_layers; i++) {
        if (deepEmbeddingTensors.find(i) == deepEmbeddingTensors.end()) {
            LOGE("Failed to get deepembedding tensor %s", ("s_emb" + std::to_string(i) + "_in").c_str());
            return RWKV_ERROR_IO;
        }
        void *ptr = qnnIOTensorUtils->getBuffer(deepEmbeddingTensors[i]);
        size_t deep_embedding_offset = (token_offset + i * deep_embedding_size) * deep_embeddings_elembytes;
        memcpy(ptr, external_deep_embeddings.get() + deep_embedding_offset, deep_embedding_size * deep_embeddings_elembytes);
    }
    return RWKV_SUCCESS;
}

int qnn_backend::copy_deep_embedding_to_qnn_tensor_prefill(int idx, int dst_offset) {
    // [vocab_size, n_layers * deep_embedding_size]
    if (external_deep_embeddings == nullptr) {
        LOGE("external_deep_embeddings is not loaded");
        return RWKV_ERROR_IO;
    }

    size_t token_offset = idx * n_layers * deep_embedding_size;
    for (int j = 0; j < n_layers; j++) {
        if (deepEmbeddingPrefillTensors.find(j) == deepEmbeddingPrefillTensors.end()) {
            LOGE("Failed to get deepembedding tensor %s", ("s_emb" + std::to_string(j) + "_in_prefill").c_str());
            return RWKV_ERROR_IO;
        }
        uint8_t *ptr = (uint8_t*)qnnIOTensorUtils->getBuffer(deepEmbeddingPrefillTensors[j]) + dst_offset * deep_embedding_size * deep_embeddings_elembytes;
        size_t deep_embedding_offset = (token_offset + j * deep_embedding_size) * deep_embeddings_elembytes;
        memcpy(ptr, external_deep_embeddings.get() + deep_embedding_offset, deep_embedding_size * deep_embeddings_elembytes);
    }
    return RWKV_SUCCESS;
}

int qnn_backend::debug_dump_state() { 
    // for (auto &[tensorName, tensor] : stateTensorsNameToTensorPointer) {
    for (int i = 0; i < 3 * n_layers; i++) {
        std::string tensorName = "state" + std::to_string(i) + "_out";
        Qnn_Tensor_t *qnntensor = (Qnn_Tensor_t*)stateTensorsNameToTensorPointer[tensorName];

        void *buffer = qnnIOTensorUtils->getBuffer(qnntensor);
        if (buffer == nullptr) {
            LOGE("%s: Failed to get buffer for tensor %s", __func__, tensorName.c_str());
            return RWKV_ERROR_IO;
        }
        std::string msg = tensorName + ": ";
        if (QNN_TENSOR_GET_DATA_TYPE(qnntensor) == QNN_DATATYPE_FLOAT_16) {
            for (int i = 0; i < 10; i++) {
                msg += std::to_string(((__fp16*)buffer)[i]) + " ";
            }
        }
        else if (QNN_TENSOR_GET_DATA_TYPE(qnntensor) == QNN_DATATYPE_FLOAT_32)
            for (int i = 0; i < 10; i++) {
                msg += std::to_string(((float*)buffer)[i]) + " ";
            }
        else {
            float *float_buffer = new float[10];
            datautil::tfNToFloat<uint16_t>(float_buffer, reinterpret_cast<uint16_t*>(buffer),
                QNN_TENSOR_GET_QUANT_PARAMS(qnntensor).scaleOffsetEncoding.offset,
                QNN_TENSOR_GET_QUANT_PARAMS(qnntensor).scaleOffsetEncoding.scale,
                10);
            for (int i = 0; i < 10; i++) {
                msg += std::to_string(float_buffer[i]) + " ";
            }
            delete[] float_buffer;
        }
        LOGI("%s", msg.c_str());
    }
    return RWKV_SUCCESS;
}

int qnn_backend::eval(int id, Tensor1D & logits) {
    {
        std::lock_guard<std::mutex> lock(g_qnn_backend_context_ptr->qnnMutex);
        if (!isTensorInitialized) {
            LOGD("qnn_backend::eval() isTensorInitialized: %d", isTensorInitialized);
            return RWKV_ERROR_EVAL;
        }

        if (tokenInputTensorBatchDecode[1] == nullptr) {
            if (external_embeddings == nullptr) {
                LOGE("The model requires external embeddings, but external embeddings are not loaded");
                return RWKV_ERROR_IO;
            }

            if (tokenInputTensorEmbd == nullptr) {
                LOGE("tokenInputTensorEmbd is not set");
                return RWKV_ERROR_IO;
            }

            // assume that the external_embeddings has elementsize = 2
            void *buffer = qnnIOTensorUtils->getBuffer(tokenInputTensorEmbd);
            if (buffer == nullptr) {
                LOGE("Failed to get tokenInputTensorEmbd");
                return RWKV_ERROR_IO;
            }
            uint16_t *emb_ptr = (uint16_t*)external_embeddings.get();
            memcpy(buffer, emb_ptr + hidden_size * id, hidden_size * sizeof(uint16_t));

            if (has_deep_embedding) {
                if (RWKV_SUCCESS != copy_deep_embedding_to_qnn_tensor_decode(id)) {
                    return RWKV_ERROR_EVAL;
                }
            }

            if (RWKV_SUCCESS != execute_emb_decode_graph()) {
                return RWKV_ERROR_EVAL;
            }
        } else {
            int *token_input = (int*)qnnIOTensorUtils->getBuffer(tokenInputTensorBatchDecode[1]);
            if (token_input == nullptr) {
                LOGE("Failed to get tokenInputTensorBatchDecode[1]");
                return RWKV_ERROR_IO;
            }
            *token_input = id;

            if (has_deep_embedding) {
                if (RWKV_SUCCESS != copy_deep_embedding_to_qnn_tensor_decode(id)) {
                    return RWKV_ERROR_EVAL;
                }
            }

            if (RWKV_SUCCESS != execute_batch_decode_graph(1)) {
                return RWKV_ERROR_EVAL;
            }
        }
    }

    // debug_dump_state();
    return post_graph_execute(logits);
}

int qnn_backend::eval(std::vector<int> ids, Tensor1D & logits) {
    if (ids.empty()) {
        return RWKV_ERROR_EVAL;
    }

    if (tokenInputTensorBatchDecode[1] == nullptr) {
        if (external_embeddings == nullptr) {
            LOGE("The model requires external embeddings, but external embeddings are not loaded");
            return RWKV_ERROR_IO;
        }

        if (tokenInputTensorEmbd == nullptr) {
            LOGE("tokenInputTensorEmbd is not set");
            return RWKV_ERROR_IO;
        }

        std::lock_guard<std::mutex> lock(g_qnn_backend_context_ptr->qnnMutex);
        int idx = 0;
        uint16_t *buffer;
        uint16_t *emb_ptr = (uint16_t*)external_embeddings.get();

        if (embdPrefillSequenceLength > 0 && tokenInputTensorEmbdPrefill != nullptr) {
            buffer = (uint16_t*)qnnIOTensorUtils->getBuffer(tokenInputTensorEmbdPrefill);
            if (buffer == nullptr) {
                LOGE("Failed to get tokenInputTensorEmbdPrefill");
                return RWKV_ERROR_IO;
            }
            for (; idx + embdPrefillSequenceLength <= ids.size(); idx += embdPrefillSequenceLength) {
                for (int i = 0; i < embdPrefillSequenceLength; i++) {
                    memcpy(buffer + i * hidden_size, emb_ptr + hidden_size * ids[idx + i], hidden_size * deep_embeddings_elembytes);
                }

                if (has_deep_embedding) {
                    for (int i = 0; i < embdPrefillSequenceLength; i++) {
                        if (RWKV_SUCCESS != copy_deep_embedding_to_qnn_tensor_prefill(ids[idx + i], i)) {
                            LOGE("Failed to copy deep embedding to qnn tensor");
                            return RWKV_ERROR_EVAL;
                        }
                    }
                }

                if (RWKV_SUCCESS != execute_emb_prefill_graph()) {
                    LOGE("Failed to execute emb prefill graph");
                    return RWKV_ERROR_EVAL;
                }
            }
        }

        buffer = (uint16_t*)qnnIOTensorUtils->getBuffer(tokenInputTensorEmbd);
        if (buffer == nullptr) {
            LOGE("Failed to get tokenInputTensorEmbd");
            return RWKV_ERROR_IO;
        }
        for (; idx < ids.size(); idx++) {
            memcpy(buffer, emb_ptr + hidden_size * ids[idx], hidden_size * deep_embeddings_elembytes);

            if (has_deep_embedding) {
                if (RWKV_SUCCESS != copy_deep_embedding_to_qnn_tensor_decode(ids[idx])) {
                    LOGE("Failed to copy deep embedding to qnn tensor");
                    return RWKV_ERROR_EVAL;
                }
            }

            if (RWKV_SUCCESS != execute_emb_decode_graph()) {
                LOGE("Failed to execute emb decode graph");
                return RWKV_ERROR_EVAL;
            }
        }
    } else {
        if (prefillSequenceLength == 0) {
            for (auto id : ids) {
                if (RWKV_SUCCESS != eval(id, logits)) {
                    LOGE("Failed to eval");
                    return RWKV_ERROR_MODEL;
                }
            }
        } else {
            std::lock_guard<std::mutex> lock(g_qnn_backend_context_ptr->qnnMutex);
            int *token_input = (int*)qnnIOTensorUtils->getBuffer(tokenInputTensorPrefill);
            if (token_input == nullptr) {
                LOGE("Failed to get tokenInputTensorPrefill");
                return RWKV_ERROR_IO;
            }
            int idx = 0;

            bool is_prefilling_usable = true;
            auto start = std::chrono::high_resolution_clock::now();
            for (; (idx + prefillSequenceLength) <= ids.size(); idx += prefillSequenceLength) {
                for (int i = 0; i < prefillSequenceLength; i++) {
                    token_input[i] = ids[idx + i];
                }
                // LOGD("Prefilling using seq mode from %d to %d", idx, idx + prefillSequenceLength);

                if (has_deep_embedding) {
                    for (int i = 0; i < prefillSequenceLength; i++) {
                        if (RWKV_SUCCESS != copy_deep_embedding_to_qnn_tensor_prefill(ids[idx + i], i)) {
                            LOGE("Failed to copy deep embedding to qnn tensor");
                            return RWKV_ERROR_EVAL;
                        }
                    }
                }

                if (RWKV_SUCCESS != execute_prefill_graph()) {
                    is_prefilling_usable = false;
                    LOGE("QNN: prefill graph not usable; falling back to decode mode");
                }
                if (!is_prefilling_usable) {
                    break;
                }
            }

            auto end = std::chrono::high_resolution_clock::now();
            prefill_speed = (ids.size() / prefillSequenceLength * prefillSequenceLength) * 1000000.0 / std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            // LOGD("Prefilling tails using decode mode from %d to %d", idx, ids.size());
            token_input = (int*)qnnIOTensorUtils->getBuffer(tokenInputTensorBatchDecode[1]);
            if (token_input == nullptr) {
                LOGE("Failed to get tokenInputTensorBatchDecode[1]");
                return RWKV_ERROR_IO;
            }
            for (; idx < ids.size(); idx++) {
                *token_input = ids[idx];

                if (has_deep_embedding) {
                    if (RWKV_SUCCESS != copy_deep_embedding_to_qnn_tensor_decode(ids[idx])) {
                        LOGE("Failed to copy deep embedding to qnn tensor");
                        return RWKV_ERROR_EVAL;
                    }
                }

                if (RWKV_SUCCESS != execute_batch_decode_graph(1)) {
                    LOGE("Failed to execute decode graph");
                    return RWKV_ERROR_EVAL;
                }
            }
        }
    }

    // copy logits
    return post_graph_execute(logits);
}

int qnn_backend::eval_with_embeddings(const float *embeddings, int n_tokens, Tensor1D & logits) {
    {
        std::lock_guard<std::mutex> lock(g_qnn_backend_context_ptr->qnnMutex);
        if (!isTensorInitialized) return RWKV_ERROR_EVAL;
        LOGD("[QNN] eval_with_embeddings: n_tokens: %d", n_tokens);

        int i = 0;
        if (embdPrefillSequenceLength > 0) {
            for (; i + embdPrefillSequenceLength <= n_tokens; i += embdPrefillSequenceLength) {
                if (RWKV_SUCCESS != copy_float_to_qnn_tensor(tokenInputTensorEmbdPrefill, embeddings + i * hidden_size, embdPrefillSequenceLength * hidden_size)) {
                    LOGE("Failed to copy embeddings to tokenInputTensorEmbdPrefill");
                    return RWKV_ERROR_IO;
                }

                if (RWKV_SUCCESS != execute_emb_prefill_graph()) {
                    return RWKV_ERROR_EVAL;
                }
            }
        }

        // leftovers
        for (; i < n_tokens; i++) {
            if (RWKV_SUCCESS != copy_float_to_qnn_tensor(tokenInputTensorEmbd, embeddings + i * hidden_size, hidden_size)) {
                return RWKV_ERROR_IO;
            }

            if (RWKV_SUCCESS != execute_emb_decode_graph()) {
                return RWKV_ERROR_EVAL;
            }
        }
    }

    return post_graph_execute(logits);
}

int qnn_backend::eval_batch(std::vector<std::vector<int>> ids, Tensor1D & logits) {
    if (supported_batch_sizes.size() == 0) {
        return RWKV_ERROR_UNSUPPORTED;
    }
    int batch_size = ids.size();

    for (int i = 0; i < batch_size; i++) {
        if (ids[i].size() != 1) {
            LOGE("eval_batch with prefilling is not supported yet");
            return RWKV_ERROR_UNSUPPORTED;
        }
    }

    {
        std::lock_guard<std::mutex> lock(g_qnn_backend_context_ptr->qnnMutex);
        if (!isTensorInitialized) {
            LOGE("qnn_backend::eval() isTensorInitialized: %d", isTensorInitialized);
            return RWKV_ERROR_EVAL;
        }

        int needed_bsz = (batch_size == 1) ? 1 : ((batch_size + 1) & ~1);
        int *token_input = (int*)qnnIOTensorUtils->getBuffer(tokenInputTensorBatchDecode[needed_bsz]); // ceil to nearest even number (except bsz=1)
        if (token_input == nullptr) {
            LOGE("Failed to get tokenInputTensor for batch size %d", needed_bsz);
            return RWKV_ERROR_IO;
        }
        for (int b = 0; b < ids.size(); b++) {
            token_input[b] = ids[b][0];
        }

        if (RWKV_SUCCESS != execute_batch_decode_graph(batch_size)) {
            LOGE("QNN: failed to execute batch decode graph");
            return RWKV_ERROR_EVAL;
        }
    }

    // return post_graph_execute_batch(logits);
    if (logits_buffer.size() < vocab_size * batch_size) {
        logits_buffer.resize(vocab_size * batch_size);
    }

    if (logitsOutputTensorSize == 0) {
        std::vector<size_t> dims;
        getTensorDims(dims, QNN_TENSOR_GET_DIMENSIONS(logitsOutputTensor), QNN_TENSOR_GET_RANK(logitsOutputTensor));
        logitsOutputTensorSize = dims[2];
    }

    {
        if (RWKV_SUCCESS != copy_qnn_tensor_to_float(logitsOutputTensor, logits_buffer.data(), vocab_size * batch_size)) {
            return RWKV_ERROR_IO;
        }
    }
    logits = Tensor1D::make(logits_buffer.data(), TensorDType::F32, (size_t)(vocab_size * batch_size));
    return RWKV_SUCCESS;
}

bool qnn_backend::is_available() {
    // TODO: Detect this
    return true;
}

int qnn_backend::zero_state() {
    return zero_state_on_batch_slot(0);
}

int qnn_backend::get_state(std::any &state) {
    return get_state_on_batch_slot(0, state);
}

int qnn_backend::set_state(std::any state) {
    return set_state_on_batch_slot(0, state);
}

int qnn_backend::free_state(std::any state) {
    if (!state.has_value()) return RWKV_SUCCESS;
    auto new_state = std::any_cast<std::vector<std::vector<uint8_t>>>(state);
    for (auto &s : new_state) {
        s.clear();
    }
    new_state.clear();
    return RWKV_SUCCESS;
}

int qnn_backend::get_state_on_batch_slot(int slot, std::any &state) {
    int max_supported_bsz = 1;
    if (!supported_batch_sizes.empty()) {
        int max_supported_bsz_index = std::max_element(supported_batch_sizes.begin(), supported_batch_sizes.end()) - supported_batch_sizes.begin();
        max_supported_bsz = supported_batch_sizes[max_supported_bsz_index];
    }
    if (slot >= max_supported_bsz) {
        return RWKV_ERROR_IO;
    }
    auto new_state = std::vector<std::vector<uint8_t>>();
    for (int i = 0; i < 3 * n_layers; i++) {
        Qnn_Tensor_t *tensor = (Qnn_Tensor_t*)stateTensorsNameToTensorPointer["state" + std::to_string(i) + "_out"];
        uint8_t *buffer = (uint8_t*)qnnIOTensorUtils->getBuffer(tensor);
        if (buffer == nullptr) {
            LOGE("%s: Failed to get buffer for state tensor %i", __func__, i);
            return RWKV_ERROR_IO;
        }
        size_t total_size = qnnIOTensorUtils->getBufferSize(tensor);
        size_t size_per_slot = total_size / max_supported_bsz;
        size_t offset = size_per_slot * slot;
        new_state.push_back(std::vector<uint8_t>(buffer + offset, buffer + offset + size_per_slot));
    }
    state = new_state;
    return RWKV_SUCCESS;
}

int qnn_backend::set_state_on_batch_slot(int slot, std::any state) {
    if (!state.has_value()) return RWKV_SUCCESS;
    int max_supported_bsz = 1;
    if (!supported_batch_sizes.empty()) {
        int max_supported_bsz_index = std::max_element(supported_batch_sizes.begin(), supported_batch_sizes.end()) - supported_batch_sizes.begin();
        max_supported_bsz = supported_batch_sizes[max_supported_bsz_index];
    }
    if (slot >= max_supported_bsz) {
        return RWKV_ERROR_IO;
    }
    auto new_state = std::any_cast<std::vector<std::vector<uint8_t>>>(state);
    for (int i = 0; i < 3 * n_layers; i++) {
        Qnn_Tensor_t *tensor = (Qnn_Tensor_t*)stateTensorsNameToTensorPointer["state" + std::to_string(i) + "_out"];
        void *buffer = qnnIOTensorUtils->getBuffer(tensor);
        if (buffer == nullptr) {
            LOGE("%s: Failed to get buffer for state tensor %i", __func__, i);
            return RWKV_ERROR_IO;
        }
        size_t total_size = qnnIOTensorUtils->getBufferSize(tensor);
        size_t size_per_slot = total_size / max_supported_bsz;
        size_t offset = size_per_slot * slot;
        memcpy((uint8_t*)buffer + offset, new_state[i].data(), new_state[i].size());
    }
    return RWKV_SUCCESS;
}

int qnn_backend::zero_state_on_batch_slot(int slot) {
    int max_supported_bsz = 1;
    if (!supported_batch_sizes.empty()) {
        int max_supported_bsz_index = std::max_element(supported_batch_sizes.begin(), supported_batch_sizes.end()) - supported_batch_sizes.begin();
        max_supported_bsz = supported_batch_sizes[max_supported_bsz_index];
    }
    if (slot >= max_supported_bsz) {
        return RWKV_ERROR_IO;
    }

    for (auto &[tensorName, tensor] : stateTensorsNameToTensorPointer) {
        size_t element_count = 1;
        Qnn_Tensor_t *qnntensor = (Qnn_Tensor_t*)tensor;
        for (int j = 0; j < QNN_TENSOR_GET_RANK(qnntensor); j++) {
            element_count *= *(QNN_TENSOR_GET_DIMENSIONS(qnntensor) + j);
        }
        uint8_t *buffer = (uint8_t*)qnnIOTensorUtils->getBuffer(qnntensor);
        if (buffer == nullptr) {
            LOGE("%s: Failed to get buffer for tensor %s", __func__, tensorName.c_str());
            return RWKV_ERROR_IO;
        }
        size_t total_size = qnnIOTensorUtils->getBufferSize(qnntensor);
        size_t size_per_slot = total_size / max_supported_bsz;
        size_t offset = size_per_slot * slot;
        if (QNN_TENSOR_GET_DATA_TYPE(qnntensor) == QNN_DATATYPE_FLOAT_16 || QNN_TENSOR_GET_DATA_TYPE(qnntensor) == QNN_DATATYPE_FLOAT_32)
            memset(buffer + offset, 0, size_per_slot);
        else {
            float fpzero = 0.0;
            uint16_t qtzero = 0;
            datautil::floatToTfN<uint16_t>(&qtzero, &fpzero,
                QNN_TENSOR_GET_QUANT_PARAMS(qnntensor).scaleOffsetEncoding.offset,
                QNN_TENSOR_GET_QUANT_PARAMS(qnntensor).scaleOffsetEncoding.scale,
                1);
            for (int j = 0; j < element_count / max_supported_bsz; j++) {
                ((uint16_t*)(buffer + offset))[j] = qtzero;
            }
        }
    }
    return RWKV_SUCCESS;
}

int qnn_backend::load_raw_states(std::vector<std::vector<half_float::half>> states) {
    zero_state();
    if (states.size() != n_layers) {
        LOGE("%s: The state size is not equal to the number of layers, expected %d, got %d", __func__, n_layers, states.size());
        return RWKV_ERROR_IO;
    }
    for (int i = 0; i < n_layers; i++) {
        Qnn_Tensor_t *tensor = (Qnn_Tensor_t*)stateTensorsNameToTensorPointer["state" + std::to_string(i * 3 + 1) + "_out"];
        void *buffer = qnnIOTensorUtils->getBuffer(tensor);
        if (buffer == nullptr) {
            LOGE("%s: Failed to get buffer for state tensor %i", __func__, i);
            return RWKV_ERROR_IO;
        }
        memcpy(buffer, states[i].data(), states[i].size() * sizeof(half_float::half));
    }

    return RWKV_SUCCESS;
}

int qnn_backend::serialize_runtime_state(std::any state, std::vector<uint8_t> &data) {
    if (!state.has_value()) return RWKV_ERROR_IO;
    auto new_state = std::any_cast<std::vector<std::vector<uint8_t>>>(state);
    for (int i = 0; i < new_state.size(); i++) {
        int32_t size = static_cast<int32_t>(new_state[i].size());
        uint8_t *p = reinterpret_cast<uint8_t*>(&size);
        for (size_t j = 0; j < sizeof(int32_t); j++) {
            data.push_back(p[j]);
        }
        data.insert(data.end(), new_state[i].begin(), new_state[i].end());
    }
    return RWKV_SUCCESS;
}

int qnn_backend::deserialize_runtime_state(std::vector<uint8_t> &data, std::any &state) {
    auto new_state = std::vector<std::vector<uint8_t>>();
    size_t offset = 0;
    while (offset + sizeof(int32_t) <= data.size()) {
        int32_t size = 0;
        for (size_t j = 0; j < sizeof(int32_t); j++) {
            ((uint8_t*)&size)[j] = data[offset + j];
        }
        offset += sizeof(int32_t);
        if (offset + size > data.size()) {
            LOGE("state data is not complete");
            return RWKV_ERROR_IO;
        }
        std::vector<uint8_t> buf(data.begin() + offset, data.begin() + offset + size);
        new_state.push_back(std::move(buf));
        offset += size;
    }
    state = new_state;
    return RWKV_SUCCESS;
}

int qnn_backend::release_model() {
    LOGI("[QNN] release_model");
    // free graphs
    if (qnnPrefillGraphsCount > 0) {
        for (int i = 0; i < qnnPrefillGraphsCount; i++) {
            auto graphInfo     = (*qnnPrefillGraphsInfo)[i];
            qnnIOTensorUtils->tearDownTensors(inputTensorsPrefill[i], graphInfo.numInputTensors);
            qnnIOTensorUtils->tearDownTensors(outputTensorsPrefill[i], graphInfo.numOutputTensors);
            inputTensorsPrefill[i]  = nullptr;
            outputTensorsPrefill[i] = nullptr;
        }

        freeGraphsInfo(&qnnPrefillGraphsInfo, qnnPrefillGraphsCount);
        qnnPrefillGraphsInfo = nullptr;
    }

    if (qnnEmbdGraphsCount > 0) {
        for (int i = 0; i < qnnEmbdGraphsCount; i++) {
            auto graphInfo     = (*qnnEmbdGraphsInfo)[i];
            qnnIOTensorUtils->tearDownTensors(inputTensorsEmbd[i], graphInfo.numInputTensors);
            qnnIOTensorUtils->tearDownTensors(outputTensorsEmbd[i], graphInfo.numOutputTensors);
            inputTensorsEmbd[i]  = nullptr;
            outputTensorsEmbd[i] = nullptr;
        }

        freeGraphsInfo(&qnnEmbdGraphsInfo, qnnEmbdGraphsCount);
        qnnEmbdGraphsInfo = nullptr;
    }

    cleanup_batch_graphs();

    for (int i = 0; i < qnnContextHandles.size(); i++) {
        if (QNN_CONTEXT_NO_ERROR !=
            g_qnn_backend_context_ptr->qnnFunctionPointers.qnnInterface.contextFree(qnnContextHandles[i], nullptr)) {
            LOGE("Could not free context");
        }
    }
    qnnContextHandles.clear();

    for (int i = 0; i < graphConfigsInfoCount; i++) {
        delete graphConfigsInfo[i];
    }
    delete graphConfigsInfo;

    delete qnnIOTensorUtils;

    if (qnnModelHandle)
        pal::dynamicloading::dlClose(qnnModelHandle);

    tokenInputTensorBatchDecode.clear();
    deepEmbeddingTensors.clear();
    deepEmbeddingPrefillTensors.clear();
    stateTensorsNameToTensorPointer.clear();
    return RWKV_SUCCESS;
}

void qnn_backend::cleanup_batch_graphs() {
    for (auto& [batchSize, count] : qnnBatchDecodeGraphsCount) {
        if (qnnBatchDecodeGraphsInfo[batchSize] != nullptr) {
            free(qnnBatchDecodeGraphsInfo[batchSize]);
            qnnBatchDecodeGraphsInfo[batchSize] = nullptr;
        }
        if (inputTensorsBatchDecode[batchSize] != nullptr) {
            delete[] inputTensorsBatchDecode[batchSize];
            inputTensorsBatchDecode[batchSize] = nullptr;
        }
        if (outputTensorsBatchDecode[batchSize] != nullptr) {
            delete[] outputTensorsBatchDecode[batchSize];
            outputTensorsBatchDecode[batchSize] = nullptr;
        }
    }
    qnnBatchDecodeGraphsCount.clear();
    qnnBatchDecodeGraphsInfo.clear();
    inputTensorsBatchDecode.clear();
    outputTensorsBatchDecode.clear();
    batchDecodeGraphsTensorNameToTensorPointer.clear();
    batchDecodeGraphsTensorNameToSize.clear();
}

int qnn_backend::release() {
    if (g_qnn_backend_context_ptr == nullptr) {
        LOGW("[QNN] qnn_backend::release: qnn_backend context is already null");
        return RWKV_SUCCESS;
    }
    if (g_qnn_backend_context_ptr->ref_count > 0) {
        // std::lock_guard<std::mutex> lock(g_qnn_backend_context_ptr->qnnMutex);
        g_qnn_backend_context_ptr->ref_count--;
        LOGI("[QNN] qnn_backend::release: qnn_backend ref_count: %d", g_qnn_backend_context_ptr->ref_count);
    }

    if (g_qnn_backend_context_ptr->ref_count == 0) {
        g_qnn_backend_context_ptr.reset();
        LOGI("[QNN] qnn_backend::release: qnn_backend context reset");
    }
    return RWKV_SUCCESS;
}

} // namespace rwkvmobile