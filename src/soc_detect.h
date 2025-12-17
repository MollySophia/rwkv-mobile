#pragma once

#include <vector>
#include <cstdint>

namespace rwkvmobile {

enum platform_type {
    PLATFORM_SNAPDRAGON,
    PLATFORM_MEDIATEK,
    PLATFORM_UNKNOWN,
};

struct snapdragon_soc_info {
    int soc_id;
    const char * soc_partname;
    const char * soc_name;
    const char * htp_arch;
};

struct mediatek_soc_info {
    const char * soc_partname;
    const char * soc_name;
};

class soc_detect {
    public:
        soc_detect();
        ~soc_detect();

        int detect_platform();

        platform_type get_platform_type();
        const char * get_platform_name();
        const char * get_soc_name();
        const char * get_soc_partname();
        const char * get_htp_arch();
    private:
        platform_type m_platform_type = PLATFORM_UNKNOWN;
        int m_soc_id = 0;
        const char * m_soc_name = "Unknown";
        const char * m_soc_partname = "Unknown";
        const char * m_htp_arch = "Unknown";
};

struct CPUGroup {
    uint32_t minFreq;
    uint32_t maxFreq;
    std::vector<int> ids;
};

const std::vector<CPUGroup> get_cpu_groups();

} // namespace rwkvmobile