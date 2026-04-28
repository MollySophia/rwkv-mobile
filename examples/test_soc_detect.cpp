#include "c_api.h"
#include "logger.h"
#include <cstdio>

int main() {
    printf("platform: %s\n", rwkvmobile_get_platform_name());
    printf("soc_name: %s\n", rwkvmobile_get_soc_name());
    printf("soc_partname: %s\n", rwkvmobile_get_soc_partname());
    printf("htp_arch: %s\n", rwkvmobile_get_htp_arch());
    return 0;
}
