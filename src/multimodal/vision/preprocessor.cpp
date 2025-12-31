#include "vision_encoder.h"
#include "logger.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <cmath>
#include <cstring>

namespace rwkvmobile {

bool VisionEncoder::load_file_to_bytes(const char* path, unsigned char** bytesOut, long *sizeOut) {
    auto file = fopen(path, "rb");
    if (file == NULL) {
        LOGE("%s: can't read file %s\n", __func__, path);
        return false;
    }

    fseek(file, 0, SEEK_END);
    auto fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    auto buffer = (unsigned char *)malloc(fileSize); // Allocate memory to hold the file data
    if (buffer == NULL) {
        LOGE("%s: failed to alloc %ld bytes for file %s\n", __func__, fileSize, path);
        perror("Memory allocation error");
        fclose(file);
        return false;
    }
    errno = 0;
    size_t ret = fread(buffer, 1, fileSize, file); // Read the file into the buffer
    if (ferror(file)) {
        LOGE("read error: %d", errno);
        free(buffer);
        fclose(file);
        return false;
    }
    if (ret != (size_t) fileSize) {
        LOGE("unexpectedly reached end of file");
        free(buffer);
        fclose(file);
        return false;
    }
    fclose(file); // Close the file

    *bytesOut = buffer;
    *sizeOut = fileSize;
    return true;
}


// Bilinear resize function
void VisionEncoder::bilinear_resize(const image_u8& src, image_u8& dst, int target_width, int target_height) {
    dst.nx = target_width;
    dst.ny = target_height;
    dst.buf.resize(3 * target_width * target_height);

    const int src_width = src.nx;
    const int src_height = src.ny;

    constexpr double bilinear_filter_support = 1.0;

    static constexpr uint32_t precision_bits = 32 - 8 - 2;

    const double scale_x = static_cast<double>(src_width) / target_width;
    const double scale_y = static_cast<double>(src_height) / target_height;

    std::vector<int32_t> bounds_horiz;
    std::vector<double> kk_horiz;
    int32_t ksize_horiz = 0;

    double filterscale_x = scale_x;
    if (filterscale_x < 1.0) {
        filterscale_x = 1.0;
    }

    const double support_x = bilinear_filter_support * filterscale_x;

    ksize_horiz = static_cast<int32_t>(ceil(support_x)) * 2 + 1;

    kk_horiz.resize(target_width * ksize_horiz);

    bounds_horiz.resize(target_width * 2);

    constexpr double half_pixel = 0.5;
    for (int32_t xx = 0; xx < target_width; ++xx) {
        double center = (xx + half_pixel) * scale_x - half_pixel;
        double ww = 0.0;
        double ss = 1.0 / filterscale_x;

        auto xmin = static_cast<int32_t>(center - support_x + half_pixel);
        if (xmin < 0) {
            xmin = 0;
        }

        auto xmax = static_cast<int32_t>(center + support_x + half_pixel);
        if (xmax > src_width) {
            xmax = src_width;
        }
        xmax -= xmin;

        double* k = &kk_horiz[xx * ksize_horiz];
        for (int32_t x = 0; x < xmax; ++x) {
            // Bilinear filter: if |x| < 1.0, return 1.0 - |x|, else return 0.0
            double filter_x = (x + xmin - center + half_pixel) * ss;
            if (filter_x < 0.0) {
                filter_x = -filter_x;
            }
            double w = (filter_x < 1.0) ? (1.0 - filter_x) : 0.0;
            k[x] = w;
            ww += w;
        }

        for (int32_t x = 0; x < xmax; ++x) {
            if (ww != 0.0) {
                k[x] /= ww;
            }
        }

        for (int32_t x = xmax; x < ksize_horiz; ++x) {
            k[x] = 0;
        }

        bounds_horiz[xx * 2 + 0] = xmin;
        bounds_horiz[xx * 2 + 1] = xmax;
    }


    std::vector<int32_t> bounds_vert;
    std::vector<double> kk_vert;
    int32_t ksize_vert = 0;

    double filterscale_y = scale_y;
    if (filterscale_y < 1.0) {
        filterscale_y = 1.0;
    }

    const double support_y = bilinear_filter_support * filterscale_y;

    ksize_vert = static_cast<int32_t>(ceil(support_y)) * 2 + 1;

    kk_vert.resize(target_height * ksize_vert);

    bounds_vert.resize(target_height * 2);

    for (int32_t yy = 0; yy < target_height; ++yy) {
        double center = (yy + half_pixel) * scale_y - half_pixel;
        double ww = 0.0;
        double ss = 1.0 / filterscale_y;

        auto ymin = static_cast<int32_t>(center - support_y + half_pixel);
        if (ymin < 0) {
            ymin = 0;
        }

        auto ymax = static_cast<int32_t>(center + support_y + half_pixel);
        if (ymax > src_height) {
            ymax = src_height;
        }
        ymax -= ymin;

        double* k = &kk_vert[yy * ksize_vert];
        for (int32_t y = 0; y < ymax; ++y) {
            // Bilinear filter: if |x| < 1.0, return 1.0 - |x|, else return 0.0
            double filter_y = (y + ymin - center + half_pixel) * ss;
            if (filter_y < 0.0) {
                filter_y = -filter_y;
            }
            double w = (filter_y < 1.0) ? (1.0 - filter_y) : 0.0;
            k[y] = w;
            ww += w;
        }

        for (int32_t y = 0; y < ymax; ++y) {
            if (ww != 0.0) {
                k[y] /= ww;
            }
        }

        for (int32_t y = ymax; y < ksize_vert; ++y) {
            k[y] = 0;
        }

        bounds_vert[yy * 2 + 0] = ymin;
        bounds_vert[yy * 2 + 1] = ymax;
    }

    const int32_t ybox_first = bounds_vert[0];
    const int32_t ybox_last = bounds_vert[target_height * 2 - 2] + bounds_vert[target_height * 2 - 1];
    std::vector<uint8_t> temp_buffer(3 * (ybox_last - ybox_first) * target_width);

    std::vector<double> kk_horiz_norm(kk_horiz.begin(), kk_horiz.end());
    constexpr auto shifted_coeff = static_cast<double>(1U << precision_bits);
    for (auto& k : kk_horiz_norm) {
        if (k < 0) {
            k = trunc(-half_pixel + k * shifted_coeff);
        } else {
            k = trunc(half_pixel + k * shifted_coeff);
        }
    }

    for (int32_t yy = 0; yy < ybox_last - ybox_first; ++yy) {
        for (int32_t xx = 0; xx < target_width; ++xx) {
            const int32_t xmin = bounds_horiz[xx * 2 + 0];
            const int32_t xmax = bounds_horiz[xx * 2 + 1];
            const double* k = &kk_horiz_norm[xx * ksize_horiz];

            for (int32_t c = 0; c < 3; ++c) {
                double ss = static_cast<double>(1U << (precision_bits - 1U)); // init_buffer
                for (int32_t x = 0; x < xmax; ++x) {
                    ss += static_cast<double>(src.buf[3 * ((yy + ybox_first) * src_width + (x + xmin)) + c]) * k[x];
                }

                auto saturate_val = static_cast<intmax_t>(ss) >> precision_bits;
                if (saturate_val < 0) {
                    temp_buffer[3 * (yy * target_width + xx) + c] = 0;
                } else if (saturate_val > 255) {
                    temp_buffer[3 * (yy * target_width + xx) + c] = 255;
                } else {
                    temp_buffer[3 * (yy * target_width + xx) + c] = static_cast<uint8_t>(saturate_val);
                }
            }
        }
    }

    for (int32_t i = 0; i < target_height; ++i) {
        bounds_vert[i * 2] -= ybox_first;
    }

    std::vector<double> kk_vert_norm(kk_vert.begin(), kk_vert.end());
    for (auto& k : kk_vert_norm) {
        if (k < 0) {
            k = trunc(-half_pixel + k * shifted_coeff);
        } else {
            k = trunc(half_pixel + k * shifted_coeff);
        }
    }

    for (int32_t yy = 0; yy < target_height; ++yy) {
        for (int32_t xx = 0; xx < target_width; ++xx) {
            const int32_t ymin = bounds_vert[yy * 2 + 0];
            const int32_t ymax = bounds_vert[yy * 2 + 1];
            const double* k = &kk_vert_norm[yy * ksize_vert];

            for (int32_t c = 0; c < 3; ++c) {
                double ss = static_cast<double>(1U << (precision_bits - 1U)); // init_buffer
                for (int32_t y = 0; y < ymax; ++y) {
                    ss += static_cast<double>(temp_buffer[3 * ((y + ymin) * target_width + xx) + c]) * k[y];
                }

                auto saturate_val = static_cast<intmax_t>(ss) >> precision_bits;
                if (saturate_val < 0) {
                    dst.buf[3 * (yy * target_width + xx) + c] = 0;
                } else if (saturate_val > 255) {
                    dst.buf[3 * (yy * target_width + xx) + c] = 255;
                } else {
                    dst.buf[3 * (yy * target_width + xx) + c] = static_cast<uint8_t>(saturate_val);
                }
            }
        }
    }
}

void VisionEncoder::rescale_image_u8_to_f32(const image_u8* src, image_f32* dst, const double scale) {
    dst->nx = src->nx;
    dst->ny = src->ny;
    dst->buf.resize(src->buf.size());

    for (size_t i = 0; i < src->buf.size(); ++i) {
        dst->buf[i] = static_cast<float>(src->buf[i]) * scale;
    }
}

void VisionEncoder::normalize_image_f32(const image_f32* src, image_f32* dst, const float mean[3], const float std[3]) {
    dst->nx = src->nx;
    dst->ny = src->ny;
    dst->buf.resize(src->buf.size());

    for (size_t i = 0; i < src->buf.size(); ++i) {
        int c = i % 3; // rgb
        dst->buf[i] = (src->buf[i] - mean[c]) / std[c];
    }
}

void VisionEncoder::preprocess(const image_u8 &img, std::vector<image_f32> &res_imgs) {
    res_imgs.resize(1);
    image_u8 resized_image;
    bilinear_resize(img, resized_image, image_size, image_size);
    rescale_image_u8_to_f32(&resized_image, &res_imgs[0], 0.00392156862745098);
    normalize_image_f32(&res_imgs[0], &res_imgs[0], image_mean, image_std);
}

bool VisionEncoder::image_u8_load_from_bytes(const unsigned char * bytes, size_t bytes_length, image_u8 &img) {
    int nx, ny, nc;
    auto * data = stbi_load_from_memory(bytes, bytes_length, &nx, &ny, &nc, 3);
    if (!data) {
        LOGE("%s: failed to decode image bytes\n", __func__);
        return false;
    }
    img.nx = nx;
    img.ny = ny;
    img.buf.resize(3 * nx * ny);
    memcpy(img.buf.data(), data, img.buf.size());
    stbi_image_free(data);
    return true;
}

} // namespace rwkvmobile