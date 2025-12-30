#include "clip.h"
#include "llava.h"

#include "llama.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define LOG_INF(...) do { fprintf(stdout, __VA_ARGS__); } while (0)
#define LOG_WRN(...) do { fprintf(stderr, __VA_ARGS__); } while (0)
#define LOG_ERR(...) do { fprintf(stderr, __VA_ARGS__); } while (0)
#define LOG_DBG(...) do { fprintf(stdout, __VA_ARGS__); } while (0)

// RGB uint8 image
struct clip_image_u8 {
    int nx;
    int ny;

    std::vector<uint8_t> buf;
};

// RGB float32 image (NHWC)
// Memory layout: RGBRGBRGB...
struct clip_image_f32 {
    int nx;
    int ny;

    std::vector<float> buf;
};

static bool encode_image_with_clip(clip_ctx * ctx_clip, int n_threads, const clip_image_u8 * img, float * image_embd, int * n_img_pos, bool force_no_postnorm) {
    // std::vector<clip_image_f32*> img_res_v; // format VectN x H x W x RGB (N x 336 x 336 x 3), so interleaved RGB - different to the python implementation which is N x 3 x 336 x 336
    clip_image_f32_batch img_res_v;
    img_res_v.size = 0;
    img_res_v.data = nullptr;
    if (!clip_image_preprocess(ctx_clip, img, &img_res_v)) {
        LOG_ERR("%s: unable to preprocess image\n", __func__);
        delete[] img_res_v.data;
        return false;
    }

    const int64_t t_img_enc_start_us = ggml_time_us();

    // const char * mm_patch_merge_type = clip_patch_merge_type(ctx_clip);

    {
        // flat / default llava-1.5 type embedding
        *n_img_pos = clip_n_patches(ctx_clip);
        bool encoded = clip_image_encode(ctx_clip, n_threads, &img_res_v.data[0], image_embd, force_no_postnorm); // image_embd shape is 576 x 4096
        delete[] img_res_v.data;
        if (!encoded) {
            LOG_ERR("Unable to encode image\n");

            return false;
        }
    }
    LOG_INF("%s: image embedding created: %d tokens\n", __func__, *n_img_pos);

    const int64_t t_img_enc_end_us = ggml_time_us();
    float t_img_enc_ms = (t_img_enc_end_us - t_img_enc_start_us) / 1000.0;

    LOG_INF("\n%s: image encoded in %8.2f ms by CLIP (%8.2f ms per image patch)\n", __func__, t_img_enc_ms, t_img_enc_ms / *n_img_pos);

    return true;
}

bool llava_image_embed_make_with_clip_img(clip_ctx * ctx_clip, int n_threads, const clip_image_u8 * img, float ** image_embd_out, int * n_img_pos_out, bool force_no_postnorm) {
    int num_max_patches = 16;
    float * image_embd;
    auto size = clip_embd_nbytes(ctx_clip)*num_max_patches;
    image_embd = (float *)malloc(size); // TODO: base on gridsize/llava mode
    if (!image_embd) {
        LOG_ERR("Unable to allocate memory for image embeddings\n");
        return false;
    }

    int n_img_pos;
    if (!encode_image_with_clip(ctx_clip, n_threads, img, image_embd, &n_img_pos, force_no_postnorm)) {
        LOG_ERR("%s: cannot encode image, aborting\n", __func__);
        free(image_embd);
        return false;
    }
    *image_embd_out = image_embd;
    *n_img_pos_out = n_img_pos;

    return true;
}

struct llava_image_embed * llava_image_embed_make_with_bytes(struct clip_ctx * ctx_clip, int n_threads, const unsigned char * image_bytes, int image_bytes_length, bool force_no_postnorm) {
    clip_image_u8 * img = clip_image_u8_init();
    if (!clip_image_load_from_bytes(image_bytes, image_bytes_length, img)) {
        clip_image_u8_free(img);
        LOG_ERR("%s: can't load image from bytes, is it a valid image?", __func__);
        return NULL;
    }

    float* image_embed = NULL;
    int n_image_pos = 0;
    bool image_embed_result = llava_image_embed_make_with_clip_img(ctx_clip, n_threads, img, &image_embed, &n_image_pos, force_no_postnorm);
    if (!image_embed_result) {
        clip_image_u8_free(img);
        LOG_ERR("%s: couldn't embed the image\n", __func__);
        return NULL;
    }

    clip_image_u8_free(img);
    auto result = (llava_image_embed*)malloc(sizeof(llava_image_embed));
    result->embed = image_embed;
    result->n_image_pos = n_image_pos;
    return result;
}

void llava_image_embed_free(struct llava_image_embed * embed) {
    free(embed->embed);
    free(embed);
}
