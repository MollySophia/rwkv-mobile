#ifndef LLAVA_H
#define LLAVA_H

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

struct clip_ctx;
struct llava_image_embed {
    float * embed;
    int n_image_pos;
};

bool llava_image_embed_make_with_clip_img(struct clip_ctx * ctx_clip, int n_threads, const struct clip_image_u8 * img, float ** image_embd_out, int * n_img_pos_out, bool force_no_postnorm = false);

/** build an image embed from image file bytes */
struct llava_image_embed * llava_image_embed_make_with_bytes(struct clip_ctx * ctx_clip, int n_threads, const unsigned char * image_bytes, int image_bytes_length, bool force_no_postnorm = false);
/** build an image embed from a path to an image filename */
struct llava_image_embed * llava_image_embed_make_with_filename(struct clip_ctx * ctx_clip, int n_threads, const char * image_path, bool force_no_postnorm = false);
/** free an embedding made with llava_image_embed_make_* */
void llava_image_embed_free(struct llava_image_embed * embed);

#ifdef __cplusplus
}
#endif

#endif
