#pragma once
#include <stdint.h>
typedef struct gs_stack gs_stack;
#pragma pack(push, 1)
typedef struct { uint8_t *base; int64_t len; } gs_rhdr;
typedef struct { gs_rhdr *hdr; gs_stack *stk; } gs_rref;
typedef struct { gs_rhdr *hdr; gs_stack *stk; gs_rhdr *fl; gs_stack *flstk; } gs_pref;

typedef struct { double e[23]; } a23_f64;

typedef struct { int32_t e[128]; } a128_i32;

typedef struct { float e[131072]; } a131072_f32;

typedef struct { float e[65536]; } a65536_f32;

typedef struct { float e[524288]; } a524288_f32;

typedef struct { float e[256]; } a256_f32;

typedef struct { float e[4194304]; } a4194304_f32;

typedef struct { float e[16384]; } a16384_f32;

typedef struct { float e[5898240]; } a5898240_f32;

typedef struct { int32_t e[64]; } a64_i32;

typedef struct { float e[64]; } a64_f32;

typedef struct { uint8_t *data; int64_t len; } sl_u8;

typedef struct { float *data; int64_t len; } sl_f32;

typedef struct ByteSet_g ByteSet_g;
typedef struct { uint8_t e[256]; } a256_b;

typedef struct { uint8_t e[3]; } a3_u8;

typedef struct { uint8_t e[16]; } a16_u8;

struct ByteSet_g {
    a256_b members_g;
    uint8_t kind_g;
    uint8_t invert_g;
    a3_u8 bytes_g;
    a16_u8 lo_g;
    a16_u8 hi_g;
};

#pragma pack(pop)

#ifdef __cplusplus
extern "C" {
#endif
void goose_init(int argc, char **argv);
double init_model(int32_t gs_export_arg0, int32_t gs_export_arg1, int32_t gs_export_arg2, int32_t gs_export_arg3, int32_t gs_export_arg4, int32_t gs_export_arg5, double gs_export_arg6, double gs_export_arg7);
double offset_of(int32_t gs_export_arg0_2, int32_t gs_export_arg1_2);
const uint8_t * buf_tokens(int64_t *gs_export_len_3);
const uint8_t * buf_logits(int64_t *gs_export_len_4);
const uint8_t * buf_record(int64_t *gs_export_len_5);
int32_t max_tokens(void);
int32_t max_context(void);
void set_record(int32_t gs_export_arg0_3);
void clear_steer(void);
int32_t add_steer(int32_t gs_export_arg0_4, int32_t gs_export_arg1_3, double gs_export_arg2_2);
int32_t forward(sl_f32 gs_export_arg0_5, int32_t gs_export_arg1_4, int32_t gs_export_arg2_3);
#ifdef __cplusplus
}
#endif
