#!/usr/bin/env bash
# Builds gen/llama.c and gen/llama.h (the C that Goose generates from
# engine/llama.goose), and web/engine.mjs and web/engine.wasm from that C.
#
# Needs the Goose compiler (GOOSE, default: goose on PATH) with its standard
# library, and Emscripten's emcc on PATH (README, Build).
set -euo pipefail

cd "$(dirname "$0")"
goose="${GOOSE:-goose}"
mkdir -p gen

# Goose -> one C file with the whole runtime and no main(). The relative
# source path keeps the path of this checkout out of the C.
"$goose" -q --standalone --header gen/llama.h -o gen/llama.c engine/llama.goose

# Every exported function of the header, for Emscripten.
exports="$(sed -n 's/^[a-z0-9_ *]*[ *]\([a-z_0-9]*\)(.*);$/_\1/p' gen/llama.h | paste -sd, -)"

# The data stack holds only the zeroed temporaries of goose_init, one global
# array at a time. The largest is rec, 128 x 30 x 1536 f32 = 23.6 MB, so the
# stack is 32 MiB. The weights (538 MB for SmolLM2-135M) are malloc'd by the
# page. The float flags let clang vectorize the dot products, which only
# changes the order of the sums.
emcc -O3 -msimd128 -fassociative-math -fno-signed-zeros -fno-trapping-math \
    gen/llama.c -o web/engine.mjs \
    -DGS_STACK_RESERVE='(32ull<<20)' -DGS_STACK_GAP=65536 \
    -sMODULARIZE -sEXPORT_ES6 -sENVIRONMENT=web,node \
    -sINITIAL_MEMORY=160MB -sALLOW_MEMORY_GROWTH -sMAXIMUM_MEMORY=2GB \
    -sEXPORTED_FUNCTIONS="_malloc,_free,$exports" \
    -sEXPORTED_RUNTIME_METHODS=HEAPU8

# The tests check these sums, so a stale or hand-edited C or wasm fails CI.
sha256sum engine/llama.goose gen/llama.c gen/llama.h web/engine.wasm > gen/sha256sums
echo "built web/engine.mjs and web/engine.wasm"
