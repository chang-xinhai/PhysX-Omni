#!/bin/bash
# 渲染启动脚本（random + fixed_grid）
# 用法：bash run_render.sh
# 断点续跑：直接重新执行即可，已完成 item 自动跳过

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

LOG="$BASE_OUT/render_main.log"
mkdir -p "$BASE_OUT"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"; }

read -r -a GPU_LIST <<< "$GPUS"
W_ARGS=(
    --workers_small "$WORKERS_SMALL"
    --workers_medium "$WORKERS_MEDIUM"
    --workers_large "$WORKERS_LARGE"
    --workers_xlarge "$WORKERS_XLARGE"
)
GPU_ARGS=(--gpus "${GPU_LIST[@]}")

cd "$TOOLKIT"

log "==== render config ===="
log "PYTHON=$PYTHON"
log "BLENDER_BIN=$BLENDER_BIN"
log "DATA_ROOT=$DATA_ROOT"
log "BASE_OUT=$BASE_OUT"
log "GPUS=$GPUS"
log "WORKERS small/medium/large/xlarge=$WORKERS_SMALL/$WORKERS_MEDIUM/$WORKERS_LARGE/$WORKERS_XLARGE"

# ── random ────────────────────────────────────────────────────────────────────
log "==== random: GT mobility =="
$PYTHON render_adaptive.py \
    --dataset_roots "$GT_MOBILITY" \
    --filter_json "$MOB_JSON" \
    --output_root "$BASE_OUT/random" "${GPU_ARGS[@]}" \
    --view_mode random --num_views $NUM_VIEWS_RANDOM \
    --resolution "$RESOLUTION" "${W_ARGS[@]}" 2>&1 | tee -a "$LOG"

log "==== random: GT verse =="
$PYTHON render_adaptive.py \
    --dataset_roots "$GT_VERSE" \
    --filter_json "$VERSE_JSON" \
    --output_root "$BASE_OUT/random" "${GPU_ARGS[@]}" \
    --view_mode random --num_views $NUM_VIEWS_RANDOM \
    --resolution "$RESOLUTION" "${W_ARGS[@]}" 2>&1 | tee -a "$LOG"

log "==== random: mob OBJ =="
$PYTHON render_adaptive.py \
    --dataset_roots "${MOB_OBJ_ROOTS[@]}" \
    --filter_json "$MOB_JSON" \
    --output_root "$BASE_OUT/random" "${GPU_ARGS[@]}" \
    --view_mode random --num_views $NUM_VIEWS_RANDOM \
    --resolution "$RESOLUTION" "${W_ARGS[@]}" 2>&1 | tee -a "$LOG"

log "==== random: mob GLB =="
$PYTHON render_adaptive.py \
    --dataset_roots "${MOB_GLB_ROOTS[@]}" \
    --filter_json "$MOB_JSON" \
    --output_root "$BASE_OUT/random" "${GPU_ARGS[@]}" \
    --view_mode random --num_views $NUM_VIEWS_RANDOM \
    --resolution "$RESOLUTION" "${W_ARGS[@]}" 2>&1 | tee -a "$LOG"

log "==== random: verse OBJ =="
$PYTHON render_adaptive.py \
    --dataset_roots "${VERSE_OBJ_ROOTS[@]}" \
    --filter_json "$VERSE_JSON" \
    --output_root "$BASE_OUT/random" "${GPU_ARGS[@]}" \
    --view_mode random --num_views $NUM_VIEWS_RANDOM \
    --resolution "$RESOLUTION" "${W_ARGS[@]}" 2>&1 | tee -a "$LOG"

log "==== random: verse GLB =="
$PYTHON render_adaptive.py \
    --dataset_roots "${VERSE_GLB_ROOTS[@]}" \
    --filter_json "$VERSE_JSON" \
    --output_root "$BASE_OUT/random" "${GPU_ARGS[@]}" \
    --view_mode random --num_views $NUM_VIEWS_RANDOM \
    --resolution "$RESOLUTION" "${W_ARGS[@]}" 2>&1 | tee -a "$LOG"

log "==== random ALL DONE =="

# ── fixed_grid ────────────────────────────────────────────────────────────────
log "==== fixed_grid: GT mobility =="
$PYTHON render_adaptive.py \
    --dataset_roots "$GT_MOBILITY" \
    --filter_json "$MOB_JSON" \
    --output_root "$BASE_OUT/fixed_grid" "${GPU_ARGS[@]}" \
    --view_mode fixed_grid \
    --resolution "$RESOLUTION" "${W_ARGS[@]}" 2>&1 | tee -a "$LOG"

log "==== fixed_grid: mob OBJ =="
$PYTHON render_adaptive.py \
    --dataset_roots "${MOB_OBJ_ROOTS[@]}" \
    --filter_json "$MOB_JSON" \
    --output_root "$BASE_OUT/fixed_grid" "${GPU_ARGS[@]}" \
    --view_mode fixed_grid \
    --resolution "$RESOLUTION" "${W_ARGS[@]}" 2>&1 | tee -a "$LOG"

log "==== fixed_grid: mob GLB =="
$PYTHON render_adaptive.py \
    --dataset_roots "${MOB_GLB_ROOTS[@]}" \
    --filter_json "$MOB_JSON" \
    --output_root "$BASE_OUT/fixed_grid" "${GPU_ARGS[@]}" \
    --view_mode fixed_grid \
    --resolution "$RESOLUTION" "${W_ARGS[@]}" 2>&1 | tee -a "$LOG"

log "==== fixed_grid: GT verse =="
$PYTHON render_adaptive.py \
    --dataset_roots "$GT_VERSE" \
    --filter_json "$VERSE_JSON" \
    --output_root "$BASE_OUT/fixed_grid" "${GPU_ARGS[@]}" \
    --view_mode fixed_grid \
    --resolution "$RESOLUTION" "${W_ARGS[@]}" 2>&1 | tee -a "$LOG"

log "==== fixed_grid: verse OBJ =="
$PYTHON render_adaptive.py \
    --dataset_roots "${VERSE_OBJ_ROOTS[@]}" \
    --filter_json "$VERSE_JSON" \
    --output_root "$BASE_OUT/fixed_grid" "${GPU_ARGS[@]}" \
    --view_mode fixed_grid \
    --resolution "$RESOLUTION" "${W_ARGS[@]}" 2>&1 | tee -a "$LOG"

log "==== fixed_grid: verse GLB =="
$PYTHON render_adaptive.py \
    --dataset_roots "${VERSE_GLB_ROOTS[@]}" \
    --filter_json "$VERSE_JSON" \
    --output_root "$BASE_OUT/fixed_grid" "${GPU_ARGS[@]}" \
    --view_mode fixed_grid \
    --resolution "$RESOLUTION" "${W_ARGS[@]}" 2>&1 | tee -a "$LOG"

log "==== ALL DONE =="
