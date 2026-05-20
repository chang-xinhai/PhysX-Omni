# =============================================================================
# 渲染流水线配置文件
# 换服务器时只需修改本文件，其余脚本无需改动
# 用法：bash run_render.sh
# =============================================================================

# ── 环境 ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

PYTHON="${PYTHON:-python3}"
export BLENDER_BIN="${BLENDER_BIN:-blender}"
export BLENDER_DEVICE="${BLENDER_DEVICE:-GPU}"

# ── 代码路径 ──────────────────────────────────────────────────────────────────
TOOLKIT="${SCRIPT_DIR}"

# ── 数据输入路径 ──────────────────────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/physx_result}"

MOB_JSON="${MOB_JSON:-${REPO_ROOT}/benchmark/assets/description/descript_mobility.json}"
VERSE_JSON="${VERSE_JSON:-${REPO_ROOT}/benchmark/assets/description/descript_verse.json}"
INTHEWILD_JSON="${INTHEWILD_JSON:-${REPO_ROOT}/benchmark/assets/description/descript_inthewild.json}"

# GT 数据集（OBJ/GLB，importer 自动处理坐标系，无需 rotate_x）
GT_MOBILITY=$DATA_ROOT/PhysX-Mobility/partseg
GT_VERSE=$DATA_ROOT/PhysXverse/physxverse/wholeglb

# Mobility OBJ 数据集（Blender OBJ importer 自动处理 Y-up→Z-up，无需 rotate_x）
MOB_OBJ_ROOTS=(
    $DATA_ROOT/ours_mobility_181500
    $DATA_ROOT/output_articulateanything_mobility
)

# Mobility GLB 数据集
MOB_GLB_ROOTS=(
    $DATA_ROOT/output_physxanything_mobility
    $DATA_ROOT/outputs_physxgen_mobility
)

# Verse OBJ 数据集（同上，无需 rotate_x）
VERSE_OBJ_ROOTS=(
    $DATA_ROOT/ours_verse_181500
    $DATA_ROOT/output_articulateanything_verse
)

# Verse GLB 数据集
VERSE_GLB_ROOTS=(
    $DATA_ROOT/output_physxanything_verse
    $DATA_ROOT/outputs_physxgen_verse
)

# In-the-wild result roots.
INTHEWILD_OBJ_ROOTS=(
    $DATA_ROOT/ours_inthewild_181500
    $DATA_ROOT/output_articulateanything_inthewild
)

INTHEWILD_GLB_ROOTS=(
    $DATA_ROOT/output_physxanything_inthewild
    $DATA_ROOT/outputs_physxgen_inthewild
)

# ── 输出路径 ──────────────────────────────────────────────────────────────────
BASE_OUT="${BASE_OUT:-${REPO_ROOT}/benchmark/benchmark_assets/rendered_views}"

# ── 渲染参数 ──────────────────────────────────────────────────────────────────
RESOLUTION=512
NUM_VIEWS_RANDOM=25
# Default to all 8 GPUs on this server. Override before sourcing if needed:
#   GPUS="0 1 2 3" bash run_render.sh
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"

WORKERS_SMALL=6
WORKERS_MEDIUM=4
WORKERS_LARGE=3
WORKERS_XLARGE=2
