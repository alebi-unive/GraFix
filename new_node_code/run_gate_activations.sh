#!/usr/bin/env bash
# run_gate_activations.sh
#
# For each dataset and gate activation defined in a JSON config:
#   1. Train symmetric   (best config from JSON)
#   2. Train asymmetric  (best config from JSON)
#   3. Visualize sym vs asym  (heatmap + all extra diagnostic plots)
#
# At the end of each dataset: cross-activation comparison panels
# (sym-only and asym-only).
#
# Usage:
#   bash run_gate_activations.sh [--config CONFIG_JSON]
#   PYTHON_BIN=python3 bash run_gate_activations.sh --config gate_activations_config.json

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─── argument parsing ─────────────────────────────────────────────────────────
CONFIG_JSON="gate_activations_config.json"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG_JSON="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ─── checks ───────────────────────────────────────────────────────────────────
if ! command -v jq &>/dev/null; then
    echo "Error: jq is required to parse the JSON config." >&2
    echo "       Install it with: apt-get install jq  (or brew install jq)" >&2
    exit 1
fi

if [[ ! -f "$CONFIG_JSON" ]]; then
    echo "Error: config file not found: $CONFIG_JSON" >&2
    exit 1
fi

# ─── tunables (overridable via env) ───────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-gate_activations}"

# ─── read common config ───────────────────────────────────────────────────────
SEED=$(jq -r       '.common.seed'        "$CONFIG_JSON")
SPLIT_INDEX=$(jq -r '.common.split_index' "$CONFIG_JSON")
EPOCHS=$(jq -r     '.common.epochs'      "$CONFIG_JSON")
PATIENCE=$(jq -r   '.common.patience'    "$CONFIG_JSON")
NUMHEADS=$(jq -r   '.common.numheads'    "$CONFIG_JSON")
GL_K=$(jq -r       '.common.GL_k'        "$CONFIG_JSON")
BATCH_SIZE=$(jq -r '.common.batch_size'  "$CONFIG_JSON")
KERNELS=$(jq -r    '.common.kernels'     "$CONFIG_JSON")

# Read gate activations list from JSON (fallback to defaults if missing)
mapfile -t GATE_ACTIVATIONS < <(jq -r '.gate_activations[]? // empty' "$CONFIG_JSON")
if [[ ${#GATE_ACTIVATIONS[@]} -eq 0 ]]; then
    GATE_ACTIVATIONS=(sigmoid tanh_shifted relu softplus tanh_abs)
fi

# ─── helpers ──────────────────────────────────────────────────────────────────

# train_model <dataset> <mode: symmetric|asymmetric> <gate_activation>
train_model() {
    local dataset="$1"
    local mode="$2"
    local act="$3"

    local hidden layers hop wl C dropout lr
    hidden=$(jq -r  ".datasets.\"${dataset}\".\"${mode}\".hidden"  "$CONFIG_JSON")
    layers=$(jq -r  ".datasets.\"${dataset}\".\"${mode}\".layers"  "$CONFIG_JSON")
    hop=$(jq -r     ".datasets.\"${dataset}\".\"${mode}\".hop"     "$CONFIG_JSON")
    wl=$(jq -r      ".datasets.\"${dataset}\".\"${mode}\".wl"      "$CONFIG_JSON")
    C=$(jq -r       ".datasets.\"${dataset}\".\"${mode}\".C"       "$CONFIG_JSON")
    dropout=$(jq -r ".datasets.\"${dataset}\".\"${mode}\".dropout" "$CONFIG_JSON")
    lr=$(jq -r      ".datasets.\"${dataset}\".\"${mode}\".lr"      "$CONFIG_JSON")

    local outdir="${OUTPUT_ROOT}/${dataset}/${mode}_${act}"
    local pna_dir="${OUTPUT_ROOT}/${dataset}/npz/${mode}_${act}"

    echo "  [${dataset} / ${mode} / ${act}] training..."

    local extra_args=()
    if [[ "$mode" == "asymmetric" ]]; then
        extra_args+=(--asymmetric-gate)
    fi

    "$PYTHON_BIN" NodeClassification_CPU.py \
        --dataset        "$dataset"      \
        --split-index    "$SPLIT_INDEX"  \
        --seed           "$SEED"         \
        --epochs         "$EPOCHS"       \
        --patience       "$PATIENCE"     \
        --num-layers     "$layers"       \
        --hop            "$hop"          \
        --wl             "$wl"           \
        --dim_hidden     "$hidden"       \
        --lr             "$lr"           \
        --dropout        "$dropout"      \
        --numheads       "$NUMHEADS"     \
        --GL_k           "$GL_K"         \
        --batch_size     "$BATCH_SIZE"   \
        --kernels        "$KERNELS"      \
        --isgnn          True            \
        --cluster-wl-features            \
        --n-clusters     "$C"            \
        "${extra_args[@]}"               \
        --gate-activation "$act"         \
        --save-pre-norm-attention        \
        --pre-norm-attention-output-dir  "$pna_dir" \
        --outdir         "$outdir"       \
        --skip-curves

    echo "  [${dataset} / ${mode} / ${act}] done."
}

# visualize_pair <dataset> <gate_activation>
# Plots symmetric vs asymmetric heatmap for a single activation.
visualize_pair() {
    local dataset="$1"
    local act="$2"
    local ds_lower="${dataset,,}"

    local sym_npz="${OUTPUT_ROOT}/${dataset}/npz/symmetric_${act}/symmetric_seed_${SEED}_last_layer_pre_norm_attention.npz"
    local asym_npz="${OUTPUT_ROOT}/${dataset}/npz/asymmetric_${act}/asymmetric_seed_${SEED}_last_layer_pre_norm_attention.npz"
    local out_dir="${OUTPUT_ROOT}/${dataset}/plots/${act}"
    mkdir -p "$out_dir"

    echo "  [${dataset} / visualize / ${act}] symmetric vs asymmetric..."
    "$PYTHON_BIN" visualize_attention.py \
        --npz          "$sym_npz" "$asym_npz"                          \
        --dataset      "$dataset"                                       \
        --matrix-key   pre_norm_attention_head_0                        \
        --titles       "${dataset} symmetric (${act})" "${dataset} asymmetric (${act})" \
        --out          "${out_dir}/${ds_lower}_sym_vs_asym_${act}.png"  \
        --extra-plots  all
    echo "  [${dataset} / visualize / ${act}] done."
}

# visualize_all_activations <dataset> <mode: symmetric|asymmetric>
# Plots a cross-activation comparison panel for one mode.
visualize_all_activations() {
    local dataset="$1"
    local mode="$2"
    local ds_lower="${dataset,,}"
    local out_dir="${OUTPUT_ROOT}/${dataset}/plots"

    local npzs=()
    local titles=()
    for act in "${GATE_ACTIVATIONS[@]}"; do
        npzs+=("${OUTPUT_ROOT}/${dataset}/npz/${mode}_${act}/${mode}_seed_${SEED}_last_layer_pre_norm_attention.npz")
        titles+=("${mode} / ${act}")
    done

    echo "  [${dataset} / cross-activation / ${mode}]..."
    "$PYTHON_BIN" visualize_attention.py \
        --npz         "${npzs[@]}"                                                    \
        --dataset     "$dataset"                                                       \
        --matrix-key  pre_norm_attention_head_0                                        \
        --titles      "${titles[@]}"                                                   \
        --out         "${out_dir}/${ds_lower}_${mode}_all_activations.png"             \
        --extra-plots class-agg entropy same-cross-class
    echo "  [${dataset} / cross-activation / ${mode}] done."
}

# ─── main loop ────────────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_ROOT"

mapfile -t DATASETS < <(jq -r '.datasets | keys[]' "$CONFIG_JSON")

for DATASET in "${DATASETS[@]}"; do
    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "  Dataset: ${DATASET}"
    echo "╚══════════════════════════════════════════════════════╝"

    mkdir -p "${OUTPUT_ROOT}/${DATASET}/npz" "${OUTPUT_ROOT}/${DATASET}/plots"

    # read the modes defined for this dataset
    mapfile -t MODES < <(jq -r ".datasets.\"${DATASET}\" | keys[]" "$CONFIG_JSON")

    # per-activation: train all modes then visualize the sym/asym pair
    for ACT in "${GATE_ACTIVATIONS[@]}"; do
        echo ""
        echo "  ── Gate activation: ${ACT} ──"
        for MODE in "${MODES[@]}"; do
            train_model "$DATASET" "$MODE" "$ACT"
        done
        # visualize only if both symmetric and asymmetric exist
        if [[ " ${MODES[*]} " == *" symmetric "* ]] && [[ " ${MODES[*]} " == *" asymmetric "* ]]; then
            visualize_pair "$DATASET" "$ACT"
        fi
    done

    # cross-activation comparison panel per mode
    echo ""
    echo "  ── Cross-activation comparisons ──"
    for MODE in "${MODES[@]}"; do
        visualize_all_activations "$DATASET" "$MODE"
    done
done

echo ""
echo "All done. Results are in: ${OUTPUT_ROOT}/"
echo "  <dataset>/npz/   — raw .npz attention files (one subdir per mode+activation)"
echo "  <dataset>/plots/ — visualizations (per-activation pairs + cross-activation panels)"
