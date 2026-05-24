#!/usr/bin/env bash
# run_cora_gate_activations.sh
#
# For each gate activation (sigmoid, tanh_shifted, relu, softplus, tanh_abs):
#   1. Train Cora symmetric   (best config from best_configs.md)
#   2. Train Cora asymmetric  (best config from best_configs.md)
#   3. Visualize sym vs asym  (heatmap + all extra diagnostic plots)
#
# At the end: cross-activation comparison panels (sym-only and asym-only).
#
# Usage:
#   bash run_cora_gate_activations.sh
#   PYTHON_BIN=python3 bash run_cora_gate_activations.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─── tunables ──────────────────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-cora_gate_activations}"
SEED=42
SPLIT_INDEX=0
EPOCHS=1000
PATIENCE=200
GATE_ACTIVATIONS=(sigmoid tanh_shifted relu softplus tanh_abs)

# ─── Cora best configs (best_configs.md, split 0, seed 42) ──────────────────
# symmetric:  hidden=64,  wl=5, layers=2, hop=3, C=32, dropout=0.2, lr=0.001
SYM_HIDDEN=64;  SYM_WL=5; SYM_LAYERS=2; SYM_HOP=3; SYM_C=32
SYM_DROPOUT=0.2; SYM_LR=0.001

# asymmetric: hidden=128, wl=5, layers=1, hop=3, C=32, dropout=0.0, lr=0.001
ASYM_HIDDEN=128; ASYM_WL=5; ASYM_LAYERS=1; ASYM_HOP=3; ASYM_C=32
ASYM_DROPOUT=0.0; ASYM_LR=0.001

# ─── helpers ────────────────────────────────────────────────────────────────

train_symmetric() {
    local act="$1"
    local outdir="${OUTPUT_ROOT}/symmetric_${act}"
    local pna_dir="${OUTPUT_ROOT}/npz/symmetric_${act}"

    echo "  [symmetric / ${act}] training..."
    "$PYTHON_BIN" NodeClassification_CPU.py \
        --dataset Cora \
        --split-index "$SPLIT_INDEX" \
        --seed "$SEED" \
        --epochs "$EPOCHS" \
        --patience "$PATIENCE" \
        --num-layers "$SYM_LAYERS" \
        --hop "$SYM_HOP" \
        --wl "$SYM_WL" \
        --dim_hidden "$SYM_HIDDEN" \
        --lr "$SYM_LR" \
        --dropout "$SYM_DROPOUT" \
        --numheads 1 \
        --GL_k 5 \
        --batch_size 32 \
        --kernels WL \
        --isgnn True \
        --cluster-wl-features \
        --n-clusters "$SYM_C" \
        --gate-activation "$act" \
        --save-pre-norm-attention \
        --pre-norm-attention-output-dir "$pna_dir" \
        --outdir "$outdir" \
        --skip-curves
    echo "  [symmetric / ${act}] done."
}

train_asymmetric() {
    local act="$1"
    local outdir="${OUTPUT_ROOT}/asymmetric_${act}"
    local pna_dir="${OUTPUT_ROOT}/npz/asymmetric_${act}"

    echo "  [asymmetric / ${act}] training..."
    "$PYTHON_BIN" NodeClassification_CPU.py \
        --dataset Cora \
        --split-index "$SPLIT_INDEX" \
        --seed "$SEED" \
        --epochs "$EPOCHS" \
        --patience "$PATIENCE" \
        --num-layers "$ASYM_LAYERS" \
        --hop "$ASYM_HOP" \
        --wl "$ASYM_WL" \
        --dim_hidden "$ASYM_HIDDEN" \
        --lr "$ASYM_LR" \
        --dropout "$ASYM_DROPOUT" \
        --numheads 1 \
        --GL_k 5 \
        --batch_size 32 \
        --kernels WL \
        --isgnn True \
        --cluster-wl-features \
        --n-clusters "$ASYM_C" \
        --asymmetric-gate \
        --gate-activation "$act" \
        --save-pre-norm-attention \
        --pre-norm-attention-output-dir "${OUTPUT_ROOT}/npz/asymmetric_${act}" \
        --outdir "$outdir" \
        --skip-curves
    echo "  [asymmetric / ${act}] done."
}

# Visualize sym vs asym for a given activation (heatmap + all extra plots).
visualize_pair() {
    local act="$1"
    local sym_npz="${OUTPUT_ROOT}/npz/symmetric_${act}/symmetric_seed_${SEED}_last_layer_pre_norm_attention.npz"
    local asym_npz="${OUTPUT_ROOT}/npz/asymmetric_${act}/asymmetric_seed_${SEED}_last_layer_pre_norm_attention.npz"
    local out_dir="${OUTPUT_ROOT}/plots/${act}"
    mkdir -p "$out_dir"

    echo "  [visualize / ${act}] symmetric vs asymmetric..."
    "$PYTHON_BIN" visualize_attention.py \
        --npz "$sym_npz" "$asym_npz" \
        --dataset Cora \
        --matrix-key pre_norm_attention_head_0 \
        --titles "Cora symmetric (${act})" "Cora asymmetric (${act})" \
        --out "${out_dir}/cora_sym_vs_asym_${act}.png" \
        --extra-plots all
    echo "  [visualize / ${act}] done."
}

# ─── main loop: for each activation, train both then visualize ────────────────
mkdir -p "${OUTPUT_ROOT}/npz" "${OUTPUT_ROOT}/plots"

for ACT in "${GATE_ACTIVATIONS[@]}"; do
    echo ""
    echo "════════════════════════════════════════════════"
    echo " Gate activation: ${ACT}"
    echo "════════════════════════════════════════════════"
    train_symmetric  "$ACT"
    train_asymmetric "$ACT"
    visualize_pair   "$ACT"
done

# ─── final cross-activation comparisons ──────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo " Final: all activations — symmetric only"
echo "════════════════════════════════════════════════"
SYM_NPZS=()
SYM_TITLES=()
for ACT in "${GATE_ACTIVATIONS[@]}"; do
    SYM_NPZS+=("${OUTPUT_ROOT}/npz/symmetric_${ACT}/symmetric_seed_${SEED}_last_layer_pre_norm_attention.npz")
    SYM_TITLES+=("sym / ${ACT}")
done
"$PYTHON_BIN" visualize_attention.py \
    --npz "${SYM_NPZS[@]}" \
    --dataset Cora \
    --matrix-key pre_norm_attention_head_0 \
    --titles "${SYM_TITLES[@]}" \
    --out "${OUTPUT_ROOT}/plots/cora_symmetric_all_activations.png" \
    --extra-plots class-agg entropy same-cross-class

echo ""
echo "════════════════════════════════════════════════"
echo " Final: all activations — asymmetric only"
echo "════════════════════════════════════════════════"
ASYM_NPZS=()
ASYM_TITLES=()
for ACT in "${GATE_ACTIVATIONS[@]}"; do
    ASYM_NPZS+=("${OUTPUT_ROOT}/npz/asymmetric_${ACT}/asymmetric_seed_${SEED}_last_layer_pre_norm_attention.npz")
    ASYM_TITLES+=("asym / ${ACT}")
done
"$PYTHON_BIN" visualize_attention.py \
    --npz "${ASYM_NPZS[@]}" \
    --dataset Cora \
    --matrix-key pre_norm_attention_head_0 \
    --titles "${ASYM_TITLES[@]}" \
    --out "${OUTPUT_ROOT}/plots/cora_asymmetric_all_activations.png" \
    --extra-plots class-agg entropy same-cross-class

echo ""
echo "All done. Results are in: ${OUTPUT_ROOT}/"
echo "  npz/        — raw .npz attention files"
echo "  plots/      — visualizations (one subfolder per activation + final comparisons)"
