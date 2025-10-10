#!/bin/bash
set -euo pipefail

# Unified launcher for GLA LoRA experiments (dynamic rounds from Round_all).
# ❗️New behavior:
# - Read configs from Round_all=() and AUTO-SPLIT into rounds of size == NUM_GPUS
# - NUM_GPUS is auto-detected; must equal 7 (this host has 7 GPUs). If not 7, exit with an error.
# - Each round launches up to NUM_GPUS parallel jobs (one per GPU).
#
# Usage examples (same as before):
#   bash scripts/train/new/gla_round_new.sh 1
#   TASK=rte SEED=127 bash scripts/train/new/gla_round_new.sh 2
#   mamba-peft/scripts/train/new/gla_round_new.sh
#   bash mamba-peft/scripts/train/new/gla_round_new.sh all
#   bash scripts/train/new/gla_round_new.sh 3 1
#   bash /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new/gla_round_new.sh E1 all
#
# Optional:
#   export GPU_IDS="0 1 2 3 4 5 6"   # Explicit GPU mapping; if set, its count must also be 7.

###############################################################################
#                               USER CONFIG HERE                              #
###############################################################################
# Master list of yaml filenames (seedless, relative to $CFG_DIR).
# 把你要跑的 YAML 全部写进这个数组即可；脚本会自动按每 7 个切一轮。
#Round_all=(
#  E0_ZS_r0_alpha0.yaml
#  E1_QKVO_DoRA_r8_alpha8.yaml
#  E1_QKVO_R16_r16_alpha16.yaml
#  E1_QKVO_RSLoRA_r8_alpha8.yaml
#  E1_QKVO_r8_alpha16.yaml
#  E1_QKVO_dropout0_r8_alpha8.yaml
#  E1_QKVO_first6_r8_alpha8.yaml
#  E1_QKVO_last6_r8_alpha8.yaml
#  E1_QKVO_lr1e-4_r8_alpha8.yaml
#  E1_QKVO_plus_GK_last6_r8_alpha8.yaml
#  E1_QKVO_r4_alpha8.yaml
#  E1_QKVO_r8_alpha8.yaml
#
#
#  E2_OMLP_DoRA_r8_alpha8.yaml
#  E2_OMLP_r8_alpha16.yaml
#  E2_OMLP_dropout0_r8_alpha8.yaml
#  E2_OMLP_last6_r8_alpha8.yaml
#  E2_OMLP_middle6_r8_alpha8.yaml
#  E2_OMLP_r6_alpha6.yaml
#  E2_OMLP_r8_alpha8.yaml
#  E3_QV_r8_alpha8.yaml
#  E4_OONLY_dropout0_r4_alpha4.yaml
#  E4_OONLY_r16_alpha16.yaml
#  E4_OONLY_r4_alpha4.yaml
#  E4_OONLY_r4_alpha8.yaml
#  E5_MLPONLY_r8_alpha8.yaml
#  E6_QKV_r8_alpha8.yaml
#  E7_GONLY_r4_alpha4.yaml
#  E7_GONLY_r8_alpha8.yaml
#  E8_QKVO_G_r8_alpha8.yaml
#  OMLP_plus_G_r8_a8.yaml
#  QKVO_plus_G_RSLoRA_r8_a8.yaml
#  QKVO_plus_G_r16_a16.yaml
#)
Round_all=()

# --- E1 Series: QKVO Fine-tuning Experiments ---
ROUND_NA=(
  # Baseline (Centerpiece of comparisons)
  "E1_QKVO_r8_alpha8.yaml"

  # --- Group 1: Capacity (Rank & Alpha) ---
  # Description: Evaluates the impact of LoRA's capacity.
  # Baseline: E1_QKVO_r8_alpha8.yaml
  "E1_QKVO_r4_alpha8.yaml"            # Lower rank
  "E1_QKVO_r8_alpha16.yaml"           # Higher alpha at same rank
  "E1_QKVO_R16_r16_alpha16.yaml"      # Higher rank and scaled alpha

  # --- Group 2: Core Target Modules ---
  # Description: Ablation study on which modules to fine-tune.
  # Baseline: E1_QKVO_r8_alpha8.yaml
  "E1_QKVO_plus_G_r8_alpha8.yaml"
  "E1_QKVO_plus_GK_r8_alpha8.yaml"
  "E1_QKVO_plus_MLP_r8_alpha8.yaml"
  "E1_QKVO_plus_G_plus_GK_r8_alpha8.yaml"
  "E1_QKVO_plus_G_plus_GK_plus_MLP_r8_alpha8.yaml"
  "QKVO_plus_G_r16_a16.yaml"          # Compares with E1_QKVO_R16_alpha16

  # --- Group 3: LoRA Variants & Training Strategy ---
  # Description: Compares different LoRA algorithms and training hyperparameters.
  # Baseline: E1_QKVO_r8_alpha8.yaml
  "E1_QKVO_DoRA_r8_alpha8.yaml"
  "E1_QKVO_RSLoRA_r8_alpha8.yaml"
  "E1_QKVO_lr1e-4_r8_alpha8.yaml"
  "E1_QKVO_dropout0_r8_alpha8.yaml"

  # --- Group 4: Layer Targeting ---
  # Description: Explores the effect of fine-tuning different layers.
  # Baseline: E1_QKVO_r8_alpha8.yaml (all layers)
  "E1_QKVO_first6_r8_alpha8.yaml"
  "E1_QKVO_last6_r8_alpha8.yaml"

  # --- Confounded Experiments (from original list) ---
  # Description: Kept for reference, but mix multiple variables.
  "E1_QKVO_plus_GK_last6_r8_alpha8.yaml"
)

ROUND_NA2=(

  # --- 实验方向 1：Alpha=2r 缩放策略检验 ---
  # 动机：发现 alpha=2r（r=8, a=16）有显著提升，验证该策略在其他配置下是否依然有效。
  "E1_QKVO_plus_G_plus_GK_r8_alpha16.yaml"   # 实验 1.1：最佳模块组合上应用 alpha=2r
  "E1_QKVO_r16_alpha32.yaml"                # 实验 1.2：高 Rank 下的 alpha=2r

  # --- 实验方向 2：RS-LoRA 与最佳模块组合 ---
  # 动机：RS-LoRA 是表现最好的变体，+G+GK 是最强模块组合，两者结合能否产生协同增益？
  "E1_QKVO_plus_G_plus_GK_RSLoRA_r8_alpha8.yaml"  # 实验 2.1

  # --- 实验方向 3：MLP 模块重新评估 ---
  # 动机：在 r=8, a=8 时 MLP 提升有限，测试在更优 alpha 或更高容量下是否“激活”其价值。
  "E1_QKVO_plus_MLP_r8_alpha16.yaml"        # 实验 3.1：MLP + alpha=2r
  "E1_QKVO_plus_MLP_r16_alpha16.yaml"       # 进一步测试高 Rank 下的 MLP 效果

  # --- 附加配置（补充探索） ---
  "E1_QKVO_plus_G_plus_GK_r16_alpha16.yaml"      # +G+GK 在高 Rank、scaled alpha=16
  "E1_QKVO_plus_G_plus_GK_DORA_r8_alpha8.yaml"   # DoRA 变体 + 最强模块组合
)


#!/usr/bin/env bash

ROUND_E1=(

  # --- 实验方向 1：精细化 Alpha 调优 & +G+GK 中间点探索 ---
  # 动机：在 alpha=8–16 之间绘制性能曲线，并探索中间地带 +G+GK 的相互作用关系。
  "E1_QKVO_r8_alpha12.yaml"                  # 实验 1.1: 探索 alpha=12 (r=8)
  "E1_QKVO_r8_alpha20.yaml"                  # 实验 1.2: 探索 alpha=20 (r=8)
  "E1_QKVO_plus_G_plus_GK_r8_alpha12.yaml"   # 实验 1.3: +G+GK @ alpha=12 (r=8)

  # --- 实验方向 2：在“alpha=2r”启发下重新审视 Rank ---
  # 动机：使用更优 alpha=2r 策略，绘制性能随 Rank 变化的曲线。
  # "E1_QKVO_r4_alpha8.yaml"    # 实验 2.1: r=4, alpha=8
  #"E1_QKVO_r8_alpha16.yaml"   # 实验 2.2: r=8, alpha=16
  "E1_QKVO_r12_alpha24.yaml"  # 实验 2.3: r=12, alpha=24
  #"E1_QKVO_r16_alpha32.yaml"  # 实验 2.4: r=16, alpha=32

  # --- 实验方向 3：在“冠军配置”上重新评估 LoRA 变体 ---
  # 动机：验证 DoRA 和 RS-LoRA 在 (r=8, alpha=16) 冠军配置下的表现。
  "E1_QKVO_DoRA_r8_alpha16.yaml"     # 实验 3.1: use_dora=true
  "E1_QKVO_RSLoRA_r8_alpha16.yaml"   # 实验 3.2: use_rslora=true

)
###############################################################################
#                           DO NOT EDIT BELOW UNLESS                          #
#                             YOU KNOW WHAT YOU DO                            #
###############################################################################

declare -a PIDS=()
declare -a COMPLETED_ROUNDS=()
declare -a RUN_QUEUE=()
declare -a DETECTED_GPUS=()

CURRENT_ROUND=""
FAILED_ROUND=""

print_interruption_summary() {
  echo ""
  echo "SUMMARY:"
  if (( ${#COMPLETED_ROUNDS[@]} > 0 )); then
    echo "  Experiments completed: ${COMPLETED_ROUNDS[*]}."
  else
    echo "  Experiments completed: none."
  fi
  if [[ -n "${CURRENT_ROUND:-}" ]]; then
    echo "  Experiment ${CURRENT_ROUND} exited abnormally (interrupted)."
  fi
}

print_failure_summary() {
  echo ""
  echo "SUMMARY:"
  if (( ${#COMPLETED_ROUNDS[@]} > 0 )); then
    echo "  Experiments completed: ${COMPLETED_ROUNDS[*]}."
  else
    echo "  Experiments completed: none."
  fi
  if [[ -n "${FAILED_ROUND:-}" ]]; then
    echo "  Experiment ${FAILED_ROUND} failed. Stopping."
  fi
}

cleanup() {
  for pid in "${PIDS[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
  sleep 1
  for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  sleep 1
  for pid in "${PIDS[@]}"; do kill -KILL "$pid" 2>/dev/null || true; done

  if [[ -n "${EXP_ROOT:-}" ]]; then
    pkill -f -- "train.py --cfg ${EXP_ROOT}/" 2>/dev/null || true
  fi

  print_interruption_summary
  exit 130
}
trap cleanup INT TERM

ROUND="${1:-1}"        # first arg kept for backward compat/docs; may be number or 'all'
TASK="${TASK:-cola}"   # informational only
SEED="${SEED:-42}"     # informational only (NOT used for training)
FORCE_SEED=127         # actual seed used in training (HP_SEED). Ignore any seed elsewhere.

# Remote workspace expected by train.py
PEFT_ROOT="/home/user/mzs_h/code/zh-LAT-peft/mamba-peft"
cd "$PEFT_ROOT"

# Env mirrors/caches (same as original)
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/home/user/mzs_h/data/hf_cache"
export HF_HUB_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export HF_EVALUATE_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export GLUE_METRIC_DIR="/home/user/mzs_h/data/hf_cache/eval_metrics/glue"
export HF_HUB_ENABLE_HF_TRANSFER=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_MODE=disabled
export WANDB_DISABLED=true
rm -rf ~/.config/wandb ~/.triton ~/.cache/torch_extensions || true

# ---------------------------------------------------------------------------
# Paths: split root and subdirs (YAML & JSON)
# ---------------------------------------------------------------------------
EXP_ROOT="${EXP_ROOT:-cfg/my_lora_exp}"     # root for this experiment family
CFG_DIR="${CFG_DIR:-${EXP_ROOT}/yaml}"      # YAML configs
PEFT_DIR="${PEFT_DIR:-${EXP_ROOT}/peft}"    # JSON assets (train.py uses as needed)

if [[ ! -d "$CFG_DIR" ]]; then
  echo "Config directory not found: $CFG_DIR" >&2
  exit 1
fi

# -------- GPU detection (must be 7 or exit) --------
parse_gpu_list() {
  # Normalize a space- or comma-separated list into DETECTED_GPUS array
  local s="${1:-}"
  s="${s//,/ }"
  DETECTED_GPUS=()
  for tok in $s; do
    [[ -n "$tok" ]] && DETECTED_GPUS+=("$tok")
  done
}

detect_gpus() {
  if [[ -n "${GPU_IDS:-}" ]]; then
    parse_gpu_list "$GPU_IDS"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    parse_gpu_list "$CUDA_VISIBLE_DEVICES"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    local cnt
    cnt="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    DETECTED_GPUS=()
    for ((i=0;i<cnt;i++)); do DETECTED_GPUS+=("$i"); done
  elif command -v rocm-smi >/dev/null 2>&1; then
    # Fallback for AMD: count "GPU" lines
    local cnt
    cnt="$(rocm-smi --showid 2>/dev/null | grep -E 'GPU\[|GPU' | wc -l | tr -d ' ')"
    DETECTED_GPUS=()
    for ((i=0;i<cnt;i++)); do DETECTED_GPUS+=("$i"); done
  else
    echo "ERROR: Could not detect GPUs (no GPU_IDS/CUDA_VISIBLE_DEVICES and nvidia-smi/rocm-smi missing)." >&2
    exit 1
  fi
}

detect_gpus
NUM_GPUS="${#DETECTED_GPUS[@]}"

# Hard requirement: server must expose exactly 7 GPUs.
if (( NUM_GPUS != 7 )); then
  echo "ERROR: Expected exactly 7 GPUs on this server, but detected ${NUM_GPUS}." >&2
  echo " - DETECTED_GPUS = ${DETECTED_GPUS[*]-<none>}" >&2
  echo "Troubleshooting:" >&2
  echo "  * If you intend to run on a subset/specific devices, set: export GPU_IDS=\"0 1 2 3 4 5 6\"" >&2
  echo "  * If CUDA_VISIBLE_DEVICES is set, ensure it lists 7 devices." >&2
  exit 1
fi

# -------------------------
# Suite selector (E1 only)
# -------------------------
if [[ "${1:-}" =~ ^(E1|e1)$ ]]; then
  SELECT_SUITE="E1"
  Round_all=("${ROUND_E1[@]}")   # replace master list with E1 subset
  shift                          # consume "E1"
  ROUND="${1:-all}"              # default to 'all' if no more args
else
  SELECT_SUITE="ALL"
fi

# -------- Dynamic round slicing from Round_all --------
if (( ${#Round_all[@]} == 0 )); then
  echo "ERROR: Round_all is empty. Please populate the Round_all array or call with 'E1' to use ROUND_E1." >&2
  exit 1
fi

# Number of dynamic rounds = ceil(len / NUM_GPUS)
TOTAL_CFGS="${#Round_all[@]}"
N_ROUNDS=$(( (TOTAL_CFGS + NUM_GPUS - 1) / NUM_GPUS ))

defined_rounds_str() {
  local out=""
  for ((r=1;r<=N_ROUNDS;r++)); do out+="${r} "; done
  printf "%s" "$out"
}

# Resolve a config entry to an absolute path under $CFG_DIR (no side effects!)
canonical_cfg_path() {
  local entry="$1"
  local path="${CFG_DIR}/${entry}"
  if [[ -f "$path" ]]; then
    printf '%s\n' "$path"; return 0
  else
    printf '%s\n' "$path"; return 1
  fi
}

# Build SELECT_SET for a given dynamic round r (1-based)
declare -a SELECT_SET=()
get_round_configs() {
  local r="$1"
  if (( r < 1 || r > N_ROUNDS )); then
    return 1
  fi
  local start=$(( (r-1)*NUM_GPUS ))
  local end=$(( r*NUM_GPUS ))
  if (( end > TOTAL_CFGS )); then end="$TOTAL_CFGS"; fi
  SELECT_SET=()
  local i
  for ((i=start;i<end;i++)); do
    SELECT_SET+=("${Round_all[i]}")
  done
  (( ${#SELECT_SET[@]} > 0 ))
}

run_round () {
  local r="$1"

  if ! get_round_configs "$r"; then
    echo "Round ${r} is empty or out of range. Valid rounds: $(defined_rounds_str)" >&2
    return 1
  fi

  # Resolve to absolute paths and verify existence
  local missing=()
  local -a RESOLVED_CFGS=()
  local resolved=""
  for f in "${SELECT_SET[@]}"; do
    if resolved="$(canonical_cfg_path "$f")"; then
      RESOLVED_CFGS+=("$resolved")
    else
      missing+=("$resolved")
    fi
  done
  if (( ${#missing[@]} > 0 )); then
    echo "Missing configs (expected under $CFG_DIR):" >&2
    printf '  %s\n' "${missing[@]}" >&2
    return 1
  fi

  local num_jobs="${#RESOLVED_CFGS[@]}"
  echo "=== Starting Round ${r} (${num_jobs} jobs; FORCE_SEED=${FORCE_SEED}; NUM_GPUS=${NUM_GPUS}) ==="
  echo "SUITE   = ${SELECT_SUITE}"
  echo "CFG_DIR = $CFG_DIR"
  echo "PEFT_DIR= $PEFT_DIR"
  echo "GPUs    = ${DETECTED_GPUS[*]}"

  # Choose GPU per job from detected list
  PIDS=()
  local i
  for i in "${!RESOLVED_CFGS[@]}"; do
    local CFG="${RESOLVED_CFGS[$i]}"
    local GPU="${DETECTED_GPUS[$i]}"
    echo "[GPU ${GPU}] ${CFG}  (HP_SEED=${FORCE_SEED}; ignoring seed in name/YAML)"
    HP_SEED=${FORCE_SEED} CUDA_VISIBLE_DEVICES="$GPU" \
      python train.py --cfg "$CFG" --overwrite &
    PIDS+=("$!")
  done

  local any_failed=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      any_failed=1
    fi
  done

  if (( any_failed )); then
    return 1
  fi

  echo "ROUND=${r} TASK=${TASK} SEED=${SEED} finished (all ran with HP_SEED=${FORCE_SEED})."
  return 0
}

# -------------------------
# Build the run queue
# -------------------------
if (( $# == 0 )); then
  if [[ "$ROUND" == "all" ]]; then
    for ((r=1;r<=N_ROUNDS;r++)); do RUN_QUEUE+=("$r"); done
  else
    RUN_QUEUE+=("$ROUND")
  fi
else
  for arg in "$@"; do
    if [[ "$arg" == "all" ]]; then
      for ((r=1;r<=N_ROUNDS;r++)); do RUN_QUEUE+=("$r"); done
    else
      RUN_QUEUE+=("$arg")
    fi
  done
fi

# Validate run queue
for r in "${RUN_QUEUE[@]}"; do
  if ! [[ "$r" =~ ^[0-9]+$ ]] || (( r < 1 || r > N_ROUNDS )); then
    echo "Invalid round '$r'. Valid values: $(defined_rounds_str) or 'all'." >&2
    exit 1
  fi
done

# -------------------------
# Execute strictly in order
# -------------------------
for r in "${RUN_QUEUE[@]}"; do
  CURRENT_ROUND="$r"
  if run_round "$r"; then
    COMPLETED_ROUNDS+=("$r")
    CURRENT_ROUND=""
  else
    FAILED_ROUND="$r"

    for pid in "${PIDS[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
    for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${PIDS[@]}"; do kill -KILL "$pid" 2>/dev/null || true; done
    if [[ -n "${EXP_ROOT:-}" ]]; then
      pkill -f -- "train.py --cfg ${EXP_ROOT}/" 2>/dev/null || true
    fi

    print_failure_summary
    exit 1
  fi
done

exit 0