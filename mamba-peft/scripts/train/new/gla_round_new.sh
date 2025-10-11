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
#!/usr/bin/env bash

# --- E1 Series: QKVO Fine-tuning Experiments (Unified & Reordered) ---
# 组织原则（控制变量）：
# 1) 以固定 (r, alpha) 的“基线”统一参照；
# 2) 单变量扫描：先在固定 rank 下扫描 alpha，再在 alpha=2r 策略下只改变 rank；
# 3) 仅改“目标模块”的消融；随后考察“模块×alpha”的互作；
# 4) 仅改“被微调层范围”的层级定位；
# 5) 仅改“算法/训练策略”的 LoRA 变体与超参；
# 6) 在“冠军配置”(r=8, a=16) 上复验算法与模块；
# 7) MLP 在更高容量下的复查；最后做高容量对照与混杂项保留。

ROUND_E1=(

  # --- 0. 基线（统一对照） ---
  # 控制变量：固定 r=8, alpha=8；后续组只改注释里的那一个因子。
  "E1_QKVO_r8_alpha8.yaml"

  # --- 1. Alpha 扫描（控制 rank=8，不改其它；只改变 alpha） ---
  # 目的：绘制 alpha 曲线，验证缩放效应；关注 alpha=2r（r=8→a=16）的关键点。
  "E1_QKVO_r8_alpha12.yaml"
  "E1_QKVO_r8_alpha16.yaml"        # 关键假设点：alpha=2r
  "E1_QKVO_r8_alpha20.yaml"

  # --- 2. Rank×Alpha 等比扩展（控制策略 alpha=2r；只改变 rank） ---
  # 目的：在等效缩放策略下( alpha=2r )观察容量-性能曲线：r=4/8/12/16 → a=8/16/24/32。
  "E1_QKVO_r4_alpha8.yaml"
  "E1_QKVO_r12_alpha24.yaml"
  "E1_QKVO_r16_alpha32.yaml"

  # --- 3. 模块消融（控制 r=8, a=8；只改变微调的目标模块） ---
  # 目的：QKVO 为基础，考察 +G、+GK、+MLP 及其组合的边际贡献。
  "E1_QKVO_plus_G_r8_alpha8.yaml"
  "E1_QKVO_plus_GK_r8_alpha8.yaml"
  "E1_QKVO_plus_MLP_r8_alpha8.yaml"
  "E1_QKVO_plus_G_plus_GK_r8_alpha8.yaml"
  "E1_QKVO_plus_G_plus_GK_plus_MLP_r8_alpha8.yaml"

  # --- 4. 模块 × Alpha 互作（控制模块或组合不变；在 r=8 下只改变 alpha） ---
  # 目的：在更优 alpha 区间检验模块组合是否被“激活”或产生协同。
  "E1_QKVO_plus_G_plus_GK_r8_alpha12.yaml"
  "E1_QKVO_plus_G_plus_GK_r8_alpha16.yaml"   # +G+GK @ alpha=2r
  "E1_QKVO_plus_MLP_r8_alpha16.yaml"         # 检查 MLP 是否在更高 alpha 下显效

  # --- 5. 层级定位（控制 r=8, a=8；只改变被微调的层范围） ---
  # 目的：定位增益主要来自前/后层还是全层。
  "E1_QKVO_first6_r8_alpha8.yaml"
  "E1_QKVO_last6_r8_alpha8.yaml"

  # --- 6. 算法/训练策略变体（控制 r=8, a=8；只改变算法或训练超参） ---
  # 目的：把算法因素与容量因素解耦；在基线容量下比较 DoRA、RS-LoRA、学习率、dropout。
  "E1_QKVO_DoRA_r8_alpha8.yaml"
  "E1_QKVO_RSLoRA_r8_alpha8.yaml"
  "E1_QKVO_lr1e-4_r8_alpha8.yaml"
  "E1_QKVO_dropout0_r8_alpha8.yaml"

  # --- 7. 冠军配置复验（控制 r=8, a=16；只改变算法或模块组合） ---
  # 目的：在更优的容量点复核算法与组合是否进一步放大收益。
  "E1_QKVO_DoRA_r8_alpha16.yaml"
  "E1_QKVO_RSLoRA_r8_alpha16.yaml"
  "E1_QKVO_plus_G_plus_GK_RSLoRA_r8_alpha8.yaml"  # 对照：同组合在 a=8 的表现
  "E1_QKVO_plus_G_plus_GK_DORA_r8_alpha8.yaml"    # 对照：同组合在 a=8 的表现（DoRA）

  # --- 8. MLP 在更高容量下的复查（控制模块=MLP；只改变 r 或 alpha） ---
  # 目的：评估容量提升是否“激活” MLP 的贡献。
  "E1_QKVO_plus_MLP_r16_alpha16.yaml"

  # --- 9. 高容量对照（r=16 相关；用于与 r=8 系列对齐比较） ---
  # 目的：在更大模型容量下对比基础与模块组合的可迁移性。
  "E1_QKVO_plus_G_plus_GK_r16_alpha16.yaml"
  "QKVO_plus_G_r16_a16.yaml"
  "E1_QKVO_R16_r16_alpha16.yaml"   # 命名看似异常（R16_r16）；请确认是否应为 "E1_QKVO_r16_alpha16.yaml"

  # --- 10. 混杂/保留项（多变量同时变化；仅作参考或 sanity check） ---
  "E1_QKVO_plus_GK_last6_r8_alpha8.yaml"
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
FORCE_SEED=83         # actual seed used in training (HP_SEED). Ignore any seed elsewhere. FORCE_SEED=127 确实能够全局控制随机性，确保所有实验（除了数据集shuffle的固定种子外）都在相同的随机种子下运行。13 21 42 87 127

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