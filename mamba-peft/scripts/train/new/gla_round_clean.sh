#!/bin/bash
set -euo pipefail

LAUNCHER_PY="train_gla_only.py"

###############################################################################
#                               USER CONFIG HERE                              #
###############################################################################
: "${ROUND_E_MASTER[@]:-}" >/dev/null 2>&1 || declare -a ROUND_E_MASTER=()
ROUND_E_MASTER=( # 70 existing configs, grouped & non-duplicated

  ############################################################
  # 0. Anchor baselines & rank/alpha sweep (LoRA, QKVO backbone)
  #    —— 主干对照组：只动 QKVO，扫 rank / α，当作所有结构 / 方法的共同锚点
  ############################################################
  "E1_QKVO_r4_alpha8.yaml"
  "E1_QKVO_r8_alpha8.yaml"
  "E1_QKVO_r8_alpha16.yaml"   # 主 anchor：r=8, α=16
  "E1_QKVO_r16_alpha32.yaml"  # 高 rank 对照，给 DoRA / RSLoRA 的 “等效 rank” 参考

  ############################################################
  # 1. Structural ablations on QKVO (LoRA)
  #    —— 只改 “插哪里 / 插什么结构”，方法固定为 LoRA
  ############################################################
  # 1.1 α sweep @ r=8
#  "E1_QKVO_plus_G_r8_alpha8.yaml"
#    "E1_QKVO_plus_GK_r8_alpha8.yaml"
#      "E1_QKVO_plus_G_plus_GK_r8_alpha8.yaml"
  "E1_QKVO_plus_G_r8_alpha16.yaml"
  "E1_QKVO_plus_GK_r8_alpha16.yaml"
  "E1_QKVO_plus_G_plus_GK_r8_alpha16.yaml"

  # 1.2 Branch 结构
  "E1_QKVO_plus_MLP_r8_alpha16.yaml"
  "E1_QKVO_plus_G_plus_GK_plus_MLP_r8_alpha16.yaml"

  # 1.3 纯 G / GK 分支对照（只加门控，不加 QKVO）
  "E1_QKVO_plus_G_only_r8_alpha16.yaml"   # 只加 G 分支，在主 budget r=8 α=16
  "E1_QKVO_plus_GK_only_r8_alpha16.yaml"  # 只加 GK 分支，在主 budget r=8 α=16

  ############################################################
  # 2. O-MLP family (LoRA, aligned backbone)
  #    —— 和 QKVO 主干对齐的 O-MLP LoRA，对比 “改 attention 还是改 MLP”
  ############################################################
  # 2.1 α sweep @ r=8
  "E2_OMLP_r8_alpha8.yaml"
  "E2_OMLP_r8_alpha16.yaml"

  # 2.2 gating interactions @ r=8 α=16
  "E2_OMLP_plus_G_r8_alpha16.yaml"
  "E2_OMLP_plus_GK_r8_alpha16.yaml"
  "E2_OMLP_plus_G_plus_GK_r8_alpha16.yaml"

  # 2.3 （潜在缺失，对 O-MLP 做纯 G / 纯 GK，只占位）
  # "E2_OMLP_plus_G_only_r8_alpha16.yaml"   # MISSING: O-MLP 只接 G adapter
  # "E2_OMLP_plus_GK_only_r8_alpha16.yaml"  # MISSING: O-MLP 只接 GK adapter

  ############################################################
  # 3. Layer-wise localization (LoRA)
  #    —— 控制 “层位置” 这个变量：前几层 / 后几层 / 中间几层
  ############################################################
  "E1_QKVO_first6_r8_alpha16.yaml"
  "E1_QKVO_last6_r8_alpha16.yaml"
  "E2_OMLP_last6_r8_alpha16.yaml"
  "E2_OMLP_middle6_r8_alpha16.yaml"

  ############################################################
  # 4. Fine-grained target sets (LoRA)
  #    —— 精细化 “更新哪些子模块”：Gating-only / QK-only / KV-only / QK+Gating / O+Head
  ############################################################
  "E3_GATINGONLY_r8_alpha16.yaml"
  "E6_QKONLY_r8_alpha16.yaml"
  "E7_KVONLY_r8_alpha16.yaml"
  "E8_QK_plus_GATING_r8_alpha16.yaml"
  "E9_OplusHEAD_r8_alpha16.yaml"

  # 4.1 更细但缺失的目标集合（只占位，方便你以后补全）
  # "E4_QONLY_r8_alpha16.yaml"      # MISSING: 只更新 Q
  # "E4_KONLY_r8_alpha16.yaml"      # MISSING: 只更新 K
  # "E4_VONLY_r8_alpha16.yaml"      # MISSING: 只更新 V
  # "E4_OONLY_r8_alpha16.yaml"      # MISSING: 只更新 O
  # "E4_HEADONLY_r8_alpha16.yaml"   # MISSING: 只更新 output head，不动 O 本体
  # "E4_GKONLY_r8_alpha16.yaml"     # MISSING: gating 里只接 GK 不接 G 的版本

  ############################################################
  # 5. Method family: DoRA (fixed structure, compare method)
  #    —— 把结构固定在 QKVO 主干 / 结构 ablation / fine-grained，换成 DoRA
  ############################################################
  # 5.1 QKVO backbone: rank sweep with DoRA
  "E1_QKVO_DoRA_r8_alpha16.yaml"
  "E1_QKVO_DoRA_r12_alpha24.yaml"

  # 5.2 Structural: +G / +GK / +G+GK with DoRA
  "E1_QKVO_plus_G_DoRA_r8_alpha16.yaml"
  "E1_QKVO_plus_GK_DoRA_r8_alpha16.yaml"
  "E1_QKVO_plus_G_plus_GK_DoRA_r8_alpha16.yaml"

  # 5.3 Fine-grained target sets with DoRA
  "E3_GATINGONLY_DoRA_r8_alpha16.yaml"
  "E6_QKONLY_DoRA_r8_alpha16.yaml"
  "E7_KVONLY_DoRA_r8_alpha16.yaml"
  "E8_QK_plus_GATING_DoRA_r8_alpha16.yaml"
  "E9_OplusHEAD_DoRA_r8_alpha16.yaml"

  # 5.4 Mixed-budget (QKVO main, Gates as aux) with DoRA on aux path
  "E3_QKVO_main_Gates_aux_DoRA_r8a16_r4a4.yaml"

  # 5.5 round4: DoRA on standard QKVO / QKVO+GK
  "round4_DoRA_QKVO_r8_a16.yaml"
  "round4_DoRA_QKVO_plus_GK_r8_a16.yaml"

  ############################################################
  # 6. Method family: RSLoRA
  #    —— 完全平行的 RSLoRA 组，和上面 DoRA 可一一对比
  ############################################################
  # 6.1 QKVO backbone: rank sweep with RSLoRA
  "E1_QKVO_RSLoRA_r8_alpha16.yaml"
  "E1_QKVO_RSLoRA_r12_alpha24.yaml"

  # 6.2 Structural: +G / +GK / +G+GK with RSLoRA
  "E1_QKVO_plus_G_RSLoRA_r8_alpha16.yaml"
  "E1_QKVO_plus_GK_RSLoRA_r8_alpha16.yaml"
  "E1_QKVO_plus_G_plus_GK_RSLoRA_r8_alpha16.yaml"

  # 6.3 Fine-grained target sets with RSLoRA
  "E3_GATINGONLY_RSLoRA_r8_alpha16.yaml"
  "E6_QKONLY_RSLoRA_r8_alpha16.yaml"
  "E7_KVONLY_RSLoRA_r8_alpha16.yaml"
  "E8_QK_plus_GATING_RSLoRA_r8_alpha16.yaml"
  "E9_OplusHEAD_RSLoRA_r8_alpha16.yaml"

  # 6.4 Mixed-budget with RSLoRA on aux path
  "E3_QKVO_main_Gates_aux_RSLoRA_r8a16_r4a4.yaml"

  # 6.5 round4: RSLoRA on standard QKVO / QKVO+GK
  "round4_RSLORA_QKVO_r8_a16.yaml"
  "round4_RSLORA_QKVO_plus_GK_r8_a16.yaml"

  ############################################################
  # 7. Other method families on QKVO backbone
  #    —— PiSSA / LoRA-GA，相当于 “方法轴” 的补完
  ############################################################
  # LoRA baseline under round4 setting
  "round4_QKVO_r8_a16.yaml"
  "round4_QKVO_plus_GK_r8_a16.yaml"

  # PiSSA variants
  "round4_PISSA_QKVO_r8_a16.yaml"
  "round4_PISSA_QKVO_plus_GK_r8_a16.yaml"

  # LoRA-GA variants
  "round4_LORAGA_QKVO_r8_a16.yaml"
  "round4_LORAGA_QKVO_plus_GK_r8_a16.yaml"

  # （如果以后有更多方法，比如 AdaLoRA / VeRA，也可以在这里按同样 pattern 补）


  ############################################################
  # 8. Mixed-budget experiments (E3 family, LoRA)
  #    —— 主 budget 固定在 QKVO r=8 a=16，再加小 budget 给 Gates / MLP
  ############################################################
  "E3_QKVO_main_Gates_aux_r8a16_r2a2.yaml"
  "E3_QKVO_main_Gates_aux_r8a16_r4a4.yaml"
  "E3_QKVO_main_Gates_aux_r8a16_r4a8.yaml"
  "E3_QKVO_main_MLP_aux_r8a16_r4a8.yaml"
  "E3_QKVO_main_GatesMLP_aux_r8a16_r4a8.yaml"
  "E3_QKVO_plus_G_only_r4a4.yaml"
  "E3_QKVO_plus_GK_only_r4a4.yaml"

  # 8.1 混合 budget 里理论上也可以做 “only Q / only K / only V” 的小 adapter（先占位）
  # "E3_QKVO_main_Q_aux_r8a16_r4a4.yaml"   # MISSING: main 在 QKVO，aux 在 Q
  # "E3_QKVO_main_K_aux_r8a16_r4a4.yaml"   # MISSING
  # "E3_QKVO_main_V_aux_r8a16_r4a4.yaml"   # MISSING

  ############################################################
  # 9. Training-strategy ablations (LoRA, decoupled from structure)
  #    —— 结构固定 QKVO r=8 α=16，只改 training hyper
  ############################################################
  "E1_QKVO_dropout0_r8_alpha16.yaml"
  "E1_QKVO_wd0.01_r8_alpha16.yaml"

  # 9.1 你之前注释掉的 lr / loradrop 也可以在这里补上，如果 yaml 文件本身存在：
  # "E1_QKVO_lr5e-5_r8_alpha16.yaml"      # OPTIONAL: 低 lr 对照
  # "E1_QKVO_lr2e-4_r8_alpha16.yaml"      # OPTIONAL: 高 lr 对照
  # "E1_QKVO_loradrop0.05_r8_alpha16.yaml" # OPTIONAL: LoRA dropout ablation
)

# 只跑 “方法轴”：DoRA 系列
ROUND_DORA=(
  "E1_QKVO_DoRA_r8_alpha16.yaml"
  "E1_QKVO_DoRA_r12_alpha24.yaml"
  "E1_QKVO_plus_G_DoRA_r8_alpha16.yaml"
  "E1_QKVO_plus_GK_DoRA_r8_alpha16.yaml"
  "E1_QKVO_plus_G_plus_GK_DoRA_r8_alpha16.yaml"
  "E3_GATINGONLY_DoRA_r8_alpha16.yaml"
  "E6_QKONLY_DoRA_r8_alpha16.yaml"
  "E7_KVONLY_DoRA_r8_alpha16.yaml"
  "E8_QK_plus_GATING_DoRA_r8_alpha16.yaml"
  "E9_OplusHEAD_DoRA_r8_alpha16.yaml"
  "E3_QKVO_main_Gates_aux_DoRA_r8a16_r4a4.yaml"
  "round4_DoRA_QKVO_r8_a16.yaml"
  "round4_DoRA_QKVO_plus_GK_r8_a16.yaml"
)

# 只跑 “方法轴”：RSLoRA 系列
ROUND_RSLoRA=(
  "E1_QKVO_RSLoRA_r8_alpha16.yaml"
  "E1_QKVO_RSLoRA_r12_alpha24.yaml"
  "E1_QKVO_plus_G_RSLoRA_r8_alpha16.yaml"
  "E1_QKVO_plus_GK_RSLoRA_r8_alpha16.yaml"
  "E1_QKVO_plus_G_plus_GK_RSLoRA_r8_alpha16.yaml"
  "E3_GATINGONLY_RSLoRA_r8_alpha16.yaml"
  "E6_QKONLY_RSLoRA_r8_alpha16.yaml"
  "E7_KVONLY_RSLoRA_r8_alpha16.yaml"
  "E8_QK_plus_GATING_RSLoRA_r8_alpha16.yaml"
  "E9_OplusHEAD_RSLoRA_r8_alpha16.yaml"
  "E3_QKVO_main_Gates_aux_RSLoRA_r8a16_r4a4.yaml"
  "round4_RSLORA_QKVO_r8_a16.yaml"
  "round4_RSLORA_QKVO_plus_GK_r8_a16.yaml"
)


ROUND_E5=( #18
  # 0. Baselines (anchor)
  "E1_QKVO_r8_alpha16.yaml"

  # Structural ablations @ r=8 α=16
  "E1_QKVO_plus_G_r8_alpha16.yaml"
  "E1_QKVO_plus_GK_r8_alpha16.yaml"
  "E1_QKVO_plus_G_plus_GK_r8_alpha16.yaml"
  "E1_QKVO_plus_G_plus_GK_plus_MLP_r8_alpha16.yaml"

    #  O-MLP (α=16)
  "E2_OMLP_r8_alpha16.yaml"
  "E2_OMLP_plus_G_r8_alpha16.yaml"
  "E2_OMLP_plus_GK_r8_alpha16.yaml"
  "E2_OMLP_plus_G_plus_GK_r8_alpha16.yaml"

  # Layer-wise localization (α=16)
  "E1_QKVO_first6_r8_alpha16.yaml"
  "E1_QKVO_last6_r8_alpha16.yaml"
  "E2_OMLP_last6_r8_alpha16.yaml"
  "E2_OMLP_middle6_r8_alpha16.yaml"

  "E1_QKVO_plus_MLP_r8_alpha16.yaml"


  # 8.1 Fine-grained target sets (r=8 α=16)
  "E3_GATINGONLY_r8_alpha16.yaml"
  "E6_QKONLY_r8_alpha16.yaml"
  "E7_KVONLY_r8_alpha16.yaml"
#  "E8_QK_plus_GATING_r8_alpha16.yaml"
  "E9_OplusHEAD_r8_alpha16.yaml"

  # 3. Module × Method (DoRA only) @ r=8 α=16
#  "E1_QKVO_plus_G_DoRA_r8_alpha16.yaml"
#  "E1_QKVO_plus_GK_DoRA_r8_alpha16.yaml"
#  "E1_QKVO_plus_G_plus_GK_DoRA_r8_alpha16.yaml"

  # 8.2 DoRA only (r=8 α=16)
#  "E3_GATINGONLY_DoRA_r8_alpha16.yaml"
#  "E6_QKONLY_DoRA_r8_alpha16.yaml"
#  "E7_KVONLY_DoRA_r8_alpha16.yaml"
#  "E8_QK_plus_GATING_DoRA_r8_alpha16.yaml"
#  "E9_OplusHEAD_DoRA_r8_alpha16.yaml"
)

ROUND_E51=( #18
  "E1_QKVO_plus_G_r8_alpha16.yaml"
  "E1_QKVO_plus_GK_r8_alpha16.yaml"
  "E2_OMLP_plus_G_r8_alpha16.yaml"
  "E2_OMLP_plus_GK_r8_alpha16.yaml"
  "E2_OMLP_plus_G_plus_GK_r8_alpha16.yaml"
  "E1_QKVO_plus_MLP_r8_alpha16.yaml"


)

ROUND_E52=( #18
  "E1_QKVO_r8_alpha16.yaml"
  "E1_QKVO_first6_r8_alpha16.yaml"
  "E1_QKVO_last6_r8_alpha16.yaml"
  "E2_OMLP_last6_r8_alpha16.yaml"
  "E2_OMLP_middle6_r8_alpha16.yaml"
  "E3_GATINGONLY_r8_alpha16.yaml"
  "E6_QKONLY_r8_alpha16.yaml"
  "E7_KVONLY_r8_alpha16.yaml"
  "E9_OplusHEAD_r8_alpha16.yaml"
)

#####################################################################
#                           Sensitive Code                          #
#####################################################################

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
    pkill -f -- "${LAUNCHER_PY} --cfg ${EXP_ROOT}/" 2>/dev/null || true
  fi

  # Best-effort email notification for interruption (controlled by env, default on; does not kill anything, only notifies once)
  _email_interrupt="${SWANLAB_EMAIL_ON_INTERRUPT:-1}"
  if [[ "${_email_interrupt}" == "1" ]] && command -v python >/dev/null 2>&1; then
    (
      cd "$(dirname "$0")/.." || true
      python -m scripts.utils.email_notify \
        --event INTERRUPTED \
        --group "suite=${SELECT_SUITE} round=${CURRENT_ROUND} data=${DATA}" \
        --yaml "${SWANLAB_EMAIL_YAML:-}" >/dev/null 2>&1 || true
    )
  fi

  print_interruption_summary
  exit 130
}
trap cleanup INT TERM

ROUND="${1:-1}"        # first arg kept for backward compat/docs; may be number or 'all'
SEED="${SEED:-42}"     # informational only (NOT used for training)
FORCE_SEED=87         # actual seed used in training (HP_SEED). Ignore any seed elsewhere. FORCE_SEED=127 确实能够全局控制随机性，确保所有实验（除了数据集shuffle的固定种子外）都在相同的随机种子下运行。13 21 42 87 127
DATA="${DATA:-glue-tvt_cola}"  # injected dataset name (can override via env: DATA=AAA)

# Remote workspace expected by train_gla_only.py
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

# ---- Echo invocation & key env overrides (for reproducible logs) ----
echo "CMD: $0 $*"
echo "ENV_OVERRIDES:"
for _k in \
  GPU_IDS \
  GPU_PLAN \
  CUDA_VISIBLE_DEVICES \
  DATA \
  SPIDER_LOCAL_DIR \
  NLTK_DATA \
    GLA_FORCE_LEFT_PAD GLA_USE_MAX_NEW_TOKENS GLA_VERBOSE \
  EVAL_GEN EVAL_GEN_MAX_LENGTH EVAL_GEN_MIN_LENGTH EVAL_GEN_NUM_BEAMS \
  PYTORCH_CUDA_ALLOC_CONF TOKENIZERS_PARALLELISM OMP_NUM_THREADS MKL_NUM_THREADS \
  GRADIENT_CHECKPOINTING \
  LOGITS_TO_KEEP \
  NUM_DATA_WORKERS \
  FORCE_SEED \
  SEED \
  HP_DATA HP_BATCH_SIZE HP_LR HP_EPOCHS HP_PREC HP_SEED \
  HP_PEFT_R HP_PEFT_ALPHA HP_PEFT_DROPOUT HP_INIT HP_PISSA_FAST \
  HP_MAX_STEPS HP_EVAL_STEPS HP_SAVE_STEPS HP_LOGGING_STEPS \
  HP_LORAGA_BATCH_SIZE HP_LORAGA_STEPS HP_LORAGA_LAYERWISE HP_LORAGA_STABLE_C
do
  v="${!_k-}"
  if [[ -n "${v:-}" ]]; then
    echo "  ${_k}=${v}"
  fi
done
if command -v env >/dev/null 2>&1; then
  echo "HP_* (all):"; env | grep -E '^HP_' | sort || true
fi

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

if (( NUM_GPUS < 1 )); then
  echo "ERROR: No GPUs detected (after considering GPU_IDS/CUDA_VISIBLE_DEVICES)." >&2
  exit 1
fi

# -------------------------
# Per-GPU concurrency plan
# -------------------------
# GPU_PLAN: comma/space separated integers per detected GPU, e.g. "3,3,3,3,0,3,3".
# - If unset: default to 1 slot per detected GPU (previous behavior)
# - If single integer provided: broadcast to all GPUs
# - If length matches NUM_GPUS: use as-is
# - Otherwise: error
GPU_PLAN_STR="${GPU_PLAN:-}"
declare -a GPU_PLAN_ARR=()
if [[ -z "$GPU_PLAN_STR" ]]; then
  for _ in "${DETECTED_GPUS[@]}"; do GPU_PLAN_ARR+=(1); done
else
  # normalize separators to spaces
  GPU_PLAN_STR="${GPU_PLAN_STR//,/ }"
  read -r -a GPU_PLAN_ARR <<<"$GPU_PLAN_STR"
  if (( ${#GPU_PLAN_ARR[@]} == 1 )); then
    val="${GPU_PLAN_ARR[0]}"; GPU_PLAN_ARR=()
    for _ in "${DETECTED_GPUS[@]}"; do GPU_PLAN_ARR+=("$val"); done
  elif (( ${#GPU_PLAN_ARR[@]} != NUM_GPUS )); then
    echo "ERROR: GPU_PLAN length (${#GPU_PLAN_ARR[@]}) must be 1 or equal to number of detected GPUs (${NUM_GPUS})." >&2
    echo " - DETECTED_GPUS = ${DETECTED_GPUS[*]}" >&2
    echo " - GPU_PLAN      = ${GPU_PLAN_STR}" >&2
    exit 1
  fi
fi

# Build GPU_SLOTS by repeating each GPU id according to its concurrency
declare -a GPU_SLOTS=()
for i in "${!DETECTED_GPUS[@]}"; do
  gpu="${DETECTED_GPUS[$i]}"
  cnt="${GPU_PLAN_ARR[$i]}"
  # treat non-positive as zero
  if [[ -z "$cnt" || "$cnt" -le 0 ]]; then cnt=0; fi
  for ((j=0;j<cnt;j++)); do GPU_SLOTS+=("$gpu"); done
done
N_SLOTS="${#GPU_SLOTS[@]}"
if (( N_SLOTS < 1 )); then
  echo "ERROR: Effective parallel slots is zero (GPU_PLAN all zeros?)." >&2
  echo " - DETECTED_GPUS = ${DETECTED_GPUS[*]}" >&2
  echo " - GPU_PLAN      = ${GPU_PLAN_ARR[*]}" >&2
  exit 1
fi

# =========================
# Suite selector (E1..E10)
# =========================
SELECT_SUITE="ALL"

append_suite_into_master() {
  local var="$1"
  if eval "[[ -v ${var} && \${#${var}[@]} -gt 0 ]]"; then
    local tmp=( $(eval "printf '%q ' \"\${${var}[@]}\"") )
    if ((${#tmp[@]} > 0)); then
      read -r -a tmp <<<"$(eval "printf '%s ' \"\${${var}[@]}\"")"
      Round_all+=("${tmp[@]}")
    fi
  fi
}

if [[ "${1:-}" =~ ^([Ee][0-9]+)$ ]]; then
  suite="${BASH_REMATCH[1]}"
  suite="${suite^^}"
  SELECT_SUITE="$suite"
  shift

  varname="ROUND_${suite}"
  if ! eval "[[ -v ${varname} && \${#${varname}[@]} -gt 0 ]]"; then
    echo "ERROR: Suite '${suite}' is not defined or empty. Please define ${varname}=() with configs." >&2
    exit 1
  fi

  Round_all=()
  append_suite_into_master "${varname}"
  ROUND="${1:-all}"
else
  if (( ${#Round_all[@]} == 0 )); then
    Round_all=()
    for i in {1..10}; do
      append_suite_into_master "ROUND_E${i}"
    done
    if (( ${#Round_all[@]} == 0 )); then
      echo "ERROR: No configs found. Either populate Round_all=() manually, or pass a suite like 'E1', 'E2', ...," >&2
      echo "       or define the corresponding ROUND_E* arrays." >&2
      exit 1
    fi
  fi
fi
# -------- Dynamic round slicing from Round_all --------
# Number of dynamic rounds = ceil(len / N_SLOTS)
TOTAL_CFGS="${#Round_all[@]}"
N_ROUNDS=$(( (TOTAL_CFGS + N_SLOTS - 1) / N_SLOTS ))

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
  local start=$(( (r-1)*N_SLOTS ))
  local end=$(( r*N_SLOTS ))
  if (( end > TOTAL_CFGS )); then end="$TOTAL_CFGS"; fi
  SELECT_SET=()
  local i
  for ((i=start;i<end;i++)); do
    SELECT_SET+=("${Round_all[i]}")
  done
  (( ${#SELECT_SET[@]} > 0 ))
}

# Create a temp YAML with injected data: <DATA>, leaving original YAML intact.
make_tmp_cfg_with_data() {
  local src="$1"; local outdir="$2"
  local base
  base="$(basename "$src")"
  local name ext
  name="${base%.*}"; ext="${base##*.}"
  local out
  out="$outdir/${name}.${ext}"
  # Ensure unique filename if duplicates exist in the same round
  if [[ -e "$out" ]]; then
    local k=1
    while :; do
      local cand="$outdir/${name}__rep${k}.${ext}"
      if [[ ! -e "$cand" ]]; then out="$cand"; break; fi
      k=$((k+1))
    done
  fi
  cp "$src" "$out"
  printf '\n# injected by gla_round_clean.sh\ndata: %s\n' "$DATA" >>"$out"
  # Highest priority num_data_workers injection (default 8 if unset)
  local ndw
  ndw="${NUM_DATA_WORKERS:-8}"
  printf 'num_data_workers: %s\n' "$ndw" >>"$out"
  # Optional gradient checkpointing (enable only when explicitly set truthy)
  if [[ -n "${GRADIENT_CHECKPOINTING:-}" ]]; then
    case "${GRADIENT_CHECKPOINTING,,}" in
      1|true|yes|on)
        printf 'gradient_checkpointing: true\n' >>"$out"
        ;;
    esac
  fi
  # Optional logits_to_keep (only if provided)
  if [[ -n "${LOGITS_TO_KEEP:-}" ]]; then
    printf 'logits_to_keep: %s\n' "$LOGITS_TO_KEEP" >>"$out"
  fi
  printf '%s\n' "$out"
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
  echo "=== Starting Round ${r} (${num_jobs} jobs; FORCE_SEED=${FORCE_SEED}; NUM_GPUS=${NUM_GPUS}; N_SLOTS=${N_SLOTS}) ==="
  echo "SUITE   = ${SELECT_SUITE}"
  echo "CFG_DIR = $CFG_DIR"
  echo "PEFT_DIR= $PEFT_DIR"
  echo "GPUs    = ${DETECTED_GPUS[*]}"
  echo "PLAN    = ${GPU_PLAN_ARR[*]}  (GPU->slots)"
  echo "SLOTS   = ${GPU_SLOTS[*]}     (flattened)"
  echo "DATA    = ${DATA}"
  echo "SPIDER_LOCAL_DIR = ${SPIDER_LOCAL_DIR:-}"
  echo "NLTK_DATA        = ${NLTK_DATA:-}"
  # Round timing (start)
  local __round_start_epoch
  __round_start_epoch="$(date +%s)"
  local __round_start_iso
  __round_start_iso="$(date +%F_%T)"
  echo "[${__round_start_iso}] ROUND=${r} START"

  # Choose GPU per job from detected list
  PIDS=()
  local i
  # Make a temp dir for this round's YAML copies with injected data
  local TMP_CFG_DIR
  TMP_CFG_DIR="$(mktemp -d /tmp/gla_data_XXXXXX)"
  for i in "${!RESOLVED_CFGS[@]}"; do
    local CFG="${RESOLVED_CFGS[$i]}"
    # choose slot by index cycling when fewer jobs than slots
    local slot_index=$(( i % N_SLOTS ))
    local GPU="${GPU_SLOTS[$slot_index]}"
    local CFG_INJ
    CFG_INJ="$(make_tmp_cfg_with_data "$CFG" "$TMP_CFG_DIR")"
    echo "[GPU ${GPU}] ${CFG_INJ}  (HP_SEED=${FORCE_SEED}; data=${DATA}; ignoring seed in name/YAML)"
    HP_SEED=${FORCE_SEED} CUDA_VISIBLE_DEVICES="$GPU" \
      python "$LAUNCHER_PY" --cfg "$CFG_INJ" --overwrite &
    PIDS+=("$!")
  done

  local any_failed=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      any_failed=1
    fi
  done

  # cleanup temp dir
  rm -rf "$TMP_CFG_DIR" || true

  # Round timing (end)
  local __round_end_epoch
  __round_end_epoch="$(date +%s)"
  local __round_end_iso
  __round_end_iso="$(date +%F_%T)"
  local __round_elapsed
  __round_elapsed=$(( __round_end_epoch - __round_start_epoch ))
  local __round_h
  local __round_m
  local __round_s
  __round_h=$(( __round_elapsed / 3600 ))
  __round_m=$(( (__round_elapsed % 3600) / 60 ))
  __round_s=$(( __round_elapsed % 60 ))
  printf '[%s] ROUND=%s END elapsed=%02d:%02d:%02d (%ds)\n' "${__round_end_iso}" "${r}" "${__round_h}" "${__round_m}" "${__round_s}" "${__round_elapsed}"

  if (( any_failed )); then
    return 1
  fi

  echo "ROUND=${r} finished (all ran with HP_SEED=${FORCE_SEED})."
  return 0
}

# -------------------------
# Build the run queue
# -------------------------
if (( $# == 0 )); then
  if [[ "${ROUND:-}" == "all" ]]; then
    for ((r=1;r<=N_ROUNDS;r++)); do RUN_QUEUE+=("$r"); done
  else
    RUN_QUEUE+=("${ROUND:-1}")
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
      pkill -f -- "${LAUNCHER_PY} --cfg ${EXP_ROOT}/" 2>/dev/null || true
    fi

    print_failure_summary
    exit 1
  fi
done

exit 0