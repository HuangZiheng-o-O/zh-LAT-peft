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
#   bash /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new/gla_round_new.sh E2 2
#   bash /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new/gla_round_new.sh E2 all
#   bash /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new/gla_round_new.sh e4 all
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
#  E2_OMLP_plus_G_r8_a8.yaml
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

ROUND_E2=(
  # --- 0. 基线 (统一对照) ---
  # 控制变量：固定 r=8, alpha=8, target=O-MLP
  "E2_OMLP_r8_alpha8.yaml"

  # --- 1. Alpha 扫描 (控制 Rank=8) ---
  # 目的：绘制 O-MLP 对 alpha 的敏感度曲线。
  "E2_OMLP_r8_alpha4.yaml"
  "E2_OMLP_r8_alpha16.yaml"
  "E2_OMLP_r8_alpha24.yaml"

  # --- 2. Rank 扫描 (控制 Alpha=Rank) ---
  # 目的：绘制 O-MLP 在 alpha=r 策略下的容量-性能曲线。
  "E2_OMLP_r4_alpha4.yaml"
  "E2_OMLP_r6_alpha6.yaml"
  "E2_OMLP_r16_alpha16.yaml"

  # --- 3. 模块消融 (控制 r=8, a=8) ---
  # 目的：分解 O-MLP，探究性能增益的核心来源。
  "E4_OONLY_r8_alpha8.yaml"  # 注意：这是 E4 系列，但用于 E2 的消融对比
  "E5_MLPONLY_r8_alpha8.yaml"  # 注意：这是 E5 系列，但用于 E2 的消融对比

  # --- 4. 与门控模块的交互作用 (控制 r=8, a=8) ---
  # 目的：测试 O-MLP 与 G/GK 门控微调的协同效应。
  "E2_OMLP_plus_G_r8_a8.yaml"
  "E2_OMLP_plus_GK_r8_alpha8.yaml"
  "E2_OMLP_plus_G_plus_GK_r8_alpha8.yaml"

  # --- 5. 算法/训练策略变体 (控制 r=8, a=8) ---
  # 目的：在 O-MLP 基线上评估不同 LoRA 变体和超参。
  "E2_OMLP_DoRA_r8_alpha8.yaml"
  "E2_OMLP_RSLoRA_r8_alpha8.yaml"
  "E2_OMLP_dropout0_r8_alpha8.yaml"
  "E2_OMLP_lr1e-4_r8_alpha8.yaml"

  # --- 6. 层级定位 (来自您原有的配置，作为补充) ---
  "E2_OMLP_last6_r8_alpha8.yaml"
  "E2_OMLP_middle6_r8_alpha8.yaml"
)

ROUND_E3=(
# ============================
# ROUND_E3  ——  第一次回复的实验清单（Core + 复核预留位）
# 设计宗旨：
#   * 仅新增、且与现有结果不重复；
#   * 控制变量严格：同一 (r, alpha) 下对比 Base / DoRA / RSLoRA；
#   * 聚焦“高概率优”的容量窗口（r10, α16/20；r12, α24）与 +GK 的小幅增益验证；
#   * Checkpoint 建议：~5k / ~7k / ~9k（必要时 ~10–12k）。
# 备注：
#   * seed 不在此处控制（由脚本 FORCE_SEED 统一），此数组仅列 YAML 文件名。
# ============================

  # ---------- A1. 纯 QKVO 的 rank/alpha 插值（r=10） ----------
  # 目标：在“中高 α”区间 (16/20) 插值 rank=10，并用 DoRA/RSLoRA 做配对对照
  # 控制：同一 seed、同一 alpha，对比 Base / DoRA / RSLoRA

  "E1_QKVO_r10_alpha16.yaml"         # 基线：QKVO, r=10, α=16
  "E1_QKVO_DoRA_r10_alpha16.yaml"    # 对照1：加 DoRA（方向-幅值分解）
  "E1_QKVO_RSLoRA_r10_alpha16.yaml"  # 对照2：加 RSLoRA（稀疏/随机子空间）

  "E1_QKVO_r10_alpha20.yaml"         # 基线：QKVO, r=10, α=20
  "E1_QKVO_DoRA_r10_alpha20.yaml"    # 对照1：r=10, α=20 + DoRA
  "E1_QKVO_RSLoRA_r10_alpha20.yaml"  # 对照2：r=10, α=20 + RSLoRA


  # ---------- A2. 将 DoRA / RSLoRA 移植到强基线 r12, α24 ----------
  # 目标：验证 DoRA / RSLoRA 在第二强“纯 QKVO”组合上的稳定增益
  # 控制：同一 (r, α)；Base 已存在，无需重复，仅补 DoRA/RSLoRA 版本

  "E1_QKVO_DoRA_r12_alpha24.yaml"    # QKVO, r=12, α=24 + DoRA
  "E1_QKVO_RSLoRA_r12_alpha24.yaml"  # QKVO, r=12, α=24 + RSLoRA


  # ---------- A3. 在强配置上验证 +GK 的“增量价值” ----------
  # 目标：对齐我们发现的 +GK 小幅稳定增益（~0.5%），并考察与 DoRA 的叠加
  # 控制：同一 (r, α)；对比 “带/不带 DoRA”的 +GK 增量

  "E1_QKVO_plus_GK_DoRA_r8_alpha16.yaml"  # r=8, α=16 +GK +DoRA（你尚未在 α16 上做过该组合）
  "E1_QKVO_plus_GK_r8_alpha16.yaml"       # r=8, α=16 +GK（无 DoRA，对照其纯增益）

# ============================
# （新家族/新打点组合，贴合 GLA 结构）
# 设计宗旨：
#   * 不与已跑过的 E1/E2 重复，转而探索更细粒度的“打点集合”；
#   * 每条结构给 Base 与 DoRA 配对（最小而有力的对照）；
#   * 统一容量：r=8, α=16（你的数据中该档位稳健），减少无关方差；
#   * 注释清楚标明目标层，便于审计与复现。
# ============================

  # ---------- E3：Gating-only（仅门控；GLA 特有） ----------
  # 目标层：g_proj（输出门 W_r）、gk_proj[0/1]（遗忘门 W_α^1 / W_α^2）
  # 动机：以极少参数直接调控 r_t 与 α_t 的动态，撬动记忆/遗忘机制
  "E3_GATINGONLY_r8_alpha16.yaml"          # Base：LoRA 仅作用于 {g_proj, gk_proj[0], gk_proj[1]}
  "E3_GATINGONLY_DoRA_r8_alpha16.yaml"     # 对照：同上 + DoRA


  # ---------- E6：QK-only（仅对齐核；不动 V/O/门控/FFN） ----------
  # 目标层：q_proj、k_proj
  # 动机：Q/K 决定相似度核；低秩更新改变对齐模式而不改写值通道
  "E6_QKONLY_r8_alpha16.yaml"              # Base：LoRA 仅作用于 {q_proj, k_proj}
  "E6_QKONLY_DoRA_r8_alpha16.yaml"         # 对照：同上 + DoRA


  # ---------- E7：KV-only（仅值与键；不动 Q/O/门控/FFN） ----------
  # 目标层：k_proj、v_proj
  # 动机：更贴近值路由/状态累积的通路，适合长序列/记忆型任务
  "E7_KVONLY_r8_alpha16.yaml"              # Base：LoRA 仅作用于 {k_proj, v_proj}
  "E7_KVONLY_DoRA_r8_alpha16.yaml"         # 对照：同上 + DoRA


  # ---------- E8：Attn+Gating（QK + 门控；不含 V/O/FFN） ----------
  # 目标层：q_proj、k_proj、g_proj、gk_proj[0]、gk_proj[1]
  # 动机：把“对齐核（QK）”与“门控（r_t、α_t）”贯通；参数远少于全 QKVO
  "E8_QK_plus_GATING_r8_alpha16.yaml"      # Base：LoRA 作用于 {q,k,g,gk0,gk1}
  "E8_QK_plus_GATING_DoRA_r8_alpha16.yaml" # 对照：同上 + DoRA


  # ---------- E9：O + Head（输出端适配；读出为主） ----------
  # 目标层：o_proj、lm_head
  # 动机：若任务偏“读出域适配”，仅输出侧的小参数更新或可足够
  "E9_OplusHEAD_r8_alpha16.yaml"           # Base：LoRA 仅作用于 {o_proj, lm_head}
  "E9_OplusHEAD_DoRA_r8_alpha16.yaml"      # 对照：同上 + DoRA
)


# 可选占位：若后续需要支持 E3/E4/...，在此处定义各自的数组（可为空，脚本会自动跳过空数组）。
#: "${ROUND_E3[@]:-}" >/dev/null 2>&1 || declare -a ROUND_E3=()
: "${ROUND_E4[@]:-}" >/dev/null 2>&1 || declare -a ROUND_E4=()
: "${ROUND_E5[@]:-}" >/dev/null 2>&1 || declare -a ROUND_E5=()
: "${ROUND_E6[@]:-}" >/dev/null 2>&1 || declare -a ROUND_E6=()
: "${ROUND_E7[@]:-}" >/dev/null 2>&1 || declare -a ROUND_E7=()
: "${ROUND_E8[@]:-}" >/dev/null 2>&1 || declare -a ROUND_E8=()
: "${ROUND_E9[@]:-}" >/dev/null 2>&1 || declare -a ROUND_E9=()
: "${ROUND_E10[@]:-}" >/dev/null 2>&1 || declare -a ROUND_E10=()

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
      pkill -f -- "train.py --cfg ${EXP_ROOT}/" 2>/dev/null || true
    fi

    print_failure_summary
    exit 1
  fi
done

exit 0