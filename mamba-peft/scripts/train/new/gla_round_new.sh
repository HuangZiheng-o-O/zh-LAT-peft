#!/bin/bash
set -euo pipefail

# Unified launcher for GLA LoRA experiments (dynamic rounds from Round_all).
# ❗️New behavior:
# - Read configs from Round_all=() and AUTO-SPLIT into rounds of size == NUM_GPUS
# - NUM_GPUS is auto-detected; must equal 7 (this host has 7 GPUs). If not 7, exit with an error.
# - Each round launches up to NUM_GPUS parallel jobs (one per GPU).
# Found the latest cached dataset configuration 'mrpc' at /home/user/mzs_h/data/hf_cache/nyu-mll___glue/mrpc/0.0.0/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c (last modified on Sun Sep 14 23:23:58 2025).
# Usage examples (same as before):
#   bash scripts/train/new/gla_round_new.sh 1
#   SEED=127 bash scripts/train/new/gla_round_new.sh 2
#   mamba-peft/scripts/train/new/gla_round_new.sh
#   bash mamba-peft/scripts/train/new/gla_round_new.sh all
#   bash scripts/train/new/gla_round_new.sh 3 1
#   bash /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new/gla_round_new.sh E1 all
#   bash /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new/gla_round_new.sh E2 2
#   bash /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new/gla_round_new.sh E2 all
#   bash /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new/gla_round_new.sh e4 all
#   bash /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new/gla_round_new.sh all
# Optional:
#   export GPU_IDS="0 1 2 3 4 5 6"   # Explicit GPU mapping; if set, its count must also be 7.
#   bash /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new/gla_tmux_nohup.sh --suite E2 --round all --seed 87 --data glue-tvt_mrpc
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
#ROUND_E1=(
#
#  # ======================
#  # 0. 统一基线（对照起点）
#  # ======================
#  "E1_QKVO_r8_alpha8.yaml"
#  "E2_OMLP_r8_alpha8.yaml"
#
#  # =====================================================
#  # 1. 纯 QKVO 容量扫描（主干：先 alpha@r=8，再等比 alpha=2r，再插值 r=10）
#  #    目标：建立容量-性能主干曲线，并插入 A/B 变体作为配对对照
#  # =====================================================
#
#  # 1.1 Alpha 扫描（r=8，改 alpha）
#  "E1_QKVO_r8_alpha12.yaml"
#  "E1_QKVO_r8_alpha16.yaml"        # 关键点：alpha=2r
#  "E1_QKVO_r8_alpha20.yaml"
#
#  # 1.2 Rank×Alpha 等比扩展（alpha=2r）
#  "E1_QKVO_r4_alpha8.yaml"
#  "E1_QKVO_r12_alpha24.yaml"
#  "E1_QKVO_r16_alpha32.yaml"
#
#  # 1.3 r=10 插值（配套 Base / DoRA / RSLoRA）
#  "E1_QKVO_r10_alpha16.yaml"
#  "E1_QKVO_DoRA_r10_alpha16.yaml"
#  "E1_QKVO_RSLoRA_r10_alpha16.yaml"
#  "E1_QKVO_r10_alpha20.yaml"
#  "E1_QKVO_DoRA_r10_alpha20.yaml"
#  "E1_QKVO_RSLoRA_r10_alpha20.yaml"
#
#  # 1.4 在强基线 r=12, a=24 上移植 DoRA / RSLoRA（补充配对，不重复 Base）
#  "E1_QKVO_DoRA_r12_alpha24.yaml"
#  "E1_QKVO_RSLoRA_r12_alpha24.yaml"
#
#  # ==========================================
#  # 2. 模块消融（r=8, a=8）
#  #    目标：分解增益来源（QKVO 基础上 +G/+GK/+MLP 及其组合）
#  # ==========================================
#  "E1_QKVO_plus_G_r8_alpha8.yaml"
#  "E1_QKVO_plus_GK_r8_alpha8.yaml"
#  "E1_QKVO_plus_MLP_r8_alpha8.yaml"
#  "E1_QKVO_plus_G_plus_GK_r8_alpha8.yaml"
#  "E1_QKVO_plus_G_plus_GK_plus_MLP_r8_alpha8.yaml"
#
#  # （用于 O-MLP 的对照消融，沿用 r=8, a=8）
#  "E4_OONLY_r8_alpha8.yaml"     # E4 系列但用于 E2 的消融对比
#  "E5_MLPONLY_r8_alpha8.yaml"   # E5 系列但用于 E2 的消融对比
#
#  # ============================================================
#  # 3. 模块 × Alpha 互作（固定 r=8；考察更优 alpha 区间的协同/激活）
#  # ============================================================
#  "E1_QKVO_plus_G_plus_GK_r8_alpha12.yaml"
#  "E1_QKVO_plus_G_plus_GK_r8_alpha16.yaml"   # +G+GK @ alpha=2r
#  "E1_QKVO_plus_MLP_r8_alpha16.yaml"         # 检查 MLP 在更高 alpha 是否显效
#
#  # （与 +GK 的增量价值复验；统一对齐到 alpha=16）
#  "E1_QKVO_plus_GK_r8_alpha16.yaml"
#  "E1_QKVO_plus_GK_DoRA_r8_alpha16.yaml"
#
#  # ==============================================
#  # 4. 层级定位（仅变“微调层范围”，容量固定 r=8, a=8）
#  # ==============================================
#  "E1_QKVO_first6_r8_alpha8.yaml"
#  "E1_QKVO_last6_r8_alpha8.yaml"
#  "E2_OMLP_last6_r8_alpha8.yaml"
#  "E2_OMLP_middle6_r8_alpha8.yaml"
#  "E1_QKVO_plus_GK_last6_r8_alpha8.yaml"   # 混杂/保留项移入层级定位，便于同类对照
#
#  # ====================================================
#  # 5. 算法 / 训练策略变体（在基线容量 r=8, a=8 下对齐比较）
#  # ====================================================
#  "E1_QKVO_DoRA_r8_alpha8.yaml"
#  "E1_QKVO_RSLoRA_r8_alpha8.yaml"
#  "E1_QKVO_lr1e-4_r8_alpha8.yaml"
#  "E1_QKVO_dropout0_r8_alpha8.yaml"
#  "E2_OMLP_DoRA_r8_alpha8.yaml"
#  "E2_OMLP_RSLoRA_r8_alpha8.yaml"
#  "E2_OMLP_dropout0_r8_alpha8.yaml"
#  "E2_OMLP_lr1e-4_r8_alpha8.yaml"
#
#  # ==========================================================
#  # 6. 冠军配置复验（a=16 视为更优容量点；含 a=8 的对照回看）
#  # ==========================================================
#  "E1_QKVO_DoRA_r8_alpha16.yaml"
#  "E1_QKVO_RSLoRA_r8_alpha16.yaml"
#  "E1_QKVO_plus_G_plus_GK_RSLoRA_r8_alpha8.yaml"  # 对照：同组合在 a=8 的表现
#  "E1_QKVO_plus_G_plus_GK_DORA_r8_alpha8.yaml"    # 对照：同组合在 a=8 的表现（DoRA）
#
#  # ============================================
#  # 7. 高容量对照（r=16 相关；与 r=8 系列对齐比较）
#  # ============================================
#  "E1_QKVO_plus_MLP_r16_alpha16.yaml"       # MLP 在更高容量下的复查
#  "E1_QKVO_plus_G_plus_GK_r16_alpha16.yaml"
#  "QKVO_plus_G_r16_a16.yaml"
#  "E1_QKVO_R16_r16_alpha16.yaml"            # 命名看似异常，保留原文件名
#
#  # ==========================================================
#  # 8. O-MLP 家族扫描与交互（E2 系列集中整理，避免分散）
#  # ==========================================================
#
#  # 8.1 Alpha 扫描（r=8）
#  "E2_OMLP_r8_alpha4.yaml"
#  "E2_OMLP_r8_alpha16.yaml"
#  "E2_OMLP_r8_alpha24.yaml"
#
#  # 8.2 Rank 扫描（策略 alpha=r）
#  "E2_OMLP_r4_alpha4.yaml"
#  "E2_OMLP_r6_alpha6.yaml"
#  "E2_OMLP_r16_alpha16.yaml"
#
#  # 8.3 与门控模块的交互（r=8, a=8）
#  "E2_OMLP_plus_G_r8_a8.yaml"
#  "E2_OMLP_plus_GK_r8_alpha8.yaml"
#  "E2_OMLP_plus_G_plus_GK_r8_alpha8.yaml"
#
#  # =================================================================
#  # 9. 统一容量的新打点集合（r=8, a=16）——贴合 GLA 结构的细粒度目标集
#  #    每条结构给 Base 与 DoRA 配对，便于最小充分对照
#  # =================================================================
#
#  # 9.1 Gating-only（g_proj, gk_proj[0], gk_proj[1]）
#  "E3_GATINGONLY_r8_alpha16.yaml"
#  "E3_GATINGONLY_DoRA_r8_alpha16.yaml"
#
#  # 9.2 QK-only（q_proj, k_proj）
#  "E6_QKONLY_r8_alpha16.yaml"
#  "E6_QKONLY_DoRA_r8_alpha16.yaml"
#
#  # 9.3 KV-only（k_proj, v_proj）
#  "E7_KVONLY_r8_alpha16.yaml"
#  "E7_KVONLY_DoRA_r8_alpha16.yaml"
#
#  # 9.4 Attn + Gating（q_proj, k_proj, g_proj, gk_proj[0], gk_proj[1]）
#  "E8_QK_plus_GATING_r8_alpha16.yaml"
#  "E8_QK_plus_GATING_DoRA_r8_alpha16.yaml"
#
#  # 9.5 O + Head（o_proj, lm_head）
#  "E9_OplusHEAD_r8_alpha16.yaml"
#  "E9_OplusHEAD_DoRA_r8_alpha16.yaml"
#)
#
# bash /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new/gla_round_new.sh E2 all


ROUND_E2=(
  "E1_QKVO_r8_alpha16.yaml"
  "E1_QKVO_r8_alpha12.yaml"
  "E1_QKVO_r8_alpha20.yaml"
  "E1_QKVO_r16_alpha32.yaml"
  "E1_QKVO_DoRA_r8_alpha16.yaml"
  "E1_QKVO_RSLoRA_r8_alpha16.yaml"
  "E1_QKVO_DoRA_r12_alpha24.yaml"
  "E1_QKVO_RSLoRA_r12_alpha24.yaml"
  "E1_QKVO_plus_G_r8_alpha16.yaml"
  "E1_QKVO_plus_GK_r8_alpha16.yaml"
  "E1_QKVO_plus_G_plus_GK_r8_alpha16.yaml"
  "E1_QKVO_plus_MLP_r8_alpha16.yaml"
  "E1_QKVO_first6_r8_alpha16.yaml"
  "E1_QKVO_last6_r8_alpha16.yaml"
  "E1_QKVO_dropout0_r8_alpha16.yaml"
  "E1_QKVO_wd0.01_r8_alpha16.yaml"
  "E2_OMLP_r8_alpha16.yaml"
)
ROUND_E3=(
  # Mixed budget: QKVO r8a16 main, Gates (g/gk) as auxiliaries
  "E3_QKVO_main_Gates_aux_r8a16_r2a2.yaml"
  "E3_QKVO_main_Gates_aux_r8a16_r4a4.yaml"
  "E3_QKVO_main_Gates_aux_r8a16_r4a8.yaml"

  # Mixed budget: add MLP as auxiliary (with/without Gates)
  "E3_QKVO_main_MLP_aux_r8a16_r4a8.yaml"
  "E3_QKVO_main_GatesMLP_aux_r8a16_r4a8.yaml"

  # Method variants at fixed mixed Gates r4a4
  "E3_QKVO_main_Gates_aux_RSLoRA_r8a16_r4a4.yaml"
  "E3_QKVO_main_Gates_aux_DoRA_r8a16_r4a4.yaml"

  # Diagnostic: add only G vs only GK on top of QKVO r8a16
  "E3_QKVO_plus_G_only_r4a4.yaml"
  "E3_QKVO_plus_GK_only_r4a4.yaml"
)
ROUND_E1=(

  # ======================
  # 0. 统一基线（多锚点）
  # ======================
  "E1_QKVO_r8_alpha16.yaml"
  "E1_QKVO_r8_alpha8.yaml"
  "E1_QKVO_r10_alpha16.yaml"
  "E2_OMLP_r8_alpha16.yaml"

  # =====================================================
  # 1. QKVO 容量主干
  # =====================================================

  # 1.1 r=8 的 α 扫描
  "E1_QKVO_r8_alpha12.yaml"
  "E1_QKVO_r8_alpha16.yaml"
  "E1_QKVO_r8_alpha20.yaml"
  # "E1_QKVO_r8_alpha24.yaml"                     # （推荐删除：高α边界信号稀疏，ROI偏低，先保 r8@{12,16,20}）

  # 1.2 等比扩展 α=2r
  "E1_QKVO_r4_alpha8.yaml"
  "E1_QKVO_r6_alpha12.yaml"                     # （一般可删除：与 r4/r8/r10/r12 的跨度价值重叠度高）
  "E1_QKVO_r8_alpha16.yaml"
  "E1_QKVO_r10_alpha20.yaml"
#  "E1_QKVO_r12_alpha24.yaml"

  # 1.3 方法在锚点上的横评
  "E1_QKVO_DoRA_r8_alpha16.yaml"
  "E1_QKVO_RSLoRA_r8_alpha16.yaml"
  # "E1_QKVO_DoRA_r10_alpha20.yaml"              # （一般可删除：已有人群在 r10@α=16 做方法对照，α=2r 版本可延后）
  # "E1_QKVO_RSLoRA_r10_alpha20.yaml"            # （一般可删除：同上）
  "E1_QKVO_DoRA_r12_alpha24.yaml"
  "E1_QKVO_RSLoRA_r12_alpha24.yaml"
  # "E1_QKVO_DoRA_r8_alpha8.yaml"                # （推荐删除：低α下方法差异小且质量偏低，先让容量主干收敛）
  # "E1_QKVO_RSLoRA_r8_alpha8.yaml"              # （推荐删除：同上）

  # ==========================================
  # 2. 模块消融（统一容量：r=8；α多点）
  # ==========================================
   "E1_QKVO_plus_G_r8_alpha8.yaml"              # （一般可删除：G 主看 a=16；a=8 的价值由 +G+GK@a=8 代表）
  # "E1_QKVO_plus_G_r8_alpha12.yaml"             # （一般可删除：与 a=16 结论高度一致时可并线）
  "E1_QKVO_plus_G_r8_alpha16.yaml"
  # "E1_QKVO_plus_G_r8_alpha20.yaml"             # （一般可删除：先聚焦 a=16 主干）

   "E1_QKVO_plus_GK_r8_alpha8.yaml"             # （一般可删除：GK 在低α不稳，优先看 a=16）
  # "E1_QKVO_plus_GK_r8_alpha12.yaml"            # （一般可删除）
  "E1_QKVO_plus_GK_r8_alpha16.yaml"
  # "E1_QKVO_plus_GK_r8_alpha20.yaml"            # （一般可删除）

  "E1_QKVO_plus_G_plus_GK_r8_alpha8.yaml"
  # "E1_QKVO_plus_G_plus_GK_r8_alpha12.yaml"     # （推荐删除：与 a=8/a=16 重叠，收敛信号稀薄）
  "E1_QKVO_plus_G_plus_GK_r8_alpha16.yaml"
  # "E1_QKVO_plus_G_plus_GK_r8_alpha20.yaml"     # （一般可删除）

  "E1_QKVO_plus_MLP_r8_alpha16.yaml"
  "E1_QKVO_plus_G_plus_GK_plus_MLP_r8_alpha16.yaml"

  # ============================================================
  # 3. 模块 × 方法 × 容量（小矩阵）
  # ============================================================
  "E1_QKVO_plus_G_DoRA_r8_alpha16.yaml"
  "E1_QKVO_plus_G_RSLoRA_r8_alpha16.yaml"
  "E1_QKVO_plus_GK_DoRA_r8_alpha16.yaml"
  "E1_QKVO_plus_GK_RSLoRA_r8_alpha16.yaml"
  "E1_QKVO_plus_G_plus_GK_DoRA_r8_alpha16.yaml"
  "E1_QKVO_plus_G_plus_GK_RSLoRA_r8_alpha16.yaml"
  # "E1_QKVO_plus_G_DoRA_r10_alpha16.yaml"       # （一般可删除：容量不变跨r复验可留到跨数据集阶段）
  # "E1_QKVO_plus_GK_DoRA_r10_alpha16.yaml"      # （一般可删除：同上）

  # ==============================================
  # 4. 层级定位（证明性，统一 α=16）
  # ==============================================
  "E1_QKVO_first6_r8_alpha16.yaml"
  "E1_QKVO_last6_r8_alpha16.yaml"
  "E2_OMLP_last6_r8_alpha16.yaml"
  "E2_OMLP_middle6_r8_alpha16.yaml"
  # "E1_QKVO_plus_GK_last6_r8_alpha16.yaml"      # （推荐删除：层位×模块的交互先避免混杂）

  # ====================================================
  # 5. 训练策略（与结构解耦）
  # ====================================================
  "E1_QKVO_dropout0_r8_alpha16.yaml"
  # "E1_QKVO_lr1e-4_r8_alpha16.yaml"             # （推荐删除：当前小集显著拉胯；跨数据集再开专轮更合适）
  "E1_QKVO_lr5e-5_r8_alpha16.yaml"
  "E1_QKVO_lr2e-4_r8_alpha16.yaml"
  "E1_QKVO_loradrop0.05_r8_alpha16.yaml"
  # "E1_QKVO_loradrop0.1_r8_alpha16.yaml"        # （一般可删除：与 0.05 高度重叠时可并线）
  "E1_QKVO_wd0.01_r8_alpha16.yaml"
  # "E1_QKVO_plus_G_plus_GK_lr1e-4_r8_alpha16.yaml"  # （推荐删除：策略×模块交互先收敛结构主干）

  # ============================================
  # 6. 高容量探针（集中于此）
  # ============================================
  # "E1_QKVO_r16_alpha24.yaml"                   # （一般可删除：r16 先固定 α=32 作上限探针）
  "E1_QKVO_r16_alpha32.yaml"
  # "E1_QKVO_DoRA_r16_alpha32.yaml"              # （一般可删除：高容量先确认是否必要，再做方法对照）
  # "E1_QKVO_RSLoRA_r16_alpha32.yaml"            # （一般可删除：同上）

  # ==========================================================
  # 7. O-MLP 家族（与主干对齐）
  # ==========================================================
  # 7.1 α 扫描（r=8）
  "E2_OMLP_r8_alpha8.yaml"
  "E2_OMLP_r8_alpha16.yaml"
  "E2_OMLP_r8_alpha24.yaml"

  # 7.2 Rank 扫描（α=2r）
  "E2_OMLP_r6_alpha12.yaml"
  "E2_OMLP_r12_alpha24.yaml"

  # 7.3 与门控交互（统一 r=8, a=16）
  "E2_OMLP_plus_G_r8_alpha16.yaml"
  "E2_OMLP_plus_GK_r8_alpha16.yaml"            # （一般可删除：优先验证 +G 与 +G+GK）
  "E2_OMLP_plus_G_plus_GK_r8_alpha16.yaml"

  # =================================================================
  # 8. 细粒度目标集（r=8,a=16 三方法横评 + 少量 α 反证）
  # =================================================================
  # 8.1 Gating-only
  "E3_GATINGONLY_r8_alpha16.yaml"
  "E3_GATINGONLY_DoRA_r8_alpha16.yaml"
  "E3_GATINGONLY_RSLoRA_r8_alpha16.yaml"

  # 8.2 QK-only
  "E6_QKONLY_r8_alpha16.yaml"
  "E6_QKONLY_DoRA_r8_alpha16.yaml"
  "E6_QKONLY_RSLoRA_r8_alpha16.yaml"

  # 8.3 KV-only
  "E7_KVONLY_r8_alpha16.yaml"
  "E7_KVONLY_DoRA_r8_alpha16.yaml"
  "E7_KVONLY_RSLoRA_r8_alpha16.yaml"

  # 8.4 QK + Gating
  "E8_QK_plus_GATING_r8_alpha16.yaml"
  "E8_QK_plus_GATING_DoRA_r8_alpha16.yaml"
  "E8_QK_plus_GATING_RSLoRA_r8_alpha16.yaml"

  # 8.5 O + Head
  "E9_OplusHEAD_r8_alpha16.yaml"
  "E9_OplusHEAD_DoRA_r8_alpha16.yaml"
  "E9_OplusHEAD_RSLoRA_r8_alpha16.yaml"

)

#: "${ROUND_E1[@]:-}" >/dev/null 2>&1 || declare -a ROUND_E1=()
# 可选占位：若后续需要支持 E3/E4/...，在此处定义各自的数组（可为空，脚本会自动跳过空数组）。
: "${ROUND_E3[@]:-}" >/dev/null 2>&1 || declare -a ROUND_E3=()
: "${ROUND_E4[@]:-}" >/dev/null 2>&1 || declare -a ROUND_E4=()
ROUND_E4=(
  # LoRA baseline (QKVO) and +G+GK
  "round4_QKVO_r8_a16_seed127.yaml"
  "round4_QKVO_plus_GK_r8_a16_seed127.yaml"
  # DoRA variants
  "round4_DoRA_QKVO_r8_a16_seed127.yaml"
  "round4_DoRA_QKVO_plus_GK_r8_a16_seed127.yaml"
  # RSLoRA variants
  "round4_RSLORA_QKVO_r8_a16_seed127.yaml"
  "round4_RSLORA_QKVO_plus_GK_r8_a16_seed127.yaml"
)
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
SEED="${SEED:-42}"     # informational only (NOT used for training)
FORCE_SEED=87         # actual seed used in training (HP_SEED). Ignore any seed elsewhere. FORCE_SEED=127 确实能够全局控制随机性，确保所有实验（除了数据集shuffle的固定种子外）都在相同的随机种子下运行。13 21 42 87 127
DATA="${DATA:-glue-tvt_cola}"  # injected dataset name (can override via env: DATA=AAA)

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
  local out="$outdir/$(basename "$src")"
  cp "$src" "$out"
  printf '\n# injected by gla_round_new.sh\ndata: %s\n' "$DATA" >>"$out"
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
      python train.py --cfg "$CFG_INJ" --overwrite &
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
      pkill -f -- "train.py --cfg ${EXP_ROOT}/" 2>/dev/null || true
    fi

    print_failure_summary
    exit 1
  fi
done

exit 0