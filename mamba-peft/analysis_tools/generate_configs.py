import os
import json
import re
from pathlib import Path

# ======================================================================================
# 脚本配置区域 (Script Configuration Area)
# ======================================================================================

# 1. 定义文件输出的基础路径
# 请确保此路径正确指向您的 `my_lora_exp` 目录
BASE_PATH = "/Users/huangziheng/PycharmProjects/code/zh-LAT-peft/mamba-peft/cfg/my_lora_exp"
PEFT_PATH = os.path.join(BASE_PATH, "peft")
YAML_PATH = os.path.join(BASE_PATH, "yaml")

# 2. 定义可复用的目标模块“积木”
# 这是所有配置正确性的核心，确保了模块组合的精确性
TARGETS = {
    "QKVO": ["attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj"],
    "OMLP": ["attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
    "G": ["attn.g_proj"],
    "GK": ["attn.gk_proj.0", "attn.gk_proj.1"],
    "MLP": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
    "O": ["attn.o_proj"],
    "HEAD": ["lm_head"],
    "QK": ["attn.q_proj", "attn.k_proj"],
    "KV": ["attn.k_proj", "attn.v_proj"],
}
# 组合“积木”
TARGETS["GATINGONLY"] = TARGETS["G"] + TARGETS["GK"]
TARGETS["QK_plus_GATING"] = TARGETS["QK"] + TARGETS["GATINGONLY"]
TARGETS["OplusHEAD"] = TARGETS["O"] + TARGETS["HEAD"]

# 3. 声明所有需要创建的实验 (YAML 文件名列表)
# 这是您在上一轮中提供的缺失文件列表（完整保留）
EXPERIMENTS_TO_CREATE = [
    "E1_QKVO_dropout0_r8_alpha16.yaml",
    "E1_QKVO_first6_r8_alpha16.yaml",
    "E1_QKVO_last6_r8_alpha16.yaml",
    "E1_QKVO_loradrop0.05_r8_alpha16.yaml",
    "E1_QKVO_lr2e-4_r8_alpha16.yaml",
    "E1_QKVO_lr5e-5_r8_alpha16.yaml",
    "E1_QKVO_plus_GK_RSLoRA_r8_alpha16.yaml",
    "E1_QKVO_plus_G_DoRA_r8_alpha16.yaml",
    "E1_QKVO_plus_G_RSLoRA_r8_alpha16.yaml",
    "E1_QKVO_plus_G_plus_GK_DoRA_r8_alpha16.yaml",
    "E1_QKVO_plus_G_plus_GK_RSLoRA_r8_alpha16.yaml",
    "E1_QKVO_plus_G_plus_GK_plus_MLP_r8_alpha16.yaml",
    "E1_QKVO_plus_G_r8_alpha16.yaml",
    "E1_QKVO_r6_alpha12.yaml",
    "E1_QKVO_wd0.01_r8_alpha16.yaml",
    "E2_OMLP_last6_r8_alpha16.yaml",
    "E2_OMLP_middle6_r8_alpha16.yaml",
    "E2_OMLP_plus_GK_r8_alpha16.yaml",
    "E2_OMLP_plus_G_plus_GK_r8_alpha16.yaml",
    "E2_OMLP_plus_G_r8_alpha16.yaml",
    "E2_OMLP_r12_alpha24.yaml",
    "E2_OMLP_r6_alpha12.yaml",
    "E3_GATINGONLY_RSLoRA_r8_alpha16.yaml",
    "E6_QKONLY_RSLoRA_r8_alpha16.yaml",
    "E7_KVONLY_RSLoRA_r8_alpha16.yaml",
    "E8_QK_plus_GATING_RSLoRA_r8_alpha16.yaml",
    "E9_OplusHEAD_RSLoRA_r8_alpha16.yaml",
]


# ======================================================================================
# 核心生成逻辑 (Core Generation Logic)
# ======================================================================================

def get_layered_targets(base_targets, layer_indices):
    """为指定层生成目标模块列表"""
    layered_modules = []
    for l_idx in layer_indices:
        for module in base_targets:
            layered_modules.append(f"layers.{l_idx}.{module}")
    return layered_modules


def generate_files():
    """主函数，遍历实验列表并生成文件"""
    print("=" * 40)
    print("Starting Experiment File Generation...")
    print("=" * 40)

    # 确保输出目录存在
    Path(PEFT_PATH).mkdir(parents=True, exist_ok=True)
    Path(YAML_PATH).mkdir(parents=True, exist_ok=True)

    for yaml_name in EXPERIMENTS_TO_CREATE:
        print(f"\nProcessing: {yaml_name}")
        try:
            # --- 1. 从文件名智能解析参数 ---
            base_name = yaml_name.replace(".yaml", "")
            parts = base_name.split('_')

            # 默认值，以冠军配置为基础
            params = {
                'r': 8, 'alpha': 16, 'dropout': 0.05,
                'lr': 0.0003, 'wd': 0.01,
                'use_dora': False, 'use_rslora': False,
                'layers': None, 'target_keys': []
            }

            # 解析目标模块
            tokens = set(parts)
            # 一级目标解析：允许在任意 token 位置出现（而非仅 parts[1]）
            if "QKVO" in tokens:
                params['target_keys'].append("QKVO")
            if "OMLP" in tokens:
                params['target_keys'].append("OMLP")
            if "GATINGONLY" in tokens:
                params['target_keys'].append("GATINGONLY")
            if "QKONLY" in tokens:
                params['target_keys'].append("QK")
            if "KVONLY" in tokens:
                params['target_keys'].append("KV")
            # 复合：QK+GATING（如 E8_QK_plus_GATING_*）
            if ("QK" in tokens) and ("GATING" in tokens or "GATINGONLY" in tokens):
                params['target_keys'].append("QK_plus_GATING")
            if "OplusHEAD" in tokens:
                params['target_keys'].append("OplusHEAD")

            if "plus" in tokens:
                if "G" in tokens: params['target_keys'].append("G")
                if "GK" in tokens: params['target_keys'].append("GK")
                if "MLP" in tokens: params['target_keys'].append("MLP")

            # 解析超参数
            for part in parts:
                if m := re.match(r'^r(\d+)$', part): params['r'] = int(m.group(1))
                if m := re.match(r'^alpha(\d+)$', part): params['alpha'] = int(m.group(1))
                if "DoRA" == part: params['use_dora'] = True
                if "RSLoRA" == part: params['use_rslora'] = True
                if "dropout0" == part: params['dropout'] = 0.0
                if "loradrop0.05" == part: params['dropout'] = 0.05
                if part.startswith("lr"):
                    # 支持 lr2e-4/lr5e-5/lr0.0002 等形式
                    val = part[2:]
                    try:
                        if 'e' in val:
                            params['lr'] = float(val)
                        elif val.startswith('e-'):
                            params['lr'] = float(f"1{val}")
                        else:
                            params['lr'] = float(val)
                    except Exception:
                        pass
                if part.startswith("wd"): params['wd'] = float(part[2:])
                if "first6" == part: params['layers'] = range(0, 6)
                if "last6" == part: params['layers'] = range(18, 24)
                if "middle6" == part: params['layers'] = range(9, 15)

            # --- 2. 构建 JSON 配置文件 ---
            final_target_modules = []
            for key in sorted(list(set(params['target_keys']))):
                final_target_modules.extend(TARGETS.get(key, []))

            if params['layers'] is not None:
                final_target_modules = get_layered_targets(final_target_modules, params['layers'])

            # 训练策略类：仅 YAML，复用 canonical JSON（QKVO α=2r@8/16）
            training_only = any(p.startswith("lr") or p.startswith("wd") or p.startswith("loradrop") or p == "dropout0" for p in parts)
            canonical_qkvo_alpha2r = "lora_qkvo_alpha2r_r8_a16.json"

            if training_only and ("QKVO" in params['target_keys']) and not params['use_dora'] and not params['use_rslora'] and params['r'] == 8 and params['alpha'] == 16 and params['layers'] is None:
                json_name = canonical_qkvo_alpha2r
                json_filepath = os.path.join(PEFT_PATH, json_name)
                create_json = False
            else:
                json_name = base_name.replace(parts[0] + "_", "lora_") + ".json"
                json_filepath = os.path.join(PEFT_PATH, json_name)
                create_json = True

            if create_json:
                json_content = {
                    "peft_type": "LORA",
                    "r": params['r'],
                    "lora_alpha": params['alpha'],
                    "lora_dropout": params['dropout'],
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                    "target_modules": sorted(list(set(final_target_modules)))
                }
                if params['use_dora']: json_content["use_dora"] = True
                if params['use_rslora']: json_content["use_rslora"] = True

            # --- 3. 构建 YAML 实验文件 ---
            yaml_filepath = os.path.join(YAML_PATH, yaml_name)
            yaml_content_lines = [
                f"batch_size: 4",
                f"data: glue-tvt_cola",
                f"learning_rate: {params['lr']}",
                f"model: /home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B",
                f"no_save: false",
                f"num_epochs: 10",
                f"peft: cfg/my_lora_exp/peft/{json_name}",
                f"prec: bf16",
            ]
            if params['wd'] != 0.01:
                yaml_content_lines.append(f"weight_decay: {params['wd']}")

            # --- 4. 写入文件 (如果不存在) ---
            if create_json:
                if not os.path.exists(json_filepath):
                    print(f"  -> Creating JSON: {json_name}")
                    with open(json_filepath, 'w') as f:
                        json.dump(json_content, f, indent=2)
                else:
                    print(f"  -> JSON exists: {json_name}")
            else:
                print(f"  -> Reusing canonical JSON: {json_name}")

            if not os.path.exists(yaml_filepath):
                print(f"  -> Creating YAML: {yaml_name}")
                with open(yaml_filepath, 'w') as f:
                    f.write("\n".join(yaml_content_lines) + "\n")
            else:
                print(f"  -> YAML exists: {yaml_name}")

        except Exception as e:
            print(f"[ERROR] Failed to process {yaml_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\\n" + "=" * 40)
    print("File generation process completed.")
    print("=" * 40)


if __name__ == "__main__":
    generate_files()