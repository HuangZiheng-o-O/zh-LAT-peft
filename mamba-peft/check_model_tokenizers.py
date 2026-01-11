#!/usr/bin/env python3
"""
检查GLA、DeltaNet、RetNet三个模型各自使用的tokenizer类型和行为。
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lat_adapter import prepare_lat_model_and_tokenizer


def check_tokenizer(model_type, model_id):
    """检查指定模型的tokenizer"""
    print(f"\n{'='*70}")
    print(f"Model Type: {model_type}")
    print(f"Model ID: {model_id}")
    print(f"{'='*70}")

    try:
        # 使用lat_adapter加载模型和tokenizer（与实际eval相同的方式）
        model, tokenizer, _ = prepare_lat_model_and_tokenizer(
            model_type=model_type,
            model_id=model_id,
            prec="bf16",
            debug=True,  # CPU模式，避免GPU占用
            peft_json_path=None,
        )

        print(f"Tokenizer class: {type(tokenizer).__name__}")
        print(f"Tokenizer name_or_path: {getattr(tokenizer, 'name_or_path', 'N/A')}")
        print(f"Vocab size: {tokenizer.vocab_size}")

        # 测试关键labels
        print("\n--- 测试数字labels (BoolQ, PIQA) ---")
        for label in ["0", "1"]:
            ids = tokenizer.encode(label, add_special_tokens=False)
            tokens = [tokenizer.decode([i]) for i in ids]
            status = "✓ SINGLE" if len(ids) == 1 else "✗ MULTI"
            print(f"  '{label}' -> {status:12} ids={ids}")

        print("\n--- 测试字母labels (Arc, SocialIQA, etc) ---")
        for label in ["A", "B", "C", "D"]:
            ids = tokenizer.encode(label, add_special_tokens=False)
            tokens = [tokenizer.decode([i]) for i in ids]
            status = "✓ SINGLE" if len(ids) == 1 else "✗ MULTI"
            print(f"  '{label}' -> {status:12} ids={ids}")

        return True, tokenizer

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    print("\n" + "="*70)
    print("检查GLA、DeltaNet、RetNet三个模型的tokenizer")
    print("="*70)

    # 从环境变量或默认值获取模型路径
    models_to_check = [
        ("gla", os.environ.get("GLA_MODEL", "/home/user/mzs_h/model/gla-1.3B-100B")),
        ("delta_net", os.environ.get("DELTA_NET_MODEL", "/home/user/mzs_h/model/delta_net-2.7B-100B")),
        ("retnet", os.environ.get("RETNET_MODEL", "/home/user/mzs_h/model/retnet-1.3B-100B")),
    ]

    results = {}
    for model_type, model_id in models_to_check:
        success, tokenizer = check_tokenizer(model_type, model_id)
        results[model_type] = {
            "success": success,
            "tokenizer": tokenizer,
            "model_id": model_id,
        }

    # 总结
    print("\n" + "="*70)
    print("总结")
    print("="*70)

    for model_type, result in results.items():
        if result["success"]:
            print(f"\n{model_type.upper()}:")
            print(f"  Model: {result['model_id']}")
            tok = result["tokenizer"]
            print(f"  Tokenizer: {type(tok).__name__}")

            # 检查是否有问题
            test_labels = ["0", "1", "A", "B"]
            problem_labels = []
            for label in test_labels:
                ids = tok.encode(label, add_special_tokens=False)
                if len(ids) != 1:
                    problem_labels.append(f"'{label}'")

            if problem_labels:
                print(f"  ⚠️  问题labels: {', '.join(problem_labels)} 不是单token")
            else:
                print(f"  ✓  所有测试labels都是单token")
        else:
            print(f"\n{model_type.upper()}: ❌ 加载失败")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
