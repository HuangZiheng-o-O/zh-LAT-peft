#!/usr/bin/env python3
"""
验证空格前缀labels在不同tokenizer下的行为。
在云端运行此脚本来确认修复方案是否正确。

Usage:
    python verify_tokenizer_fix.py
"""

from transformers import AutoTokenizer


def test_labels(tokenizer, tokenizer_name):
    """测试各种labels在给定tokenizer下的编码行为"""
    print(f"\n{'='*70}")
    print(f"Testing: {tokenizer_name}")
    print(f"{'='*70}")

    # 测试数字labels (BoolQ, PIQA, WinoGrande)
    print("\n[数字labels - 无空格前缀]")
    for label in ["0", "1", "2"]:
        ids = tokenizer.encode(label, add_special_tokens=False)
        tokens = [tokenizer.decode([i]) for i in ids]
        status = "✓ SINGLE" if len(ids) == 1 else "✗ MULTI"
        print(f"  '{label}' -> {status:12} ids={ids}, tokens={tokens}")

    print("\n[数字labels - 有空格前缀 (修复后)]")
    for label in [" 0", " 1", " 2"]:
        ids = tokenizer.encode(label, add_special_tokens=False)
        tokens = [tokenizer.decode([i]) for i in ids]
        status = "✓ SINGLE" if len(ids) == 1 else "✗ MULTI"
        print(f"  '{label}' -> {status:12} ids={ids}, tokens={tokens}")

    # 测试字母labels (Arc, SocialIQA, HellaSwag, OpenBookQA)
    print("\n[字母labels - 无空格前缀]")
    for label in ["A", "B", "C", "D", "E"]:
        ids = tokenizer.encode(label, add_special_tokens=False)
        tokens = [tokenizer.decode([i]) for i in ids]
        status = "✓ SINGLE" if len(ids) == 1 else "✗ MULTI"
        print(f"  '{label}' -> {status:12} ids={ids}, tokens={tokens}")

    print("\n[字母labels - 有空格前缀 (修复后)]")
    for label in [" A", " B", " C", " D", " E"]:
        ids = tokenizer.encode(label, add_special_tokens=False)
        tokens = [tokenizer.decode([i]) for i in ids]
        status = "✓ SINGLE" if len(ids) == 1 else "✗ MULTI"
        print(f"  '{label}' -> {status:12} ids={ids}, tokens={tokens}")


def main():
    print("\n" + "="*70)
    print("验证tokenizer fix方案：空格前缀labels")
    print("="*70)

    # Test 1: RetNet/GLA (Llama tokenizer) - 主要问题所在
    print("\n\n[1/2] RetNet/GLA Tokenizer (Llama-based)")
    try:
        tokenizer_retnet = AutoTokenizer.from_pretrained(
            "fla-hub/gla-1.3B-100B",
            trust_remote_code=True
        )
        test_labels(tokenizer_retnet, "RetNet/GLA (Llama)")
    except Exception as e:
        print(f"ERROR loading RetNet tokenizer: {e}")

    # Test 2: Mamba (GPT-NeoX tokenizer) - 确保兼容性
    print("\n\n[2/2] Mamba Tokenizer (GPT-NeoX-based)")
    try:
        tokenizer_mamba = AutoTokenizer.from_pretrained(
            "state-spaces/mamba-1.4b-hf",
            trust_remote_code=True
        )
        test_labels(tokenizer_mamba, "Mamba (GPT-NeoX)")
    except Exception as e:
        print(f"ERROR loading Mamba tokenizer: {e}")

    # 总结
    print("\n" + "="*70)
    print("验证完成！")
    print("="*70)
    print("\n期望结果：")
    print("  - RetNet: 空格前缀的labels应该都是 ✓ SINGLE")
    print("  - Mamba: 空格前缀的labels应该也都是 ✓ SINGLE")
    print("\n如果上述条件满足，说明修复方案正确，可以上传新代码。")


if __name__ == "__main__":
    main()
