#!/usr/bin/env python3
"""Test tokenizer behavior on numeric vs letter labels for retnet (llama) and mamba (gpt-neox)."""

from transformers import AutoTokenizer

def test_tokenizer(model_name, desc):
    print(f'\n{"="*60}')
    print(f'{desc}')
    print(f'{"="*60}')

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Test numeric labels
    print('\nNumeric labels (no space prefix):')
    for label in ['0', '1', '2']:
        ids = tokenizer.encode(label, add_special_tokens=False)
        tokens = [tokenizer.decode([i]) for i in ids]
        print(f'  "{label}" -> ids={ids}, tokens={tokens}, len={len(ids)}')

    # Test letter labels
    print('\nLetter labels (no space prefix):')
    for label in ['A', 'B', 'C', 'D', 'E']:
        ids = tokenizer.encode(label, add_special_tokens=False)
        tokens = [tokenizer.decode([i]) for i in ids]
        print(f'  "{label}" -> ids={ids}, tokens={tokens}, len={len(ids)}')

    # Test with space prefix (common in llama tokenizers)
    print('\nWith space prefix:')
    for label in [' 0', ' 1', ' 2', ' A', ' B']:
        ids = tokenizer.encode(label, add_special_tokens=False)
        tokens = [tokenizer.decode([i]) for i in ids]
        print(f'  "{label}" -> ids={ids}, tokens={tokens}, len={len(ids)}')

# Test retnet (llama-based tokenizer)
test_tokenizer('fla-hub/gla-1.3B-100B', 'RetNet/GLA Tokenizer (Llama-based)')

# Test mamba (gpt-neox tokenizer)
try:
    test_tokenizer('state-spaces/mamba-1.4b-hf', 'Mamba Tokenizer (GPT-NeoX-based)')
except Exception as e:
    print(f'\nSkipping Mamba test: {e}')
