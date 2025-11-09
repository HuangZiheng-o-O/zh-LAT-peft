#!/usr/bin/env python3
import os
import sys
import traceback

# Ensure repo root is on sys.path BEFORE importing project modules
CODE_ROOT = os.environ.get("CODE_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

import torch
from transformers import AutoTokenizer

from mamba_ssm_peft import load_mamba
from mamba_ssm_peft.utils.generation import InferenceParams
from mamba_ssm_peft.utils.beam_search import get_logits_recurrent, get_logits_parallel


def truthy(x: str | None) -> bool:
    if x is None:
        return False
    return str(x).lower() in ("1", "true", "yes", "on")


def main():
    # Model/tokenizer
    model_dir = os.environ.get("MODEL_DIR")
    if not model_dir:
        print("[ERROR] Please set MODEL_DIR to your (Mamba) model checkpoint or HF id")
        sys.exit(1)

    tokenizer_dir = os.environ.get("TOKENIZER_DIR", None)
    if tokenizer_dir:
        tok = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    else:
        # Fall back to the project’s default Mamba tokenizer
        from mamba_ssm_peft import load_mamba_tokenizer
        tok = load_mamba_tokenizer()

    # Load model (no PEFT merge/attach, dtype and device inferred from env if any)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.bfloat16, "fp32": torch.float32}.get(
        os.environ.get("HP_PREC", "bf16").lower(), torch.bfloat16
    )
    model = load_mamba(model_dir, dtype=dtype, device="cuda")["model"]
    model.eval()
    device = next(iter(model.parameters())).device

    # Inputs
    prompt = os.environ.get("PROMPT", "DART debug: simple test prompt.")
    num_beams = int(os.environ.get("EVAL_GEN_NUM_BEAMS", "5"))
    max_new_tokens = int(os.environ.get("EVAL_GEN_MAX_LENGTH", "16"))

    enc = tok(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    batch_beam_input = input_ids.repeat(num_beams, 1)  # [num_beams, prompt_len]

    print("==== Model / Tokenizer ====")
    print(f"MODEL_DIR={model_dir}")
    print(f"TOKENIZER_DIR={tokenizer_dir or '(default mamba tokenizer)'}")
    print(f"device={device} dtype={dtype}")
    print()

    print("==== Inputs ====")
    print(f"prompt='{prompt[:80]}'...")
    print(f"input_ids.shape={tuple(input_ids.shape)}  (batch=1, seqlen={input_ids.shape[1]})")
    print(f"num_beams={num_beams}")
    print()

    # Prepare dummy inference params (only used by recurrent path)
    inf_params = InferenceParams(max_seqlen=input_ids.shape[1] + max_new_tokens, max_batch_size=num_beams)

    with torch.no_grad():
        # 1) Raw forward (no special args)
        logits_full = model(input_ids=batch_beam_input).logits
        print("==== Raw forward logits ====")
        print(f"model(...).logits.shape = {tuple(logits_full.shape)}  # expect [B*K, seq_len, vocab]")
        print()

        # 2) Raw forward with logits_to_keep=1 (model contracts to last step)
        try:
            logits_keep1 = model(input_ids=batch_beam_input, logits_to_keep=1).logits
            print("==== Raw forward logits_to_keep=1 ====")
            print(f"model(..., logits_to_keep=1).logits.shape = {tuple(logits_keep1.shape)}  # expect [B*K, 1, vocab]")
        except Exception as e:
            print("model(..., logits_to_keep=1) raised:", repr(e))
        print()

        # 3) Parallel path (explicit last-token slice)
        try:
            logits_parallel = get_logits_parallel(model, batch_beam_input, inf_params)
            print("==== get_logits_parallel ====")
            print(f"get_logits_parallel(...) -> {tuple(logits_parallel.shape)}  # expect [B*K, vocab]")
        except Exception as e:
            print("get_logits_parallel raised:", repr(e))
        print()

        # 4) Recurrent path (uses num_last_tokens=1 argument)
        try:
            logits_recurrent = get_logits_recurrent(model, batch_beam_input, inf_params)
            print("==== get_logits_recurrent ====")
            print(f"get_logits_recurrent(...) -> {tuple(logits_recurrent.shape)}  # EXPECT [B*K, vocab]")
            print("Note: If this prints [B*K, seq_len, vocab] instead, model ignored num_last_tokens=1.")
        except Exception as e:
            print("get_logits_recurrent raised:", repr(e))
        print()

    print("==== Diagnosis hint ====")
    print("- If recurrent returned 3D [B*K, seq_len, vocab], it means the model forward ignored num_last_tokens.")
    print("- Beam search then tries to add beam_scores [B*K,1] to a 3D tensor, causing the expand_as size mismatch.")
    print("- Two robust fixes:")
    print("  (A) Use parallel path for logits in beam search (always slice [:, -1]).")
    print("  (B) Or call model(..., logits_to_keep=1) in the recurrent path and squeeze dim=1.")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("[FATAL]", repr(e))
        traceback.print_exc()
        sys.exit(1)


