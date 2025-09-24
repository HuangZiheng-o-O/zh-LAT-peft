import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download

def grab(repo_id: str, out_dir: str, revision: str | None):
    # 只下载常见权重/配置/分词器文件，避免把大文件之外的杂项都拉下来
    allow = [
        "*.bin", "*.safetensors", "*.pt", "*.onnx",
        "*.json", "tokenizer.*", "merges.txt", "vocab.*", "spiece.*",
        "config.*", "generation_config.*", "special_tokens_map.*", "README*"
    ]
    local_dir = Path(out_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,     # 方便打包/拷贝
        allow_patterns=allow,
        resume_download=True
    )
    print(f"[OK] downloaded to: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="state-spaces/mamba-370m-hf",
                        help="HuggingFace repo id，比如 state-spaces/mamba-370m-hf")
    parser.add_argument("--tokenizer_id", default="EleutherAI/gpt-neox-20b",
                        help="分词器（与原脚本默认一致）")
    parser.add_argument("--out_dir", default="./weights/hf"),
                        # e.g. ./weights/hf/state-spaces/mamba-370m-hf
    parser.add_argument("--revision", default=None, help="可选：tag/commit hash/branch")
    args = parser.parse_args()

    # 可选：提升下载速度（需安装 hf_transfer）
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    # 预下模型
    grab(args.model_id, Path(args.out_dir) / args.model_id, args.revision)
    # 预下分词器（可选，但训练基本都会用到）
    grab(args.tokenizer_id, Path(args.out_dir) / args.tokenizer_id, args.revision)