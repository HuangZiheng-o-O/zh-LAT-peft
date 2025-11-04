#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataset_downloader_v3.py
- 扫描已有 -> 跳过
- HuggingFace 数据集快照到 <project>/data/<owner_repo>（或 <project>/data/<repo>）
- MNIST/CIFAR 走 torchvision（已装才下）
- Spider 自动补 tables.json（优先 HF，其次 GitHub，最后 git clone 兜底）
- 429/网络错误自动重试（指数退避）
- samsum 自动回退到 GEM/samsum
- 失败不中断并记录 log
"""

import argparse
import os
import sys
import time
import shutil
from pathlib import Path
from datetime import datetime
import traceback

# 可选：hf_transfer 加速
try:
    import hf_transfer  # noqa: F401
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
except Exception:
    pass

from huggingface_hub import snapshot_download, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
import requests

_TV_OK = None  # torchvision lazy import flag


def log(msg, log_file):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def dir_non_empty(p: Path) -> bool:
    return p.exists() and p.is_dir() and any(p.iterdir())


def ensure_parent(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _snapshot_with_retries(repo_id, repo_type, local_dir, max_retries=5, log_file=None):
    # 指数退避：1s,2s,4s,8s,8s …
    backoffs = [1, 2, 4, 8, 8]
    for i in range(max_retries):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,  # 新版会忽略，但保留不影响
            )
            return True
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            # 429/网络型错误重试
            if ("429" in msg or "Too Many Requests" in msg or
                "LocalEntryNotFoundError" in msg or
                isinstance(e, HfHubHTTPError)):
                if i < max_retries - 1:
                    t = backoffs[i]
                    log(f"⚠️  {repo_id} 拉取失败，{t}s 后重试（{i+1}/{max_retries}） | {msg}", log_file)
                    time.sleep(t)
                    continue
            # 其他错误直接抛
            raise
    return False


def download_hf_dataset(repo_ids, targets, log_file, resync=False):
    """
    repo_ids: [候选repo id]（比如 ["samsum","GEM/samsum"]）
    targets:  与 repo_ids 一一对应的目标目录 Path
    规则：
      - 只要有一个 target 目录非空，就认为“已存在”，直接跳过
      - 否则按顺序尝试每个 repo_id，成功一个即返回 True
    """
    # 如已有任何一个候选目标目录非空 -> 跳过
    for t in targets:
        if dir_non_empty(t) and not resync:
            log(f"⏭️  skip (exists): {t}", log_file)
            return True

    # 否则尝试逐个 repo
    for rid, tgt in zip(repo_ids, targets):
        ensure_parent(tgt)
        try:
            log(f"⬇️  snapshot {rid}  ->  {tgt}", log_file)
            _snapshot_with_retries(rid, "dataset", tgt, log_file=str(log_file))
            log(f"✅ snapshot ok: {rid}", log_file)
            return True
        except Exception as e:
            log(f"❌ snapshot failed: {rid} | {e}", log_file)
            log(traceback.format_exc(), log_file)
            # 尝试下一个候选
            continue
    return False


def download_spider_tables(target_root: Path, log_file: str):
    """
    目标：<data_root>/xlangai_spider/spider/tables.json
    候选来源（按优先级）：
      1) HF dataset taoyds/spider 里的 spider/tables.json 或 tables.json / database/tables.json
      2) GitHub raw (main/master + 3种路径)
      3) git clone 仓库到临时目录后查找 tables.json
    """
    dst = target_root / "spider" / "tables.json"
    if dst.exists() and dst.stat().st_size > 100:  # 粗略视为有效
        log(f"⏭️  skip spider tables (exists): {dst}", log_file)
        return True

    ensure_parent(dst.parent)

    # 1) HF 优先
    candidates = ["spider/tables.json", "tables.json", "database/tables.json"]
    for fname in candidates:
        try:
            log(f"⬇️  spider tables via HF: taoyds/spider::{fname}", log_file)
            cached = hf_hub_download(
                repo_id="taoyds/spider",
                repo_type="dataset",
                filename=fname,
            )
            shutil.copyfile(cached, dst)
            log("✅ spider tables.json ok (HF)", log_file)
            return True
        except Exception:
            pass

    # 2) GitHub raw
    for branch in ["main", "master"]:
        for path in candidates:
            url = f"https://raw.githubusercontent.com/taoyds/spider/{branch}/{path}"
            try:
                log(f"⬇️  spider tables via GitHub: {url}", log_file)
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                content = r.content
                # 极短/HTML 404 文本视为失败
                if len(content) < 100 or content.startswith(b"404"):
                    raise RuntimeError("content too small or 404 page")
                dst.write_bytes(content)
                log("✅ spider tables.json ok (GitHub)", log_file)
                return True
            except Exception:
                continue

    # 3) git clone 兜底
    import tempfile, subprocess
    with tempfile.TemporaryDirectory() as tmp:
        try:
            log("⬇️  spider tables via git clone ...", log_file)
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/taoyds/spider", tmp],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            found = None
            for rel in candidates:
                p = Path(tmp) / rel
                if p.exists():
                    found = p
                    break
            if found is None:
                # 再全盘搜
                for p in Path(tmp).rglob("tables.json"):
                    found = p
                    break
            if found and found.stat().st_size > 100:
                shutil.copyfile(found, dst)
                log(f"✅ spider tables.json ok (git clone at {found})", log_file)
                return True
            else:
                raise RuntimeError("tables.json not found after clone")
        except Exception as e:
            log(f"❌ spider tables.json failed: {e}", log_file)
            log(traceback.format_exc(), log_file)
            return False


def _need_tv():
    global _TV_OK
    if _TV_OK is None:
        try:
            import torchvision  # noqa: F401
            _TV_OK = True
        except Exception:
            _TV_OK = False
    return _TV_OK


def download_mnist(target: Path, log_file: str):
    if not _need_tv():
        log("❌ torchvision 未安装，跳过 MNIST（pip install torchvision）", log_file)
        return False
    try:
        from torchvision import datasets, transforms
        ensure_parent(target)
        log(f"⬇️  MNIST -> {target}", log_file)
        datasets.MNIST(root=str(target), train=True, download=True, transform=transforms.ToTensor())
        datasets.MNIST(root=str(target), train=False, download=True, transform=transforms.ToTensor())
        log("✅ MNIST ok", log_file)
        return True
    except Exception as e:
        log(f"❌ MNIST failed: {e}", log_file)
        log(traceback.format_exc(), log_file)
        return False


def download_cifar10(target: Path, log_file: str):
    if not _need_tv():
        log("❌ torchvision 未安装，跳过 CIFAR10（pip install torchvision）", log_file)
        return False
    try:
        from torchvision import datasets, transforms
        ensure_parent(target)
        log(f"⬇️  CIFAR10 -> {target}", log_file)
        datasets.CIFAR10(root=str(target), train=True, download=True, transform=transforms.ToTensor())
        datasets.CIFAR10(root=str(target), train=False, download=True, transform=transforms.ToTensor())
        log("✅ CIFAR10 ok", log_file)
        return True
    except Exception as e:
        log(f"❌ CIFAR10 failed: {e}", log_file)
        log(traceback.format_exc(), log_file)
        return False


def main():
    parser = argparse.ArgumentParser(description="Bulk dataset downloader (skip-if-exists, resilient).")
    parser.add_argument("--project", required=True, help="mamba-peft 项目根目录（包含 data 子目录）")
    parser.add_argument("--only", nargs="*", default=None,
                        help="只下这些（用空格或逗号分隔）。可用 job 名：alpaca/alpaca_eval/samsum/boolq/piqa/glue/arc/spider/mmlu/mnist/cifar10/dart 或 repo id 如 GEM/dart")
    parser.add_argument("--resync", action="store_true", help="即使目录存在也做 snapshot 校验/补齐")
    args = parser.parse_args()

    project = Path(args.project).expanduser().resolve()
    data_root = project / "data"
    ensure_parent(data_root)

    log_file = project / "download_datasets.log"
    log("========== start download ==========", str(log_file))

    jobs = []

    def add_hf(name, repo_id_or_list):
        if isinstance(repo_id_or_list, str):
            repo_ids = [repo_id_or_list]
        else:
            repo_ids = list(repo_id_or_list)
        targets = [data_root / rid.replace("/", "_") for rid in repo_ids]
        jobs.append(("hf", name, repo_ids, targets))

    def add_tv(name):
        jobs.append(("tv", name, None, None))

    # HF 数据集
    add_hf("alpaca",       "yahma/alpaca-cleaned")
    add_hf("alpaca_eval",  "tatsu-lab/alpaca_eval")
    add_hf("samsum",       ["samsum", "GEM/samsum"])   # 自动回退
    add_hf("boolq",        "google/boolq")
    add_hf("piqa",         "piqa")
    add_hf("glue",         "nyu-mll/glue")
    add_hf("arc",          "allenai/ai2_arc")
    add_hf("spider",       "xlangai/spider")
    add_hf("mmlu",         "cais/mmlu")
    add_hf("dart",         "GEM/dart")                 # 修正为 GEM/dart

    # torchvision
    add_tv("mnist")
    add_tv("cifar10")

    # 处理 --only（支持逗号/空格 & 支持 job 名或 repo id）
    if args.only:
        raw = []
        for token in args.only:
            raw.extend([t for t in token.split(",") if t])
        only = set(raw)
        def keep(job):
            kind, name, repo_ids, targets = job
            if name in only:
                return True
            if kind == "hf" and any(r in only for r in repo_ids):
                return True
            return False
        jobs = [j for j in jobs if keep(j)]

    ok = 0
    fail = 0
    for kind, name, repo_ids, targets in jobs:
        try:
            if kind == "hf":
                if download_hf_dataset(repo_ids, targets, str(log_file), resync=args.resync):
                    # Spider 追加 tables.json
                    if name == "spider":
                        if not download_spider_tables(targets[0], str(log_file)):
                            fail += 1
                            continue
                    ok += 1
                else:
                    fail += 1
            elif kind == "tv":
                if name == "mnist":
                    target = data_root / "mnist"
                    ok += 1 if download_mnist(target, str(log_file)) else 0
                    fail += 0 if target.exists() else 1
                elif name == "cifar10":
                    target = data_root / "cifar"
                    ok += 1 if download_cifar10(target, str(log_file)) else 0
                    fail += 0 if target.exists() else 1
                else:
                    log(f"❌ unknown torchvision job: {name}", str(log_file))
                    fail += 1
            else:
                log(f"❌ unknown job kind: {kind}", str(log_file))
                fail += 1
        except KeyboardInterrupt:
            log("🛑 停止（Ctrl-C）", str(log_file))
            break
        except Exception as e:
            log(f"❌ job failed ({name}): {e}", str(log_file))
            log(traceback.format_exc(), str(log_file))
            fail += 1

    log(f"========== done ({ok} ok, {fail} fail) ==========", str(log_file))


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    sys.exit(main())