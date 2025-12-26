"""
统一的 Linear Attention 训练入口脚本（Unified Linear Attention Training Entry Point）。

本脚本的目标：
---------------------------------
提供一个统一接口，用于训练 FLA（Fast Linear Attention / Flash Linear Attention 等相关实现）生态中的
多种 Linear Attention / SSM 模型结构，包括但不限于：
- GLA（Gated Linear Attention）
- RetNet（Retentive Network）
- Mamba2（State Space Model 变体）

设计原则（Design Principles）：
---------------------------------
1. **向后兼容（Backward Compatibility）**
   - 当 model_type="gla" 或默认行为路径下，训练行为应当与历史脚本 train_gla_only.py 保持一致；
   - 也就是说：同样的 cfg、同样的 env 覆盖、同样的数据加载、同样的训练器参数，输出应当一致或近似一致。

2. **统一接口（Unified Interface）**
   - 不管使用 GLA/RetNet/Mamba2，训练流程都走同一套：
     - 统一读取 YAML config
     - 统一读取环境变量覆盖
     - 统一 prepare_lat_model_and_tokenizer() 加载模型与 tokenizer
     - 统一使用 GenericLMTrainer / GenericLMTrainingArguments
     - 统一保存 cfg.yaml 与 checkpoint
     - 统一评估逻辑（可选 generation）

3. **自动检测（Auto-Detection）**
   - 允许用户传入 model_type="auto"
   - 允许用环境变量 MODEL_TYPE 覆盖
   - （如果你后续扩展）也可以从配置文件（或模型名）推断模型类型

支持模型（Supported Models）：
---------------------------------
- gla: Gated Linear Attention (https://arxiv.org/abs/2312.06635)
- retnet: Retentive Network (https://arxiv.org/abs/2307.08621)
- mamba2: Mamba2 State Space Model (https://arxiv.org/abs/2405.21060)

环境变量（Environment Variables）：
---------------------------------
- MODEL_TYPE: 模型类型强制覆盖（gla / retnet / mamba2 / auto）
- LAT_* / GLA_*: 其他训练或模型相关开关（此脚本通过 get_lat_env/get_lat_env_bool 等读取部分）

使用方式（Usage）：
---------------------------------
    # 显式指定模型类型
    python train_lat.py --cfg configs/gla.yaml --model-type gla

    # 自动检测（由 MODEL_TYPE env 或未来扩展的 config 推断）
    python train_lat.py --cfg configs/model.yaml --model-type auto

    # 兼容 GLA 的老行为（等价于 train_gla_only.py）
    python train_lat.py --cfg configs/gla.yaml
"""

import sys
from pathlib import Path

# --- 确保在从 mamba-peft/ 目录运行时，本地的 fla 子模块可导入 ---
# 背景：
#   - 一些 repo 的结构是：根目录下有 fla/ 子目录（可能是 submodule 或 symlink）
#   - 但是 python 的 import 搜索路径 sys.path 可能不包含该根目录
#   - 于是 import fla 失败
# 策略：
#   - 先尝试直接 import fla
#   - 如果失败，尝试推断 repo_root，把它插入 sys.path，再 import fla
try:
    import fla  # noqa: F401
except Exception:
    try:
        # Path(__file__).resolve(): 当前脚本的绝对路径
        # .parents[1]：往上两级目录（父目录的父目录）
        # 这里注释写的是：.../zh-LAT-peft
        repo_root = Path(__file__).resolve().parents[1]  # .../zh-LAT-peft

        # 如果根目录下有 fla（可能是 symlink），就把 repo_root 加到 sys.path
        fla_symlink = repo_root / "fla"
        if fla_symlink.exists():
            sys.path.insert(0, str(repo_root))
            import fla  # noqa: F401
    except Exception:
        # 这里选择吞掉异常：
        #   - 因为即便 fla 导入失败，后面可能仍能继续（例如某些路径不需要 fla）
        #   - 但实际运行若依赖 fla 的地方会再报错
        pass

import json
import os
import shutil
from typing import Optional, Dict

import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader  # noqa: F401  # 兼容性保留：可能以前某些脚本需要显式引用 DataLoader

import yaml

# W&B 项目名固定写死为 mamba-peft
# 背景：
#   - 许多训练框架会读取 WANDB_PROJECT 来决定日志归属项目
#   - 你这里后面其实 report_to="none"，但保留项目名设置对某些外部 hook 也有用
os.environ["WANDB_PROJECT"] = "mamba-peft"

# 数据集加载函数：你项目内部 dataset.py
# 负责：
#   - 根据 data 名称（glue/spider/samsum...）加载对应 dataset、collator、metric 逻辑
from dataset import load_dataset

# 统一 Trainer：封装 HF Trainer / 自定义 Trainer 的一个统一接口
# GenericLMTrainingArguments：类似 TrainingArguments 的封装
from trainer.generic_lm_trainer import GenericLMTrainer, GenericLMTrainingArguments

# PEFT（参数高效微调）工具：
# get_trainable_parameters_ratio：计算可训练参数占比
# print_trainable_parameter_names：打印可训练参数名（常用于 sanity check）
from mamba_ssm_peft import get_trainable_parameters_ratio, print_trainable_parameter_names

# 统一 LAT 模型加载适配器：
# prepare_lat_model_and_tokenizer：
#   - 根据 model_type（gla/retnet/mamba2）加载正确模型类
#   - 应用精度 prec（bf16/fp16/fp32）
#   - 应用 peft 配置（如 LoRA、prefix、adapter）
from lat_adapter import prepare_lat_model_and_tokenizer

# generation 评估时用于创建 decoder：
# create_lat_decoder：
#   - 封装 model.generate 的调用参数
#   - 统一适配不同模型类型的 generation 细节
from mamba_ssm_peft.utils.lat_decoder import create_lat_decoder

# 环境变量读取工具
# get_lat_env: 读取某些 LAT 相关环境变量
# get_lat_env_bool: 读取 bool 类型环境变量（通常支持 "1/true/yes/on"）
from mamba_ssm_peft.utils.lat_model_loader import get_lat_env, get_lat_env_bool


def _env_bool(name: str, default: bool) -> bool:
    """
    读取布尔型环境变量的小工具。

    参数：
    - name: 环境变量名
    - default: 当环境变量不存在时的默认值

    返回：
    - True/False

    解释：
    - 如果环境变量不存在 -> 返回 default
    - 如果存在 -> 对值做 lower()，并判断是否在 ("1","true","yes","on") 集合里
    """
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes", "on")





def _env_float(name: str, default: float) -> float:
    """
    读取 float 型环境变量的小工具（带异常保护）。

    用途：
    - 例如 warmup_ratio、lr 等配置可能来自 env
    - env 内容可能写错（非数字），所以这里用 try/except 防崩

    参数：
    - name: 环境变量名
    - default: env 不存在或解析失败时返回这个默认值

    返回：
    - float
    """
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError as e:
        raise ValueError(
            f"Environment variable '{name}' must be a float, got '{v}'"
        )


def _lock_share(name: str, model_type: str = "LAT") -> bool:
    """
    在 share/lock/<name> 下创建一个简单“文件锁”，用来避免多进程/多脚本重复训练同一个 output_dir。

    这个锁机制非常朴素，但在集群/多卡/多任务脚本里很常见。

    参数：
    - name:
        通常传 output_dir（字符串），因为 output_dir 唯一对应一次实验
    - model_type:
        用于打印日志 tag（例如 [GLA] 或 [MAMBA2]）

    返回语义（重要！）：
    - True  -> 锁已经存在（说明别的进程正在训练/已经占用），调用者应该 skip / return
    - False -> 锁创建成功（当前进程获得锁），调用者可以继续训练；训练结束应删除锁文件

    实现细节：
    - path.exists(): 先快速判断
    - open(path, "x"): 以“独占创建”方式创建文件
        - 若文件已存在，会抛 OSError
    """
    path = Path("share/lock") / name
    path.parent.mkdir(parents=True, exist_ok=True)

    # 如果锁文件已经存在，说明已经有人先跑了
    if path.exists():
        print(f"[{model_type}][lock] {path} exists; skipping this run to avoid duplicate training.")
        return True

    try:
        # "x" 模式：文件必须不存在，才能创建成功；否则抛异常
        with open(path, "x"):
            pass
        print(f"[{model_type}][lock] acquired {path}")
        return False
    except OSError:
        # 这里 double-check：即便上面 exists() 通过，
        # 也可能出现竞态条件（另一个进程刚刚创建了文件）
        print(f"[{model_type}][lock] {path} exists; skipping this run.")
        return True


def build_and_run_trainer_lat(
    *,
    model,
    tokenizer,
    model_type: str,
    output_dir: str,
    cfg: Dict,
    cfg_path: str,
    learning_rate: float,
    total_steps: int,
    logging_steps: int,
    gradient_accumulation_steps: int,
    num_data_workers: int,
    batch_size: int,
    eval_batch_size: int,
    eval_epochs: int,
    skip_eval: bool,
    no_save: bool,
    eval_steps_override: Optional[int],
    save_steps_override: Optional[int],
    eval_gen: Optional[Dict],
    resume_from_checkpoint: bool,
    min_eval_metric_after_epoch,
    seed: int,
    data: str,
    val_data: Optional[str],
    val_data_split: str,
    debug: bool,
    gradient_checkpointing: bool = False,
    logits_to_keep: int | None = None,
):
    """
    统一 Linear Attention 训练与评估的核心入口。

    你可以把它理解为：
    - 一切准备都在这里做完：
      - 数据集准备（train/val）
      - metrics 函数绑定
      - generation decoder（如果需要）
      - HF Trainer-like 的 args 构建
      - callbacks（例如 SwanLab）
      - 最终 trainer.train()

    - run_train() 负责更“上层”的逻辑：
      - cfg 合并
      - 输出目录决定
      - resume/overwrite/lock 等策略
      - 计算 total_steps
      - 选择并加载模型

    参数解释（重点字段）：
    - model / tokenizer: 已经加载好的模型与 tokenizer（prepare_lat_model_and_tokenizer 返回）
    - model_type: 'gla'/'retnet'/'mamba2'/'auto'
    - output_dir: 训练输出目录（cfg.yaml、checkpoint、日志等）
    - cfg/cfg_path: 原始配置与路径（用于记录和可追溯）
    - total_steps: 总训练步数（max_steps）
    - eval_gen: 若不为 None，说明该任务需要 generation 评估，使用 create_lat_decoder
    - resume_from_checkpoint: 如果为 str 或 bool（你这里传入 resume_arg），用于 trainer.train()
    - logits_to_keep: 可能用于减少计算/内存（例如只保留 top-k logits），放入 info 记录
    """

    # log_tag 用于所有 print 日志的前缀
    # 若 model_type 仍是 "auto"，则标签统一显示 "LAT"
    log_tag = model_type.upper() if model_type != "auto" else "LAT"

    # 打印可训练参数名：用于确认 PEFT 是否生效
    # 例如 LoRA 注入后，你希望看到只有 lora_A/lora_B 或某些 adapter 参数是 trainable
    print_trainable_parameter_names(model, output_dir=output_dir, cfg_path=cfg_path)
    print("Loaded model")

    # 强制左 padding（decoder-only 生成强烈建议 left padding）
    # 原因：
    #   - decoder-only 模型 generate 时通常假设“右侧是未来 token”
    #   - 如果右 padding，会出现 attention mask / position id 处理不一致风险
    #   - Transformers 也会对 decoder-only + right padding 发 warning
    try:
        _force_left = get_lat_env_bool("FORCE_LEFT_PAD", "1")
        if _force_left and hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "left"

            # 如果 tokenizer 没有 pad_token_id，但有 eos_token，则把 pad_token 设置成 eos_token
            # 常见于 GPT-like tokenizer：默认没有 pad token
            if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token

            if get_lat_env_bool("VERBOSE"):
                print(f"[{log_tag}] Using left padding for decoder-only generation.")
    except Exception as _e:
        # 这里不让失败中断训练：只是提示
        print(f"[{log_tag}][warn] Failed to enforce left padding policy early: {_e}")

    # ----------------------------
    # 构建训练集数据模块
    # ----------------------------
    # load_dataset(..., return_module=True) 表示返回一个“模块对象”，通常包含：
    # - dataset: torch Dataset
    # - data_collator: batch 拼接逻辑
    # - 可能还有 tokenizer/formatting 等
    train_data_module = load_dataset(data, tokenizer, "train", return_module=True)

    # ----------------------------
    # 保存 cfg.yaml 到 output_dir
    # ----------------------------
    # 训练可追溯性非常重要：必须记录 cfg，以便复现实验
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "cfg.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    # ----------------------------
    # 构建 generation decoder（如果需要）
    # ----------------------------
    # eval_gen 非 None 表示：
    # - 任务属于 generation 任务（例如 samsum/dart/spider）
    # - validation 时不是简单算 perplexity/accuracy，而是 generate 输出再 compute_metrics
    eval_generator = None
    if eval_gen is not None:
        _eval = dict(eval_gen)

        # max_length/min_length 默认值
        max_length = int(_eval.get("max_length", 1024))
        min_length = int(_eval.get("min_length", 5))

        # create_lat_decoder 内部会封装 generate 参数
        # do_sample=False 表示使用 greedy 或 beam（取决于内部实现），这里明确不采样
        eval_generator = create_lat_decoder(
            tokenizer,
            model_type=model_type,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
        )

    # ----------------------------
    # 构建验证集数据模块
    # ----------------------------
    # val_data_split 通常是 "val"（也可以通过 env 覆盖为 train/test）
    # mode：
    #   - 若 eval_generator 为 None -> "lm"：语言模型评估（loss/acc等）
    #   - 若 eval_generator 不为 None -> "gen"：生成评估（需要 prompt/target 格式）
    val_data_module = load_dataset(
        val_data if val_data is not None else data,
        tokenizer,
        val_data_split,
        mode="lm" if eval_generator is None else "gen",
        return_module=True,
    )

    # 计算指标函数通常挂在 dataset 上（你项目自己的约定）
    compute_metrics = val_data_module.dataset.compute_metrics

    # ----------------------------
    # Debug 模式：截断数据集
    # ----------------------------
    # 用于快速跑通 pipeline：
    # - train 只取 8 条
    # - val 只取 2 条
    if debug:
        train_data_module.dataset = torch.utils.data.Subset(
            train_data_module.dataset, range(8)
        )
        val_data_module.dataset = torch.utils.data.Subset(
            val_data_module.dataset, range(2)
        )

    # ----------------------------
    # Gradient Checkpointing 参数
    # ----------------------------
    # _gc_kwargs 只在 gradient_checkpointing=True 时启用
    # use_reentrant=False 是一些模型在 PyTorch 新版本下的推荐（避免某些 reentrant 的 edge case）
    _gc_kwargs = {"use_reentrant": False} if gradient_checkpointing else None

    # ----------------------------
    # 是否保存 optimizer state（用于 resume 更完整）
    # ----------------------------
    # SAVE_OPTIMIZER_STATE env 若为真，则保存 optimizer state
    # 否则可能只保存模型权重，resume 时优化器状态重置
    _sos_env = str(os.environ.get("SAVE_OPTIMIZER_STATE", "")).lower()
    _save_optimizer_state = _sos_env in ("1", "true", "yes", "on")

    # ----------------------------
    # DataLoader 参数（环境变量控制）
    # ----------------------------
    # 这些参数用于提升性能或避免某些多进程数据加载问题
    def _env_int(name: str, default: int) -> int:
        try:
            v = os.environ.get(name)
            return int(v) if v is not None else default
        except Exception:
            return default

    _prefetch = _env_int("DATALOADER_PREFETCH_FACTOR", 2)
    _pin_memory = _env_bool("DATALOADER_PIN_MEMORY", True)
    _persist_workers = _env_bool("DATALOADER_PERSISTENT_WORKERS", False)
    _eval_acc_steps = _env_int("EVAL_ACCUMULATION_STEPS", 128)

    # ----------------------------
    # LR scheduler 配置（环境变量控制）
    # ----------------------------
    _lr_scheduler_type = os.environ.get("LR_SCHEDULER_TYPE", "constant")

    # warmup 既支持 warmup_steps，也支持 warmup_ratio
    # 优先级：
    #   - 如果 env 里设置了 LR_WARMUP_STEPS，则用 steps
    #   - 否则如果 ratio > 0，则根据 total_steps 计算
    _warmup_steps = _env_int("LR_WARMUP_STEPS", None)
    _warmup_ratio = _env_float("LR_WARMUP_RATIO", 0.1)
    if _warmup_steps is None and _warmup_ratio > 0:
        _warmup_steps = int(_warmup_ratio * total_steps)

    # ----------------------------
    # SwanLab 可选集成（类似 wandb，但你这里 report_to="none"）
    # ----------------------------
    callbacks = []
    _sl_enable = str(os.environ.get("SWANLAB_ENABLE", "")).lower() in ("1", "true", "yes", "on", "cloud", "local")
    if _sl_enable:
        try:
            # 过滤 transformers 的 padding_side warning（你明确强制 left padding）
            import warnings
            warnings.filterwarnings("ignore", message=".*For correct generation results, please set.*padding_side.*left.*", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*decoder-only architecture is being used, but right-padding was detected.*", category=UserWarning)

            from swanlab.integration.transformers import SwanLabCallback

            # swanlab project 名称默认：<model_type>-peft
            sl_project = os.environ.get("SWANLAB_PROJECT", f"{model_type}-peft")

            # 实验名前缀：用于统一管理
            exp_prefix = os.environ.get("SWANLAB_EXPERIMENT_PREFIX", "")

            # exp_name 默认取 output_dir 最后一级目录名
            exp_name = Path(output_dir).name
            if exp_prefix:
                exp_name = f"{exp_prefix}_{exp_name}"

            # swanlab mode（cloud/local）可选
            sl_mode = os.environ.get("SWANLAB_MODE", "")
            if sl_mode:
                callbacks.append(SwanLabCallback(project=sl_project, experiment_name=exp_name, mode=sl_mode))
            else:
                callbacks.append(SwanLabCallback(project=sl_project, experiment_name=exp_name))

            # ----------------------------
            # SwanLab Email 回调（可选）
            # ----------------------------
            # 说明：
            # - 你这里额外支持在训练开始/结束/失败时发邮件提醒
            # - email 配置从 YAML 读取（dangerous/email_notify.yaml）
            try:
                import swanlab
                from swanlab.plugin.notification import EmailCallback

                email_yaml = os.environ.get("SWANLAB_EMAIL_YAML", "dangerous/email_notify.yaml")
                if Path(email_yaml).is_file():
                    with open(email_yaml, "r") as _ef:
                        _ecfg = yaml.safe_load(_ef) or {}

                    # 必须包含这些字段才认为配置完整
                    if all(k in _ecfg for k in ("sender_email", "receiver_email", "password", "smtp_server", "port")):
                        _email_cb = EmailCallback(
                            sender_email=str(_ecfg["sender_email"]),
                            receiver_email=str(_ecfg["receiver_email"]),
                            password=str(_ecfg["password"]),
                            smtp_server=str(_ecfg["smtp_server"]),
                            port=int(_ecfg.get("port", 587)),
                            language=str(_ecfg.get("language", "zh")),
                        )

                        # 注册回调到 swanlab
                        swanlab.register_callbacks([_email_cb])

                        # 是否在训练开始时发送邮件（默认开启）
                        _start_env = str(os.environ.get("SWANLAB_EMAIL_ON_START", "1")).lower()
                        if _start_env in ("1", "true", "yes", "on"):
                            try:
                                _msg = f"Output: {output_dir}\nData: {cfg.get('data')}\nSeed: {cfg.get('seed')}\nCfg: {cfg_path}"
                                _email_cb.send_email(subject=f"SwanLab | STARTED | {exp_name}", content=_msg)
                            except Exception:
                                # 邮件失败不影响训练
                                pass
            except Exception:
                # swanlab email 组件导入或注册失败，不影响训练主流程
                pass
        except Exception as e:
            print(f"[{log_tag}][swanlab][warn] Failed to initialize SwanLabCallback: {e}")

    # 评估 batch size：优先从 cfg 里取
    # 注意：这里 eval_batch_size 参数传进来了，但你又用 cfg.get("eval_batch_size",1) 覆盖一次
    # 这意味着 cfg.yaml 中的 eval_batch_size 更权威（如果 cfg 有值）
    _eval_batch_size = int(cfg.get("eval_batch_size", 1) or 1)

    # ----------------------------
    # 构造 GenericLMTrainer
    # ----------------------------
    # GenericLMTrainingArguments 内部类似 HF TrainingArguments
    # 你在这里集中设置：
    # - 学习率、步数、batch、梯度累积、GC
    # - scheduler、warmup
    # - dataloader 多进程参数
    # - 保存/评估策略
    # - seed 等
    trainer = GenericLMTrainer(
        model=model,
        train_dataset=train_data_module.dataset,
        tokenizer=tokenizer,
        args=GenericLMTrainingArguments(
            learning_rate=float(learning_rate),
            max_steps=total_steps,

            # per_device_train_batch_size：单卡 batch size
            per_device_train_batch_size=batch_size,

            # per_device_eval_batch_size：验证单卡 batch size
            per_device_eval_batch_size=_eval_batch_size,

            gradient_accumulation_steps=gradient_accumulation_steps,

            # gradient checkpointing：通过重算换显存
            gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs=_gc_kwargs,

            # optim：默认 adamw_torch
            optim=cfg.get("optim", "adamw_torch"),

            # scheduler 类型：constant/cosine/linear 等（取决于你 TrainingArguments 支持）
            lr_scheduler_type=_lr_scheduler_type,
            warmup_steps=_warmup_steps,

            output_dir=output_dir,

            # logging_steps：多少 step 打一次 log
            logging_steps=logging_steps,

            # dataloader 配置
            dataloader_num_workers=num_data_workers,
            dataloader_prefetch_factor=_prefetch,
            dataloader_pin_memory=_pin_memory,
            dataloader_persistent_workers=_persist_workers,

            # eval_accumulation_steps：评估时累积多少步再做一次 gather
            # 用于避免评估时显存暴涨
            eval_accumulation_steps=_eval_acc_steps,

            # info：自定义信息，会被记录到日志或 checkpoint 元数据中
            info={
                "trainable_params": get_trainable_parameters_ratio(model),
                "cfg_path": cfg_path,
                "logits_to_keep": logits_to_keep,
                "model_type": model_type,
            },

            # 是否保存 optimizer state
            save_optimizer_state=_save_optimizer_state,

            # save_strategy / evaluation_strategy：
            # - 如果 no_save=True -> "no"：不保存
            # - 否则按 steps 保存
            save_strategy="steps" if not no_save else "no",

            # - 如果 skip_eval=True -> "no"：不评估
            # - 否则按 steps 评估
            evaluation_strategy="steps" if not skip_eval else "no",

            # save_steps / eval_steps 的默认计算逻辑：
            # - 这里用 eval_epochs 乘以 “每 epoch 的 iteration 数”
            # - 但注意：你这里的 iteration 数是用 dataset_len // batch_size 估算，
            #   并且没有考虑 gradient_accumulation_steps，也没有考虑多卡（DDP）world_size
            #   这通常是“老脚本兼容”的写法
            save_steps=(
                save_steps_override
                if save_steps_override is not None
                else int(
                    eval_epochs
                    * (
                        len(train_data_module.dataset) // batch_size
                        + (len(train_data_module.dataset) % batch_size > 0)
                    )
                )
            ),
            eval_steps=(
                eval_steps_override
                if eval_steps_override is not None
                else int(
                    eval_epochs
                    * (
                        len(train_data_module.dataset) // batch_size
                        + (len(train_data_module.dataset) % batch_size > 0)
                    )
                )
            ),

            # drop_last=True：丢弃最后不足一个 batch 的数据
            # 好处：避免 batch size 不一致导致的一些 shape 问题
            # 坏处：训练数据少一点点
            dataloader_drop_last=True,

            # report_to="none"：禁用 HF 内置 wandb/tensorboard 报告
            report_to="none",

            seed=seed,
        ),
        compute_metrics=compute_metrics,
        data_collator=train_data_module.data_collator,
        eval_dataset=val_data_module.dataset,

        # callbacks：如果启用了 swanlab，就会注入 SwanLabCallback
        callbacks=callbacks or None,

        # eval_generator：generation 任务需要它
        eval_generator=eval_generator,

        # min_eval_metric_after_epoch：
        # 你项目的自定义逻辑：可能用于 early stop 或 “达到阈值后才开始记录最好结果”
        min_eval_metric_after_epoch=min_eval_metric_after_epoch,
    )

    # ----------------------------
    # 开始训练：带 best-effort 邮件通知
    # ----------------------------
    try:
        # resume_from_checkpoint：
        # - 你在 run_train() 里传入的是 resume_arg（字符串路径）或 None
        # - 这里直接交给 trainer.train()
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # 训练结束邮件（可选）
        try:
            _fin_env = str(os.environ.get("SWANLAB_EMAIL_ON_FINISH", "1")).lower()
            if _sl_enable and _fin_env in ("1", "true", "yes", "on"):
                from scripts.utils.email_notify import send_event_email
                send_event_email("FINISHED", group=Path(output_dir).name, details=f"Finished OK: {output_dir}")
        except Exception:
            pass

    except Exception as _e:
        # 训练失败邮件（可选）
        try:
            from scripts.utils.email_notify import send_event_email
            import traceback
            tb = "".join(traceback.format_exception_only(type(_e), _e))
            send_event_email("FAILED", group=Path(output_dir).name, details=f"Failed: {tb}")
        except Exception:
            pass

        # 失败后必须继续抛异常，保证外层脚本/调度系统能捕获到失败状态
        raise


def run_train(
    output_dir,
    cfg_path,
    model,
    data,
    model_type: str = "auto",  # 新增：模型类型参数
    val_data=None,
    val_data_split="val",
    tokenizer="EleutherAI/gpt-neox-20b",  # 为了配置兼容而保留，但实际不使用（真正 tokenizer 来自 prepare_lat_model_and_tokenizer）
    num_epochs=10,
    prec="bf16",
    peft=None,
    optim="adamw_torch",
    learning_rate=5e-4,
    gradient_accumulation_steps=1,
    num_data_workers=8,
    batch_size=4,
    eval_batch_size=1,
    eval_gen=None,
    backend="cuda",  # 兼容保留，不使用
    debug=False,
    resume=False,
    overwrite=False,
    lock=False,
    no_save=False,
    skip_eval=False,
    eval_epochs=1,
    min_eval_metric_after_epoch=None,
    seed=42,
    is_sdlora=False,  # 兼容保留
    gradient_checkpointing=False,
    logits_to_keep=None,
):
    """
    统一 Linear Attention 的 run_train 入口（更上层的控制逻辑）。

    它负责：
    - 决定 model_type（auto -> env -> 参数）
    - overwrite/resume/lock 的输出目录策略
    - 加载模型与 tokenizer（prepare_lat_model_and_tokenizer）
    - 计算每 epoch iteration 数、总 steps
    - 将控制参数传给 build_and_run_trainer_lat

    注意：
    - 这里是“训练调度层”；真正 Trainer 构建在 build_and_run_trainer_lat
    """

    # ------------------------------------------------------------
    # 决定模型类型（model_type）：
    # ------------------------------------------------------------
    # 如果传入 model_type="auto"
    #   -> 再读 env MODEL_TYPE
    #       若 env 不是 "auto"，则 env 覆盖
    if model_type == "auto":
        env_model_type = os.environ.get("MODEL_TYPE", "auto")
        if env_model_type != "auto":
            model_type = env_model_type

    # 日志标签
    log_tag = model_type.upper() if model_type != "auto" else "LAT"

    # ------------------------------------------------------------
    # Legacy：SD-LoRA 特殊逻辑（兼容保留）
    # ------------------------------------------------------------
    # overwrite + is_sdlora 时要求 output_dir 已存在
    # 这通常用于“在已有输出目录上继续写入”
    if overwrite and is_sdlora:
        assert Path(output_dir).exists()

    # ------------------------------------------------------------
    # cfg 快照：
    # ------------------------------------------------------------
    # locals() 会把当前函数的所有局部变量打包成 dict
    # 这用于保存 cfg.yaml，保证你把所有训练参数记录下来
    # 注意：
    #   - locals() 里会包含 model 对象等不可序列化内容吗？
    #   - 这里 locals() 发生在 model 加载之前，所以 model 还是字符串入参，不是实际模型对象
    cfg = {**locals()}

    # ------------------------------------------------------------
    # 输出目录策略：lock / overwrite / resume
    # ------------------------------------------------------------
    created_lock = False
    if not overwrite:
        if lock:
            # _lock_share 返回 True 表示“锁已存在，应该 skip”
            if _lock_share(str(output_dir), log_tag):
                return
            created_lock = True

        # 如果 output_dir/cfg.yaml 已存在，说明该实验可能已经跑过
        if (Path(output_dir) / "cfg.yaml").exists():
            if resume:
                # resume=True：允许继续训练
                resume_from_checkpoint = True
            else:
                # resume=False：直接报错，防止无意覆盖实验
                assert False, str(Path(output_dir) / "cfg.yaml") + " exists!"
        else:
            resume_from_checkpoint = False
    else:
        # overwrite=True：无论是否存在 cfg.yaml，都不从 checkpoint resume（重新开始）
        resume_from_checkpoint = False

    # ------------------------------------------------------------
    # 多 epoch + no_save 的安全提示
    # ------------------------------------------------------------
    # 这段逻辑写得比较“反直觉”，但它意图是：
    #   - 某些小任务（glue/spider_1000）允许不保存 checkpoint 跑多 epoch
    #   - 否则如果你跑多 epoch 但 no_save=True，会很危险（断了就全没了）
    if not (
        data.startswith("glue_")
        or data in ("glue_rte", "glue_mrpc", "glue_cola", "spider_1000")
        or not (no_save and num_epochs > 1)
    ):
        print("Training for more than one epoch without saving ckpts!")

    # ------------------------------------------------------------
    # 加载模型（统一适配器）
    # ------------------------------------------------------------
    print(f"Loading {log_tag} model: {model}")
    model_id = model

    # prepare_lat_model_and_tokenizer 返回：
    # - model：已加载的 torch.nn.Module（可能已注入 PEFT）
    # - tokenizer_obj：正确的 tokenizer
    # - 第三个返回值这里用 _ 忽略（可能是 config 或其他元信息）
    model, tokenizer_obj, _ = prepare_lat_model_and_tokenizer(
        model_type=model_type,
        model_id=model_id,
        prec=prec,
        debug=debug,
        peft_json_path=peft,
    )

    # ------------------------------------------------------------
    # 再次强制 left padding（与 build_and_run_trainer_lat 的策略一致）
    # ------------------------------------------------------------
    force_left = get_lat_env_bool("FORCE_LEFT_PAD", "1")
    if force_left:
        try:
            tokenizer_obj.padding_side = "left"
            if getattr(tokenizer_obj, "pad_token_id", None) is None and getattr(tokenizer_obj, "eos_token", None) is not None:
                tokenizer_obj.pad_token = tokenizer_obj.eos_token
            print(f"[{log_tag}] Using left padding for decoder-only generation.")
        except Exception as e:
            print(f"[{log_tag}][warn] Failed to enforce left padding policy: {e}")
    else:
        print(f"[{log_tag}] Respecting tokenizer's original padding policy.")

    # ------------------------------------------------------------
    # 加载一次训练集，仅用于计算长度（its_per_epoch）
    # ------------------------------------------------------------
    # 为什么要“for_len”？
    #   - 你后面要计算每 epoch 的 step 数（its_per_epoch）
    #   - 为避免重复构造大 trainer，这里先轻量加载一次模块拿 dataset 长度
    train_data_module_for_len = load_dataset(
        data, tokenizer_obj, "train", return_module=True
    )

    # its_per_epoch = ceil(len(dataset) / batch_size)
    # 注意：
    #   - 这里没有考虑 DDP world_size
    #   - 也没有考虑 gradient_accumulation_steps
    #   - 但保持老脚本逻辑一致（兼容）
    its_per_epoch = int(
        np.ceil(len(train_data_module_for_len.dataset) / batch_size)
    )

    # ------------------------------------------------------------
    # logging_steps：默认每 50 step 或每 epoch step 数更小者
    # env HP_LOGGING_STEPS 可覆盖
    # ------------------------------------------------------------
    env = os.environ
    logging_steps = min(50, its_per_epoch)
    try:
        if env.get("HP_LOGGING_STEPS"):
            logging_steps = int(env.get("HP_LOGGING_STEPS"))
    except Exception:
        pass

    # ------------------------------------------------------------
    # total_steps：默认 num_epochs * its_per_epoch
    # env HP_MAX_STEPS 可覆盖（最高优先级）
    # ------------------------------------------------------------
    total_steps = int(num_epochs * its_per_epoch)
    try:
        if env.get("HP_MAX_STEPS"):
            total_steps = int(env.get("HP_MAX_STEPS"))
    except Exception:
        pass

    # ------------------------------------------------------------
    # eval_steps/save_steps override（可选）
    # ------------------------------------------------------------
    eval_steps_override = None
    save_steps_override = None
    try:
        if env.get("HP_EVAL_STEPS"):
            eval_steps_override = int(env.get("HP_EVAL_STEPS"))
    except Exception:
        pass
    try:
        if env.get("HP_SAVE_STEPS"):
            save_steps_override = int(env.get("HP_SAVE_STEPS"))
    except Exception:
        pass

    # W&B run name：通常用 output_dir 去掉 weights/ 前缀
    os.environ["WANDB_NAME"] = str(output_dir).replace("weights/", "")

    print("Dropping last batch")

    # ------------------------------------------------------------
    # Resume：找到最新 checkpoint-* 目录
    # ------------------------------------------------------------
    resume_arg = None
    if resume_from_checkpoint:
        last_ckpt = _find_last_checkpoint(Path(output_dir))
        if last_ckpt is None:
            raise RuntimeError(f"[{log_tag}] --resume was set but no checkpoint-* found under {output_dir}")
        resume_arg = str(last_ckpt)
        print(f"[{log_tag}] Resuming from checkpoint: {resume_arg}")

    # ------------------------------------------------------------
    # 调用训练器构建 + 开训
    # ------------------------------------------------------------
    try:
        build_and_run_trainer_lat(
            model=model,
            tokenizer=tokenizer_obj,
            model_type=model_type,
            output_dir=str(output_dir),
            cfg=cfg,
            cfg_path=cfg_path,
            learning_rate=learning_rate,
            total_steps=total_steps,
            logging_steps=logging_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_data_workers=num_data_workers,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            eval_epochs=eval_epochs,
            skip_eval=skip_eval,
            no_save=no_save,
            eval_steps_override=eval_steps_override,
            save_steps_override=save_steps_override,
            eval_gen=eval_gen,
            resume_from_checkpoint=resume_arg,
            min_eval_metric_after_epoch=min_eval_metric_after_epoch,
            seed=seed,
            data=data,
            val_data=val_data,
            val_data_split=val_data_split,
            debug=debug,
            gradient_checkpointing=gradient_checkpointing,
            logits_to_keep=logits_to_keep,
        )
    finally:
        # ------------------------------------------------------------
        # finally：如果创建了 lock，则释放 lock
        # ------------------------------------------------------------
        if created_lock:
            try:
                lock_path = Path("share/lock") / str(output_dir)
                lock_path.unlink(missing_ok=True)
                print(f"[{log_tag}][lock] released {lock_path}")
            except Exception as e:
                print(f"[{log_tag}][lock][warn] failed to remove lock: {e}")


def get_output_path_for_cfg(cfg_path, cfg):
    """
    根据 cfg 路径与 cfg 内容决定输出目录。

    目标路径规则：
      /home/user/mzs_h/output/benchmark/glue/<data>_seed<seed>/<yaml_stem>

    其中：
    - yaml_stem = Path(cfg_path).stem（配置文件名去掉扩展名）
    - data = cfg["data"]
    - seed = cfg["seed"]

    fallback：
    - 如果 data 或 seed 缺失：
      /home/user/mzs_h/output/benchmark/glue/cola_gla/<yaml_stem>

    注意：
    - 这里路径写死在 /home/user/mzs_h/...，强耦合你的运行环境
    - 若换机器，需要修改这里或通过外部传参
    """
    yaml_stem = Path(cfg_path).stem
    data = cfg.get("data")
    seed = cfg.get("seed")

    if data and seed is not None:
        folder = f"{data}_seed{seed}"
        return Path("/home/user/mzs_h/output/benchmark/glue") / folder / yaml_stem
    return Path("/home/user/mzs_h/output/benchmark/glue/cola_gla") / yaml_stem


def _find_last_checkpoint(root: Path) -> Optional[Path]:
    """
    在 output_dir 下寻找 checkpoint-* 子目录，并返回 step 最大的那个。

    约定：
    - checkpoint 目录命名形如：checkpoint-1000、checkpoint-2000
    - step_of() 用 p.name.split("-")[-1] 解析最后一段数字作为 step

    返回：
    - Path：最新 checkpoint 目录
    - None：没有任何 checkpoint-* 目录或 root 不存在
    """
    if not root.exists():
        return None
    try:
        candidates = [p for p in root.glob("checkpoint-*") if p.is_dir()]
        if not candidates:
            return None

        def step_of(p: Path) -> int:
            try:
                return int(p.name.split("-")[-1])
            except Exception:
                return -1

        candidates.sort(key=step_of)
        return candidates[-1] if candidates else None
    except Exception:
        return None


def main():
    """
    CLI 入口：
    - 解析命令行参数
    - 读取 YAML cfg
    - 应用 env 覆盖（MODEL_TYPE / HP_* 等）
    - 自动注入 eval_gen（针对 generation 任务）
    - 决定 output_dir
    - 合并参数后调用 run_train()
    """
    parser = argparse.ArgumentParser(description="Unified Linear Attention Training")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--model-type", type=str, default="auto",
                        help="Model type: gla, retnet, mamba2, or auto (default: auto)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--lock", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--prec")
    parser.add_argument("--device")
    args = parser.parse_args()

    # --device：用于设置某种自定义的可见设备变量 VISIBLE_DEVICES
    # 注意：
    # - 常见的是 CUDA_VISIBLE_DEVICES
    # - 但你这里用 VISIBLE_DEVICES，可能你项目内部或启动脚本会读取它
    if args.device is not None:
        os.environ["VISIBLE_DEVICES"] = args.device

    # 读取 YAML 配置
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)

    # ------------------------------------------------------------
    # 应用环境变量覆盖
    # ------------------------------------------------------------
    env = os.environ

    def _maybe(v, cast):
        """
        辅助函数：
        - 若 v 为 None 或 ""，返回 None
        - 否则返回 cast(v)
        用于简化 HP_* env 的读取
        """
        return cast(v) if v is not None and v != "" else None

    # MODEL_TYPE env override：最高优先级覆盖命令行
    model_type_env = env.get("MODEL_TYPE")
    if model_type_env:
        args.model_type = model_type_env

    # ------------------------------------------------------------
    # HP_DATA：任务别名映射（尤其是 GLUE）
    # ------------------------------------------------------------
    # 用途：
    # - 允许通过 env 传入简写：rte/mrpc/cola...
    # - 脚本会自动变成 glue-tvt_<task>
    data_env = env.get("HP_DATA")
    if data_env:
        glue_tasks = {
            "rte", "mrpc", "cola", "sst2", "qnli", "qqp", "mnli", "wnli",
        }
        accepted_prefixes = (
            "glue", "samsum", "dart", "spider", "mnist", "cifar", "piqa", "boolq", "arc",
        )
        if data_env in glue_tasks:
            cfg["data"] = f"glue-tvt_{data_env}"
        elif data_env == "cifar":
            cfg["data"] = "cifar-tvt"
        elif data_env == "spider":
            cfg["data"] = "spider-tvt"
        else:
            # 如果 data_env 不是标准 glue alias，就原样写入
            # accepted_prefixes 目前没有真正用于过滤（两分支一样），但保留结构方便以后扩展校验
            cfg["data"] = (
                data_env
                if data_env.startswith(accepted_prefixes)
                else data_env
            )

    # batch size 覆盖
    bs_env = _maybe(env.get("HP_BATCH_SIZE"), int)
    if bs_env is not None:
        cfg["batch_size"] = bs_env

    # learning rate 覆盖
    lr_env = _maybe(env.get("HP_LR"), float)
    if lr_env is not None:
        cfg["learning_rate"] = lr_env

    # epochs 覆盖
    epochs_env = _maybe(env.get("HP_EPOCHS"), int)
    if epochs_env is not None:
        cfg["num_epochs"] = epochs_env

    # precision 覆盖
    prec_env = env.get("HP_PREC")
    if prec_env:
        cfg["prec"] = prec_env

    # seed 覆盖
    seed_env = _maybe(env.get("HP_SEED"), int)
    if seed_env is not None:
        cfg["seed"] = seed_env

    # no_save 覆盖（字符串转 bool）
    no_save_env = env.get("HP_NO_SAVE")
    if no_save_env is not None:
        cfg["no_save"] = str(no_save_env).lower() in ("1", "true", "yes", "on")

    # val split 覆盖：只能是 train/val/test
    val_split_env = env.get("HP_VAL_SPLIT")
    if val_split_env in {"train", "val", "test"}:
        cfg["val_data_split"] = val_split_env

    # eval batch size 覆盖
    eval_bs_env = _maybe(env.get("HP_EVAL_BATCH_SIZE"), int)
    if eval_bs_env is not None and eval_bs_env > 0:
        cfg["eval_batch_size"] = eval_bs_env

    # ------------------------------------------------------------
    # eval_gen 自动注入：针对 generation 类任务
    # ------------------------------------------------------------
    def _truthy(x: Optional[str]) -> bool:
        if x is None:
            return False
        return str(x).lower() in ("1", "true", "yes", "on")

    data_name = str(cfg.get("data", ""))
    is_gen_task = any([
        data_name.startswith("samsum"),
        data_name.startswith("dart"),
        data_name.startswith("spider"),
    ])

    # EVAL_GEN env 强制开启 generation 评估（即使任务名不是上述前缀）
    force_eval_gen = _truthy(env.get("EVAL_GEN"))

    # 如果 cfg 里没有 eval_gen，并且该任务是 generation 或被 env 强制
    # 则自动注入 eval_gen 的 max/min length
    if (cfg.get("eval_gen") is None) and (is_gen_task or force_eval_gen):
        max_len = _maybe(env.get("EVAL_GEN_MAX_LENGTH"), int) or 1024
        min_len = _maybe(env.get("EVAL_GEN_MIN_LENGTH"), int) or 5
        cfg["eval_gen"] = {
            "max_length": int(max_len),
            "min_length": int(min_len),
        }

    # ------------------------------------------------------------
    # 输出目录
    # ------------------------------------------------------------
    output_dir = get_output_path_for_cfg(args.cfg, cfg)

    # ------------------------------------------------------------
    # 合并 cfg 与 CLI args -> run_train 的参数
    # ------------------------------------------------------------
    train_args = {
        **cfg,
        **{k: v for k, v in vars(args).items() if v is not None},
        "output_dir": str(output_dir),
        "model_type": args.model_type,  # 显式写入 model_type
    }

    # run_train 期待 cfg_path 字段，而 CLI 参数名是 cfg
    train_args["cfg_path"] = train_args.pop("cfg")

    # device 参数在 run_train 里不需要（已经用于设置 env）
    if "device" in train_args:
        del train_args["device"]

    # 调用训练入口
    run_train(**train_args)


if __name__ == "__main__":
    main()