"""
Unified Linear Attention Model Adapter.

本模块的目标：给“线性注意力（Linear Attention）”家族模型提供一个**统一的准备入口**，
用于训练/微调前的模型与 tokenizer 加载、以及可选的 PEFT/LoRA 挂载配置。
同时保留旧的 GLA 入口函数以保证历史代码完全兼容。

-------------------------------------------------------------------------------
这段 docstring 是“对外说明书”，面向使用者/调用方，而不是面向实现者。
它解释了：模块是干什么的、支持哪些模型、有哪些环境变量可以覆盖配置、以及如何使用。

Design Principles（设计原则）:
==================
1. **Backward Compatibility（向后兼容）**
   - `prepare_gla_model_and_tokenizer()` 必须继续存在
   - 行为必须和原实现一致（尤其是 dtype 映射、PEFT 参数覆盖策略等）
   - 旧训练脚本不需要改代码即可继续运行

2. **Unified Interface（统一接口）**
   - `prepare_lat_model_and_tokenizer()` 作为新统一入口
   - 能对所有支持的 Linear Attention 模型类型工作（gla/retnet/mamba2/auto）

3. **PEFT Support（PEFT 支持）**
   - 同一套 PEFT/LoRA JSON 配置加载与环境变量覆盖逻辑，适用于所有模型类型
   - target_modules 若未指定，按模型类型提供合理默认值

Supported Models（支持模型）:
================
- gla: Gated Linear Attention（门控线性注意力）
- retnet: Retentive Network（RetNet）
- mamba2: Mamba2 State Space Model（Mamba2 状态空间模型）

Environment Variables for PEFT overrides（用于覆盖 PEFT 配置的环境变量）:
========================================
- HP_PEFT_R: 覆盖 LoRA rank（秩）
- HP_PEFT_ALPHA: 覆盖 LoRA alpha
- HP_PEFT_DROPOUT: 覆盖 LoRA dropout
- HP_INIT: 覆盖 init_lora_weights（如 "pissa", "pissa_niter_4"）
- HP_PISSA_FAST: 若设置且 init 为 "pissa"，则切换到 "pissa_niter_4"（更快的 SVD 初始化）

Usage（用法示例）:
======
    from lat_adapter import prepare_lat_model_and_tokenizer

    # For GLA (backward compatible)
    model, tokenizer, peft_cfg = prepare_lat_model_and_tokenizer(
        model_type="gla",
        model_id="fla-hub/gla-1.3B-100B",
        prec="bf16",
        debug=False,
        peft_json_path="configs/peft_lora.json",
    )

    # For RetNet
    model, tokenizer, peft_cfg = prepare_lat_model_and_tokenizer(
        model_type="retnet",
        model_id="fla-hub/retnet-1.3B",
        prec="bf16",
        debug=False,
        peft_json_path="configs/peft_lora.json",
    )

    # With auto-detection
    model, tokenizer, peft_cfg = prepare_lat_model_and_tokenizer(
        model_type="auto",
        model_id="fla-hub/gla-1.3B-100B",
        ...
    )
"""

# ---------------------------
# 标准库导入
# ---------------------------
import json
import os
from typing import Any, Dict, Optional, Tuple

# ---------------------------
# 第三方库导入
# ---------------------------
import torch

# ---------------------------
# 统一模型加载器导入
# ---------------------------
# 这里从 mamba_ssm_peft.utils.lat_model_loader 导入统一的加载接口：
# - load_lat_model: 负责根据 model_type/model_id 加载模型与 tokenizer
# - get_lat_env / get_lat_env_bool: “看起来”是用于读取环境变量的工具函数
#   注意：在当前文件中，它们被导入但没有被使用（可能是为了未来扩展，或历史遗留）
from mamba_ssm_peft.utils.lat_model_loader import (
    load_lat_model,
    get_lat_env,
    get_lat_env_bool,
)


def _dtype_from_prec(prec: str) -> torch.dtype:
    """
    Convert precision string to torch dtype.
    将“字符串形式的精度参数”转换为 torch.dtype。

    这个函数是一个“小型映射器”：
    - 输入: "bf16" / "fp16" / "fp32"
    - 输出: torch.bfloat16 / torch.float32 等

    注意点（非常关键）:
    Note: fp16 is mapped to bfloat16 for consistency with the original implementation.
    - 这里明确声明：即使用户传 "fp16"，也会映射到 torch.bfloat16
    - 这通常是为了保持历史训练脚本的行为一致
    - 也可能是因为某些线性注意力/特定算子在 bf16 上更稳定，
      或底层实现/硬件（例如某些 GPU）对 bf16 更友好
    - 这属于“向后兼容/既有行为保留”，不是常规意义的 fp16

    Args:
        prec: 精度字符串。理论上必须是 mapping 的键之一。

    Returns:
        torch.dtype: torch 的 dtype，用于模型权重/计算精度设置。

    Raises:
        ValueError:
            如果 prec 不是已知 key，会抛出异常，避免静默使用错误 dtype 导致训练不一致。
    """
    # mapping 是一个“白名单”：
    # 只有出现在这里的 prec 才被接受。
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.bfloat16,  # 兼容旧行为：fp16 -> bfloat16（非常规但刻意为之）
        "fp32": torch.float32,
    }

    # 如果用户输入不在 mapping 中，说明调用方传参错误或拼写错误
    # 这里直接 raise，避免出现“偷偷用默认值”的危险行为。
    if prec not in mapping:
        raise ValueError(f"Unknown precision '{prec}'. Supported: {list(mapping.keys())}")

    # 返回对应 dtype
    return mapping[prec]


def _apply_peft_env_overrides(peft_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides to PEFT configuration.
    将“环境变量”对 PEFT/LoRA JSON 配置进行覆盖（override）。

    为什么要做环境变量覆盖？
    - 在训练平台/集群环境中（例如 Slurm、K8s、CI/CD、超参搜索）
      很常见用环境变量临时改 LoRA 的 r/alpha/dropout/init，而不改 JSON 文件本身
    - 这样同一份配置文件可以复用，训练脚本也不需要再加很多 CLI 参数

    支持的覆盖项（本函数只管这些）:
    - HP_PEFT_R: LoRA rank (int)
    - HP_PEFT_ALPHA: LoRA alpha (int)
    - HP_PEFT_DROPOUT: LoRA dropout (float)
    - HP_INIT: init_lora_weights (str, 例如 "pissa", "pissa_niter_4")
    - HP_PISSA_FAST: 如果为真 且 init 为 "pissa"，则切换到 "pissa_niter_4"

    重要的实现策略：
    - 覆盖是“尽力而为”（best-effort）:
      解析失败（ValueError/TypeError）就 pass，不让程序崩
      这是为了适应环境变量可能被误配（例如空字符串、非数字）
    - HP_INIT 优先级高于 HP_PISSA_FAST:
      如果用户显式设置了 HP_INIT，就完全按 HP_INIT 来，不再考虑 fast 标志
    - 对 HP_PISSA_FAST 的“真值判断”采用字符串黑名单：
      "0"/"false"/"no"/"off" 视为 False，其它非空视为 True（非常常见的实践）

    Args:
        peft_json: 原始的 PEFT 配置 dict（从 JSON 文件读出来的）

    Returns:
        dict: 可能被覆盖后的 PEFT 配置 dict（原 dict 会被就地修改，因为是同一个对象）
    """
    # os.environ 是一个类似 dict 的对象，读取当前进程环境变量
    env = os.environ

    # ---------------------------
    # 1) HP_PEFT_R：覆盖 LoRA rank
    # ---------------------------
    # LoRA rank 控制低秩分解的秩，影响参数量/表达能力/显存等
    r_env = env.get("HP_PEFT_R")
    if r_env is not None:
        # 只要环境变量存在（即使是空字符串），我们就尝试解析成 int
        # 如果解析失败，静默忽略（pass）
        try:
            peft_json["r"] = int(r_env)
        except (ValueError, TypeError):
            # ValueError: 例如 r_env="abc"
            # TypeError: 例如 r_env 不是可 int() 的类型（较少见）
            pass

    # ---------------------------
    # 2) HP_PEFT_ALPHA：覆盖 LoRA alpha
    # ---------------------------
    # alpha 通常用于缩放 LoRA 更新（与 r 有关联，常见组合：alpha = r 或 2r 等）
    alpha_env = env.get("HP_PEFT_ALPHA")
    if alpha_env is not None:
        try:
            peft_json["lora_alpha"] = int(alpha_env)
        except (ValueError, TypeError):
            pass

    # ---------------------------
    # 3) HP_PEFT_DROPOUT：覆盖 LoRA dropout
    # ---------------------------
    # dropout 用于 LoRA 分支的正则化，通常是 0.0 ~ 0.1 之类
    drop_env = env.get("HP_PEFT_DROPOUT")
    if drop_env is not None:
        try:
            peft_json["lora_dropout"] = float(drop_env)
        except (ValueError, TypeError):
            pass

    # ---------------------------
    # 4) HP_INIT：覆盖 init_lora_weights
    # ---------------------------
    # init_lora_weights 控制 LoRA 权重的初始化方式。
    # 例如：
    # - 默认初始化
    # - "pissa"：可能是一种 SVD/分解类初始化策略（具体依赖 peft 或扩展实现）
    # - "pissa_niter_4"：可能代表迭代次数更少的快速近似
    init_env = env.get("HP_INIT")
    if init_env:
        # 注意：这里判断条件是 if init_env: —— 空字符串会被当 False
        # 即：HP_INIT="" 等价于未提供
        peft_json["init_lora_weights"] = init_env
    else:
        # ---------------------------
        # 5) HP_PISSA_FAST：仅在未显式设置 HP_INIT 时才生效
        # ---------------------------
        # 如果 fast 标志为真，且当前 init 为 "pissa"，则替换为 "pissa_niter_4"
        # 意图：保留 pissa 初始化的风格，但降低计算开销
        fast_pissa_env = env.get("HP_PISSA_FAST")
        try:
            # fast_pissa_env 可能是 None/""/"0"/"false"/"1"/"true" 等
            # 这里用字符串方式做“宽松真值判断”
            if fast_pissa_env and str(fast_pissa_env).lower() not in ("0", "false", "no", "off"):
                init_val = peft_json.get("init_lora_weights", None)
                # 只在 init_val 是字符串且等于 "pissa"（忽略大小写）时才替换
                if isinstance(init_val, str) and init_val.lower() == "pissa":
                    peft_json["init_lora_weights"] = "pissa_niter_4"
        except Exception:
            # 这里 catch 所有异常（更宽松），避免环境变量/配置异常导致整体崩溃
            pass

    # 返回修改后的 dict（注意：这个 dict 通常已经被原地修改）
    return peft_json


def _get_target_modules_for_model(model_type: str, model: Any) -> Optional[list]:
    """
    Get default LoRA target modules for a specific model type.
    为不同模型类型提供“默认的 LoRA target_modules”。

    背景解释：
    - 在 PEFT 的 LoRA 配置里，target_modules 指定“哪些线性层/投影层要插 LoRA”
    - 不同架构的模块命名不一样（例如 Transformer 常见 q_proj/k_proj/v_proj/o_proj）
    - 如果用户的 peft_json 没写 target_modules，就需要给一个“合理默认值”
    - 若不给默认值，有时 PEFT 可能尝试自动检测，但：
      1) 自动检测可能不稳定/依赖实现细节
      2) 对非标准架构（如 mamba2）自动检测可能失败或覆盖不全

    注意：本函数签名里有 model 参数，但当前实现并没有使用它。
    - 它的存在通常表示未来可能会“检查模型结构”来动态推断模块名
    - 或者在不同版本模型里模块命名变化时，用 model 做兜底检测

    Args:
        model_type: 模型类型字符串（"gla"/"retnet"/"mamba2" 等）
        model: 已加载的模型对象（当前未用到）

    Returns:
        list[str] 或 None
        - 返回 list 表示明确指定这些模块名作为 target_modules
        - 返回 None 表示：不提供默认值，交给外部/PEFT 自己处理
    """
    # defaults 是按模型类型给出的“经验默认模块名列表”
    # 一般来说，这些都是影响注意力/FFN 的关键投影层。
    defaults = {
        # 对 gla 与 retnet：按类 Transformer 的投影命名习惯
        # - q_proj/k_proj/v_proj/o_proj：注意力的 QKV 与输出投影
        # - gate_proj/up_proj/down_proj：常见于 gated-MLP 或 SwiGLU 等 FFN 结构
        "gla": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "retnet": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],

        # 对 mamba2：典型结构不同，不一定有 q/k/v/o
        # 常见的是输入投影/输出投影等模块名
        "mamba2": ["in_proj", "out_proj"],
    }

    # dict.get：若 model_type 不在 defaults 中，返回 None
    # 这意味着未知模型类型时不强行猜测 target_modules
    return defaults.get(model_type)


def prepare_lat_model_and_tokenizer(
    model_type: str,
    model_id: str,
    prec: str,
    debug: bool,
    peft_json_path: Optional[str],
) -> Tuple[Any, Any, Optional[Any]]:
    """
    Prepare a Linear Attention model + tokenizer and (optionally) attach PEFT LoRA.
    统一入口：加载（线性注意力家族）模型与 tokenizer，并按需挂载 LoRA（PEFT）。

    这是本模块的核心函数。调用方只需要提供：
    - model_type：模型类型（gla/retnet/mamba2/auto）
    - model_id：HF 模型名或本地路径
    - prec：精度字符串（bf16/fp16/fp32）
    - debug：是否调试模式（调试时用 CPU，避免 GPU 依赖或节省资源）
    - peft_json_path：LoRA 配置 JSON 路径；为 None 则不启用 PEFT

    内部流程概览：
    1) 根据 debug 决定 device（cpu/cuda）
    2) 根据 prec 决定 dtype（通过 _dtype_from_prec）
    3) 通过统一 loader：load_lat_model(...) 加载 model/tokenizer
       - loader 也会返回“解析后的真实 model_type”（例如 auto 推断结果）
    4) 如果提供 peft_json_path：
       - 读取 JSON
       - 应用环境变量覆盖（_apply_peft_env_overrides）
       - 如果 JSON 未指定 target_modules，则按模型类型给默认值
       - 构造 LoraConfig 并用 get_peft_model 包装模型
    5) 返回 (model, tokenizer, peft_cfg)

    Args:
        model_type:
            模型类型字符串：
            - "gla"/"retnet"/"mamba2"：明确指定
            - "auto"：由 loader 根据 model_id 或模型配置自动推断
        model_id:
            HuggingFace 模型 ID（例如 "fla-hub/gla-1.3B-100B"）
            或本地权重路径
        prec:
            "bf16"/"fp16"/"fp32"
            注意：这里的 "fp16" 仍会映射到 bfloat16（向后兼容行为）
        debug:
            True 表示使用 CPU；False 表示使用 CUDA
            常见用途：
            - 单元测试
            - 本地无 GPU 环境快速跑通代码
            - 快速验证加载/配置逻辑
        peft_json_path:
            LoRA 配置 JSON 文件路径；None 则跳过 PEFT

    Returns:
        (model, tokenizer, peft_cfg)
        - model: 可能已经被 get_peft_model 包装（即模型内部插入 LoRA Adapter）
        - tokenizer: 从 loader 返回的 tokenizer
        - peft_cfg: LoraConfig 对象；若未启用 PEFT 则为 None
    """
    # ---------------------------
    # Step 1: 决定 device 与 dtype
    # ---------------------------
    # debug=True -> 用 CPU（避免 CUDA 依赖）
    # debug=False -> 默认用 GPU（"cuda"）
    device = "cpu" if debug else "cuda"

    # 将字符串 prec 转成 torch.dtype
    # 如果传入未知 prec，会在这里直接抛 ValueError
    dtype = _dtype_from_prec(prec)

    # ---------------------------
    # Step 2: 使用统一 loader 加载模型与 tokenizer
    # ---------------------------
    # load_lat_model 的返回值是一个 dict-like（这里假设是 dict）
    # 约定字段：
    # - "model": 已加载的模型对象
    # - "tokenizer": tokenizer
    # - "model_type": loader 解析/确认后的 model_type（尤其对 "auto" 很重要）
    #
    # trust_remote_code=True:
    # - 允许 HuggingFace 加载仓库里的自定义代码
    # - 这对很多非标准架构（特别是线性注意力/状态空间）是必须的
    # - 风险点：远端代码执行（一般需要你信任模型仓库）
    loaded = load_lat_model(
        model_type=model_type,
        model_id=model_id,
        trust_remote_code=True,
        device=device,
        dtype=dtype,
    )

    # 取出模型与 tokenizer
    model = loaded["model"]
    tokenizer = loaded["tokenizer"]

    # 解析后的模型类型：例如 model_type="auto" 时，
    # loader 可能根据模型配置推断出实际是 "gla" 或 "retnet" 等
    resolved_model_type = loaded["model_type"]

    # ---------------------------
    # Step 3: 根据 peft_json_path 决定是否启用 PEFT/LoRA
    # ---------------------------
    peft_cfg = None  # 默认不启用 PEFT 时返回 None

    if peft_json_path is not None:
        # Lazy import（延迟导入）：
        # - 只有在真正需要 PEFT 时才 import peft
        # - 避免在“只加载模型不加 LoRA”的场景里强依赖 peft 包
        # - 也能减少启动开销
        from peft import LoraConfig, get_peft_model

        # 读取 JSON 配置文件
        # - 这里假设 peft_json_path 路径有效且可读
        # - 若文件不存在/JSON 格式错误，会抛异常（当前代码不捕获）
        with open(peft_json_path, "r") as f:
            peft_json = json.load(f)

        # 应用环境变量覆盖
        # - 这一步会根据 HP_PEFT_R/ALPHA/DROPOUT/INIT/PISSA_FAST 调整 peft_json
        # - 解析失败会静默忽略（保持原值）
        peft_json = _apply_peft_env_overrides(peft_json)

        # 如果 JSON 没有指定 target_modules，或显式指定为 None：
        # - 我们按 resolved_model_type 提供默认 target_modules
        # - 这能提升“开箱即用”的成功率，避免 PEFT 自动检测失败
        if "target_modules" not in peft_json or peft_json["target_modules"] is None:
            default_targets = _get_target_modules_for_model(resolved_model_type, model)
            if default_targets:
                peft_json["target_modules"] = default_targets

        # 用最终 peft_json 构造 LoraConfig
        # 注意：LoraConfig(**peft_json) 要求 peft_json 的 key 与 LoraConfig 参数匹配
        # 如果 JSON 里有拼错/无效字段，会在这里抛 TypeError
        peft_cfg = LoraConfig(**peft_json)

        # 用 PEFT 包装原模型：
        # - get_peft_model 会在 target_modules 指定的层上注入 LoRA adapter
        # - 返回的新 model 仍是 torch.nn.Module，但内部结构已经包含可训练的 LoRA 参数
        # - 通常 base model 权重会被冻结（取决于 peft 设置与训练脚本）
        model = get_peft_model(model, peft_cfg)

    # 最终返回：
    # - model：可能已注入 LoRA，也可能是原模型
    # - tokenizer：原样返回
    # - peft_cfg：LoraConfig 或 None
    return model, tokenizer, peft_cfg


# ============================================================================
# BACKWARD COMPATIBILITY: GLA-specific function
# ============================================================================
# 这部分是“向后兼容层”：
# - 旧代码可能只知道 prepare_gla_model_and_tokenizer
# - 新实现用 prepare_lat_model_and_tokenizer 统一实现
# - 但必须保留旧函数名与旧签名，并保证行为一致
def prepare_gla_model_and_tokenizer(
    model_id: str,
    prec: str,
    debug: bool,
    peft_json_path: Optional[str],
) -> Tuple[Any, Any, Optional[Any]]:
    """
    Prepare GLA model + tokenizer and (optionally) attach HF PEFT LoRA.
    专门给 GLA 提供的旧接口（兼容历史脚本）。

    该函数的“存在意义”：
    - 让旧训练入口（例如 train_gla_adapter.py、train.py 的旧逻辑）
      不需要改任何调用方式
    - 内部直接调用统一入口 prepare_lat_model_and_tokenizer(model_type="gla", ...)

    文档中强调：
    This function provides exact backward compatibility with the original
    train_gla_adapter.py implementation.

    Behavior is intentionally identical to the inlined logic in train.py:
    - Uses load_gla(...) to get model & tokenizer
      注意：这里的注释是历史描述；当前实现通过 load_lat_model 统一加载，
      但对调用方“外部可观察行为”应保持一致（例如 dtype 行为、PEFT 覆盖逻辑）
    - When peft_json_path is provided, loads JSON and applies env overrides:
      HP_PEFT_R, HP_PEFT_ALPHA, HP_PEFT_DROPOUT, HP_INIT, HP_PISSA_FAST
      Then builds peft.LoraConfig and wraps with peft.get_peft_model(...)

    Args:
        model_id:
            HuggingFace model ID 或本地路径
            这里仍叫 model_id（与统一入口一致）
        prec:
            精度字符串（bf16/fp16/fp32）
        debug:
            True -> CPU；False -> CUDA
        peft_json_path:
            LoRA 配置 JSON 路径；None -> 不挂 LoRA

    Returns:
        (model, tokenizer, peft_cfg)
        - peft_cfg 为 None 表示未启用 PEFT
    """
    # 直接复用统一入口，但固定 model_type="gla"
    # 这样：
    # - 旧函数名仍可用
    # - 逻辑维护只需要维护一份（统一入口）
    return prepare_lat_model_and_tokenizer(
        model_type="gla",
        model_id=model_id,
        prec=prec,
        debug=debug,
        peft_json_path=peft_json_path,
    )