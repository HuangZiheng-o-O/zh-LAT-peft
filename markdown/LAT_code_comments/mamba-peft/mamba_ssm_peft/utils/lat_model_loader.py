"""
Unified Linear Attention Model Loader.

本模块目标：
-----------
提供一个“统一入口”的模型加载器，用来加载 FLA（Flash Linear Attention）库中不同类型的线性注意力/线性序列模型，
例如：GLA、RetNet、Mamba2 等。

为什么需要统一入口？
-------------------
在实践中，线性注意力/状态空间模型的生态里，不同模型的：
- 配置类（ConfigClass）不同
- 模型类（ForCausalLM）不同
- cache 的形态不同（past_key_values vs cache_params）
- 内部“主干模型”的属性名不同（model vs backbone）
导致下游代码不得不写很多 if/else。

本模块通过 MODEL_REGISTRY + detect_model_type + load_lat_model 实现：
- 一个统一的 load_lat_model() 用于加载所有支持的模型
- 兼容旧接口 load_gla()，确保历史代码不需要修改

设计原则（原文）：
-------------------
1. Backward Compatibility（向后兼容）：
   - GLA 的加载方式必须与原先 load_gla() 一致（接口上、行为上尽量一致）
2. Unified Interface（统一接口）：
   - 所有模型均通过 load_lat_model() 入口加载
3. Auto-Detection（自动检测）：
   - 能从 HuggingFace 的 config.json 读取 model_type 并自动映射到 registry key
4. Extensibility（可扩展）：
   - 新模型类型只需在 MODEL_REGISTRY / CONFIG_MODEL_TYPE_MAP 中增加配置即可

支持的模型（第一批）：
-----------------------
- gla: Gated Linear Attention (https://arxiv.org/abs/2312.06635)
- retnet: Retentive Network (https://arxiv.org/abs/2307.08621)
- mamba2: Mamba2 State Space Model (https://arxiv.org/abs/2405.21060)

环境变量（LAT_* 优先，GLA_* 兼容回退）：
---------------------------------------
- LAT_FORCE_LEFT_PAD / GLA_FORCE_LEFT_PAD: 强制左侧 padding（本文件中暂未直接使用，可能由下游/其它模块使用）
- LAT_VERBOSE / GLA_VERBOSE: 控制是否打印调试日志
- LAT_USE_FUSED_SWIGLU / GLA_USE_FUSED_SWIGLU: 是否启用 fused SwiGLU（默认禁用）

使用方式示例：
--------------
    from mamba_ssm_peft.utils.lat_model_loader import load_lat_model, detect_model_type

    # 1) 先自动检测 config.json 的 model_type
    model_type = detect_model_type("fla-hub/gla-1.3B-100B")

    # 2) 指定类型加载
    result = load_lat_model("gla", "fla-hub/gla-1.3B-100B")
    model, tokenizer = result["model"], result["tokenizer"]

    # 3) 直接 auto 自动检测并加载
    result = load_lat_model("auto", "fla-hub/gla-1.3B-100B")
"""

# 标准库：json 用于读取 config.json；os 用于读取环境变量；sys 未使用但保留（可能未来扩展）；typing 用于类型标注
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple

# PyTorch：模型加载与 dtype 设置依赖 torch
import torch

# transformers：AutoTokenizer 用于 tokenizer；CONFIG_NAME 是 config 文件名（通常为 "config.json"）
from transformers import AutoTokenizer
from transformers.utils import CONFIG_NAME

# cached_file：HuggingFace Hub 工具函数
# 用于把 model_id（repo 或本地路径）上的某个文件（config.json）解析到本地缓存文件路径
from transformers.utils.hub import cached_file


# ============================================================================
# MODEL REGISTRY
# ============================================================================
# 这里是“核心扩展点”：新增模型只要往这里填一条，基本可以接入统一加载逻辑
#
# 格式：model_type -> (module_path, config_class_name, model_class_name, special_handling)
#
# - module_path：Python import 路径（例如 "fla.models.gla"）
# - config_class_name：该 module 中 config 类名称（字符串）
# - model_class_name：该 module 中 ForCausalLM 类名称（字符串）
# - special_handling：模型特殊行为开关（字典）
#
# special_handling 的字段说明：
#   - has_fuse_swiglu:
#       是否存在 fuse_swiglu 相关选项（如果 True，则可能需要根据环境变量禁用 fused kernel）
#   - cache_type:
#       该模型推理缓存的类型语义
#       * "past_key_values"：类似传统 Transformer cache（GLA、RetNet）
#       * "cache_params"：Mamba2 的缓存结构不同（通常是状态空间相关参数缓存）
#       注意：本文件中只是记录该信息，真正使用可能在下游生成/推理逻辑中
#   - inner_model_attr:
#       模型内部主干属性名：
#       * GLA/RetNet：通常是 model.model
#       * Mamba2：常见命名 backbone
#       这里也只是记录，方便下游统一访问内部结构
#
MODEL_REGISTRY: Dict[str, Tuple[str, str, str, Dict[str, Any]]] = {
    "gla": (
        "fla.models.gla",        # 模型定义所在的模块路径
        "GLAConfig",             # Config 类名
        "GLAForCausalLM",        # 模型类名（用于因果语言模型）
        {
            "has_fuse_swiglu": True,               # 该模型支持 fuse_swiglu（通常意味着可能有 triton fused kernel）
            "cache_type": "past_key_values",       # 推理缓存采用 past_key_values 语义
            "inner_model_attr": "model"            # 内部主干属性名为 model（如 model.model）
        },
    ),
    "retnet": (
        "fla.models.retnet",
        "RetNetConfig",
        "RetNetForCausalLM",
        {
            "has_fuse_swiglu": True,
            "cache_type": "past_key_values",
            "inner_model_attr": "model"
        },
    ),
    "mamba2": (
        "fla.models.mamba2",
        "Mamba2Config",
        "Mamba2ForCausalLM",
        {
            "has_fuse_swiglu": False,              # Mamba2 这里标注为没有 fuse_swiglu 开关
            "cache_type": "cache_params",          # Mamba2 的缓存语义不同
            "inner_model_attr": "backbone"         # 内部主干属性名为 backbone
        },
    ),
}

# CONFIG_MODEL_TYPE_MAP：
# ----------------------
# HuggingFace 的 config.json 通常会包含字段 "model_type"（例如 "gla"、"retnet"）。
# 这里将 config.json 里的 model_type 映射到本 loader 的 registry key。
#
# 注意：有的库会在 config.json 写其它名字（比如 "fla_gla" 或 "gla2"），那你需要在这里补一条映射。
#
CONFIG_MODEL_TYPE_MAP: Dict[str, str] = {
    "gla": "gla",
    "retnet": "retnet",
    "mamba2": "mamba2",
}


# ============================================================================
# ENVIRONMENT VARIABLE HELPERS
# ============================================================================
def get_lat_env(key: str, default: str = "0") -> str:
    """
    读取环境变量（以 LAT_* 为主，GLA_* 为兼容回退）。

    背景：
    -----
    你原来的工程可能只定义了 GLA_VERBOSE、GLA_USE_FUSED_SWIGLU 等变量。
    现在统一为 LAT_*，但不能破坏旧环境配置，因此做“双前缀读取”。

    优先级：
    --------
    1) LAT_{key}
    2) GLA_{key}
    3) default

    参数：
    -----
    key:
        不含前缀的变量名，例如 "VERBOSE"
    default:
        当两个变量都不存在时返回的默认值，默认 "0"

    示例：
    -----
    get_lat_env("VERBOSE") 会依次查：
    - LAT_VERBOSE
    - GLA_VERBOSE
    - 若都没有则返回 "0"
    """
    # 统一前缀 LAT_
    lat_key = f"LAT_{key}"
    # 兼容旧前缀 GLA_
    gla_key = f"GLA_{key}"

    # os.getenv(key, fallback)：
    # 若 lat_key 不存在，则用 gla_key 的值；若 gla_key 也不存在，则用 default
    return os.getenv(lat_key, os.getenv(gla_key, default))


def get_lat_env_bool(key: str, default: str = "0") -> bool:
    """
    将环境变量读取结果转换为 bool。

    判真规则：
    ----------
    环境变量（字符串）转小写后，若属于以下集合则认为 True：
    ("1", "true", "yes", "on")

    例如：
    - LAT_VERBOSE=1     => True
    - LAT_VERBOSE=true  => True
    - LAT_VERBOSE=0     => False
    - LAT_VERBOSE=FALSE => False（lower 后为 "false" 不在集合中）
    """
    return get_lat_env(key, default).lower() in ("1", "true", "yes", "on")


def _verbose_print(msg: str) -> None:
    """
    受控打印：只有在 LAT_VERBOSE 或 GLA_VERBOSE 打开时才打印。

    目的：
    -----
    - 默认保持安静（不污染日志）
    - 需要排障时可打开 verbose 输出关键步骤

    备注：
    -----
    这里统一打印前缀 "[LAT]" 便于检索日志。
    """
    if get_lat_env_bool("VERBOSE"):
        print(f"[LAT] {msg}")


# ============================================================================
# MODEL TYPE DETECTION
# ============================================================================
def detect_model_type(model_id: str, trust_remote_code: bool = True) -> str:
    """
    从 HuggingFace 的 config.json 自动检测模型类型。

    核心思路：
    ---------
    1) 使用 transformers 的 cached_file() 定位并缓存 config.json
    2) 读取 JSON 内容
    3) 从 config_dict["model_type"] 取出类型
    4) 用 CONFIG_MODEL_TYPE_MAP 映射到本 loader 支持的 registry key

    参数：
    -----
    model_id:
        - 既可以是 HuggingFace repo id（例如 "fla-hub/gla-1.3B-100B"）
        - 也可以是本地路径（例如 "/path/to/model_dir"）
    trust_remote_code:
        - 理论上与 cached_file 的行为关系不大（cached_file 主要拉文件）
        - 但保留此参数是为了与 transformers/加载流程一致，也方便未来扩展
        - 该参数在本函数中没有直接使用（仅作为接口一致性）

    返回：
    -----
    返回值是 registry key（"gla"/"retnet"/"mamba2"）

    异常：
    -----
    - 如果 config.json 不存在、无法读取、JSON 解析失败等：抛 ValueError 并提示用户显式指定 model_type
    - 如果 config.json 没有 "model_type" 字段：抛 ValueError
    - 如果 model_type 不在 CONFIG_MODEL_TYPE_MAP：抛 ValueError 并列出支持类型
    """
    _verbose_print(f"Detecting model type from: {model_id}")

    # Step 1：尝试读取 config.json
    try:
        # cached_file 的作用：
        # - 在 HF Hub 模式下：下载并缓存 config.json，返回本地缓存路径
        # - 在本地目录模式下：定位该目录下的 config.json
        #
        # _raise_exceptions_for_missing_entries=True：
        # - 如果文件缺失或无法访问，将抛出异常，便于我们捕获并给出更明确的错误信息
        resolved_config = cached_file(
            model_id,                         # 模型 repo id 或本地路径
            CONFIG_NAME,                      # transformers 约定的配置文件名（一般是 "config.json"）
            _raise_exceptions_for_missing_entries=True,
        )

        # Step 2：打开该 config 文件并解析 JSON
        with open(resolved_config, "r") as f:
            config_dict = json.load(f)

    except Exception as e:
        # 这里用 ValueError（而不是 RuntimeError）是为了表达：
        # “无法自动推断类型，因此需要用户提供更多信息（显式指定 model_type）”
        raise ValueError(
            f"[LAT] Failed to load config.json from '{model_id}': {e}. "
            f"Please specify model_type explicitly."
        ) from e

    # Step 3：从 config.json 取出 model_type 字段
    config_model_type = config_dict.get("model_type")
    if config_model_type is None:
        # 如果 config.json 里没有 model_type，这通常意味着：
        # - repo 不是标准 HF 结构
        # - 或者模型作者没有在 config 里标注 model_type
        # 这种情况下无法自动推断，只能要求用户显式指定
        raise ValueError(
            f"[LAT] config.json for '{model_id}' does not contain 'model_type'. "
            f"Please specify model_type explicitly."
        )

    # Step 4：映射到 registry key
    model_type = CONFIG_MODEL_TYPE_MAP.get(config_model_type)
    if model_type is None:
        # 注意这里列出的是 config.json 中支持的原始类型（CONFIG_MODEL_TYPE_MAP 的 key）
        # 而不是 registry key（虽然这里它们恰好一样）
        supported = ", ".join(CONFIG_MODEL_TYPE_MAP.keys())
        raise ValueError(
            f"[LAT] Unsupported model_type '{config_model_type}' in config.json. "
            f"Supported types: {supported}"
        )

    _verbose_print(f"Detected model type: {model_type}")
    return model_type


# ============================================================================
# FUSED OPERATIONS PATCHING
# ============================================================================
def _apply_swiglu_patch() -> None:
    """
    禁用 fused SwiGLU：把 fused 实现替换成纯 PyTorch 实现。

    背景：
    -----
    一些 FLA/相关库会提供 fused 的 SwiGLU（比如 Triton kernel）以提升性能。
    但 fused kernel 可能带来：
    - 不同 GPU 架构/驱动/编译环境下的不兼容（尤其在多机、多环境、或某些云环境）
    - 推理时偶发错误（例如 kernel 编译失败、dtype 不支持等）

    设计选择：
    ----------
    默认禁用 fused SwiGLU（除非用户通过环境变量显式打开）。
    禁用方式包括两部分：
    1) config.fuse_swiglu = False（如果 config 有该字段）
    2) 运行时 monkey patch：替换 fla.modules.mlp / fla.modules.activations 内的 swiglu 与 swiglu_linear

    风险与注意：
    ------------
    - monkey patch 是“运行时全局替换”，会影响同一 Python 进程中所有使用该模块的模型/组件
    - 如果库内部函数签名变更，这里可能 patch 失败（会打印 warn）
    - 性能可能下降（纯 PyTorch 版本通常慢于 fused kernel）
    """
    try:
        # torch.nn.functional 用于实现 silu 和 linear
        import torch.nn.functional as F
        from importlib import import_module

        # 动态导入：
        # - fla.modules.mlp：通常定义 MLP 相关实现
        # - fla.modules.activations：通常定义激活函数相关实现
        _mlp = import_module("fla.modules.mlp")
        _act = import_module("fla.modules.activations")

        # 纯 PyTorch 版本的 SwiGLU：
        # SwiGLU 典型形式：silu(x) * y
        def _pt_swiglu(x, y):
            return F.silu(x) * y

        # 纯 PyTorch 版本的 swiglu_linear：
        # 先做 swiglu 激活，再接一个线性层（weight, bias）
        def _pt_swiglu_linear(x, y, weight, bias):
            return F.linear(F.silu(x) * y, weight, bias)

        # monkey patch：替换模块中的实现
        _mlp.swiglu = _pt_swiglu
        _mlp.swiglu_linear = _pt_swiglu_linear
        _act.swiglu = _pt_swiglu
        _act.swiglu_linear = _pt_swiglu_linear

        _verbose_print("fuse_swiglu disabled; using PyTorch SwiGLU.")

    except Exception as patch_err:
        # patch 失败不应该直接导致加载失败，因此这里仅给 warning
        print(f"[LAT][warn] Failed to apply SwiGLU runtime patch: {patch_err}")


# ============================================================================
# MODEL LOADING
# ============================================================================
def _import_model_classes(model_type: str) -> Tuple[Any, Any]:
    """
    根据 model_type（registry key）动态导入对应的 Config 类 与 Model 类。

    为什么要动态导入？
    -----------------
    - 避免在文件 import 时就强依赖所有模型实现（有些用户只装了部分组件）
    - 方便扩展：新增模型只要更新 registry，无需写一堆 if/else
    - 若某模型模块不存在，可以给出更明确的错误信息

    参数：
    -----
    model_type:
        必须是 MODEL_REGISTRY 中存在的 key

    返回：
    -----
    (ConfigClass, ModelClass)

    异常：
    -----
    - model_type 不在 registry：ValueError
    - 模块导入失败（例如未安装 flash-linear-attention）：ImportError
    """
    if model_type not in MODEL_REGISTRY:
        supported = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"[LAT] Unknown model_type '{model_type}'. Supported: {supported}")

    # 解包 registry 条目
    module_path, config_cls_name, model_cls_name, _ = MODEL_REGISTRY[model_type]

    try:
        from importlib import import_module

        # import_module("fla.models.gla") 这种
        module = import_module(module_path)

        # 从 module 里拿到类对象
        config_cls = getattr(module, config_cls_name)
        model_cls = getattr(module, model_cls_name)

        return config_cls, model_cls

    except ImportError as e:
        # 这里强调“请确保安装 flash-linear-attention”
        # 方便用户定位问题：不是 HF 模型下不下来，而是本地缺少实现代码
        raise ImportError(
            f"[LAT] Failed to import {module_path}. "
            f"Ensure flash-linear-attention is installed. Error: {e}"
        ) from e


def load_lat_model(
    model_type: str,
    model_id: str,
    trust_remote_code: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Any]:
    """
    统一入口：加载线性注意力/状态空间模型 + tokenizer。

    该函数是本模块的“主入口”，支持：
    - 明确指定类型：model_type="gla"/"retnet"/"mamba2"
    - 自动检测：model_type="auto"

    参数详解：
    ---------
    model_type:
        - "gla" / "retnet" / "mamba2"
        - 或 "auto"：会调用 detect_model_type(model_id) 从 config.json 推断
    model_id:
        HF repo id 或本地路径
    trust_remote_code:
        是否信任远程代码（AutoTokenizer.from_pretrained 可能需要它）
        注意：若 repo 包含自定义 tokenizer 或其它自定义逻辑，可能必须启用 True
    device:
        - "cuda"：加载后 .to("cuda")
        - "cpu"：加载后 .to("cpu")
        - "auto"：使用 transformers 的 device_map="auto" 进行自动分层/分配
          （通常用于多卡或显存不足时的自动切分）
    dtype:
        模型权重 dtype，默认 bfloat16
        备注：如果硬件不支持 bfloat16（例如部分旧 GPU），可能需要改成 float16 或 float32

    返回值（Dict）：
    ---------------
    - "model": 已加载的模型对象
    - "tokenizer": tokenizer
    - "model_type": 最终解析后的 model_type（若传入 auto，则这里是推断结果）
    - "special_handling": registry 中记录的该模型特殊行为配置

    重要行为与坑位：
    --------------
    1) fuse_swiglu 默认禁用：
       - 对于 has_fuse_swiglu=True 的模型，如果没有设置 LAT_USE_FUSED_SWIGLU，
         则会把 config.fuse_swiglu 置 False，并应用运行时 patch
    2) device="auto" 与 model.to(device) 互斥：
       - 当 device="auto" 时，from_pretrained 传 device_map="auto"
         这种情况下模型可能被分散到多个 device（甚至 CPU+GPU 混合）
         所以不能再 model.to(device)，否则会破坏 device_map 的分布并引发错误
    """
    _verbose_print(f"Loading model: model_type={model_type}, model_id={model_id}")

    # ----------------------------------------------------------------------
    # 1) 自动检测 model_type
    # ----------------------------------------------------------------------
    if model_type == "auto":
        # 注意：detect_model_type 内部会读取 config.json 的 model_type 字段
        # 如果失败，会抛 ValueError，提示用户手动指定 model_type
        model_type = detect_model_type(model_id, trust_remote_code)

    # ----------------------------------------------------------------------
    # 2) 校验 model_type 必须在 registry 中
    # ----------------------------------------------------------------------
    if model_type not in MODEL_REGISTRY:
        supported = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"[LAT] Unknown model_type '{model_type}'. Supported: {supported}")

    # 从 registry 取出 special_handling（其它字段本函数暂时不需要）
    _, _, _, special_handling = MODEL_REGISTRY[model_type]

    # ----------------------------------------------------------------------
    # 3) 动态导入 ConfigClass / ModelClass
    # ----------------------------------------------------------------------
    ConfigClass, ModelClass = _import_model_classes(model_type)

    # ----------------------------------------------------------------------
    # 4) 加载 config
    # ----------------------------------------------------------------------
    try:
        # ConfigClass.from_pretrained：
        # - 会从 HF repo 或本地路径读取 config
        # - config 中可能包含模型架构、维度、dropout、fuse 开关等信息
        config = ConfigClass.from_pretrained(model_id)
    except Exception as e:
        # 这里用 RuntimeError，是因为“加载模型的必要环节失败”
        raise RuntimeError(
            f"[LAT] Failed to load {ConfigClass.__name__}.from_pretrained('{model_id}'). "
            f"Error: {e}"
        ) from e

    # ----------------------------------------------------------------------
    # 5) 根据 special_handling 对 config/运行时做 patch
    # ----------------------------------------------------------------------
    if special_handling.get("has_fuse_swiglu", False):
        # 对支持 fused swiglu 的模型（GLA/RetNet）：
        # 如果用户没有显式打开 fused，则默认禁用
        if not get_lat_env_bool("USE_FUSED_SWIGLU"):
            # 部分 config 类可能有 fuse_swiglu 字段（不同版本可能字段名不同）
            if hasattr(config, "fuse_swiglu"):
                # 显式关闭配置开关：告诉模型不要走 fused path
                config.fuse_swiglu = False

            # 再做运行时 patch：避免某些地方仍调用 fused 实现
            _apply_swiglu_patch()

    # ----------------------------------------------------------------------
    # 6) 加载 tokenizer
    # ----------------------------------------------------------------------
    # AutoTokenizer 是最通用的 tokenizer 加载方式：
    # - 根据 model_id 目录/仓库中的 tokenizer_config.json / vocab 文件等自动构建
    # - 对于自定义 tokenizer，可能需要 trust_remote_code=True 才能正确加载
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )

    # ----------------------------------------------------------------------
    # 7) 加载模型权重
    # ----------------------------------------------------------------------
    # ModelClass.from_pretrained：
    # - config=config：用我们可能已 patch 的 config
    # - torch_dtype=dtype：指定权重 dtype（常见 bfloat16/float16/float32）
    # - device_map：
    #   * 如果 device == "auto"，则 device_map="auto" 让 transformers 自动分配层到设备
    #   * 否则传 None（表示不使用 device_map，由用户后续 model.to(device) 控制）
    #
    # 注意：
    # - 如果你传了 device_map="auto"，transformers 通常会返回一个已被“dispatch”的模型对象
    #   后续再调用 model.to("cuda") 可能会报错或破坏分布
    model = ModelClass.from_pretrained(
        model_id,
        config=config,
        torch_dtype=dtype,
        device_map="auto" if device == "auto" else None,
    )

    # ----------------------------------------------------------------------
    # 8) 若不是 device_map="auto"，则手动移动到指定 device
    # ----------------------------------------------------------------------
    if device != "auto" and device is not None:
        # 常见情况 device="cuda"：
        # - 单卡加载，权重已在 CPU（或某默认 device），此处移动到 CUDA
        # device="cpu"：
        # - 强制 CPU
        model = model.to(device=device)

    _verbose_print(f"Model loaded successfully: {model_type}")

    # ----------------------------------------------------------------------
    # 9) 返回统一结构（model + tokenizer + meta 信息）
    # ----------------------------------------------------------------------
    return {
        "model": model,
        "tokenizer": tokenizer,
        "model_type": model_type,                 # 最终确定的模型类型（auto 会被替换）
        "special_handling": special_handling,     # 供下游判断 cache_type/inner_model_attr 等
    }


def load_lat_tokenizer(
    model_id: str,
    trust_remote_code: bool = True,
) -> Any:
    """
    只加载 tokenizer 的便捷函数。

    使用场景：
    ---------
    - 你只需要 tokenizer 做数据预处理/离线分词
    - 或者你在某些场景只想先验证 tokenizer 能否正确加载
    - 也可能用于“先加载 tokenizer 再加载模型”的拆分流程

    参数：
    -----
    model_id:
        HF repo id 或本地路径
    trust_remote_code:
        是否允许执行远程自定义代码（某些自定义 tokenizer 必须开启）

    返回：
    -----
    tokenizer 对象（具体类型依赖模型仓库）
    """
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)


# ============================================================================
# BACKWARD COMPATIBILITY: GLA-specific functions
# ============================================================================
def load_gla(
    model_id: str,
    trust_remote_code: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Any]:
    """
    兼容旧版接口：load_gla()。

    目标：
    -----
    让历史代码无需改动，仍能通过 load_gla() 得到：
    {"model": ..., "tokenizer": ...}

    实现方式：
    ---------
    直接调用新的统一入口 load_lat_model("gla", ...)，然后裁剪返回字段。

    参数：
    -----
    model_id, trust_remote_code, device, dtype：
        含义同 load_lat_model

    返回：
    -----
    仅返回 "model" 与 "tokenizer"，不返回 "model_type" 与 "special_handling"
    这是为了保持旧接口的输出结构稳定。
    """
    result = load_lat_model("gla", model_id, trust_remote_code, device, dtype)

    # 为了严格向后兼容，只返回 model/tokenizer
    return {"model": result["model"], "tokenizer": result["tokenizer"]}


def load_gla_tokenizer(
    model_id: str = "fla-hub/gla-1.3B-100B",
    trust_remote_code: bool = True,
) -> Any:
    """
    兼容旧版接口：load_gla_tokenizer()。

    行为：
    -----
    - 默认 model_id 指向 "fla-hub/gla-1.3B-100B"
    - 内部调用 load_lat_tokenizer()

    注意：
    -----
    这里保留默认参数，是为了与原先代码保持一致（减少迁移成本）。
    """
    return load_lat_tokenizer(model_id, trust_remote_code)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_model_info(model_type: str) -> Dict[str, Any]:
    """
    从 registry 查询某个 model_type 的详细信息（用于调试/下游逻辑）。

    参数：
    -----
    model_type:
        registry key，例如 "gla"/"retnet"/"mamba2"

    返回：
    -----
    {
        "model_type": ...,
        "module_path": ...,
        "config_class": ...,
        "model_class": ...,
        "special_handling": {...},
    }

    典型用途：
    ---------
    - 在 CLI 或 notebook 中查看当前 loader 支持哪些模型及其类名
    - 下游根据 cache_type / inner_model_attr 做统一推理封装
    """
    if model_type not in MODEL_REGISTRY:
        supported = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model_type '{model_type}'. Supported: {supported}")

    module_path, config_cls, model_cls, special_handling = MODEL_REGISTRY[model_type]

    return {
        "model_type": model_type,
        "module_path": module_path,
        "config_class": config_cls,
        "model_class": model_cls,
        "special_handling": special_handling,
    }


def list_supported_models() -> list:
    """
    列出当前 loader 支持的所有模型类型（registry keys）。

    返回：
    -----
    例如 ["gla", "retnet", "mamba2"]

    用途：
    -----
    - CLI 参数校验
    - 文档展示
    - 单元测试：确保 registry 不为空
    """
    return list(MODEL_REGISTRY.keys())