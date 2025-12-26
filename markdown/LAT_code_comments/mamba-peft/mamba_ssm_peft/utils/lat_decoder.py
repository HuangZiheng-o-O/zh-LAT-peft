"""
Unified Linear Attention HuggingFace Decoder for text generation.

本文件/模块的目标：
==================
提供一个“统一的 HuggingFace 解码器（decoder）封装”，用于线性注意力（Linear Attention）系列模型做文本生成（text generation）。
它将 HuggingFace 的 model.generate() 调用封装起来，并为来自 FLA（Fast Linear Attention / FLA library）生态的多类模型
（例如 GLA、RetNet、Mamba2）提供一致的调用接口与一致的生成后处理逻辑。

为什么需要这个封装：
==================
不同线性注意力模型在 HuggingFace 的 generate() 过程中，可能存在：
- attention_mask 的使用与构造差异（尤其是 padding_side 的影响）
- 生成长度语义差异（max_length vs max_new_tokens）
- 缓存参数差异（例如 Mamba2 可能有 cache_params 或类似机制）
- 老版本 transformers 对某些参数不支持（例如 min_new_tokens）

因此这里通过一个“统一 decoder”：
- 统一输入输出格式
- 尽可能保持旧接口兼容
- 在必要时做兼容性兜底
- 自动裁剪 prompt，使下游指标只看生成 token（而不是把 prompt 也当作输出）

Design Principles（设计原则）:
==============================
1. Backward Compatibility（向后兼容）:
   - 原本 GLAHFDecoder 的行为必须保持一致。
   - 即使新增了 LATHFDecoder（统一 decoder），原先依赖 GLAHFDecoder 的代码仍可无缝运行。

2. Unified Interface（统一接口）:
   - 不管底层模型是 gla / retnet / mamba2，都使用同一个 decoder 类来调用 generate()。
   - 调用形式一致：decoder(model, input_ids, attention_mask=...)

3. Model-Specific Handling（模型特定处理）:
   - 某些模型（例如 Mamba2）可能需要额外缓存处理（cache_params 等）。
   - 这里的设计意图是“透明处理”：调用端无需了解这些细节。

Environment Variables（环境变量）:
=================================
本模块支持通过环境变量改变运行行为。
采用 LAT_* 作为新前缀，同时保留 GLA_* 作为兼容旧环境变量的 fallback（回退），以避免旧脚本失效。

- LAT_USE_MAX_NEW_TOKENS / GLA_USE_MAX_NEW_TOKENS
  默认=1（启用）
  含义：使用 HuggingFace generate(max_new_tokens=...) 的“新增 token 语义”
  如果设置为 0：使用 legacy（旧式）generate(max_length=prompt_len + max_length) 的“总长度语义”

- LAT_VERBOSE / GLA_VERBOSE
  默认=0（关闭）
  含义：启用更详细的日志打印，例如显示当前使用的长度语义、padding 检查等警告

- LAT_STRICT_LEFT_PAD / GLA_STRICT_LEFT_PAD
  默认=0（关闭）
  含义：如果检测到 right-padding，则直接 raise RuntimeError，而不是仅打印 warning
  注意：right-padding 在自回归生成（autoregressive generation）中经常会引发问题（尤其对某些模型/实现）。

Supported Models（支持模型）:
============================
- gla：Gated Linear Attention（门控线性注意力）
- retnet：Retentive Network（保留/滞留网络）
- mamba2：Mamba2（状态空间模型一类，可能涉及特殊 cache）

Usage（用法示例）:
==================
    from mamba_ssm_peft.utils.lat_decoder import LATHFDecoder, create_lat_decoder

    # 创建 decoder（推荐用工厂函数）
    decoder = create_lat_decoder(tokenizer, max_length=256)

    # 或者直接实例化（如果你想显式指定 model_type，便于未来扩展模型特定逻辑）
    decoder = LATHFDecoder(tokenizer=tokenizer, model_type="mamba2", max_length=256)

    # 生成
    outputs = decoder(model, input_ids, attention_mask=attention_mask)
    generated_tokens = outputs.sequences  # 注意：这里已经自动裁剪掉 prompt，只保留生成部分
"""

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


def _get_lat_env(key: str, default: str = "0") -> str:
    """
    从环境变量中读取配置项，并支持 LAT_* 与 GLA_* 的双前缀策略。

    设计目的：
    ----------
    - 新版本优先读取 LAT_{key}（例如 LAT_VERBOSE）
    - 为了兼容旧脚本/旧环境，若 LAT_{key} 未设置，则读取 GLA_{key}
    - 两者都不存在时，使用 default

    参数：
    ----
    key:
        不含前缀的 key 名称，比如 "VERBOSE"、"USE_MAX_NEW_TOKENS"
        函数内部会拼出 "LAT_VERBOS" / "GLA_VERBOSE" 这种键名

    default:
        当两个环境变量都不存在时返回的默认值（字符串形式）
        这里用字符串是为了与 os.getenv 的返回类型一致，同时为 bool 解析做准备

    返回：
    ----
    str：
        环境变量值（字符串），或 default
    """
    lat_key = f"LAT_{key}"   # 新前缀
    gla_key = f"GLA_{key}"   # 旧前缀（兼容）
    return os.getenv(lat_key, os.getenv(gla_key, default))


def _get_lat_env_bool(key: str, default: str = "0") -> bool:
    """
    将环境变量解析成布尔值。

    解析规则：
    --------
    - 先调用 _get_lat_env 拿到字符串
    - lower() 后，如果属于 ("1", "true", "yes", "on") 则认为 True
    - 其它字符串（包含空字符串、"0"、"false"、"no" 等）都视为 False

    为什么这么做：
    ------------
    - 环境变量天生是字符串
    - 用户可能习惯写 1/0，也可能写 true/false 或 yes/on
    """
    return _get_lat_env(key, default).lower() in ("1", "true", "yes", "on")


@dataclass
class LATHFDecoder:
    """
    统一的 HuggingFace decoder：用于线性注意力模型生成文本。

    核心职责：
    --------
    1) 封装 HuggingFace model.generate()
    2) 统一 attention_mask 的构造/传入
    3) 可选地检查 padding_side（尤其避免 right-padding）
    4) 生成完成后自动裁剪 prompt，确保 outputs.sequences 只包含“新生成的 token”
    5) 处理 transformers 版本差异（例如 min_new_tokens 可能不支持）

    属性（Attributes）:
    ------------------
    tokenizer:
        任意 tokenizer 对象（通常是 transformers 的 tokenizer）
        用于读取：
        - pad_token_id：构造 attention_mask 与 generate 的 pad_token_id
        - eos_token_id：传给 generate 以确定终止 token
        注意：某些 tokenizer 可能没有 pad_token_id，此时会 fallback 到 eos_token_id

    model_type:
        模型类型，用于未来/扩展场景下做模型特定处理。
        允许值示例： "gla", "retnet", "mamba2", "auto"
        当前代码中，model_type 主要是一个“标记”，并未直接影响 generate 参数（除了子类默认值）。

    max_length:
        生成长度上限的配置值。
        注意：这里的含义取决于环境变量 LAT_USE_MAX_NEW_TOKENS：
        - 如果 LAT_USE_MAX_NEW_TOKENS=1（默认）：max_length 被解释为 max_new_tokens
          即“最多生成多少个新 token”
        - 如果 LAT_USE_MAX_NEW_TOKENS=0：max_length 被解释为“在 prompt 基础上增加多少”
          然后内部会转换成 generate(max_length = prompt_len + self.max_length)

    min_length:
        生成长度下限（可选）。
        同样受 LAT_USE_MAX_NEW_TOKENS 影响：
        - 新语义：min_new_tokens
        - 旧语义：min_length = prompt_len + self.min_length
        注意：某些 transformers 版本不支持 min_new_tokens，会触发 TypeError，代码里有兜底报错指引。

    num_beams:
        beam search 的 beam 数。
        - None 或 <=1：默认 greedy（或 sampling）
        - >1：启用 beam search，同时强制 do_sample=False（beam search 通常不与 sampling 同用）

    do_sample:
        是否启用采样（sampling）。
        注意：如果启用了 beam search（num_beams>1），会被覆盖为 False。
    """

    tokenizer: Any
    model_type: str = "auto"  # "gla", "retnet", "mamba2", or "auto"
    max_length: int = 1024  # 默认作为 max_new_tokens（当 LAT_USE_MAX_NEW_TOKENS=1）
    min_length: int = 0     # 默认不设置下限
    num_beams: Optional[int] = None
    do_sample: bool = False

    def __call__(
        self,
        model: Any,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Any:
        """
        执行生成（generation），返回 HuggingFace GenerateOutput。

        输入：
        ----
        model:
            任意实现了 .generate(**kwargs) 的模型对象。
            通常是 transformers.PreTrainedModel 或兼容接口的 FLA 模型包装。

        input_ids:
            输入 token ids，形状通常是 [batch_size, seq_len]
            注意：该 seq_len 即 prompt 长度（含 padding 时实际长度可能更长）

        attention_mask:
            可选注意力 mask，形状通常是 [batch_size, seq_len]
            - 1 表示有效 token
            - 0 表示 padding token
            如果不提供，函数会尝试基于 pad_id 自动构造。

        输出：
        ----
        GenerateOutput（或兼容对象）：
            - 必须至少包含 .sequences（如果 HF 返回了这个字段）
            - 本函数会对 outputs.sequences 做“裁剪 prompt”处理：
              outputs.sequences = outputs.sequences[:, prompt_len:]
              这样下游评测时只看到生成部分，避免把 prompt 也当输出。
        """

        # ---------------------------
        # 1) 确定 pad_token_id
        # ---------------------------
        # 生成过程里 pad_token_id 有两个重要用途：
        # - 构造 attention_mask（如果用户没传）
        # - 传给 generate()，让 HF 在 batch 内对齐时知道 padding token 是什么
        #
        # 注意：某些 tokenizer 没有 pad_token_id（比如某些纯 causal LM tokenizer）
        # 此时我们用 eos_token_id 作为替代（常见的实践：pad==eos）
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", None)

        # ---------------------------
        # 2) 若 attention_mask 未提供，尝试自动构造
        # ---------------------------
        # 构造策略：
        # - 如果我们能拿到 pad_id：
        #     attention_mask = input_ids.ne(pad_id)
        #   即 token != pad_id 的位置视为有效（1），等于 pad_id 视为 padding（0）
        # - 如果连 pad_id 都拿不到：
        #     attention_mask = None
        #   让 model.generate 自行处理（某些模型可能能不依赖 attention_mask）
        if attention_mask is None:
            attention_mask = input_ids.ne(pad_id) if pad_id is not None else None

        # ---------------------------
        # 3) 读取环境变量：长度语义与 verbose
        # ---------------------------
        # use_max_new:
        #   True（默认）：使用 max_new_tokens/min_new_tokens 语义（更符合“生成多少个新 token”）
        #   False：使用旧语义（max_length = prompt_len + max_length）
        #
        # verbose:
        #   True：打印日志并启用 padding 检查提示
        use_max_new = _get_lat_env_bool("USE_MAX_NEW_TOKENS", "1")
        verbose = _get_lat_env_bool("VERBOSE", "0")

        # ---------------------------
        # 4) 构造 generate 的基础参数 gen_kwargs
        # ---------------------------
        # 这里尽量与 HF 的 generate() 参数兼容。
        # return_dict_in_generate=True：
        #   返回 GenerateOutput（包含 sequences、scores 等字段），便于统一处理
        # output_scores=False：
        #   不输出 token-level scores（节省内存/时间）
        # do_sample：
        #   是否采样，由 self.do_sample 决定，后续若启用 beam search 会强制关闭
        gen_kwargs = dict(
            input_ids=input_ids,
            eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
            pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
            return_dict_in_generate=True,
            output_scores=False,
            do_sample=bool(self.do_sample),
        )

        # ---------------------------
        # 5) 设置长度相关参数：新语义 vs 旧语义
        # ---------------------------
        # 新语义（推荐）：
        #   max_new_tokens = self.max_length
        #   min_new_tokens = self.min_length（如果 >0）
        #
        # 旧语义（legacy）：
        #   max_length = prompt_len + self.max_length
        #   min_length = prompt_len + self.min_length（如果 >0）
        #
        # 为什么要支持两种：
        # - 历史上不少代码使用“max_length=总长度”的语义
        # - 但对评测/训练 pipeline 来说，“max_new_tokens=新增长度”更清晰、更不易出错
        if use_max_new:
            if verbose:
                print("[LAT] Using HF generate(max_new_tokens/min_new_tokens) semantics.")
            gen_kwargs["max_new_tokens"] = int(self.max_length)
            # 只有当 min_length > 0 才传入，避免传入无意义或引发兼容问题
            if self.min_length and self.min_length > 0:
                gen_kwargs["min_new_tokens"] = int(self.min_length)
        else:
            # legacy：把 max_length/min_length 当作“新增多少”，内部转换为“总长度”
            if verbose:
                print("[LAT] Using legacy generate(max_length=prompt+max_length) semantics.")
            gen_kwargs["max_length"] = int(input_ids.shape[1] + self.max_length)
            if self.min_length and self.min_length > 0:
                gen_kwargs["min_length"] = int(input_ids.shape[1] + self.min_length)

        # ---------------------------
        # 6) 传入 attention_mask，并可选做 padding_side 检查
        # ---------------------------
        # 注意：
        # - 对许多 causal LM 的 generate，自回归生成通常要求“有效 token 在右侧对齐”，即左 padding（padding_side='left'）
        # - 如果你的 batch 做了 right padding（padding_side='right'），那么序列末尾可能是 padding（attention_mask 最后一列是 0）
        #   这会导致某些实现产生错误或性能异常（尤其是 cache/kv 相关实现）
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

            # verbose 模式下才检查并输出警告/报错
            if verbose:
                self._check_padding_side(attention_mask)

        # ---------------------------
        # 7) Beam search 配置
        # ---------------------------
        # num_beams > 1：
        # - 启用 beam search
        # - 并强制 do_sample=False（beam search 常规设定）
        #
        # 注意：如果用户同时设置 do_sample=True 与 num_beams>1，
        #       这里会覆盖用户设置，优先 beam search 的确定性行为。
        if self.num_beams is not None and self.num_beams > 1:
            gen_kwargs["num_beams"] = int(self.num_beams)
            gen_kwargs["do_sample"] = False

        # ---------------------------
        # 8) 调用 model.generate() 执行生成
        # ---------------------------
        # 这里包一层 try/except 主要用于处理 transformers 参数兼容问题：
        # - 老版本 transformers 可能不支持 min_new_tokens
        #   这会在 generate(**gen_kwargs) 时抛 TypeError
        #
        # 我们对这个特定情况给出更明确的报错指导：
        # - 要么设置 LAT_USE_MAX_NEW_TOKENS=0 使用旧语义（不传 min_new_tokens）
        # - 要么升级 transformers
        try:
            outputs = model.generate(**gen_kwargs)
        except TypeError as e:
            # 只针对“启用 max_new_tokens 语义 + 报错信息包含 min_new_tokens”的情况进行特判
            if use_max_new and "min_new_tokens" in str(e):
                raise RuntimeError(
                    "min_new_tokens is not supported by the current transformers version. "
                    "Set LAT_USE_MAX_NEW_TOKENS=0 to fall back to legacy max_length semantics, "
                    "or upgrade transformers."
                ) from e
            # 其它 TypeError 原样抛出，避免掩盖真实问题
            raise

        # ---------------------------
        # 9) 裁剪 prompt：只保留生成 token
        # ---------------------------
        # HuggingFace 的 outputs.sequences 通常包含：
        #   [prompt_tokens + generated_tokens]
        #
        # 但很多评测/下游逻辑希望“输出仅包含生成部分”，否则会：
        # - 重复计算 prompt 部分
        # - 影响 perplexity/bleu/rouge 等指标或导致对齐混乱
        #
        # 因此这里强制裁剪：
        #   outputs.sequences = outputs.sequences[:, prompt_len:]
        #
        # 兼容性判断：
        # - outputs 需要有 sequences 属性
        # - sequences 需要是 2D 张量（[B, T]）
        # - input_ids 也需要是 2D
        if hasattr(outputs, "sequences"):
            seq = outputs.sequences
            if seq is not None and seq.dim() == 2 and input_ids is not None and input_ids.dim() == 2:
                # 永远裁剪掉原 prompt，保证 outputs.sequences 只含新生成 token
                outputs.sequences = seq[:, input_ids.shape[1]:]

        return outputs

    def _check_padding_side(self, attention_mask: torch.Tensor) -> None:
        """
        检查 attention_mask 是否存在 right-padding（右侧 padding）。

        right-padding 的判定方式（这里采用一个很直接的启发式）：
        ----------------------------------------------------
        如果 attention_mask 的最后一列（即最后一个 time step）存在 0，
        说明至少有一个样本在序列末尾是 padding，
        即有效 token 没有“右对齐”，常见于 padding_side='right'。

        为什么 right-padding 可能是问题：
        ------------------------------
        对自回归生成而言，模型通常期望“序列末尾是最后一个有效 token”，
        这样 cache 的增量生成才能在末端追加。
        如果末端是 padding，某些实现会：
        - 在错误的位置继续生成
        - 产生非预期行为或严重性能下降
        - 在某些自定义 kernel / linear attention 实现中直接出错

        行为：
        ----
        - 默认：检测到 right-padding 时打印 warning
        - 若 LAT_STRICT_LEFT_PAD=1：直接 raise RuntimeError，强制用户修正数据处理

        注意：
        ----
        该函数用 try/except 包裹并吞掉异常：
        - 这是为了避免在某些异常张量/设备/边界情况下影响生成流程
        - 但也意味着：极端情况下检查可能静默失败（属于“尽力而为”的诊断工具）
        """
        try:
            # attention_mask.size(1) > 0：确保 seq_len > 0
            # (attention_mask[:, -1] == 0).any()：检查最后一列是否存在 padding（0）
            if attention_mask.size(1) > 0 and (attention_mask[:, -1] == 0).any():
                msg = (
                    "[LAT][warn] Right-padding detected in attention_mask during generation. "
                    "Ensure tokenizer.padding_side='left' and collator applies left padding."
                )
                # strict 模式下直接报错，阻止继续运行，以免产生“看似能跑但结果有问题”的隐患
                if _get_lat_env_bool("STRICT_LEFT_PAD", "0"):
                    raise RuntimeError(msg)
                # 非 strict 模式：仅提示
                print(msg)
        except Exception:
            # 任何异常都吞掉，避免影响主流程（诊断不应成为硬依赖）
            pass


# ============================================================================
# FACTORY FUNCTIONS（工厂函数）
# ============================================================================
def create_lat_decoder(
    tokenizer: Any,
    model_type: str = "auto",
    max_length: int = 1024,
    min_length: int = 0,
    num_beams: Optional[int] = None,
    do_sample: bool = False,
    **kwargs,
) -> LATHFDecoder:
    """
    创建统一 Linear Attention decoder 的工厂函数。

    为什么用工厂函数：
    --------------
    - 方便外部调用（尤其是训练/评测脚本）
    - 保持与旧代码 create_gla_decoder 的调用习惯一致
    - 允许传入一些额外 kwargs 而不报错（兼容旧签名/未来扩展）

    参数：
    ----
    tokenizer:
        tokenizer 对象（提供 pad_token_id / eos_token_id 等）

    model_type:
        模型类型标记，默认 auto
        目前不强制影响行为，但为将来“模型特定参数注入”预留入口

    max_length / min_length / num_beams / do_sample:
        与 LATHFDecoder 字段同义

    **kwargs:
        额外参数被忽略（ignored for compatibility）
        这样做的意义：
        - 如果旧代码传了多余参数，不会崩
        - 可逐步迁移，不用一次性修改所有调用点

    返回：
    ----
    LATHFDecoder 实例
    """
    return LATHFDecoder(
        tokenizer=tokenizer,
        model_type=model_type,
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        do_sample=do_sample,
    )


# ============================================================================
# BACKWARD COMPATIBILITY（向后兼容）：GLA-specific aliases
# ============================================================================
@dataclass
class GLAHFDecoder(LATHFDecoder):
    """
    为向后兼容保留的 GLA decoder 别名类。

    背景：
    ----
    历史代码可能直接 import GLAHFDecoder，并假设其默认 model_type 为 "gla"。
    为避免破坏这些调用点，这里让 GLAHFDecoder 继承 LATHFDecoder，
    仅将 model_type 默认值设为 "gla"。

    重要说明：
    --------
    - 该类本质上与 LATHFDecoder 完全相同（当前实现）
    - 只是一个“名字与默认参数”的兼容层
    """
    model_type: str = field(default="gla")


def create_gla_decoder(
    tokenizer: Any,
    max_length: int = 1024,
    min_length: int = 0,
    num_beams: Optional[int] = None,
    do_sample: bool = False,
    **kwargs,
) -> GLAHFDecoder:
    """
    创建 GLA decoder（保持与原 create_gla_decoder() 完全一致的调用方式）。

    参数：
    ----
    tokenizer:
        tokenizer 对象

    max_length / min_length / num_beams / do_sample:
        与 LATHFDecoder 字段同义

    **kwargs:
        额外参数忽略（兼容旧代码签名）

    返回：
    ----
    GLAHFDecoder 实例（本质上是 LATHFDecoder 的子类 + 默认 model_type="gla"）
    """
    return GLAHFDecoder(
        tokenizer=tokenizer,
        model_type="gla",
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        do_sample=do_sample,
    )