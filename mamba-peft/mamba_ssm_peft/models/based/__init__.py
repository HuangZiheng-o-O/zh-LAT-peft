# -*- coding: utf-8 -*-
"""
Based Model Implementation (在自己的仓库中)

这是 Based 模型的实现，放在 mamba-peft 仓库中以避免修改 3rdparty。
Based 模型依然使用 FLA 的 BasedLinearAttention layer，但配置和注册在此处。

Reference: https://arxiv.org/abs/2402.18668
"""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from mamba_ssm_peft.models.based.configuration_based import BasedConfig
from mamba_ssm_peft.models.based.modeling_based import BasedForCausalLM, BasedModel

# 自动注册到 HuggingFace transformers
AutoConfig.register(BasedConfig.model_type, BasedConfig, exist_ok=True)
AutoModel.register(BasedConfig, BasedModel, exist_ok=True)
AutoModelForCausalLM.register(BasedConfig, BasedForCausalLM, exist_ok=True)

__all__ = ['BasedConfig', 'BasedForCausalLM', 'BasedModel']
