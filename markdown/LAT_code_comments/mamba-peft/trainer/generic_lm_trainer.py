# =========================
# 这里的 docstring 是文件级说明（模块说明）。
# 你要求“不要压缩”，因此保持原样不做任何删改。
# =========================
"""
Generic Language Model Trainer for GLA (Gated Linear Attention) and similar architectures.

This trainer is designed to work with the FLA (Flash Linear Attention) library's GLA models.
It properly handles attention_mask to ensure correct training behavior.

Key Design Considerations for GLA:
==================================

1. **attention_mask is REQUIRED for GLA**
   Unlike Transformer's softmax attention, GLA uses linear state accumulation:
       S_t = Diag(α_t) · S_{t-1} + k_t^T ⊗ v_t

   Without attention_mask, padding tokens pollute the hidden state S_t.
   The FLA library implements an "unpadding" strategy that completely removes
   padding tokens from computation when attention_mask is provided.

2. **Left Padding for Generation**
   Generation tasks should use left padding (tokenizer.padding_side = "left")
   to ensure the last tokens are always valid (not padding).

3. **Loss Calculation**
   Loss is only computed at positions where label_ids != -100, so padding
   positions in labels are correctly ignored.

Environment Variables:
- GLA_LOG_PADDING_STATS=1: Log padding ratio every 500 steps for debugging
- GLA_FORCE_LEFT_PAD=1: Force left padding in tokenizer (set in train_gla_only.py)

References:
- GLA Paper: "Gated Linear Attention Transformers with Hardware-Efficient Training"
  https://arxiv.org/abs/2312.06635
- FLA Library: https://github.com/sustcsonglin/flash-linear-attention
"""

# =========================
# 标准库/三方库导入
# =========================

# dataclasses 用来声明 TrainingArguments 的子类字段（更像配置对象）
from dataclasses import dataclass, field

# pathlib 的 Path 用于跨平台的路径拼接与文件存在判断
from pathlib import Path

# typing: 用于类型标注，让 IDE 和读者更好理解入参/返回值
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import os  # 读取环境变量、创建目录、路径判断等
import torch  # PyTorch 主库
from torch import nn  # nn.Module 等核心抽象
from torch.optim.lr_scheduler import LambdaLR  # scheduler 类型标注与兼容
from torch.optim.optimizer import Optimizer as Optimizer  # 优化器抽象（类型标注用）
from torch.utils.data import Dataset  # HuggingFace Trainer 使用的 dataset 抽象

# HuggingFace Transformers Trainer 相关
from transformers import Trainer, TrainerCallback
from transformers.modeling_utils import PreTrainedModel  # 预训练模型基类（类型标注）
from transformers.tokenization_utils_base import PreTrainedTokenizerBase  # tokenizer 基类（类型标注）
from transformers.trainer import logger  # Transformers 内置 logger（统一日志）
from transformers.trainer_callback import TrainerCallback  # callback 类型
from transformers.trainer_utils import EvalPrediction  # metrics 计算时用的结构
from transformers.training_args import TrainingArguments  # 官方训练参数配置类

# PEFT: Parameter-Efficient Fine-Tuning（如 LoRA）模型封装
from peft import PeftModel

# 你的工程内部模块：训练 loss 与评估工具
from trainer.loss import CrossEntropy, Accuracy
from trainer.eval_utils import (
    EvalPredictionWithText,   # 带文本解码/保存预测结果的结构
    TrainLossEarlyStop,       # 根据训练 loss 做 early stop 的逻辑
    BadEvalEarlyStop,         # 根据 eval 指标阈值做 early stop 的逻辑
)


# =========================
# TrainingArguments 子类：增加自定义字段
# =========================
@dataclass
class GenericLMTrainingArguments(TrainingArguments):
    # info: 用于注入任意额外信息（例如实验元数据），默认 None
    # 注意：field(default=None) 意味着 dataclass 初始化时可以不传
    info: Dict[str, Any] = field(default=None)

    # save_full_model: 是否保存完整 torch 模型（torch.save(self.model, ...))
    # 一般情况下 HuggingFace 推荐 save_pretrained；
    # 这里保留了一个额外开关，避免误保存导致磁盘爆炸。
    save_full_model: bool = False

    # save_optimizer_state: 是否保存优化器/调度器/rng_state 到 checkpoint
    # 默认 False：减少磁盘占用（特别是大模型训练时 optimizer.pt 很大）
    # 注意：此字段会影响 _save_optimizer_and_scheduler 和 _save_rng_state 的行为
    save_optimizer_state: bool = False


# =========================
# 自定义 Trainer：面向 GLA/Linear Attention 的训练器
# =========================
class GenericLMTrainer(Trainer):
    def __init__(self,
                 # model: 兼容 HuggingFace 的 PreTrainedModel 或任意 nn.Module
                 model: PreTrainedModel | nn.Module = None,

                 # args: 训练配置（包括输出目录、batch size、fp16、保存策略等）
                 args: TrainingArguments = None,

                 # data_collator: 将 dataset 样本拼成 batch 的函数/对象
                 data_collator: Any | None = None,

                 # train_dataset: 训练集
                 train_dataset: Dataset | None = None,

                 # eval_dataset: 验证集；HF 允许传入 Dataset 或 Dict[str, Dataset]
                 eval_dataset: Dataset | Dict[str, Dataset] | None = None,

                 # tokenizer: 用于 decode/encode，在 log_train_seq / EvalPredictionWithText 会用到
                 tokenizer: PreTrainedTokenizerBase | None = None,

                 # model_init: 延迟初始化模型（HF Trainer 用于超参搜索等）
                 model_init: Callable[[], PreTrainedModel] | None = None,

                 # compute_metrics: 评估函数，输入 EvalPrediction，输出字典指标
                 compute_metrics: Callable[[EvalPrediction], Dict] | None = None,

                 # callbacks: TrainerCallback 列表（如 early stop、日志等）
                 callbacks: List[TrainerCallback] | None = None,

                 # optimizers: (optimizer, lr_scheduler)，允许外部注入
                 optimizers: Tuple[Optimizer, LambdaLR] = (None, None),

                 # preprocess_logits_for_metrics: HF 在算 metrics 前可先处理 logits
                 preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,

                 # eval_generator: 自定义生成器（generation-based eval）；
                 # 若不为 None，则 evaluate() 走 evaluate_generation
                 eval_generator=None,

                 # min_eval_metric_after_epoch: 某个 epoch 后如果指标低于阈值则 early stop（BadEvalEarlyStop）
                 min_eval_metric_after_epoch=None,

                 # **kwargs: 透传给父类 Trainer 的其他参数（如 class_weight 等）
                 **kwargs):

        # callbacks 若为空，则初始化为空列表，避免传 None 给 super() 产生不一致行为
        if callbacks is None:
            callbacks = []

        # 调用 HuggingFace Trainer 的初始化
        # 注意：HF Trainer 的 __init__ 参数顺序较固定，这里保持对齐
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer,
                         model_init, compute_metrics, callbacks,
                         optimizers, preprocess_logits_for_metrics, **kwargs)

        # =========================
        # 训练/验证用的指标与 early stop 逻辑
        # =========================

        # train_crit：训练阶段的损失函数封装（这里是 CrossEntropy）
        # 注意：你的 CrossEntropy 很可能内部封装 ignore_index = -100
        self.train_crit = CrossEntropy()

        # val_crits：验证阶段可能用的一组指标（这里放 Accuracy）
        # 但此类 Trainer 的 evaluate() 可能主要走 compute_metrics
        self.val_crits = [Accuracy()]

        # 训练 loss 的 early stop：例如当 loss 异常、NaN、或长期不下降时触发
        self.train_loss_early_stop = TrainLossEarlyStop()

        # generation-based eval 的生成器（自定义逻辑：如 beam search、采样等）
        self.eval_generator = eval_generator

        # eval 指标阈值 early stop：
        # 若传入 min_eval_metric_after_epoch，则构造 BadEvalEarlyStop
        self.min_eval_metric_after_epoch_early_stop = (
            BadEvalEarlyStop(min_eval_metric_after_epoch)
            if min_eval_metric_after_epoch is not None
            else None
        )

        # =========================
        # 可选：模型特定 hook（配置加载）
        # =========================
        # 有些自定义模型会提供 load_config(output_dir) 用于从目录读取额外配置
        # 这里做“存在就调用”，并且 try/except 防止因为文件缺失导致训练崩溃
        if hasattr(model, "load_config"):
            try:
                model.load_config(self.args.output_dir)
            except Exception:
                # 安全吞掉异常：保证训练主流程不被配置加载失败阻断
                pass

    # =========================
    # 调试函数：打印一条样本的输入/标签/预测（文本形式）
    # =========================
    def log_train_seq(self, input_ids, label_ids, lm_logits, idx=0):
        # 取 batch 中第 idx 个样本，便于人工检查
        input_ids, label_ids, lm_logits = input_ids[idx], label_ids[idx], lm_logits[idx]

        # logits.argmax(-1) 得到每个位置预测的 token id
        output_ids = lm_logits.argmax(-1)

        # label_ids != -100 的位置才是有效监督位置（padding 或不参与 loss 的位置为 -100）
        valid_ids = label_ids != -100

        # decode: 将 token id 转为文本（注意 tokenizer.decode 默认会把特殊符号也解出来）
        input_txt = self.tokenizer.decode(input_ids)
        input_txt_valid = self.tokenizer.decode(input_ids[valid_ids])
        label_txt_valid = self.tokenizer.decode(label_ids[valid_ids])
        output_txt_valid = self.tokenizer.decode(output_ids[valid_ids])

        # 打印完整输入
        print(input_txt)

        # 打印有效输入片段 -> 有效标签片段
        print(input_txt_valid, "->", label_txt_valid)

        # 打印模型预测（有效位置）== 标签（有效位置）
        print(output_txt_valid, "==", label_txt_valid)

    # =========================
    # 核心前向：训练与评估共享
    # =========================
    def _forward(self, model, inputs):
        """
        Forward pass for training and evaluation.

        IMPORTANT: For GLA (Gated Linear Attention) models from the FLA library,
        References:
        - GLA Paper: https://arxiv.org/abs/2312.06635
        - FLA Implementation: fla/layers/gla.py (see get_unpad_data usage)
        """
        # inputs 来自 data_collator：通常是 dict[str, tensor]
        # 这里假设 collator 已经把 key 命名为 input_ids / label_ids / attention_mask
        input_ids = inputs["input_ids"]
        label_ids = inputs["label_ids"]

        # attention_mask 可能存在也可能不存在：
        # - 对于 GLA/线性注意力模型：强烈建议存在，否则 padding 会污染状态累积
        # - 对于常规 softmax attention：没有也通常能跑（但 padding 位置可能产生无效 attention）
        attention_mask = inputs.get("attention_mask")

        # 构造额外输入参数（传给 model.forward）
        add_inputs = {}

        # =========================
        # GLA 关键：传入 attention_mask 以触发 FLA 的 unpadding
        # =========================
        # FLA 的策略一般是：
        # 1) 根据 attention_mask 找到有效 token 的索引，把 padding token 从算子输入里剔除
        # 2) 使用 cu_seqlens（每条序列的边界）告诉 kernel 每条序列长度
        # 3) kernel 输出后再 pad 回原始 batch 形状
        # 这样做的结果是：
        # - 计算更快（少算 padding）
        # - 训练更正确（padding 不会进入线性状态累积 S_t）
        if attention_mask is not None:
            add_inputs["attention_mask"] = attention_mask

        # =========================
        # PEFT 模型兼容：某些 base_model.forward 可能接受 label_ids
        # =========================
        # PeftModel 是一个 wrapper；它的 base_model 才是实际模型主体
        # 这里通过检查 base.forward 的参数名，决定是否把 label_ids 传进去
        # 注意：这不是 HF 标准接口（HF 一般用 labels），但你的模型/工程可能定义了 label_ids
        if isinstance(model, PeftModel):
            base = model.base_model

            # __code__.co_varnames 包含函数形参名（含 self 等）
            # 这里是“弱反射”判断：如果 base.forward 的形参里写了 label_ids，就传
            if "label_ids" in base.forward.__code__.co_varnames:
                add_inputs["label_ids"] = label_ids

        # =========================
        # 可选调试：打印 padding 比例
        # =========================
        # 环境变量 GLA_LOG_PADDING_STATS=1 时启用
        # 每 500 step 打一次日志，避免过度刷屏
        if attention_mask is not None and os.environ.get("GLA_LOG_PADDING_STATS", "0") == "1":
            # self.state.global_step 由 HF Trainer 管理（训练过程递增）
            if hasattr(self, 'state') and self.state.global_step % 500 == 0:
                # attention_mask.sum(): 有效 token 总数（假设 mask 为 0/1）
                valid_tokens = attention_mask.sum().item()

                # attention_mask.numel(): mask 张量元素总数 = batch_size * seq_len
                total_tokens = attention_mask.numel()

                # 有效比例；total_tokens 为 0 属于极端情况（防除零）
                valid_ratio = valid_tokens / total_tokens if total_tokens > 0 else 1.0

                # 用 transformers.trainer.logger 统一输出
                logger.info(
                    f"[Step {self.state.global_step}] Padding stats: "
                    f"valid={valid_ratio:.1%}, pad={1-valid_ratio:.1%}, "
                    f"batch_shape={tuple(input_ids.shape)}"
                )

        # =========================
        # 执行模型前向
        # =========================
        # 这里显式取 .logits，说明 model(...) 返回的是类似 CausalLMOutput 的对象
        # 若你的模型不是 HF 标准输出结构，这里会报错，需要适配
        lm_logits = model(input_ids, **add_inputs).logits

        # 返回 input/label/logits：供 compute_loss / prediction_step 使用
        return input_ids, label_ids, lm_logits

    # =========================
    # 训练 loss 计算（HF Trainer 会在 training_step 中调用）
    # =========================
    def compute_loss(self, model, inputs, return_outputs=False):
        # 前向得到 logits
        input_ids, label_ids, lm_logits = self._forward(model, inputs)

        # 计算交叉熵损失：
        # 一般形状：
        # - lm_logits: [B, T, V]
        # - label_ids: [B, T]
        # CrossEntropy 内部应 flatten 并 ignore_index=-100
        lm_loss = self.train_crit(lm_logits, label_ids)

        # =========================
        # 模型自带的停止开关：should_training_stop
        # =========================
        # 某些模型可能在训练过程中达到某条件后希望终止训练（例如收敛判定/异常判定）
        if getattr(model, "should_training_stop", False):
            # 如果模型有 save_config，则尽量把配置保存到输出目录
            if hasattr(model, "save_config"):
                try:
                    model.save_config(self.args.output_dir)
                except Exception:
                    pass
                # 控制 Trainer 结束训练（HF 的 control 信号机制）
                self.control.should_training_stop = True

        # =========================
        # 训练 loss early stop（外部策略）
        # =========================
        # TrainLossEarlyStop 通常会检查：
        # - loss 是否为 NaN/Inf
        # - loss 是否长期不改善
        # - 或满足其它自定义停止条件
        self.train_loss_early_stop(self.control, lm_loss)

        # HF Trainer 只需要返回 loss（return_outputs 未用到）
        return lm_loss

    # =========================
    # optimizer_step: 这里没有改逻辑，只是留了 hook
    # =========================
    def optimizer_step(self, *args, **kwargs):
        # 仍然用父类实现（支持梯度裁剪、AMP 等）
        super().optimizer_step(*args, **kwargs)

    # =========================
    # prediction_step: HF evaluate/predict 时调用
    # =========================
    @torch.no_grad()
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        # 同样走共享 _forward
        input_ids, label_ids, lm_logits = self._forward(model, inputs)

        # 计算 loss（这里复用 train_crit；有些项目会区分 train/eval loss，这里未区分）
        lm_loss = self.train_crit(lm_logits, label_ids)

        # =========================
        # 关键：只保留有效监督位置的 logits 与 labels
        # =========================
        # HF 默认 prediction_step 通常返回：
        # (loss, logits, labels)
        # 这里自定义为“列表形式的变长张量”，因为每条样本有效 token 数可能不同
        logits_valid = []
        label_ids_valid = []

        # 遍历 batch，每条样本做一次过滤
        for i, (logits_sample, label_ids_sample) in enumerate(zip(lm_logits, label_ids)):
            # valid_pos: label != ignore_index 的位置
            valid_pos = label_ids_sample != self.train_crit.ignore_index

            # 过滤 logits：从 [T, V] 变为 [T_valid, V]
            logits_sample_valid = logits_sample[valid_pos]

            # 过滤 labels：从 [T] 变为 [T_valid]
            label_ids_sample_valid = label_ids_sample[valid_pos]

            logits_valid.append(logits_sample_valid)
            label_ids_valid.append(label_ids_sample_valid)

        # 返回 (loss, preds, labels)
        # 注意：这里 preds 是 logits_valid（list[Tensor]），不是单个 Tensor
        return (lm_loss, logits_valid, label_ids_valid)

    # =========================
    # generation_step: 对一个 batch 做生成（用于 generation-based eval）
    # =========================
    def generation_step(self, generator, model, inputs):
        """
        Run generation for a single batch.

        IMPORTANT: Inputs from dataloader are on CPU by default.
        We must move them to the model's device before generation.
        """
        # inputs 可能为空（某些 dataloader/过滤逻辑），直接返回空
        if inputs is None:
            return ([], [])

        # 安全读取 dict 中字段
        input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else None
        label_ids = inputs.get("label_ids") if isinstance(inputs, dict) else None
        attention_mask = inputs.get("attention_mask") if isinstance(inputs, dict) else None

        # input_ids/label_ids 是必须的，否则无法生成/评估
        if input_ids is None or label_ids is None:
            return ([], [])

        # =========================
        # 关键：将输入搬到模型所在 device（通常 GPU）
        # =========================
        # DataLoader 默认产出 CPU tensor；
        # 如果 model 在 CUDA 上，直接 generator(model, input_ids) 会报 device mismatch
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)

        # attention_mask 同理，如果存在也要搬到同一 device
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # label_ids 这里刻意不搬：
        # - 因为后续只是作为“对照标签”用于评估/保存
        # - 保持 CPU 可减少 GPU 显存占用
        # - 同时避免后续 EvalPredictionWithText 处理时出现 device 混乱

        # =========================
        # 调用 generator 进行生成
        # =========================
        # generator 的契约：generator(model, input_ids, attention_mask=...) -> out_seq
        # out_seq 可能是：
        # - 直接 tensor [B, T_gen]
        # - 或 transformers 的 GenerateOutput，包含 .sequences
        out_seq = generator(model, input_ids, attention_mask=attention_mask)

        # 如果是 GenerateOutput 类结构，则取 sequences
        if hasattr(out_seq, "sequences"):
            out_seq = out_seq.sequences

        # 对齐维度：确保 out_seq 与 label_ids 至少是二维 [B, T]
        if out_seq.dim() == 1:
            out_seq = out_seq.unsqueeze(0)
        if label_ids.dim() == 1:
            label_ids = label_ids.unsqueeze(0)

        # =========================
        # 将生成结果搬回 CPU，便于后处理/保存/比较
        # =========================
        out_seq = out_seq.cpu()

        # 返回 list[Tensor]，保持与 EvalPredictionWithText 的接口一致
        pred_list = [row for row in out_seq]
        label_list = [row for row in label_ids]
        return (pred_list, label_list)

    # =========================
    # save_model: 自定义保存逻辑
    # =========================
    def save_model(self, output_dir, _internal_call):
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 如果没有显式开启 save_full_model，则直接返回（不保存 torch 模型对象）
        # 这样可以防止误把包含 GPU tensor 的整个对象 dump 导致巨大文件/兼容问题
        if not getattr(self.args, "save_full_model", False):
            return

        # 保存整个 self.model（pickle 风格），通常不如 save_pretrained 通用
        torch.save(self.model, f"{output_dir}/model.pt")

    # =========================
    # _maybe_log_save_evaluate: HF 训练中周期性触发日志/保存/评估
    # =========================
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        # 如果训练 loss early stop 判定应停止，则禁用 evaluate（减少无意义评估）
        if self.train_loss_early_stop.should_stop:
            self.control.should_evaluate = False

        # 调用父类逻辑（包含：log、save checkpoint、evaluate 等）
        return super()._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)

    # =========================
    # _save_optimizer_and_scheduler: 控制是否保存 optimizer/scheduler 状态
    # =========================
    def _save_optimizer_and_scheduler(self, output_dir: str):
        # 注意：这里使用 getattr(self.args, "save_optimizer_state", True)
        # 这意味着如果 args 没有该属性，则默认 True（会保存）。
        # 但你的 GenericLMTrainingArguments 里默认是 False；
        # 若你确实用的是 GenericLMTrainingArguments，则默认不会保存。
        if not getattr(self.args, "save_optimizer_state", True):
            # 即便不保存，也尽量创建目录，防止上层逻辑依赖目录存在
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception:
                pass
            return

        # 要保存时，同样确保目录存在
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception:
            pass

        # 调用父类保存 optimizer.pt 与 scheduler.pt
        return super()._save_optimizer_and_scheduler(output_dir)

    # =========================
    # _save_rng_state: 控制是否保存随机数状态（保证可复现）
    # =========================
    def _save_rng_state(self, output_dir: str):
        # save_optimizer_state=False 时也会跳过 rng_state 保存
        # 取舍：更省空间，但恢复训练时不可完全复现
        if not getattr(self.args, "save_optimizer_state", True):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception:
                pass
            return

        # 调用父类保存 rng_state（通常包含 python/random、numpy、torch、cuda rng 等）
        return super()._save_rng_state(output_dir)

    # =========================
    # _get_collator_with_removed_columns:
    # HF Trainer 默认会根据 model.forward 的签名移除多余列，避免传入无关字段。
    # 这里直接返回原 collator，等于“禁用列裁剪”。
    # =========================
    def _get_collator_with_removed_columns(
        self, data_collator: Callable, description: Optional[str] = None
    ):
        return data_collator

    # =========================
    # reset_optimizer: 训练中途重置优化器/调度器
    # =========================
    def reset_optimizer(self):
        print("Resetting optimzer")  # 原文拼写保持不改动（你要求不要压缩/不改）
        self.optimizer = None
        self.lr_scheduler = None

        # 重新创建 optimizer & scheduler
        # max_steps - global_step：剩余训练步数
        self.create_optimizer_and_scheduler(self.args.max_steps - self.state.global_step)

    # =========================
    # evaluate: 覆盖 HF 的 evaluate，以支持 generation-based eval
    # =========================
    def evaluate(self, eval_dataset: Dataset | Dict[str, Dataset] | None = None, ignore_keys: List[str] | None = None, metric_key_prefix: str = "eval") -> Dict[str, float]:
        # 如果提供了 eval_generator，则走生成评估：
        # - 先生成文本
        # - 再 compute_metrics（通常是基于文本/序列级别的指标，如 EM/F1/ROUGE 等）
        if self.eval_generator is not None:
            metrics = self.evaluate_generation(self.eval_generator, metric_key_prefix=metric_key_prefix)
        else:
            # 否则走父类 evaluate（基于 prediction_step 的 logits/labels）
            metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # 如果设置了 min_eval_metric_after_epoch_early_stop，则用评估指标做 early stop 判定
        if self.min_eval_metric_after_epoch_early_stop is not None:
            self.min_eval_metric_after_epoch_early_stop(self.control, metrics)

        return metrics

    # =========================
    # evaluate_generation: 生成式评估（可缓存预测结果）
    # =========================
    @torch.no_grad()
    def evaluate_generation(self, generator, use_cache=True, skip_metrics=False, metric_key_prefix="eval"):
        # 预测缓存文件名包含 global_step，避免不同 step 覆盖
        eval_pred_file = Path(self.args.output_dir) / f"predictions-{self.state.global_step}.yaml"

        # use_cache=False 或 缓存文件不存在 时：重新跑一遍生成
        if not use_cache or not eval_pred_file.is_file():
            model = self.model
            model.eval()

            # HF Trainer 提供 eval dataloader（会自动处理 sampler、batch_size 等）
            dataloader = self.get_eval_dataloader()

            # 收集全部 input/pred/label 序列（用于 EvalPredictionWithText）
            input_ids_all = []
            pred_ids_all = []
            label_ids_all = []

            # 遍历 eval dataloader
            for step, inputs in enumerate(dataloader):
                if inputs is None:
                    continue

                # 对 batch 进行生成
                pred_ids, label_ids = self.generation_step(generator, model, inputs)
                if not pred_ids or not label_ids:
                    continue

                # 把原始 input_ids 也保存下来，方便 later decode 成 prompt / 输入文本
                batch_input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else None
                if batch_input_ids is None:
                    continue

                # 注意：这里 input_ids_all += [*batch_input_ids]
                # batch_input_ids 是一个 batch 张量（CPU），迭代后得到每条序列 tensor
                input_ids_all += [*batch_input_ids]
                pred_ids_all += [*pred_ids]
                label_ids_all += [*label_ids]

            # 构造 EvalPredictionWithText：
            # - 内部通常会把 input/pred/label decode 成文本
            # - remove_eos=True：在对比/保存时去掉 eos，避免指标被 eos 干扰
            # - save_file: 指定输出 yaml 文件路径
            eval_pred = EvalPredictionWithText(
                generator.tokenizer,
                input_ids_all,
                pred_ids_all,
                label_ids_all,
                save_file=eval_pred_file,
                remove_eos=True
            )

            # 保存预测结果到 yaml（用于复用/调试/追踪）
            eval_pred.save()

        else:
            # 缓存存在且 use_cache=True：可直接使用缓存
            if not skip_metrics:
                print(f"Loading prediction {eval_pred_file}")

        # =========================
        # 指标计算阶段
        # =========================
        if not skip_metrics:
            # 从缓存文件反序列化预测结果
            eval_pred = EvalPredictionWithText.from_file(str(eval_pred_file))

            # compute_metrics 是你在 Trainer 初始化时传入的回调
            # 这里传入的是 EvalPredictionWithText（不是 HF 原生 EvalPrediction）
            metrics = self.compute_metrics(eval_pred)

            # 加 metric_key_prefix 前缀（HF 常用：eval_xxx）
            if metric_key_prefix != "":
                metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}

            # Trainer.log: 记录到日志系统（也会进 TensorBoard/WandB 等）
            self.log(metrics)

            # callback：触发 on_evaluate，让 callbacks 有机会响应评估结果
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

            return metrics
        else:
            # skip_metrics=True：只做生成/保存，不算指标
            return None