from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import os
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils.data import Dataset

# HuggingFace Transformers Trainer 相关
from transformers import Trainer, TrainerCallback
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import logger
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments

# PEFT（Parameter-Efficient Fine-Tuning）相关：LoRA / Adapter 等
from peft import PeftModel

# 项目内自定义：训练损失与指标、评估工具、早停策略等
from trainer.loss import CrossEntropy, Accuracy
from trainer.eval_utils import (
    EvalPredictionWithText,
    TrainLossEarlyStop,
    BadEvalEarlyStop,
)


@dataclass
class GenericLMTrainingArguments(TrainingArguments):
    """
    继承自 Transformers 的 TrainingArguments，用于扩展自定义训练参数。

    TrainingArguments 是 HF Trainer 的核心配置容器（batch size、lr、logging、save、eval 等）。
    这里额外加了三个字段：
      - info:       任意信息字典（通常用于实验元信息/配置快照/注释等）
      - save_full_model: 是否保存完整模型对象（torch.save(self.model, ... )），而不是 HF 标准的 save_pretrained
      - save_optimizer_state: 是否保存 optimizer/scheduler/rng_state
            默认 False：节省磁盘，尤其是大模型+频繁 checkpoint 的场景
            注意：不保存 optimizer state 会导致断点续训“无法恢复优化器动量/状态”，训练轨迹会偏离
    """
    info: Dict[str, Any] = field(default=None)
    save_full_model: bool = False

    # Control whether to save optimizer state (optimizer.pt), scheduler state and rng_state during checkpointing
    # Default False to minimize disk usage unless explicitly enabled
    save_optimizer_state: bool = False


class GenericLMTrainer(Trainer):
    """
    一个面向“语言模型/自回归 LM”任务的自定义 Trainer。

    主要改动点（相对于 HF 原生 Trainer）：
      1) 自定义 compute_loss：使用项目自定义 CrossEntropy，并接入 TrainLossEarlyStop
      2) 自定义 prediction_step：返回“按样本去掉 ignore_index 的 logits/labels”用于自定义评估
      3) 支持 evaluate_generation：用 generator 做生成式评估，并将预测落盘缓存（yaml）
      4) 自定义 save_model：可选保存完整 torch 模型对象（save_full_model）
      5) 控制是否保存 optimizer/scheduler/rng_state（save_optimizer_state）
      6) _get_collator_with_removed_columns：直接返回原 collator，避免 HF 自动删字段（某些自定义输入需要）
    """

    def __init__(
        self,
        model: PreTrainedModel | nn.Module = None,
        args: TrainingArguments = None,
        data_collator: Any | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | Dict[str, Dataset] | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        model_init: Callable[[], PreTrainedModel] | None = None,
        compute_metrics: Callable[[EvalPrediction], Dict] | None = None,
        callbacks: List[TrainerCallback] | None = None,
        optimizers: Tuple[Optimizer, LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        eval_generator=None,
        min_eval_metric_after_epoch=None,
        **kwargs
    ):
        """
        与 HF Trainer.__init__ 保持一致的参数结构，同时额外引入：
          - eval_generator: 用于生成式评估的 generator（通常封装了 tokenizer、generate 参数、后处理等）
          - min_eval_metric_after_epoch: 某个评估指标低于阈值时触发早停（BadEvalEarlyStop）
        """
        # callbacks 若为 None，则置为空列表，避免 super().__init__ 里迭代时报错
        if callbacks is None:
            callbacks = []

        # 调用 HF Trainer 初始化：
        #  - 设置 self.model, self.args, self.tokenizer 等
        #  - 构造 dataloader、训练循环、回调系统等
        super().__init__(
            model, args, data_collator, train_dataset, eval_dataset, tokenizer,
            model_init, compute_metrics, callbacks,
            optimizers, preprocess_logits_for_metrics, **kwargs
        )

        # 训练损失（criterion）：自定义 CrossEntropy
        # 通常内部会定义 ignore_index（例如 -100）等与 label mask 配合
        self.train_crit = CrossEntropy()

        # 验证指标列表：这里仅 Accuracy
        # 注意：HF Trainer 通常在 compute_metrics 阶段计算指标，这里是自定义框架可能会用到
        self.val_crits = [Accuracy()]

        # 训练损失早停器：通常用于监控 loss 是否 NaN、是否爆炸、是否无下降等，满足条件设置 control.should_training_stop
        self.train_loss_early_stop = TrainLossEarlyStop()

        # 生成式评估的 generator（可选）
        self.eval_generator = eval_generator

        # “评估指标过差”早停器（可选）
        # min_eval_metric_after_epoch 可能是阈值字典/配置，BadEvalEarlyStop 内部负责判定
        self.min_eval_metric_after_epoch_early_stop = (
            BadEvalEarlyStop(min_eval_metric_after_epoch)
            if min_eval_metric_after_epoch is not None
            else None
        )

        # 可选：模型自定义 hook（load_config）
        # 目的：允许模型从 output_dir 读取某些配置（例如自定义 head、动态参数等）
        # 安全处理：如果不存在或报错就忽略，确保 Trainer 不被模型差异拖死
        if hasattr(model, "load_config"):
            try:
                model.load_config(self.args.output_dir)
            except Exception:
                pass

    def log_train_seq(self, input_ids, label_ids, lm_logits, idx=0):
        """
        调试用：打印某个样本的输入文本、有效 label 区间的输入/输出对比。

        参数：
          input_ids:  (batch, seq)
          label_ids:  (batch, seq)，通常 -100 表示忽略位置
          lm_logits:  (batch, seq, vocab)

        输出：
          - 完整 input 解码
          - 有效 token 区间（label != -100）的 input 与 label
          - 预测输出与 label 对比
        """
        input_ids, label_ids, lm_logits = input_ids[idx], label_ids[idx], lm_logits[idx]

        # argmax 得到每个位置预测 token id
        output_ids = lm_logits.argmax(-1)

        # 有效位置：label != -100（HF 语言模型常规 mask 约定）
        valid_ids = label_ids != -100

        # decode：将 token id 转文本
        input_txt = self.tokenizer.decode(input_ids)
        input_txt_valid = self.tokenizer.decode(input_ids[valid_ids])
        label_txt_valid = self.tokenizer.decode(label_ids[valid_ids])
        output_txt_valid = self.tokenizer.decode(output_ids[valid_ids])

        print(input_txt)
        print(input_txt_valid, "->", label_txt_valid)
        print(output_txt_valid, "==", label_txt_valid)

    def _forward(self, model, inputs):
        """
        自定义 forward 包装层：统一从 inputs 取出 input_ids/label_ids，并返回 logits。

        这里做了一个关键兼容：
          - 如果 model 是 PeftModel（LoRA 等），其 base_model.forward 可能支持额外参数 label_ids
          - 某些模型作者会在 forward 中使用 label_ids 做额外逻辑（例如动态 masking / 特殊损失项）
          - 这里通过检查 base.forward 的参数名是否包含 "label_ids" 来决定是否传入

        返回：
          input_ids, label_ids, lm_logits
        """
        input_ids = inputs["input_ids"]
        label_ids = inputs["label_ids"]

        # 额外输入参数容器：默认空
        add_inputs = {}

        # PEFT 模型：外层是 PeftModel，实际 forward 往往委托给 base_model
        if isinstance(model, PeftModel):
            base = model.base_model

            # 通过 introspection 判断 base.forward 是否接受 label_ids
            # 注意：这是比较脆弱的方式（依赖 __code__.co_varnames），但在多数情况下可用
            if "label_ids" in base.forward.__code__.co_varnames:
                add_inputs["label_ids"] = label_ids

        # 调用模型 forward：
        # 对于 HF causal LM：model(input_ids).logits -> (batch, seq, vocab)
        lm_logits = model(input_ids, **add_inputs).logits
        return input_ids, label_ids, lm_logits

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        HF Trainer 的关键钩子：训练时每个 step 都会调用 compute_loss。

        在 HF Trainer 中：
          - training_step 会调用 compute_loss 获取 loss
          - loss 用于反向传播与优化器 step

        这里的行为：
          1) 调用 _forward 获取 logits
          2) 用自定义 CrossEntropy 计算 lm_loss（考虑 ignore_index）
          3) 如果模型设置了 should_training_stop，则触发保存 config 并停止训练
          4) TrainLossEarlyStop 监控 loss，可能设置 control.should_training_stop 等
          5) 返回 loss（注意：未返回 outputs，即便 return_outputs=True 也未处理）
        """
        input_ids, label_ids, lm_logits = self._forward(model, inputs)

        # 自定义交叉熵损失（通常内部做 shift/mask/ignore_index 等）
        lm_loss = self.train_crit(lm_logits, label_ids)

        # 允许模型主动请求停止训练（例如检测到某种条件）
        # should_training_stop 不是 HF 标准字段，是项目约定
        if getattr(model, "should_training_stop", False):
            # 如果模型提供 save_config，则保存配置到 output_dir
            if hasattr(model, "save_config"):
                try:
                    model.save_config(self.args.output_dir)
                except Exception:
                    pass
            # 通过 TrainerControl 停止训练循环
            self.control.should_training_stop = True

        # 训练 loss 早停器：可能基于 loss 数值（NaN/Inf/阈值/长期无改善等）控制训练流程
        self.train_loss_early_stop(self.control, lm_loss)

        return lm_loss

    def optimizer_step(self, *args, **kwargs):
        """
        覆盖 optimizer_step，但目前只是直接调用父类实现。
        预留扩展点：例如梯度裁剪、自定义 AMP、梯度累积后的特殊逻辑、日志等。
        """
        super().optimizer_step(*args, **kwargs)

    @torch.no_grad()
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        HF Trainer 在 evaluate/predict 时调用的关键钩子：prediction_step。

        标准 Trainer.prediction_step 通常返回：
          (loss, logits, labels)
        并且 logits/labels 是 batch 级别张量，供 compute_metrics 计算指标。

        这里的自定义点：
          - 先计算 lm_loss
          - 然后按“每个样本”提取有效位置（label != ignore_index）的 logits/labels
          - 返回 logits_valid 和 label_ids_valid 为“列表”，每个元素是变长张量
            这更适合某些自定义评估（比如只评估有效 token，不希望被 padding/ignore 影响）

        注意：这会改变 HF 默认的评估数据结构，需要 compute_metrics 能处理这种结构。
        """
        input_ids, label_ids, lm_logits = self._forward(model, inputs)

        # 评估 loss：这里复用 train_crit
        lm_loss = self.train_crit(lm_logits, label_ids)

        logits_valid = []
        label_ids_valid = []

        # 逐样本过滤有效位置：
        # label == ignore_index 的位置不参与评估
        for i, (logits_sample, label_ids_sample) in enumerate(zip(lm_logits, label_ids)):
            valid_pos = label_ids_sample != self.train_crit.ignore_index
            logits_sample_valid = logits_sample[valid_pos]
            label_ids_sample_valid = label_ids_sample[valid_pos]
            logits_valid.append(logits_sample_valid)
            label_ids_valid.append(label_ids_sample_valid)

        # 返回结构：(loss, predictions, labels)
        # predictions/labels 是 List[Tensor]（变长），而不是单个 Tensor
        return (lm_loss, logits_valid, label_ids_valid)

    def generation_step(self, generator, model, inputs):
        """
        单步“生成式评估”的执行函数：从 inputs 提取 input_ids/label_ids/attention_mask，
        使用 generator(model, ...) 生成输出序列，并返回预测序列与 label 序列列表。

        参数：
          generator: 一个可调用对象，约定为 generator(model, input_ids, attention_mask=...) -> out_seq
                     out_seq 可能是：
                       - 直接 Tensor (batch, seq)
                       - 或 transformers generate 输出对象，含 .sequences
          model: 被评估模型
          inputs: dataloader 输出，通常是 dict，包含 input_ids、label_ids、attention_mask 等

        返回：
          (pred_list, label_list)
          pred_list:  List[Tensor(seq)]
          label_list: List[Tensor(seq)]
        """
        if inputs is None:
            return ([], [])

        input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else None
        label_ids = inputs.get("label_ids") if isinstance(inputs, dict) else None
        attention_mask = inputs.get("attention_mask") if isinstance(inputs, dict) else None

        # 如果关键字段缺失，则本 batch 跳过
        if input_ids is None or label_ids is None:
            return ([], [])

        # 调用 generator 进行生成
        out_seq = generator(model, input_ids, attention_mask=attention_mask)

        # 如果 generator 返回的是 generate 输出对象，取 sequences
        if hasattr(out_seq, "sequences"):
            out_seq = out_seq.sequences

        # 兼容：如果返回 (seq,) 一维，则扩展为 (1, seq)
        if out_seq.dim() == 1:
            out_seq = out_seq.unsqueeze(0)

        # 兼容：label 也确保是 batch 维
        if label_ids.dim() == 1:
            label_ids = label_ids.unsqueeze(0)

        # 返回为 Python list：便于后续追加与保存
        pred_list = [row for row in out_seq]
        label_list = [row for row in label_ids]
        return (pred_list, label_list)

    def save_model(self, output_dir, _internal_call):
        """
        覆盖 HF Trainer.save_model。

        行为：
          - 若 output_dir 不存在则创建
          - 若 args.save_full_model=False，则直接 return（不保存）
          - 否则使用 torch.save(self.model, output_dir/model.pt) 保存完整模型对象

        注意/风险：
          1) torch.save 保存的是 Python 对象图，强依赖代码版本与类定义路径；
             不同环境加载可能失败（尤其是自定义模块路径变化时）。
          2) HF 标准做法是 model.save_pretrained 保存 state_dict + config，更可移植。
          3) 若是大模型 + DDP/FSDP 等，直接 torch.save(self.model) 可能产生额外问题。
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 默认不保存全量模型，以减少磁盘占用
        if not getattr(self.args, "save_full_model", False):
            return

        # 保存完整模型对象（包括结构与参数）
        torch.save(self.model, f"{output_dir}/model.pt")

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        """
        HF Trainer 内部每隔 logging_steps/save_steps/eval_steps 触发的综合钩子：
          - log
          - save checkpoint
          - evaluate

        这里插入逻辑：
          - 若 train_loss_early_stop.should_stop 为 True，则禁止 evaluate（减少无意义评估/避免评估崩溃）
        """
        if self.train_loss_early_stop.should_stop:
            self.control.should_evaluate = False
        return super()._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)

    def _save_optimizer_and_scheduler(self, output_dir: str):
        """
        覆盖 HF Trainer 的 optimizer/scheduler checkpoint 保存逻辑。

        行为：
          - 若 args.save_optimizer_state 为 False：
              - 尝试确保 output_dir 存在（即使不保存，也避免上层流程依赖目录存在）
              - 直接 return，不写 optimizer.pt / scheduler.pt
          - 若为 True：调用父类实现保存

        注意：不保存 optimizer state 会导致从 checkpoint 恢复时无法恢复动量等状态。
        """
        if not getattr(self.args, "save_optimizer_state", True):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception:
                pass
            return

        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception:
            pass

        return super()._save_optimizer_and_scheduler(output_dir)

    def _save_rng_state(self, output_dir: str):
        """
        覆盖 HF Trainer 的随机数状态保存（rng_state）。

        行为与 _save_optimizer_and_scheduler 一致：
          - save_optimizer_state=False 时不保存 rng_state，减少磁盘占用
          - 代价：断点恢复时随机性不可复现（dropout、采样等会变化）
        """
        if not getattr(self.args, "save_optimizer_state", True):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception:
                pass
            return
        return super()._save_rng_state(output_dir)

    def _get_collator_with_removed_columns(self, data_collator: Callable, description: Optional[str] = None):
        """
        HF Trainer 会在某些情况下包装 data_collator，自动移除模型 forward 不需要的 columns。
        但在自定义任务里，这种“自动删字段”可能误删 label_ids、meta 信息等，导致训练/评估异常。

        这里直接返回原始 data_collator，等价于：
          - 不做 columns 的过滤/裁剪
          - 由你的 collator 决定输出字段集合

        代价：
          - 如果 dataloader 输出包含大量模型用不到的字段，会增加显存/传输开销
        """
        return data_collator

    def reset_optimizer(self):
        """
        运行时重置优化器与学习率调度器。

        典型用途：
          - 某些训练策略需要在训练中期“重启”优化器（例如阶段性训练、学习率重置）
          - 或者当检测到坏状态时尝试恢复

        行为：
          1) self.optimizer = None, self.lr_scheduler = None
          2) 调用 create_optimizer_and_scheduler 重新创建，remaining steps = max_steps - global_step
        """
        print("Resetting optimzer")
        self.optimizer = None
        self.lr_scheduler = None

        # 注意：remaining steps 计算使用 max_steps - 已走的 global_step
        self.create_optimizer_and_scheduler(self.args.max_steps - self.state.global_step)

    def evaluate(
        self,
        eval_dataset: Dataset | Dict[str, Dataset] | None = None,
        ignore_keys: List[str] | None = None,
        metric_key_prefix: str = "eval"
    ) -> Dict[str, float]:
        """
        覆盖 HF Trainer.evaluate。

        行为分支：
          - 若 self.eval_generator 不为 None：
              -> 走生成式评估 evaluate_generation（通常用于开放式生成任务）
          - 否则：
              -> 走父类 evaluate（基于 prediction_step 的 logits/labels 评估）

        然后：
          - 若配置了 min_eval_metric_after_epoch_early_stop，则根据 metrics 决定是否 early stop
        """
        if self.eval_generator is not None:
            metrics = self.evaluate_generation(self.eval_generator, metric_key_prefix=metric_key_prefix)
        else:
            metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # 指标下限早停（例如某个 epoch 后 eval 指标一直低于阈值则停止）
        if self.min_eval_metric_after_epoch_early_stop is not None:
            self.min_eval_metric_after_epoch_early_stop(self.control, metrics)

        return metrics

    @torch.no_grad()
    def evaluate_generation(self, generator, use_cache=True, skip_metrics=False, metric_key_prefix="eval"):
        """
        生成式评估：
          - 从 eval dataloader 遍历 batch
          - 对每个 batch 调用 generation_step 生成 pred_ids
          - 收集 input_ids/pred_ids/label_ids
          - 构造 EvalPredictionWithText（含 tokenizer 解码、可保存为 yaml）
          - 将预测保存到 output_dir/predictions-{global_step}.yaml 作为缓存
          - 如 skip_metrics=False，则从缓存/文件读取并计算 metrics，写入日志与回调系统

        参数：
          generator: 生成器对象（约定 generator.tokenizer 存在，且 generator 可调用）
          use_cache: 若 True 且预测文件已存在，则跳过重新生成，直接复用文件
          skip_metrics: 若 True，仅生成/缓存，不计算 metrics（或仅加载提示）
          metric_key_prefix: 指标前缀（默认 "eval"），会生成 eval_xxx 键名
        """
        # 预测缓存文件：用 global_step 区分不同 checkpoint 步的评估预测
        eval_pred_file = Path(self.args.output_dir) / f"predictions-{self.state.global_step}.yaml"

        # 若不使用缓存或文件不存在：重新跑生成
        if not use_cache or not eval_pred_file.is_file():
            model = self.model
            model.eval()

            # 获取 eval dataloader（HF Trainer 会处理 sampler、batch size 等）
            dataloader = self.get_eval_dataloader()

            # 三类序列收集：
            #   input_ids_all: 原输入（便于还原上下文）
            #   pred_ids_all:  生成输出
            #   label_ids_all: 参考答案/标签序列
            input_ids_all = []
            pred_ids_all = []
            label_ids_all = []

            for step, inputs in enumerate(dataloader):
                if inputs is None:
                    continue

                # 生成预测与取 label
                pred_ids, label_ids = self.generation_step(generator, model, inputs)
                if not pred_ids or not label_ids:
                    continue

                # 取 batch 输入
                batch_input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else None
                if batch_input_ids is None:
                    continue

                # 追加到总列表
                input_ids_all += [*batch_input_ids]
                pred_ids_all += [*pred_ids]
                label_ids_all += [*label_ids]

            # 构建带文本的 EvalPrediction（通常内部会 decode 并支持保存/加载）
            # remove_eos=True：可能用于把生成/label 的 eos 去掉，避免影响字符串比较/指标
            eval_pred = EvalPredictionWithText(
                generator.tokenizer,
                input_ids_all,
                pred_ids_all,
                label_ids_all,
                save_file=eval_pred_file,
                remove_eos=True
            )

            # 保存为 yaml：便于人工检查、也便于复用缓存
            eval_pred.save()

        else:
            # 文件存在且 use_cache=True：可直接复用
            if not skip_metrics:
                print(f"Loading prediction {eval_pred_file}")

        # 若需要计算指标
        if not skip_metrics:
            # 从文件加载（保证与 compute_metrics 输入一致）
            eval_pred = EvalPredictionWithText.from_file(str(eval_pred_file))

            # compute_metrics 是在 __init__ 传入的函数：接受 EvalPrediction（或其子类）返回 dict
            metrics = self.compute_metrics(eval_pred)

            # 添加 eval_ 前缀（与 HF 习惯一致）
            if metric_key_prefix != "":
                metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}

            # 记录日志到 Trainer 的 log 系统（会写到 stdout/W&B/TensorBoard 等）
            self.log(metrics)

            # 触发回调：on_evaluate（例如 EarlyStoppingCallback 之类）
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

            return metrics
        else:
            # 只生成不算指标
            return None