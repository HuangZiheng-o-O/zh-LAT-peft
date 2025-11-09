# 紧急修复：decoder.py UnboundLocalError

**时间**: 2025-11-08  
**问题**: 训练正常，但在第一次评估时崩溃  
**错误**: `UnboundLocalError: local variable 'BeamSearchScorer' referenced before assignment`

---

## 问题描述

**症状**：
- ✅ 训练前 10 分钟正常（2000 steps）
- ✅ 损失正常下降（2.88 → 1.05）
- ❌ 第一次评估时崩溃（`Evaluate: 0%`）
- ❌ 所有 GPU 任务陆续失败

**错误栈**：
```
File "mamba_ssm_peft/utils/decoder.py", line 116, in forward
    if BeamSearchScorer is None:
UnboundLocalError: local variable 'BeamSearchScorer' referenced before assignment
```

**根本原因**：
Python 作用域问题。在 `forward()` 方法中：
1. 第 116 行：检查 `if BeamSearchScorer is None:`
2. 第 120 行：尝试赋值 `BeamSearchScorer = _BSS`
3. Python 看到赋值，认为 `BeamSearchScorer` 是局部变量
4. 但在赋值前就引用了它 → `UnboundLocalError`

---

## 修复方案

### 方法 1: 使用局部变量（推荐）

**文件**: `mamba-peft/mamba_ssm_peft/utils/decoder.py`  
**位置**: 第 112-125 行

```python
# 修复前（错误）：
def forward(self, model, input_ids):
    device = input_ids.device
    
    # instantiate beam scorer
    if BeamSearchScorer is None:  # ← 错误：引用了局部变量
        try:
            from transformers.generation.beam_search import BeamSearchScorer as _BSS
            BeamSearchScorer = _BSS  # ← 这里赋值使它变成局部变量
        except Exception as e:
            raise ImportError("BeamSearchScorer is not available") from e
    
    beam_scorer = BeamSearchScorer(  # ← 使用局部变量

# 修复后（正确）：
def forward(self, model, input_ids):
    device = input_ids.device
    
    # instantiate beam scorer
    _BeamSearchScorer = BeamSearchScorer  # ← 先复制到局部变量
    if _BeamSearchScorer is None:
        try:
            from transformers.generation.beam_search import BeamSearchScorer as _BSS
            _BeamSearchScorer = _BSS
        except Exception as e:
            raise ImportError("BeamSearchScorer is not available") from e
    
    beam_scorer = _BeamSearchScorer(  # ← 使用局部变量
```

---

## 验证修复

```bash
cd /mnt/data4/mzs_h/code/zh-LAT-peft/mamba-peft

# 检查修复
echo "=== 检查第 116 行 ==="
sed -n '116p' mamba_ssm_peft/utils/decoder.py
# 应该看到: _BeamSearchScorer = BeamSearchScorer

echo "=== 检查第 117 行 ==="
sed -n '117p' mamba_ssm_peft/utils/decoder.py
# 应该看到: if _BeamSearchScorer is None:

echo "=== 检查第 121 行 ==="
sed -n '121p' mamba_ssm_peft/utils/decoder.py
# 应该看到: _BeamSearchScorer = _BSS

echo "=== 检查第 125 行 ==="
sed -n '125p' mamba_ssm_peft/utils/decoder.py
# 应该看到: beam_scorer = _BeamSearchScorer(

# 清理 Python 缓存
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete

# 快速测试（不会真正运行评估，只是检查导入）
python -c "from mamba_ssm_peft.utils.decoder import MambaBeamSearchDecoder; print('✓ Import successful')"
```

---

## 重启训练

```bash
cd /mnt/data4/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

# 如果之前的训练有 checkpoint，可以恢复：
# 训练会自动从最后一个 checkpoint 恢复

# 重新运行训练
export DART_LOCAL_DIR=/mnt/data4/mzs_h/code/zh-LAT-peft/mamba-peft/data/GEM_dart/
EVAL_GEN=1 \
EVAL_GEN_MAX_LENGTH=1024 \
EVAL_GEN_MIN_LENGTH=5 \
EVAL_GEN_NUM_BEAMS=5 \
HP_EVAL_STEPS=2000 \
HP_SAVE_STEPS=2000 \
HP_LOGGING_STEPS=200 \
SWANLAB_ENABLE=1 \
SWANLAB_MODE=cloud \
SWANLAB_PROJECT="gla-mamba-dart-new-1-4090E11" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUM_DATA_WORKERS=4 \
GRADIENT_CHECKPOINTING=true \
LOGITS_TO_KEEP=1 \
./gla_batch_tmux.sh --suite E11 --round all \
  --pairs "87:dart" \
  --gpus "0 1 2 3 4 5 6 7" \
  --gpu-plan "1,1,1,1,1,1,1,1"
```

---

## 为什么之前没有发现这个 bug？

1. **训练阶段不使用 beam search**：只有评估（生成）阶段才会调用 `MambaBeamSearchDecoder`
2. **评估在第 2000 步才触发**：前 10 分钟都在训练，没有触发评估
3. **GLUE 数据集不使用生成模式**：之前的 GLUE 实验都是分类任务，不需要 beam search
4. **DART 是第一个生成任务**：这是第一次真正测试 beam search decoder

---

## 影响范围

**受影响的数据集**（所有生成任务）：
- ✅ DART（RDF → 文本）
- ✅ SAMSum（对话摘要）
- ✅ Spider（Text-to-SQL）
- ✅ 任何其他使用 `mode="gen"` 的数据集

**不受影响的数据集**（分类任务）：
- ✅ GLUE（所有子任务）
- ✅ PIQA
- ✅ BoolQ
- ✅ ARC
- ✅ MNIST
- ✅ CIFAR-10

---

## 快速部署到所有服务器

```bash
# 从 Mac 上传修复后的文件到所有服务器
for server in server1 server2 server3; do
    echo "=== 上传到 $server ==="
    scp /Users/huangziheng/PycharmProjects/all_code/codeH1_4090/code/zh-LAT-peft/mamba-peft/mamba_ssm_peft/utils/decoder.py \
        user@$server:/mnt/data4/mzs_h/code/zh-LAT-peft/mamba-peft/mamba_ssm_peft/utils/decoder.py
    
    # 清理缓存
    ssh user@$server "cd /mnt/data4/mzs_h/code/zh-LAT-peft/mamba-peft && \
        find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
        find . -name '*.pyc' -delete"
done
```

---

## 总结

| 项目 | 状态 |
|------|------|
| **问题类型** | Python 作用域 bug（UnboundLocalError） |
| **影响** | 所有生成任务的评估阶段 |
| **修复难度** | ⭐ 简单（3 行代码） |
| **修复时间** | < 1 分钟 |
| **测试** | 重启训练，等待第一次评估（2000 steps） |
| **优先级** | 🔥 紧急（阻塞所有生成任务） |

---

**修复完成后，训练将能够正常进行评估，不会再在 2000 steps 时崩溃！**


