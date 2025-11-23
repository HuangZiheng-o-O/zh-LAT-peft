# DART 数据集优化日志

## 优化时间
2025年11月23日

## 优化背景
严格检查了项目流程对DART数据集的适配性，发现需要按照官方GEM/DART标准进行优化。

## 主要优化内容

### 1. 评估指标扩展 (compute_metrics)
**优化前**: 只使用BLEU和METEOR
```python
meteor_score = meteor.compute(predictions=predictions, references=references)["meteor"]
bleu_score = bleu.compute(predictions=predictions, references=references)["bleu"]
results = {"meteor": meteor_score, "bleu": bleu_score}
```

**优化后**: 添加chrF指标，符合GEM/DART标准
```python
bleu_score = bleu.compute(predictions=predictions, references=references)["bleu"]
meteor_score = meteor.compute(predictions=predictions, references=references)["meteor"]
chrf_score = chrf.compute(predictions=predictions, references=references)["score"]
results = {"bleu": bleu_score, "meteor": meteor_score, "chrf": chrf_score}
```

### 2. 文本规范化 (Text Normalization)
**新增**: 预测文本和参考文本的基本规范化
```python
# Basic normalization: trim whitespace; split labels into multi-refs by sep_token
predictions = [p.strip() if isinstance(p, str) else "" for p in predictions]
references = [
    [r.strip() for r in (rs.split(self.sep_token) if isinstance(rs, str) else []) if r.strip()]
    for rs in references
]
```

### 3. 本地评估日志 (_save_local_eval_log)
**新增**: 类似Spider数据集的本地评估日志功能
- 只在 `SWANLAB_MODE=cloud` 时保存
- 保存到 `my_swanlog/local_eval_logs/` 目录
- 只记录低重叠度样本 (Jaccard相似度 ≤ 0.20)
- 限制记录数量 (最多200个样本)
- 文件命名包含实验组信息: `{suite}_r{round}_s{seed}_{data}`

**日志格式**:
```
=== EVALUATION SUMMARY (DART) ===
bleu: 0.1234
meteor: 0.5678
chrf: 0.9012

=== LOW-OVERLAP EXAMPLES (≤0.20 Jaccard) ===
[42] jaccard=0.150 ratio=0.180
pred: generated text here
ref1: reference text 1
ref2: reference text 2

logged_examples: 150
```

## 技术细节

### 评估指标说明
- **BLEU**: Bilingual Evaluation Understudy，基于n-gram匹配
- **METEOR**: Metric for Evaluation of Translation with Explicit ORdering，考虑词干、同义词等
- **chrF**: Character n-gram F-score，基于字符级别的n-gram匹配，更适合形态丰富的语言

### 生成设置
- 继续使用HF generate的max_new_tokens语义
- 默认贪心解码 (greedy decoding)
- 支持beam search (通过EVAL_GEN_NUM_BEAMS设置)

### 数据加载
- 支持GEM/dart数据集的标准格式
- 自动处理多参考文本 (multi-reference)
- 鲁棒的三元组线性化 (triple linearization)

## 使用建议

### 环境变量设置
```bash
# DART评估设置
export EVAL_GEN=1
export EVAL_GEN_MAX_LENGTH=96
export EVAL_GEN_MIN_LENGTH=8

# SwanLab日志
export SWANLAB_MODE=cloud
export SWANLAB_EMAIL_ON_START=1
export SWANLAB_EMAIL_ON_FINISH=1
```

### 评估结果解读
- **BLEU**: 0-1之间，越高越好，通常在0.1-0.4之间
- **METEOR**: 0-1之间，越高越好，通常在0.2-0.5之间
- **chrF**: 0-1之间，越高越好，通常在0.3-0.6之间

### 本地日志查看
运行DART实验后，查看:
```bash
ls my_swanlog/local_eval_logs/eval_log_dart_*.txt
```

## 兼容性确认
- ✅ 与现有HF generate流程完全兼容
- ✅ 与SwanLab集成无冲突
- ✅ 与其他数据集(Spider, SamSum)评估流程并存
- ✅ 支持多参考评估
- ✅ 符合GEM/DART官方标准

## 下一步优化建议
1. 可考虑添加ROUGE-L指标 (可选)
2. 可测试beam search对chrF的影响
3. 可根据具体实验结果调整Jaccard阈值

## 验证方法
```bash
# 检查预测文本是否不同
python - <<'PY'
from pathlib import Path
import yaml
files = list(Path("output/").glob("**/predictions-*.yaml"))
if len(files) >= 2:
    pa = yaml.safe_load(open(files[0]))["preds"]
    pb = yaml.safe_load(open(files[1]))["preds"]
    n = min(len(pa), len(pb))
    same = sum(1 for i in range(n) if pa[i].strip()==pb[i].strip())
    print(f"总样本: {n}, 相同预测: {same} ({same/n:.1%})")
PY
```</content>
</xai:function_call">Writing file...
