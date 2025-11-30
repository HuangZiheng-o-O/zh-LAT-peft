# SST-2 GLUE 训练卡住问题诊断报告

## 问题描述

SST-2 (Stanford Sentiment Treebank) GLUE 任务训练在打印 `torch.compile is not available in Python 3.10, using identity decorator instead` 之后卡住不动，其他数据集正常。

## 时间线

- **2025-12-01 ~06:00**: 首次发现问题，SST-2 训练卡住
- **2025-12-01 ~06:35**: 添加详细 debug 输出定位问题
- **2025-12-01 ~06:46**: 发现卡在缓存构建的等待锁阶段
- **2025-12-01 ~07:00**: 确认僵尸锁文件问题，添加自动检测机制

## 问题分析

### 初始症状
```bash
# 正常输出后卡住
torch.compile is not available in Python 3.10, using identity decorator instead
Loading GLA model: /home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B
[GLA] fuse_swiglu disabled; using PyTorch SwiGLU.
[GLA] Respecting tokenizer's original padding policy (GLA_FORCE_LEFT_PAD=0).
# 这里卡住，无后续输出
```

### 排查步骤

#### 1. 添加调试输出
在关键模块添加带时间戳的 debug 输出：
- `train_gla_only.py`: 主导入和初始化流程
- `mamba_ssm_peft/__init__.py`: PEFT 包导入
- `mamba_ssm_peft/utils/hf.py`: GLA 模型加载
- `dataset/__init__.py`: 数据集模块导入
- `dataset/glue.py`: GLUE 数据集加载
- `dataset/base.py`: 数据集缓存构建

#### 2. 定位卡住点
通过 debug 输出发现卡在：
```
[DEBUG][06:46:41.685] [glue.py]   Calling super().__init__ (NluDatasetBase)...
[DEBUG][06:46:41.685] [base.py] DatasetBase.__init__ START: path=nyu-mll/glue, split=train, use_cache=True, num_parallel_workers=16
[DEBUG][06:46:41.685] [base.py]   cache_file_stem = cache_sst2-tvt_train
[DEBUG][06:46:41.685] [base.py]   cache_file = data/nyu-mll_glue/cache_sst2-tvt_train.pkl
[DEBUG][06:46:41.685] [base.py]   cache_file.exists() = False, lock_file.exists() = True
[DEBUG][06:46:41.685] [base.py]   SLOW-PATH: Cache miss or locked, need to build data...
[DEBUG][06:46:41.685] [base.py]   Lock already held by another process
[DEBUG][06:46:41.685] [base.py]   WAITER: Spinning until cache file appears (lock held by another process)...
```

#### 3. 根本原因
**僵尸锁文件问题**：
- 之前的 SST-2 训练在缓存构建过程中被中断（Ctrl+C 或异常退出）
- 锁文件 `cache_sst2-tvt_train.pkl.lock` 被创建但未被删除
- 新进程检测到锁存在，误以为有其他进程正在构建缓存
- 进入无限等待循环，每 10 秒打印一次等待状态

## 解决方案

### 即时修复
```bash
# 删除僵尸锁文件
rm -f /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/nyu-mll_glue/cache_sst2-tvt_train.pkl.lock
rm -f /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data/nyu-mll_glue/cache_sst2-tvt_val.pkl.lock
```

### 预防措施

#### 1. 僵尸锁自动检测
在 `dataset/base.py` 添加锁文件年龄检测：

```python
# 检查僵尸锁（超过 10 分钟的锁）
if lock_file.exists():
    lock_age = time.time() - os.path.getmtime(str(lock_file))
    if lock_age > 600:  # 10 分钟
        print(f"STALE LOCK DETECTED: lock is {lock_age:.0f}s old (>600s), removing...")
        os.remove(lock_file)
```

#### 2. 缓存预热避免冲突
建议先用单进程预热缓存，再进行多进程训练：

```bash
# 方案1: 专门的预热脚本
export NUM_DATA_WORKERS=0
python -c "
from transformers import AutoTokenizer
from dataset import load_dataset
tok = AutoTokenizer.from_pretrained('path/to/tokenizer')
for split in ('train','val'):
    dm = load_dataset('glue-tvt_sst2', tok, split, return_module=True)
    print(f'Pre-warmed cache for split={split}')
"

# 方案2: 单卡预热训练
./gla_batch_tmux_clean.sh \
  --suite E15 --round all \
  --pairs "87:glue-tvt_sst2" \
  --gpus "3" --gpu-plan "1"
# 然后再用多卡运行
```

#### 3. 缓存隔离
通过环境变量隔离不同实验的缓存：
```bash
export DATA_CACHE_TAG=sst2_exp1  # 会生成 cache_sst2-tvt_train_sst2_exp1.pkl
```

## 代码修改总结

### 修改的文件
1. **train_gla_only.py**: 添加 10 步导入调试
2. **mamba_ssm_peft/__init__.py**: PEFT 包导入调试
3. **mamba_ssm_peft/utils/hf.py**: GLA 加载调试
4. **dataset/__init__.py**: 数据集模块导入调试
5. **dataset/glue.py**: GLUE 数据集加载调试，移除刷屏日志
6. **dataset/base.py**: 缓存构建调试，僵尸锁检测

### 关键修复
- 在 `dataset/base.py` 添加僵尸锁检测和自动清理
- 移除高频调用的刷屏 debug 输出
- 保留关键路径的调试信息

## 后续建议

### 缓存管理最佳实践
1. **定期清理锁文件**：
   ```bash
   find /path/to/mamba-peft/data -name "*.lock" -mmin +10 -delete
   ```

2. **预热策略**：
   - 新数据集首次使用时先单进程预热
   - 批量训练前统一预热所有需要的缓存

3. **监控和报警**：
   - 训练脚本中添加锁文件年龄监控
   - 长时间等待时发出警告

### 调试经验
- 卡住问题通常是网络请求或锁等待
- 离线环境需确保所有依赖已缓存
- 并行进程间协调需要可靠的锁机制
- 适当的超时和重试机制很重要

## 测试验证
- 删除锁文件后 SST-2 训练正常启动
- 缓存构建完成后，后续运行走 FAST-PATH
- 僵尸锁检测机制有效防止类似问题再发

---
**报告生成时间**: 2025-12-01
**问题解决状态**: ✅ 已解决
**预防机制**: ✅ 已实施
