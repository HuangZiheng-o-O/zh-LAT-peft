# Spider Cache 元数据问题

## 修改摘要

**文件:** `dataset/spider_data.py`

---

## 1. Cache版本号机制 ✅

```python
# 新增版本常量
SPIDER_CACHE_VERSION = "v2"

# get_cache_name() 中嵌入版本号
def get_cache_name(self):
    name = f"cache_{name}_{self.split}_seqlen{self.max_seqlen}_{SPIDER_CACHE_VERSION}"
    #                                                         ^^^^^^^^^^^^^^^^^^^^^^
    # 新cache文件名: cache_xlangai_spider-tvt_test_seqlen1536_v2.pkl
    # 旧cache文件名: cache_xlangai_spider-tvt_test_seqlen1536.pkl (自动失效)
```

---

## 2. 运行时格式检测 ✅

```python
def _detect_cache_format(self) -> str:
    """
    Returns:
        "v2": 新格式 ((input_ids, label_ids), {"db_id": ..., "query": ...})
        "v1": 旧格式 (input_ids, label_ids)
        "unknown": 无法识别
    """
```

---

## 3. 自动迁移机制 ✅

```python
def _migrate_cache_v1_to_v2(self) -> bool:
    """
    从HF数据集重建meta信息，将v1格式迁移到v2
    - 自动检测旧cache
    - 尝试从HF数据集补充 db_id 和 query
    - 发出警告提示用户删除旧cache
    """
```

---

## 4. 多层Fallback机制 ✅

```python
def get_db_id(self, idx) -> Optional[str]:
    # 优先级:
    # 1. 从cache的meta中获取
    # 2. 从HF数据集获取 (fallback)

def get_query(self, idx) -> Optional[str]:
    # 同上
```

---

## 5. compute_metrics增强 ✅

```python
def compute_metrics(self, eval_preds, eval_mask=None):
    # 使用getter方法获取metadata，自带fallback
    for i in range(size):
        db_id = self.get_db_id(i)   # 有fallback
        query = self.get_query(i)   # 有fallback

    # 清晰的错误信息
    if db_id is None or query is None:
        raise RuntimeError(
            "Remediation options:\n"
            "1. Delete cache: rm -rf data/xlangai_spider*/cache_*.pkl\n"
            "2. Set new cache tag: export DATA_CACHE_TAG=v2_rebuild\n"
            "3. Check HF dataset availability"
        )
```

---

## 兼容性保证

| 场景                    | 行为                  |
| ----------------------- | --------------------- |
| 新cache (v2)            | 直接使用 ✅            |
| 旧cache (v1) + HF可用   | 自动迁移 + 警告 ✅     |
| 旧cache (v1) + HF不可用 | 清晰错误 + 修复指引 ✅ |
| 未知格式                | 尝试继续 + 警告 ✅     |

---

## Debug模式

```bash
# 开启详细日志
export SPIDER_DEBUG=1
```

---

## 旧cache清理命令

```bash
# 删除所有旧版本cache (没有_v2后缀的)
find data/xlangai_spider* -name "cache_*.pkl" ! -name "*_v2*.pkl" -delete
```
