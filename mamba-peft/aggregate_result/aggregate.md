
```bash
 pip install matplotlib
```
```bash

conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft

python -m aggregate_result.main \
  --base_dir /home/user/mzs_h/output/benchmark/glue \
  --dataset mrpc \
  --output /home/user/mzs_h/output/benchmark/glue_agg \
  --workers 8

```

```bash

conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft

python -m aggregate_result.main \
  --base_dir /home/user/mzs_h/output/benchmark/glue \
  --dataset cola \
  --output /home/user/mzs_h/output/benchmark/glue_agg \
  --workers 8

```


```bash

conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft

python -m aggregate_result.main \
  --base_dir /home/user/mzs_h/output/benchmark/glue \
  --dataset rte \
  --output /home/user/mzs_h/output/benchmark/glue_agg \
  --workers 8

```

  根据 aggregate_result/main.py:20-21 的代码：

  p.add_argument("--dataset", type=str, default=None,
                 help="Optional dataset filter (e.g., glue-tvt_rte or rte). If omitted, process all under base_dir.")

  两种使用方式对比：

  1️⃣ 你选中的命令（无 --dataset 参数）

  python -m aggregate_result.main \
    --base_dir /home/user/mzs_h/output/1/Jan4/ \
    --output /home/user/mzs_h/output/1/glue_aggJan4 \
    --workers 8
  功能：聚合 /home/user/mzs_h/output/benchmark/glue 下的所有数据集
  - CoLA
  - SST-2
  - MRPC
  - QQP
  - MNLI
  - QNLI
  - RTE
  - 等等...

  2️⃣ 指定单个数据集（有 --dataset 参数）

  python -m aggregate_result.main \
    --base_dir /home/user/mzs_h/output/benchmark/glue \
    --dataset qqp \
    --output /home/user/mzs_h/output/benchmark/glue_agg \
    --workers 8
  功能：只聚合 QQP 数据集的结果

  聚合脚本的功能

  从代码来看，它会为每个数据集生成：
  1. CSV 文件：实验结果汇总表（main.py:94）
  2. 数据集级别的汇总：每个数据集的统计信息（main.py:89-98）
  3. 总体汇总 JSON：aggregate_summary.json（main.py:100-101）

  输出目录结构会是：
  /home/user/mzs_h/output/benchmark/glue_agg/
  ├── glue-tvt_cola_seed87/
  │   ├── [各个实验的聚合结果]
  ├── glue-tvt_qqp_seed87/
  │   ├── [各个实验的聚合结果]
  ├── glue-tvt_mnli_seed87/
  │   ├── [各个实验的聚合结果]
  └── aggregate_summary.json  # 总汇总

  所以你的命令会一次性处理所有 GLUE 数据集的实验结果。