
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