### Evaluation in zh-LAT-peft (commonsense + PEFT batch)

This repo supports two evaluation paths:

- **Path A (recommended, fully local/offline-friendly)**: `eval_lat.py`
  - Uses your existing training framework (LAT loader/adapter + dataset/*).
  - Prefers local dataset repos under `mamba-peft/data/` (or `$LAT_DATA_DIR` / `$DATA_DIR`).
  - Writes outputs to `mamba-peft/outputs/lm_eval/`.

- **Path B (optional, lm-evaluation-harness)**: `scripts/eval/lat_lm_harness_eval.py`
  - Reuses your **model loader/adapter**, but **datasets are managed by lm_eval**.
  - If you need strict “local dataset dir” control, use Path A.
  - You can also run Path B **through `eval_lat.py`** by setting `EVAL_BACKEND=lm_eval`
    (this integrates cleanly with `lat_round.sh / lat_batch_tmux.sh`'s `--eval-after-train`).

---

### Path A: `eval_lat.py` (recommended)

#### 1) Offline/local datasets

Put dataset repos under:
- `mamba-peft/data/` (default), or set:
  - `export LAT_DATA_DIR=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/data`

Preprocessed `.pkl` caches are written under:
- `mamba-peft/data/cache/` by default
- override with `LAT_DATA_CACHE_DIR` if you want

#### 2) Evaluate a trained adapter (from the same YAML you trained with)

```bash
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft
export MODEL_TYPE=retnet
export LAT_MODEL=/home/user/mzs_h/model/retnet-1.3B-100B/
export LAT_PREC=bf16

export EVAL_TASKS='boolq,social_iqa,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa'
python eval_lat.py --cfg cfg/my_lora_exp/yaml/E1_QKVO_plus_MLP_r8_alpha16.yaml --model-type retnet
```

#### 3) Evaluate with an explicit PEFT checkpoint directory

```bash
python eval_lat.py \
  --model-type retnet \
  --model /home/user/mzs_h/model/retnet-1.3B-100B/ \
  --prec bf16 \
  --peft-weights /path/to/checkpoint-3200 \
  --tasks boolq,piqa
```

---

### Path A + batch: reuse `lat_round.sh / lat_batch_tmux.sh`

You can now batch **train→eval** across LoRA configs in a suite/round:

```bash
export EVAL_AFTER_TRAIN=1
export EVAL_TASKS='boolq,social_iqa,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa'
export EVAL_BATCH_SIZE=64

./lat_batch_tmux.sh --suite E14 --round all --pairs "87:glue-tvt_cola" --model-type retnet --eval-after-train
```

Eval-only mode (no training, just evaluate adapters already saved under the output dirs):

```bash
./lat_batch_tmux.sh --suite E14 --round all --pairs "87:glue-tvt_cola" --model-type retnet --eval-only --eval-tasks "$EVAL_TASKS"
```

Outputs:
- `mamba-peft/outputs/lm_eval/<model_type>_<model>_<adapter>/summary.json`

---

### Path B (optional): lm-evaluation-harness

Install:

```bash
pip install lm-eval
# or (if needed)
pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git
```

Run:

```bash
TASKS='boolq,social_iqa,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa'

python scripts/eval/lat_lm_harness_eval.py \
  --model LAT \
  --model_args pretrained="fla-hub/gla-1.3B-100B,model_type=gla,prec=bf16,peft_weights=/path/to/adapter,trust_remote_code=True" \
  --tasks $TASKS \
  --output_path outputs/lm_eval/lm_harness
```

---

### Path B + batch (recommended for your commonsense_170k workflow)

Goal: **train once on `commonsense_170k` (mixed), then evaluate on 8 tasks via lm_eval_harness**.

Key idea:
- Training data is fixed by `--pairs "SEED:commonsense_170k"`.
- Evaluation tasks are controlled by `EVAL_TASKS=...` (lm_eval runs each task separately, but one command can run the list).
- Switch eval backend with: `EVAL_BACKEND=lm_eval`.

Example (train→lm_eval on the same GPU per LoRA config):

```bash
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

export MODEL_TYPE=delta_net
export LAT_MODEL=/home/user/mzs_h/model/delta_net-1.3B-100B/
export LAT_PREC=bf16

export EVAL_AFTER_TRAIN=1
export EVAL_BACKEND=lm_eval
export EVAL_TASKS='boolq,social_iqa,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa'
export EVAL_BATCH_SIZE=64
export EVAL_OUTPUT_ROOT=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/outputs/lm_eval

./lat_batch_tmux.sh \
  --suite E14 \
  --round all \
  --pairs "87:commonsense_170k" \
  --gpus "0 1 2 3 4 5 6 7" \
  --gpu-plan "1,1,1,1,1,1,1,1" \
  --model-type delta_net \
  --eval-after-train \
  --eval-backend lm_eval \
  --eval-tasks "$EVAL_TASKS" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --eval-output-root "$EVAL_OUTPUT_ROOT"
```



