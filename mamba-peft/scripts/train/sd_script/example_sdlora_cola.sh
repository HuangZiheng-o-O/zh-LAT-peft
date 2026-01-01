
# =========================
# SD-LoRA CoLA 训练命令示例
# =========================
# 批量测试15个SD-LoRA配置在CoLA数据集上
#
# SD-LoRA配置 (15个):
#   KV:   train01, train05, train10, train20, train30 (5个)
#   QKVO: train01, train05, train10, train20, train30 (5个)
#   其他: v, vo, qkvog, qkvo_plus_mlp, omlp (各1个, 仅train05) (5个)

############################
# 1. 环境激活
############################
conda activate mzsz
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft/scripts/train/new

############################
# 2. 模型配置
############################
export MODEL_TYPE=gla
export LAT_MODEL=/home/user/mzs_h/model/second-gla-1.3B-100B/gla-1.3B-100B
export LAT_PREC=bf16

############################
# 3. 严格离线 + 本地 HF 缓存
############################
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/user/mzs_h/data/hf_cache
export HF_DATASETS_CACHE="$HF_HOME"
export HF_EVALUATE_CACHE="$HF_HOME"

############################
# 4. 任务配置
############################
# CoLA: 语法可接受性判断 (二分类)
export EVAL_GEN=0
export HP_VAL_SPLIT=test      # 按 glue-tvt 约定

############################
# 5. 训练超参
############################
# CoLA 训练集约 8.5k 样本，属于小数据任务
export HP_EPOCHS=10           # 小数据任务多迭代
export HP_BATCH_SIZE=8
export HP_LR=0.0005           # 5e-4

############################
# 6. 评测 / 保存 / 日志
############################
export HP_EVAL_BATCH_SIZE=64
export HP_EVAL_STEPS=100      # 小任务频繁评测
export HP_SAVE_STEPS=200
export HP_LOGGING_STEPS=20

############################
# 7. 学习率调度
############################
export LR_SCHEDULER_TYPE=cosine
export LR_WARMUP_RATIO=0.1

############################
# 8. DataLoader / Runtime
############################
export NUM_DATA_WORKERS=4
export DATALOADER_PREFETCH_FACTOR=2
export DATALOADER_PIN_MEMORY=1
export DATALOADER_PERSISTENT_WORKERS=0
export GRADIENT_CHECKPOINTING=true
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error

############################
# 9. SwanLab 监控
############################
export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="gla-sdlora-cola-Jan01"
export SWANLAB_EMAIL_ON_START=0
export SWANLAB_EMAIL_ON_FINISH=0
export SWANLAB_EMAIL_ON_INTERRUPT=0

############################
# 10. 启动交错间隔 (可选)
############################
# 避免同时加载模型导致显存峰值
export LAT_LAUNCH_STAGGER_MINUTES=0

############################
# 11. 启动 SD-LoRA 批量训练
############################
# 15个配置 × 7个GPU = 约3轮
# 每轮并行训练7个配置
./lat_batch_tmux_sparse.sh \
  --round all \
  --pairs "87:glue-tvt_cola" \
  --gpus "0 1 2 3 4 5 6 7" \
  --gpu-plan "1,1,1,1,1,1,1,1" \
  --model-type gla
