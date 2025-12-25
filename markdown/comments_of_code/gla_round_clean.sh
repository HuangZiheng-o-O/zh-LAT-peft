#!/bin/bash
# ^ 指定使用 bash 解释器执行脚本（而不是 sh）。
#   重要：某些 bash 特性（数组、[[ ]]、${var,,}、declare -a 等）在 sh 下不可用。

set -euo pipefail
# -e：任意命令返回非 0 时立即退出（但在 if/while/! 等结构中有例外）。
# -u：引用未定义变量时立即报错退出（可防止拼写错误导致 silent bug）。
# -o pipefail：管道中任意一个命令失败，则整个管道视为失败。
# 组合效果：让脚本“更严格”，更早暴露错误；但也可能让一些“非关键失败”需要显式 `|| true` 兜底。

LAUNCHER_PY="train_gla_only.py"
# Python 启动器脚本名。后面所有训练任务均通过：python train_gla_only.py --cfg <yaml> 启动。

: "${ROUND_E_MASTER[@]:-}" >/dev/null 2>&1 || declare -a ROUND_E_MASTER=()
# 这一段是一个“兼容性/防御性”写法：
# - `: ...` 是空操作命令（noop），只用于触发参数展开检查。
# - `"${ROUND_E_MASTER[@]:-}"`：如果数组未定义，就展开为空（避免 set -u 触发错误）。
# - 如果上面失败（通常是未声明数组导致），则 declare -a ROUND_E_MASTER=() 初始化为空数组。

#####################################################################
# 全局运行态变量（数组/状态）初始化
#####################################################################

declare -a PIDS=()
# 保存当前 round 启动的后台训练进程 PID 列表，用于 wait、以及失败/中断时批量 kill。

declare -a COMPLETED_ROUNDS=()
# 保存成功完成的 round 编号，用于中断/失败时打印总结。

declare -a RUN_QUEUE=()
# 保存计划要执行的 round 队列（可能来自命令行参数或默认规则）。

declare -a DETECTED_GPUS=()
# 保存检测到的 GPU ID 列表（字符串数组），来源可能是 GPU_IDS/CUDA_VISIBLE_DEVICES/nvidia-smi。

CURRENT_ROUND=""
# 记录当前正在运行的 round 编号（字符串）。用于中断时输出“哪一轮被打断”。

FAILED_ROUND=""
# 记录失败的 round 编号。用于失败总结。

print_interruption_summary() {
  # 这个函数用于“被中断”时（例如 Ctrl+C / kill TERM）打印摘要。
  echo ""
  echo "SUMMARY:"
  if (( ${#COMPLETED_ROUNDS[@]} > 0 )); then
    # 数组长度 > 0：打印已经完成的 rounds
    echo "  Experiments completed: ${COMPLETED_ROUNDS[*]}."
  else
    echo "  Experiments completed: none."
  fi
  if [[ -n "${CURRENT_ROUND:-}" ]]; then
    # CURRENT_ROUND 非空：说明中断发生在某一轮执行过程中
    echo "  Experiment ${CURRENT_ROUND} exited abnormally (interrupted)."
  fi
}

print_failure_summary() {
  # 这个函数用于“失败退出”时打印摘要（和 interruption 类似，但强调失败）。
  echo ""
  echo "SUMMARY:"
  if (( ${#COMPLETED_ROUNDS[@]} > 0 )); then
    echo "  Experiments completed: ${COMPLETED_ROUNDS[*]}."
  else
    echo "  Experiments completed: none."
  fi
  if [[ -n "${FAILED_ROUND:-}" ]]; then
    echo "  Experiment ${FAILED_ROUND} failed. Stopping."
  fi
}

cleanup() {
  # cleanup 是 trap 的处理函数：收到 INT/TERM 时执行。
  # 目标：
  # 1) 尝试优雅中断当前所有后台训练进程
  # 2) 尝试清理可能残留的 launcher 子进程
  # 3) 可选发送“中断通知邮件”
  # 4) 打印 summary 并以 130 退出（130 通常表示被 SIGINT 中断）

  for pid in "${PIDS[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
  # 先发 SIGINT（相当于 Ctrl+C），给程序机会做优雅退出/保存。

  sleep 1
  # 留一点时间让 Python/训练框架处理 SIGINT。

  for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  # 再发 SIGTERM：更强硬的“请退出”。

  sleep 1

  for pid in "${PIDS[@]}"; do kill -KILL "$pid" 2>/dev/null || true; done
  # 最后 SIGKILL：强制杀死（不可捕获、不可忽略）。
  # 风险：可能导致日志/ckpt 未写完；但用于保证脚本停止，不留僵尸任务。

  if [[ -n "${EXP_ROOT:-}" ]]; then
    pkill -f -- "${LAUNCHER_PY} --cfg ${EXP_ROOT}/" 2>/dev/null || true
  fi
  # 额外兜底：杀掉“可能不是当前 PIDS 记录到的进程”，但命令行匹配到 launcher+cfg 路径的进程。
  # - pkill -f：匹配完整命令行（不是仅进程名）。
  # - 注意：这可能误杀同机器上其它用户/其它会话启动的相似命令（取决于命令行是否相同）。
  # - `--`：防止后面的 pattern 被当作选项解析。

  # Best-effort email notification for interruption (controlled by env, default on; does not kill anything, only notifies once)
  _email_interrupt="${SWANLAB_EMAIL_ON_INTERRUPT:-1}"
  # SWANLAB_EMAIL_ON_INTERRUPT：是否在中断时发邮件，默认 1（开启）。
  # 使用 :-1 防止 set -u 因未定义而报错。

  if [[ "${_email_interrupt}" == "1" ]] && command -v python >/dev/null 2>&1; then
    # 只有在“开启邮件通知”且系统存在 python 命令时，才尝试发邮件。
    (
      cd "$(dirname "$0")/.." || true
      # 子 shell 执行，避免影响主 shell 当前工作目录。
      # "$(dirname "$0")"：脚本所在目录；再 /.. ：一般是项目根目录（按你的仓库结构推断）。
      # `|| true`：即使 cd 失败也不让脚本因 set -e 退出（best-effort）。

      python -m scripts.utils.email_notify \
        --event INTERRUPTED \
        --group "suite=${SELECT_SUITE} round=${CURRENT_ROUND} data=${DATA}" \
        --yaml "${SWANLAB_EMAIL_YAML:-}" >/dev/null 2>&1 || true
      # 通过 python 模块发送通知：
      # --event INTERRUPTED：事件类型
      # --group：把 suite/round/data 拼成一条 group 标识，便于邮件聚类/过滤
      # --yaml：邮件配置（可能包含 SMTP/收件人等）；允许为空
      # >/dev/null 2>&1：完全静默（避免干扰主日志）
      # || true：即便邮件发送失败也不影响 cleanup 的正常结束
    )
  fi

  print_interruption_summary
  # 打印本次中断摘要（已完成 rounds + 当前 round）

  exit 130
  # 退出码 130：惯例表示脚本被 SIGINT 中断（128 + 2）。
}
trap cleanup INT TERM
# trap：捕获信号并执行 cleanup。
# INT：通常来自 Ctrl+C
# TERM：常见的“请求退出”信号（例如 kill 默认发送 TERM）

ROUND="${1:-1}"        # first arg kept for backward compat/docs; may be number or 'all'
# ROUND：默认取第一个参数，否则默认 1。
# 注：后面存在 suite 解析（E1..E10），可能 shift 参数，所以这里只是早期默认值/兼容用途。

SEED="${SEED:-42}"     # informational only (NOT used for training)
# SEED：只是“信息性变量”（注释说不用于训练），默认 42。
# 你后面实际训练用的是 FORCE_SEED -> HP_SEED。

FORCE_SEED=87         # actual seed used in training (HP_SEED). Ignore any seed elsewhere. FORCE_SEED=127 ...
# FORCE_SEED：真正用于训练的随机种子（通过环境变量 HP_SEED 注入到 launcher）。
# 注释强调：忽略其它 seed（包括 YAML/文件名里写的 seed）。
# 这里列了几个候选 seed：13 21 42 87 127（你说明 127 能全局控制随机性）。

DATA="${DATA:-glue-tvt_cola}"  # injected dataset name (can override via env: DATA=AAA)
# DATA：数据集名称，默认 glue-tvt_cola，可通过环境变量 DATA 覆盖。
# 这个值后面会被注入到临时 YAML（data: <DATA>）。

# Remote workspace expected by train_gla_only.py
PEFT_ROOT="/home/user/mzs_h/code/zh-LAT-peft/mamba-peft"
cd "$PEFT_ROOT"
# 切到项目工作目录。train_gla_only.py 以及相对路径依赖可能都假定在此目录下运行。
# 如果路径不对，脚本会因 set -e（cd 失败返回非 0）直接退出。

# Env mirrors/caches (same as original)
export HF_ENDPOINT="https://hf-mirror.com"
# Hugging Face 镜像站点：用于国内/受限网络加速（避免直接访问 huggingface.co）。

export HF_HOME="/home/user/mzs_h/data/hf_cache"
# HF_HOME：HF 的缓存根目录。

export HF_HUB_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export HF_EVALUATE_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
# 将 hub/datasets/evaluate/transformers 缓存统一指向同一个目录，方便管理、避免重复下载。

export GLUE_METRIC_DIR="/home/user/mzs_h/data/hf_cache/eval_metrics/glue"
# GLUE 指标脚本或缓存位置（取决于你的评估实现方式）。

export HF_HUB_ENABLE_HF_TRANSFER=1
# 启用 hf_transfer 加速（需要环境支持）；对下载速度可能有帮助。

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# 禁用 NCCL P2P 和 IB：
# - 可能是为了规避某些集群/驱动/拓扑下的 NCCL hang 或性能不稳定。
# - 代价：多卡通信可能变慢，但此脚本是“单卡多进程并发”，所以影响可能有限。

export WANDB_MODE=disabled
export WANDB_DISABLED=true
# 禁用 wandb（避免联网、避免日志污染、避免产生 wandb 目录）。

rm -rf ~/.config/wandb ~/.triton ~/.cache/torch_extensions || true
# 清理一些常见缓存/编译产物：
# - wandb config
# - triton cache（可能和 kernel 编译有关）
# - torch_extensions（C++/CUDA 扩展编译缓存）
# `|| true`：即使删除失败（无权限/不存在）也不退出。

# ---- Echo invocation & key env overrides (for reproducible logs) ----
echo "CMD: $0 $*"
# 打印实际执行命令（脚本名 + 参数），便于复现。

echo "ENV_OVERRIDES:"
# 打印关键环境变量覆盖情况（只打印非空的）。
for _k in \
  GPU_IDS \
  GPU_PLAN \
  CUDA_VISIBLE_DEVICES \
  DATA \
  HP_VAL_SPLIT \
  SPIDER_LOCAL_DIR \
  NLTK_DATA \
    GLA_FORCE_LEFT_PAD GLA_USE_MAX_NEW_TOKENS GLA_VERBOSE \
  EVAL_GEN EVAL_GEN_MAX_LENGTH EVAL_GEN_MIN_LENGTH EVAL_GEN_NUM_BEAMS \
  PYTORCH_CUDA_ALLOC_CONF TOKENIZERS_PARALLELISM OMP_NUM_THREADS MKL_NUM_THREADS \
  GRADIENT_CHECKPOINTING \
  LOGITS_TO_KEEP \
  NUM_DATA_WORKERS \
  FORCE_SEED \
  SEED \
  HP_DATA HP_BATCH_SIZE HP_LR HP_EPOCHS HP_EVAL_BATCH_SIZE HP_PREC HP_SEED \
  HP_PEFT_R HP_PEFT_ALPHA HP_PEFT_DROPOUT HP_INIT HP_PISSA_FAST \
  HP_MAX_STEPS HP_EVAL_STEPS HP_SAVE_STEPS HP_LOGGING_STEPS \
  HP_LORAGA_BATCH_SIZE HP_LORAGA_STEPS HP_LORAGA_LAYERWISE HP_LORAGA_STABLE_C \
  LR_SCHEDULER_TYPE LR_WARMUP_STEPS LR_WARMUP_RATIO \
  GLA_LAUNCH_STAGGER_MINUTES
do
  v="${!_k-}"
  # ${!_k-}：间接展开（取名为 _k 的变量的值）；若未定义则为空（避免 set -u 报错）。

  if [[ -n "${v:-}" ]]; then
    echo "  ${_k}=${v}"
  fi
done

if command -v env >/dev/null 2>&1; then
  echo "HP_* (all):"; env | grep -E '^HP_' | sort || true
  # 打印当前所有 HP_ 开头的环境变量，排序后展示。
  # `|| true`：如果 grep 没匹配到会返回 1，避免 set -e 退出。
fi

# ---------------------------------------------------------------------------
# Paths: split root and subdirs (YAML & JSON)
# ---------------------------------------------------------------------------
EXP_ROOT="${EXP_ROOT:-cfg/my_lora_exp}"     # root for this experiment family
# 实验族根目录：默认 cfg/my_lora_exp，可通过环境变量 EXP_ROOT 覆盖。

CFG_DIR="${CFG_DIR:-${EXP_ROOT}/yaml}"      # YAML configs
# YAML 配置目录：默认 <EXP_ROOT>/yaml

PEFT_DIR="${PEFT_DIR:-${EXP_ROOT}/peft}"    # JSON assets (train.py uses as needed)
# PEFT 相关 JSON 资产目录：默认 <EXP_ROOT>/peft
# 注意：本脚本只打印它，不直接使用；实际由 train_gla_only.py / cfg 引用。

if [[ ! -d "$CFG_DIR" ]]; then
  echo "Config directory not found: $CFG_DIR" >&2
  exit 1
fi
# 如果 YAML 配置目录不存在，直接失败退出（此处必须保证配置存在）。

# -------- GPU detection (must be 7 or exit) --------
# 注：你这行注释写“must be 7”，但实际代码并未强制必须 7 张卡，
#     而是 “NUM_GPUS < 1 就退出”。所以这里的注释可能是旧需求残留。

parse_gpu_list() {
  # Normalize a space- or comma-separated list into DETECTED_GPUS array
  local s="${1:-}"
  # 参数默认空，避免 set -u

  s="${s//,/ }"
  # 把逗号替换为空格：统一分隔符（支持 "0,1,2" 或 "0 1 2"）。

  DETECTED_GPUS=()
  # 清空目标数组，保证每次调用不叠加。

  for tok in $s; do
    # 注意：这里是“基于 IFS 的单词分割”；如果 s 中有多余空格也没问题。
    [[ -n "$tok" ]] && DETECTED_GPUS+=("$tok")
    # 非空 token 才加入数组（避免连续空格产生空项）。
  done
}

detect_gpus() {
  # 优先级：
  # 1) GPU_IDS（脚本自定义的显式指定）
  # 2) CUDA_VISIBLE_DEVICES（CUDA 标准变量）
  # 3) nvidia-smi 自动检测 NVIDIA GPU 数量
  # 4) rocm-smi 自动检测 AMD GPU 数量
  # 否则报错退出。

  if [[ -n "${GPU_IDS:-}" ]]; then
    parse_gpu_list "$GPU_IDS"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    parse_gpu_list "$CUDA_VISIBLE_DEVICES"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    local cnt
    cnt="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    # nvidia-smi -L：列出每块 GPU；wc -l 统计行数即 GPU 数量。
    # tr -d ' '：去掉空格，确保是纯数字。

    DETECTED_GPUS=()
    for ((i=0;i<cnt;i++)); do DETECTED_GPUS+=("$i"); done
    # 默认生成 0..cnt-1 的 GPU id 列表。
    # 注意：这是假设 GPU id 与 nvidia-smi 列表顺序一致（通常成立）。

  elif command -v rocm-smi >/dev/null 2>&1; then
    # Fallback for AMD: count "GPU" lines
    local cnt
    cnt="$(rocm-smi --showid 2>/dev/null | grep -E 'GPU\[|GPU' | wc -l | tr -d ' ')"
    # rocm-smi 输出中匹配 GPU 行并计数（比较粗略，但可用）。

    DETECTED_GPUS=()
    for ((i=0;i<cnt;i++)); do DETECTED_GPUS+=("$i"); done

  else
    echo "ERROR: Could not detect GPUs (no GPU_IDS/CUDA_VISIBLE_DEVICES and nvidia-smi/rocm-smi missing)." >&2
    exit 1
  fi
}

detect_gpus
# 执行 GPU 检测，填充 DETECTED_GPUS。

NUM_GPUS="${#DETECTED_GPUS[@]}"
# GPU 数量 = 数组长度。

if (( NUM_GPUS < 1 )); then
  echo "ERROR: No GPUs detected (after considering GPU_IDS/CUDA_VISIBLE_DEVICES)." >&2
  exit 1
fi
# 没检测到任何 GPU：直接退出。
# 这比“后面跑起来才发现没有 CUDA 设备”更可控。

# -------------------------
# Per-GPU concurrency plan
# -------------------------
# GPU_PLAN: comma/space separated integers per detected GPU, e.g. "3,3,3,3,0,3,3".
# - If unset: default to 1 slot per detected GPU (previous behavior)
# - If single integer provided: broadcast to all GPUs
# - If length matches NUM_GPUS: use as-is
# - Otherwise: error
GPU_PLAN_STR="${GPU_PLAN:-}"
# GPU_PLAN_STR：从环境变量 GPU_PLAN 读取并默认空。
# GPU_PLAN 的语义：每张 GPU 上允许并发跑几个 job（slot 数）。
# 例如：GPU_PLAN="3,3,3,3,0,3,3" 表示第 5 张卡（索引 4）禁用，其余每张卡并发 3 个任务。

declare -a GPU_PLAN_ARR=()
# 存放解析后的 per-GPU slot 数组。

if [[ -z "$GPU_PLAN_STR" ]]; then
  # 未指定 GPU_PLAN：每张检测到的 GPU 默认 1 个 slot（即一张卡只跑一个任务）。
  for _ in "${DETECTED_GPUS[@]}"; do GPU_PLAN_ARR+=(1); done
else
  # normalize separators to spaces
  GPU_PLAN_STR="${GPU_PLAN_STR//,/ }"
  # 把逗号换空格，方便 read -a 解析。

  read -r -a GPU_PLAN_ARR <<<"$GPU_PLAN_STR"
  # 将字符串拆成数组。

  if (( ${#GPU_PLAN_ARR[@]} == 1 )); then
    # 如果只给了一个整数：广播到所有 GPU（每张卡同样并发数）
    val="${GPU_PLAN_ARR[0]}"; GPU_PLAN_ARR=()
    for _ in "${DETECTED_GPUS[@]}"; do GPU_PLAN_ARR+=("$val"); done

  elif (( ${#GPU_PLAN_ARR[@]} != NUM_GPUS )); then
    # 如果既不是 1 个值，也不是 NUM_GPUS 个值：报错（避免配置不一致导致错绑 GPU）
    echo "ERROR: GPU_PLAN length (${#GPU_PLAN_ARR[@]}) must be 1 or equal to number of detected GPUs (${NUM_GPUS})." >&2
    echo " - DETECTED_GPUS = ${DETECTED_GPUS[*]}" >&2
    echo " - GPU_PLAN      = ${GPU_PLAN_STR}" >&2
    exit 1
  fi
fi

# Build GPU_SLOTS by repeating each GPU id according to its concurrency
declare -a GPU_SLOTS=()
# GPU_SLOTS：一个“扁平化 slot 列表”，例如：
# DETECTED_GPUS = [0 1 2], GPU_PLAN_ARR=[2 1 3]
# -> GPU_SLOTS = [0 0 1 2 2 2]
# 后续 job i 会绑定到 slot_index = i % N_SLOTS，再取 GPU_SLOTS[slot_index] 得到 GPU。

for i in "${!DETECTED_GPUS[@]}"; do
  gpu="${DETECTED_GPUS[$i]}"
  cnt="${GPU_PLAN_ARR[$i]}"

  # treat non-positive as zero
  if [[ -z "$cnt" || "$cnt" -le 0 ]]; then cnt=0; fi
  # cnt<=0 的 GPU 视为禁用（不提供 slot）。

  for ((j=0;j<cnt;j++)); do GPU_SLOTS+=("$gpu"); done
  # 将 gpu 重复 cnt 次加入 GPU_SLOTS。
done

N_SLOTS="${#GPU_SLOTS[@]}"
# 实际并发 slot 总数（跨所有 GPU 累计）。

if (( N_SLOTS < 1 )); then
  echo "ERROR: Effective parallel slots is zero (GPU_PLAN all zeros?)." >&2
  echo " - DETECTED_GPUS = ${DETECTED_GPUS[*]}" >&2
  echo " - GPU_PLAN      = ${GPU_PLAN_ARR[*]}" >&2
  exit 1
fi
# 如果所有 GPU 都被配置为 0 slot：无法运行任何任务，直接退出。

# =========================
# Suite selector (E1..E10)
# =========================
SELECT_SUITE="ALL"
# suite 选择器：允许你用参数 E1/E2/... 来选择某个 ROUND_E* 列表作为 Round_all。
# 默认 ALL：表示使用全部 suites 或者已预先定义的 Round_all。

append_suite_into_master() {
  # 将某个 suite 数组（例如 ROUND_E1）追加到 Round_all 中。
  # 这里采用 eval 读取变量名对应的数组内容，属于“动态变量名”技巧。

  local var="$1"
  # var 是变量名字符串，例如 "ROUND_E1"。

  if eval "[[ -v ${var} && \${#${var}[@]} -gt 0 ]]"; then
    # [[ -v VAR ]]：bash 4.2+，判断变量是否已定义（数组也可）。
    # 再判断数组长度 > 0。
    # 用 eval 是因为 var 是字符串，需要展开成真实变量名。

    local tmp=( $(eval "printf '%q ' \"\${${var}[@]}\"") )
    # 这一行属于“历史遗留/冗余”风格：
    # - printf '%q' 会对每个元素进行 shell 转义
    # - 再用 $(...) 进行一次命令替换
    # 但下一行又重新 read -a 覆盖 tmp，因此这里的 tmp 初始化其实不关键。

    if ((${#tmp[@]} > 0)); then
      read -r -a tmp <<<"$(eval "printf '%s ' \"\${${var}[@]}\"")"
      # 将 suite 数组内容以空格拼接为字符串再读回数组 tmp。
      # 注意：如果元素中包含空格，这种方式可能破坏边界（但一般 config 文件名不会有空格）。

      Round_all+=("${tmp[@]}")
      # 追加到 Round_all（全局数组，后续会按 round 切片）。
    fi
  fi
}

if [[ "${1:-}" =~ ^([Ee][0-9]+)$ ]]; then
  # 如果第一个参数形如 E1 / e2 / E10：表示选择某个 suite。
  suite="${BASH_REMATCH[1]}"
  suite="${suite^^}"
  # ${var^^}：转大写，统一为 E1..E10。

  SELECT_SUITE="$suite"
  shift
  # shift：移除第一个参数（suite），后续参数用于 round（如 all 或具体数字）。

  varname="ROUND_${suite}"
  # 目标数组变量名：例如 ROUND_E1。

  if ! eval "[[ -v ${varname} && \${#${varname}[@]} -gt 0 ]]"; then
    echo "ERROR: Suite '${suite}' is not defined or empty. Please define ${varname}=() with configs." >&2
    exit 1
  fi
  # suite 未定义或为空：直接报错，避免误跑“空任务”。

  Round_all=()
  # 重建 Round_all，仅包含指定 suite。

  append_suite_into_master "${varname}"

  ROUND="${1:-all}"
  # suite 之后的第一个参数：如果有则作为 ROUND，否则默认 all（跑该 suite 全部 rounds）。

else
  # 没有显式 suite：使用 Round_all（如果已经预定义），否则从 E1..E10 汇总。

  if (( ${#Round_all[@]} == 0 )); then
    # 如果 Round_all 为空：自动把 ROUND_E1..ROUND_E10 追加进去（只要它们存在且非空）。

    Round_all=()
    for i in {1..10}; do
      append_suite_into_master "ROUND_E${i}"
    done

    if (( ${#Round_all[@]} == 0 )); then
      echo "ERROR: No configs found. Either populate Round_all=() manually, or pass a suite like 'E1', 'E2', ...," >&2
      echo "       or define the corresponding ROUND_E* arrays." >&2
      exit 1
    fi
  fi
fi

# -------- Dynamic round slicing from Round_all --------
# 核心思想：
# - Round_all 是所有配置文件名列表（相对 CFG_DIR）
# - 计算 N_SLOTS（并发 slot 总数）
# - 将 Round_all 按每轮最多 N_SLOTS 个配置进行切片
# - round 数量 N_ROUNDS = ceil(TOTAL_CFGS / N_SLOTS)

TOTAL_CFGS="${#Round_all[@]}"
N_ROUNDS=$(( (TOTAL_CFGS + N_SLOTS - 1) / N_SLOTS ))
# ceil 除法： (a+b-1)/b

defined_rounds_str() {
  # 输出合法 round 编号列表，例如 "1 2 3 "
  local out=""
  for ((r=1;r<=N_ROUNDS;r++)); do out+="${r} "; done
  printf "%s" "$out"
}

canonical_cfg_path() {
  # 将 entry（相对配置文件名）解析为 CFG_DIR 下的绝对路径，并检查是否存在。
  # 注意：这里“绝对路径”其实是 “CFG_DIR 前缀 + entry”，并不做 realpath。
  # 返回：
  # - 文件存在：打印路径并 return 0
  # - 文件不存在：打印“期望路径”并 return 1（便于上层收集 missing 列表）

  local entry="$1"
  local path="${CFG_DIR}/${entry}"

  if [[ -f "$path" ]]; then
    printf '%s\n' "$path"; return 0
  else
    printf '%s\n' "$path"; return 1
  fi
}

declare -a SELECT_SET=()
# SELECT_SET：某一轮 round 切片出来的 config 列表（相对路径名）。

get_round_configs() {
  # 给定 round 编号 r（1-based），计算它应包含 Round_all 的哪一段配置。
  # 输出：填充全局数组 SELECT_SET。
  # 返回值：SELECT_SET 非空 -> true；否则 false。

  local r="$1"
  if (( r < 1 || r > N_ROUNDS )); then
    return 1
  fi

  local start=$(( (r-1)*N_SLOTS ))
  local end=$(( r*N_SLOTS ))
  if (( end > TOTAL_CFGS )); then end="$TOTAL_CFGS"; fi

  SELECT_SET=()
  local i
  for ((i=start;i<end;i++)); do
    SELECT_SET+=("${Round_all[i]}")
  done

  (( ${#SELECT_SET[@]} > 0 ))
  # bash 中：(( expr )) 的退出码：expr!=0 为 0（true），expr==0 为 1（false）
}

make_tmp_cfg_with_data() {
  # 重要函数：生成一个“临时 YAML 配置副本”，并在末尾注入：
  # - data: <DATA>
  # - num_data_workers: <NUM_DATA_WORKERS or 8>
  # - gradient_checkpointing: true（若显式开启）
  # - logits_to_keep: <LOGITS_TO_KEEP>（若提供）
  # 目的：不修改原始 YAML，确保实验配置可复用/可对比。

  local src="$1"; local outdir="$2"

  local base
  base="$(basename "$src")"
  # 取文件名（去掉路径），用于构造输出文件名。

  local name ext
  name="${base%.*}"; ext="${base##*.}"
  # name：去扩展名，ext：扩展名（例如 yaml/yml）

  local out
  out="$outdir/${name}.${ext}"
  # 默认输出路径：<tmpdir>/<name>.<ext>

  # Ensure unique filename if duplicates exist in the same round
  if [[ -e "$out" ]]; then
    # 同一轮里可能出现同名配置（或不同路径但 basename 相同），避免覆盖。
    local k=1
    while :; do
      local cand="$outdir/${name}__rep${k}.${ext}"
      if [[ ! -e "$cand" ]]; then out="$cand"; break; fi
      k=$((k+1))
    done
  fi

  cp "$src" "$out"
  # 复制原始 YAML 到临时文件。

  printf '\n# injected by gla_round_clean.sh\ndata: %s\n' "$DATA" >>"$out"
  # 在末尾追加注入字段：
  # - 先换行
  # - 写注释行，标记这是脚本注入
  # - 写 data: <DATA>
  # YAML 的“后写覆盖”是否生效取决于你的解析逻辑：
  # - 如果 train_gla_only.py 用的是常规 YAML loader，同名 key 后写通常会覆盖前写（但严格 YAML 语义是后者覆盖前者）。
  # - 如果代码自定义合并策略，则可能不同；但一般都以最后出现为准。

  # Highest priority num_data_workers injection (default 8 if unset)
  local ndw
  ndw="${NUM_DATA_WORKERS:-8}"
  printf 'num_data_workers: %s\n' "$ndw" >>"$out"
  # 同理注入 num_data_workers，默认 8。

  # Optional gradient checkpointing (enable only when explicitly set truthy)
  if [[ -n "${GRADIENT_CHECKPOINTING:-}" ]]; then
    # 只有当环境变量存在且非空才进一步判断真假值。
    case "${GRADIENT_CHECKPOINTING,,}" in
      # ${var,,}：转小写，允许 TRUE/True/true 等形式。
      1|true|yes|on)
        printf 'gradient_checkpointing: true\n' >>"$out"
        ;;
    esac
  fi

  # Optional logits_to_keep (only if provided)
  if [[ -n "${LOGITS_TO_KEEP:-}" ]]; then
    printf 'logits_to_keep: %s\n' "$out"
  fi

  printf '%s\n' "$out"
  # 输出临时 YAML 路径给调用方。
}

run_round () {
  # 运行某一轮 round：
  # - 根据 round 切片取到本轮 configs
  # - 检查文件存在
  # - 为每个 config 选择一个 GPU slot
  # - 生成临时注入 YAML
  # - 启动后台 python 任务并记录 PID
  # - wait 所有 PID
  # - 任一失败则 round 失败（返回 1）

  local r="$1"

  if ! get_round_configs "$r"; then
    echo "Round ${r} is empty or out of range. Valid rounds: $(defined_rounds_str)" >&2
    return 1
  fi

  # Resolve to absolute paths and verify existence
  local missing=()
  local -a RESOLVED_CFGS=()
  local resolved=""

  for f in "${SELECT_SET[@]}"; do
    if resolved="$(canonical_cfg_path "$f")"; then
      RESOLVED_CFGS+=("$resolved")
    else
      missing+=("$resolved")
    fi
  done

  if (( ${#missing[@]} > 0 )); then
    echo "Missing configs (expected under $CFG_DIR):" >&2
    printf '  %s\n' "${missing[@]}" >&2
    return 1
  fi
  # 有缺失配置：直接让此 round 失败（但不会 kill 其它 round，因为还没启动任务）。

  local num_jobs="${#RESOLVED_CFGS[@]}"
  echo "=== Starting Round ${r} (${num_jobs} jobs; FORCE_SEED=${FORCE_SEED}; NUM_GPUS=${NUM_GPUS}; N_SLOTS=${N_SLOTS}) ==="
  # 打印 round 的核心参数：job 数量、强制随机种子、GPU 数量、slot 数量。

  echo "SUITE   = ${SELECT_SUITE}"
  echo "CFG_DIR = $CFG_DIR"
  echo "PEFT_DIR= $PEFT_DIR"
  echo "GPUs    = ${DETECTED_GPUS[*]}"
  echo "PLAN    = ${GPU_PLAN_ARR[*]}  (GPU->slots)"
  echo "SLOTS   = ${GPU_SLOTS[*]}     (flattened)"
  echo "DATA    = ${DATA}"
  echo "SPIDER_LOCAL_DIR = ${SPIDER_LOCAL_DIR:-}"
  echo "NLTK_DATA        = ${NLTK_DATA:-}"
  # 额外环境相关路径也输出，便于日志定位（即使本脚本不使用也可能被训练脚本使用）。

  # Round timing (start)
  local __round_start_epoch
  __round_start_epoch="$(date +%s)"
  local __round_start_iso
  __round_start_iso="$(date +%F_%T)"
  echo "[${__round_start_iso}] ROUND=${r} START"
  # 记录开始时间：epoch 用于计算耗时，iso 用于人读日志。

  # Choose GPU per job from detected list
  PIDS=()
  # 清空 PIDS，确保本轮只 wait/kill 本轮任务。

  local i

  # Make a temp dir for this round's YAML copies with injected data
  local TMP_CFG_DIR
  TMP_CFG_DIR="$(mktemp -d /tmp/gla_data_XXXXXX)"
  # mktemp -d：创建临时目录。round 结束后会 rm -rf。
  # 目录位于 /tmp：通常在节点本地盘，写入快；但重启会丢失（这是预期）。

  local _stagger_min="${GLA_LAUNCH_STAGGER_MINUTES:-0}"
  # 任务启动“错峰”参数（分钟）。用于避免同时启动导致 IO/编译/显存抢占尖峰。

  # normalize to integer minutes if possible; non-numeric -> 0
  if ! [[ "${_stagger_min}" =~ ^[0-9]+$ ]]; then
    _stagger_min=0
  fi
  # 只允许纯数字；否则置 0（防止传入 "1.5" 或 "abc" 破坏算术）。

  for i in "${!RESOLVED_CFGS[@]}"; do
    local CFG="${RESOLVED_CFGS[$i]}"

    # choose slot by index cycling when fewer jobs than slots
    local slot_index=$(( i % N_SLOTS ))
    # 取模绑定 slot：
    # - i=0 -> slot 0
    # - i=1 -> slot 1
    # ...
    # - i=N_SLOTS -> slot 0（循环）
    # 意味着：如果本 round 的 job 数 > slot 数，则会在同一张 GPU 上并发多 job（由 GPU_PLAN 控制 slot 数）。
    # 重要：这不是“动态负载均衡”，只是静态轮询分配；某些 job 更慢会导致尾部等待。

    local GPU="${GPU_SLOTS[$slot_index]}"
    # GPU_SLOTS 里存的是 GPU id（字符串），直接用于 CUDA_VISIBLE_DEVICES。

    local CFG_INJ
    CFG_INJ="$(make_tmp_cfg_with_data "$CFG" "$TMP_CFG_DIR")"
    # 为这个 job 生成注入后的临时 YAML。

    echo "[GPU ${GPU}] ${CFG_INJ}  (HP_SEED=${FORCE_SEED}; data=${DATA}; ignoring seed in name/YAML)"
    # 打印 job 绑定信息：
    # - GPU
    # - 临时 cfg 路径
    # - 强制种子/数据集说明

    if (( _stagger_min > 0 )); then
      local _delay_sec=$(( _stagger_min * 60 * i ))
      # 第 i 个 job 延迟 i * stagger 分钟（线性递增错峰）
      # i=0 不延迟，i=1 延迟 1 倍，i=2 延迟 2 倍...

      if (( _delay_sec > 0 )); then
        echo "[GPU ${GPU}] delaying launch by ${_delay_sec}s (stagger ${_stagger_min} min per job)"
        sleep "${_delay_sec}"
      fi
    fi

    HP_SEED=${FORCE_SEED} CUDA_VISIBLE_DEVICES="$GPU" \
      python "$LAUNCHER_PY" --cfg "$CFG_INJ" --overwrite &
    # 在后台启动训练：
    # - HP_SEED 环境变量注入：训练脚本应读取它作为随机种子。
    # - CUDA_VISIBLE_DEVICES="$GPU"：将当前进程“只看到”指定 GPU。
    #   注意：这里传的是单个 id 字符串（比如 "3"），表示只暴露一张卡。
    # - --cfg "$CFG_INJ"：使用临时注入 YAML。
    # - --overwrite：让训练脚本覆盖已有输出（具体语义看 train_gla_only.py）。
    # - 最后的 &：后台执行，立刻返回以便启动下一个 job。

    PIDS+=("$!")
    # $!：最近一个后台命令的 PID，记录下来后续 wait/kill 用。
  done

  local any_failed=0
  # 标记本轮是否出现失败（任一子进程失败即失败）。

  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      # wait：等待该 PID 结束，并返回其退出码。
      # 如果退出码非 0，则 if 条件成立。
      any_failed=1
    fi
  done
  # 这里是“逐个 wait”，保证所有任务都 wait 完（不会因某个失败就立刻中断 wait 循环）。
  # 结果：即使一个 job 失败，仍会等其它 job 完成后再判定 round 失败（这是你的策略选择）。

  # cleanup temp dir
  rm -rf "$TMP_CFG_DIR" || true
  # 清理临时 YAML 目录。
  # `|| true`：避免 rm 失败触发 set -e（例如文件被占用/权限异常）。

  # Round timing (end)
  local __round_end_epoch
  __round_end_epoch="$(date +%s)"
  local __round_end_iso
  __round_end_iso="$(date +%F_%T)"

  local __round_elapsed
  __round_elapsed=$(( __round_end_epoch - __round_start_epoch ))

  local __round_h
  local __round_m
  local __round_s
  __round_h=$(( __round_elapsed / 3600 ))
  __round_m=$(( (__round_elapsed % 3600) / 60 ))
  __round_s=$(( __round_elapsed % 60 ))

  printf '[%s] ROUND=%s END elapsed=%02d:%02d:%02d (%ds)\n' "${__round_end_iso}" "${r}" "${__round_h}" "${__round_m}" "${__round_s}" "${__round_elapsed}"
  # 打印 round 结束与耗时（同时给出格式化的 hh:mm:ss 与总秒数）。

  if (( any_failed )); then
    return 1
  fi
  # 本轮任一 job 失败：round 失败（交由外层逻辑处理 kill/summary/exit）。

  echo "ROUND=${r} finished (all ran with HP_SEED=${FORCE_SEED})."
  return 0
}

# -------------------------
# Build the run queue
# -------------------------
# 目标：根据命令行参数构建 RUN_QUEUE（要跑哪些 round）
# 规则：
# - 如果没有额外参数（$# == 0）：
#   - ROUND=all -> 1..N_ROUNDS
#   - 否则 -> 只跑 ROUND（默认 1）
# - 如果有参数：逐个解析
#   - 遇到 all -> 展开 1..N_ROUNDS
#   - 遇到数字 -> 加入队列

if (( $# == 0 )); then
  if [[ "${ROUND:-}" == "all" ]]; then
    for ((r=1;r<=N_ROUNDS;r++)); do RUN_QUEUE+=("$r"); done
  else
    RUN_QUEUE+=("${ROUND:-1}")
  fi
else
  for arg in "$@"; do
    if [[ "$arg" == "all" ]]; then
      for ((r=1;r<=N_ROUNDS;r++)); do RUN_QUEUE+=("$r"); done
    else
      RUN_QUEUE+=("$arg")
    fi
  done
fi

# Validate run queue
for r in "${RUN_QUEUE[@]}"; do
  if ! [[ "$r" =~ ^[0-9]+$ ]] || (( r < 1 || r > N_ROUNDS )); then
    echo "Invalid round '$r'. Valid values: $(defined_rounds_str) or 'all'." >&2
    exit 1
  fi
done
# 校验：RUN_QUEUE 每个元素必须是数字，且范围在 [1, N_ROUNDS]。
# 这样可以在执行前尽早失败，避免跑一半才发现参数错。

# -------------------------
# Execute strictly in order
# -------------------------
# 严格串行执行每个 round：
# - 同一 round 内并发（按 slot）
# - round 之间串行（上一轮完全结束后才开始下一轮）
# 失败策略：
# - 任一 round 失败 -> 立即停止整体脚本
# - 在失败时强制杀掉当前 round 子进程 + 尝试 pkill launcher 残留

for r in "${RUN_QUEUE[@]}"; do
  CURRENT_ROUND="$r"
  # 标记当前 round（用于中断 summary / 邮件 group）

  if run_round "$r"; then
    # round 成功
    COMPLETED_ROUNDS+=("$r")
    CURRENT_ROUND=""
    # 清空 CURRENT_ROUND：表示当前不在异常状态下运行某轮
  else
    # round 失败
    FAILED_ROUND="$r"

    # 强制杀掉当前 round 已启动的子进程（即 PIDS）。
    # 注意：run_round 的策略是 wait 完所有 pid 后才返回失败，
    #       所以理论上此处 PIDS 对应的进程多半已经结束。
    #       但如果中间出现异常返回或 wait 未覆盖的子进程，这里仍有兜底价值。

    for pid in "${PIDS[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
    for pid in "${PIDS[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${PIDS[@]}"; do kill -KILL "$pid" 2>/dev/null || true; done

    if [[ -n "${EXP_ROOT:-}" ]]; then
      pkill -f -- "${LAUNCHER_PY} --cfg ${EXP_ROOT}/" 2>/dev/null || true
    fi
    # 与 cleanup 中一致的 pkill 兜底逻辑。

    print_failure_summary
    exit 1
  fi
done

exit 0
# 全部 rounds 成功：退出 0。