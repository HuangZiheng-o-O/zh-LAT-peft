#!/usr/bin/env bash
# 监控使用 GPU 的 train_gla_only.py 进程是否疑似被swanlab挂住
# 不看运行时长，只看一个时间窗口内 GPU 是否一直 0 利用率 + IO 是否几乎不动

set -u

TRAIN_PATTERN="${TRAIN_PATTERN:-train_gla_only.py}"
SAMPLE_TOTAL="${SAMPLE_TOTAL:-30}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-5}"
SAMPLES=$(( SAMPLE_TOTAL / SAMPLE_INTERVAL ))
[ "$SAMPLES" -lt 1 ] && SAMPLES=1
MIN_VRAM_MIB="${MIN_VRAM_MIB:-1000}"
IO_DELTA_THRESHOLD="${IO_DELTA_THRESHOLD:-4096}"

TMP_DIR="${TMP_DIR:-/tmp/hung_gpu_check}"
mkdir -p "$TMP_DIR"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[-] 缺少命令: $1" >&2
    exit 1
  fi
}

need_cmd nvidia-smi
need_cmd ps
need_cmd awk
need_cmd tr

# uuid -> index 映射
uuid_index_map=$(nvidia-smi --query-gpu=uuid,index --format=csv,noheader 2>/dev/null || true)
gpu_uuid_to_index() {
  local uuid="$1"
  echo "$uuid_index_map" | awk -F',' -v U="$uuid" '
    {
      gsub(/ /,"",$1); gsub(/ /,"",$2);
      if ($1==U) {print $2}
    }' | head -n1
}

# 收集占 GPU 的 python 训练进程
declare -A PID_TO_GPUS
declare -A PID_TO_MEM

apps_raw=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
          --format=csv,noheader,nounits 2>/dev/null || true)

while IFS=',' read -r gpu_uuid pid pname mem; do
  [ -z "$pid" ] && continue
  pid=$(echo "$pid" | tr -d ' ')
  pname=$(echo "$pname" | xargs)
  mem=$(echo "$mem" | tr -d ' ')
  if ! echo "$pname" | grep -qi "python"; then
    continue
  fi
  cmdline=$(tr '\0' '\n' </proc/"$pid"/cmdline 2>/dev/null | tr '\n' ' ')
  cmdline=$(echo "$cmdline" | sed 's/  */ /g;s/^ *//;s/ *$//')
  if ! echo "$cmdline" | grep -q "$TRAIN_PATTERN"; then
    continue
  fi
  gpu_uuid_clean=$(echo "$gpu_uuid" | tr -d ' ')
  gidx=$(gpu_uuid_to_index "$gpu_uuid_clean")
  [ -z "$gidx" ] && continue

  old="${PID_TO_GPUS[$pid]:-}"
  if [ -z "$old" ]; then
    PID_TO_GPUS["$pid"]="$gidx"
  else
    if ! echo ",$old," | grep -q ",$gidx,"; then
      PID_TO_GPUS["$pid"]="$old,$gidx"
    fi
  fi

  old_mem="${PID_TO_MEM[$pid]:-0}"
  if [ "$mem" -gt "$old_mem" ]; then
    PID_TO_MEM["$pid"]="$mem"
  fi
done <<< "$apps_raw"

if [ "${#PID_TO_GPUS[@]}" -eq 0 ]; then
  echo "[*] 没有使用 GPU 的 $TRAIN_PATTERN 进程。"
  exit 0
fi

# 第一次 IO 采样
declare -A PID_IO_R1
declare -A PID_IO_W1

for pid in "${!PID_TO_GPUS[@]}"; do
  if [ -r "/proc/$pid/io" ]; then
    read r1 w1 < <(
      awk '
        $1=="read_bytes:"  {r=$2}
        $1=="write_bytes:" {w=$2}
        END {if (r=="") r=0; if (w=="") w=0; print r, w}
      ' "/proc/$pid/io" 2>/dev/null
    )
  else
    r1=0; w1=0
  fi
  PID_IO_R1["$pid"]="$r1"
  PID_IO_W1["$pid"]="$w1"
done

# GPU 利用率采样
all_gpus=""
for pid in "${!PID_TO_GPUS[@]}"; do
  gstr="${PID_TO_GPUS[$pid]}"
  for g in $(echo "$gstr" | tr ',' ' '); do
    if ! echo " $all_gpus " | grep -q " $g "; then
      all_gpus="$all_gpus $g"
    fi
  done
done

declare -A GPU_EVER_ACTIVE
for g in $all_gpus; do GPU_EVER_ACTIVE["$g"]=0; done

for ((i=1; i<=SAMPLES; i++)); do
  util_raw=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true)
  if [ -n "$util_raw" ]; then
    while IFS=',' read -r idx util; do
      idx=$(echo "$idx" | tr -d ' ')
      util=$(echo "$util" | tr -d ' ')
      if echo " $all_gpus " | grep -q " $idx "; then
        if [ "$util" -gt 0 ]; then
          GPU_EVER_ACTIVE["$idx"]=1
        fi
      fi
    done <<< "$util_raw"
  fi
  [ "$i" -lt "$SAMPLES" ] && sleep "$SAMPLE_INTERVAL"
done

# 第二次 IO 采样
declare -A PID_IO_R2
declare -A PID_IO_W2

for pid in "${!PID_TO_GPUS[@]}"; do
  if [ -r "/proc/$pid/io" ]; then
    read r2 w2 < <(
      awk '
        $1=="read_bytes:"  {r=$2}
        $1=="write_bytes:" {w=$2}
        END {if (r=="") r=0; if (w=="") w=0; print r, w}
      ' "/proc/$pid/io" 2>/dev/null
    )
  else
    r2="${PID_IO_R1[$pid]:-0}"
    w2="${PID_IO_W1[$pid]:-0}"
  fi
  PID_IO_R2["$pid"]="$r2"
  PID_IO_W2["$pid"]="$w2"
done

# 输出结果
printf "%-8s %-6s %-6s %-8s %-22s %s\n" \
  "PID" "CPU" "VRAM" "GPU" "STATUS" "CMD"

for pid in "${!PID_TO_GPUS[@]}"; do
  [ -d "/proc/$pid" ] || continue
  cpu=$(ps -p "$pid" -o %cpu= 2>/dev/null | awk '{printf("%d",$1+0)}')
  [ -z "$cpu" ] && cpu=0
  mem="${PID_TO_MEM[$pid]:-0}"
  gstr="${PID_TO_GPUS[$pid]}"

  cmdline=$(tr '\0' '\n' </proc/"$pid"/cmdline 2>/dev/null | tr '\n' ' ')
  cmdline=$(echo "$cmdline" | sed 's/  */ /g;s/^ *//;s/ *$//')

  if [ "$mem" -lt "$MIN_VRAM_MIB" ]; then
    status="IGNORE_SMALL_VRAM"
    printf "%-8s %-6s %-6s %-8s %-22s %s\n" \
      "$pid" "${cpu}%" "${mem}MiB" "[$gstr]" "$status" "$cmdline"
    continue
  fi

  gpu_idle=1
  for g in $(echo "$gstr" | tr ',' ' '); do
    if [ "${GPU_EVER_ACTIVE[$g]:-0}" -eq 1 ]; then
      gpu_idle=0
      break
    fi
  done

  r1="${PID_IO_R1[$pid]:-0}"
  w1="${PID_IO_W1[$pid]:-0}"
  r2="${PID_IO_R2[$pid]:-0}"
  w2="${PID_IO_W2[$pid]:-0}"
  dr=$(( r2 - r1 )); [ "$dr" -lt 0 ] && dr=0
  dw=$(( w2 - w1 )); [ "$dw" -lt 0 ] && dw=0
  total_delta=$(( dr + dw ))

  io_state="ACTIVE"
  if [ "$total_delta" -le "$IO_DELTA_THRESHOLD" ]; then
    io_state="IDLE"
  fi

  status="OK"
  if [ "$gpu_idle" -eq 1 ] && [ "$io_state" = "IDLE" ]; then
    status="SUSPECT_HUNG_GPU_IDLE"
  elif [ "$gpu_idle" -eq 1 ]; then
    status="SUSPECT_GPU_IDLE"
  fi

  printf "%-8s %-6s %-6s %-8s %-22s %s\n" \
    "$pid" "${cpu}%" "${mem}MiB" "[$gstr]" "$status" "$cmdline"
done