#!/usr/bin/env bash
# move_models_spaceaware.sh
# 按剩余磁盘空间“聪明”搬运：
# - 同一文件系统：并行 mv（不占额外空间，几乎瞬间完成）
# - 跨文件系统：按文件大小从小到大一个个搬（每次先检查剩余空间>文件大小+预留）
# 配置：JOBS, RESERVE_GB, HASH, PRIORITY
# 建议：高 I/O 优先级、并行处理同FS部分；跨FS部分按剩余空间逐个搬
# chmod +x move_models.sh
# PRIORITY=-1 JOBS=16 RESERVE_GB=12 HASH=0 bash move_models.sh
set -euo pipefail

# ===== 可调参数 =====
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 8)}"   # 同一文件系统部分的并行度
RESERVE_GB="${RESERVE_GB:-10}"                   # 目标盘预留空间（避免打满）
HASH="${HASH:-0}"                                 # 是否计算 SHA256（跨FS部分，默认跳过以提速）
PRIORITY="${PRIORITY:-0}"                         # I/O 优先级：1=低，0=默认，-1=高
DEST="/home/user/mzs_h/uselessBigFiles"
MANIFEST="$DEST/moved_models_manifest.tsv"
LOCKFILE="$DEST/moved_models_manifest.lock"

mkdir -p "$DEST"
if [[ ! -f "$MANIFEST" ]]; then
  echo -e "experiment\tcheckpoint\tsrc_path\tnew_path\tbytes\tsha256" > "$MANIFEST"
fi
: > "$LOCKFILE"

# ===== 你的“精简后的列表”（已去除 checkpoint-17100）=====
CHECKPOINT_DIRS=(
  # E0_ZS（Top1: 17100）
  "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/weights/benchmark/glue/cola_gla-firstround/E0_ZS/checkpoint-15390"
  "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/weights/benchmark/glue/cola_gla-firstround/E0_ZS/checkpoint-1710"

  # E1_QKVO（Top1: 8550）→ 该实验原本有 5 个含 model.pt
  "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/weights/benchmark/glue/cola_gla-firstround/E1_QKVO/checkpoint-17100"
  "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/weights/benchmark/glue/cola_gla-firstround/E1_QKVO/checkpoint-3420"
  "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/weights/benchmark/glue/cola_gla-firstround/E1_QKVO/checkpoint-5130"
  "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/weights/benchmark/glue/cola_gla-firstround/E1_QKVO/checkpoint-1710"

  # E2_OMLP（Top1: 5130）
  "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/weights/benchmark/glue/cola_gla-firstround/E2_OMLP/checkpoint-17100"
  "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/weights/benchmark/glue/cola_gla-firstround/E2_OMLP/checkpoint-6840"
  "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/weights/benchmark/glue/cola_gla-firstround/E2_OMLP/checkpoint-8550"

  # E3_QV（Top1: 5130）
  "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/weights/benchmark/glue/cola_gla-firstround/E3_QV/checkpoint-17100"
  "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/weights/benchmark/glue/cola_gla-firstround/E3_QV/checkpoint-6840"
  "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/weights/benchmark/glue/cola_gla-firstround/E3_QV/checkpoint-8550"

  # E4_OONLY（Top1: 10260）
  "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/weights/benchmark/glue/cola_gla-firstround/E4_OONLY/checkpoint-17100"
  "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/weights/benchmark/glue/cola_gla-firstround/E4_OONLY/checkpoint-5130"

  # E5_MLPONLY（Top1: 5130）
  "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/weights/benchmark/glue/cola_gla-firstround/E5_MLPONLY/checkpoint-10260"
  "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/weights/benchmark/glue/cola_gla-firstround/E5_MLPONLY/checkpoint-17100"
  "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/weights/benchmark/glue/cola_gla-firstround/E5_MLPONLY/checkpoint-3420"

  # E6_QKV（Top1: 5130）→ 注意 6840 下没有 model.pt
  "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/weights/benchmark/glue/cola_gla-firstround/E6_QKV/checkpoint-17100"
  "/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/weights/benchmark/glue/cola_gla-firstround/E6_QKV/checkpoint-3420"
)

bytes_free() {
  df -B1 --output=avail "$DEST" | tail -1 | tr -d ' '
}

bytes_reserve() {
  awk -v g="$RESERVE_GB" 'BEGIN{printf "%.0f\n", g*1024*1024*1024}'
}

same_fs() {
  local a="$1" b="$2"
  local da db
  da="$(stat -c %d "$a" 2>/dev/null || echo "")"
  db="$(stat -c %d "$b" 2>/dev/null || echo "")"
  [[ -n "$da" && -n "$db" && "$da" == "$db" ]]
}

io_prefix=()
case "$PRIORITY" in
  1)  io_prefix=(ionice -c2 -n7);;
  -1) io_prefix=(ionice -c2 -n0);;
  *)  io_prefix=();;
esac

log_manifest() {
  local exp="$1" step="$2" src="$3" new="$4" size="$5" sha="$6"
  {
    flock -x 9
    echo -e "${exp}\t${step}\t${src}\t${new}\t${size}\t${sha}" >> "$MANIFEST"
  } 9>>"$LOCKFILE"
}

move_same_fs() {
  local list=("$@")
  export -f same_fs
  for d in "${list[@]}"; do
    local exp ckdir step src new size sha="(skipped)"
    exp="$(basename "$(dirname "$d")")"
    ckdir="$(basename "$d")"
    step="${ckdir#checkpoint-}"
    src="$d/model.pt"
    [[ -f "$src" ]] || { echo "WARN: $src 不存在，跳过"; continue; }
    new="$DEST/${exp}-${ckdir}-model.pt"
    [[ -e "$new" ]] && new="$DEST/${exp}-${ckdir}-model_$(date +%s)-$$.pt"
    "${io_prefix[@]}" mv "$src" "$new"
    size="$(stat -c%s "$new" 2>/dev/null || stat -f%z "$new" 2>/dev/null || echo 0)"
    log_manifest "$exp" "$step" "$src" "$new" "$size" "$sha"
    echo "OK(SAMEFS): $src -> $new"
  done
}

move_cross_fs() {
  # 收集 (size_bytes, dir) 并按 size 升序排序，最大化装箱概率
  local tmplist
  tmplist="$(mktemp)"
  for d in "$@"; do
    local src="$d/model.pt"
    [[ -f "$src" ]] || { echo "WARN: $src 不存在，跳过"; continue; }
    local sz
    sz="$(stat -c%s "$src" 2>/dev/null || stat -f%z "$src" 2>/dev/null || echo 0)"
    printf "%012d\t%s\n" "$sz" "$d" >> "$tmplist"
  done
  sort -n "$tmplist" -o "$tmplist"

  local reserve
  reserve="$(bytes_reserve)"
  while IFS=$'\t' read -r sz d; do
    local avail
    avail="$(bytes_free)"
    if (( avail - reserve < sz )); then
      echo "WAIT: 可用空间不足 (avail=$(printf %.2f $(echo "$avail/1024/1024/1024" | bc -l)) GiB, 需~$(printf %.2f $(echo "$sz/1024/1024/1024" | bc -l)) GiB + 预留 ${RESERVE_GB}GiB)。请清理或调小 RESERVE_GB 后重试此文件：$d"
      continue
    fi

    local exp ckdir step src new sha size_after
    exp="$(basename "$(dirname "$d")")"
    ckdir="$(basename "$d")"
    step="${ckdir#checkpoint-}"
    src="$d/model.pt"
    new="$DEST/${exp}-${ckdir}-model.pt"
    [[ -e "$new" ]] && new="$DEST/${exp}-${ckdir}-model_$(date +%s)-$$.pt"

    if command -v rsync >/dev/null 2>&1; then
      "${io_prefix[@]}" rsync -a --whole-file --inplace --no-compress "$src" "$new"
      # 粗校验：大小一致
      if [[ -s "$new" ]] && [[ "$(stat -c%s "$src")" == "$(stat -c%s "$new")" ]]; then
        rm -f "$src"
      else
        echo "ERROR: 拷贝后大小不一致，保留源：$src"
        continue
      fi
    else
      "${io_prefix[@]}" cp -a "$src" "$new" && rm -f "$src"
    fi

    size_after="$(stat -c%s "$new" 2>/dev/null || stat -f%z "$new" 2>/dev/null || echo 0)"
    if [[ "$HASH" == "1" ]]; then
      if command -v sha256sum >/dev/null 2>&1; then
        sha="$(sha256sum "$new" | awk '{print $1}')"
      else
        sha="$(shasum -a 256 "$new" 2>/dev/null | awk '{print $1}')"
      fi
    else
      sha="(skipped)"
    fi
    log_manifest "$exp" "$step" "$src" "$new" "$size_after" "$sha"
    echo "OK(CROSSFS): $src -> $new"
  done < "$tmplist"
  rm -f "$tmplist"
}

# 分拣同FS/跨FS
samefs_list=()
crossfs_list=()
for d in "${CHECKPOINT_DIRS[@]}"; do
  if [[ -f "$d/model.pt" ]]; then
    if same_fs "$d" "$DEST"; then
      samefs_list+=("$d")
    else
      crossfs_list+=("$d")
    fi
  fi
done

echo "同文件系统数量: ${#samefs_list[@]}  | 跨文件系统数量: ${#crossfs_list[@]}"
# 同FS部分：并行执行（不消耗额外空间）
if ((${#samefs_list[@]})); then
  if command -v parallel >/dev/null 2>&1; then
    printf "%s\n" "${samefs_list[@]}" | parallel -j "$JOBS" --will-cite --no-notice bash -lc 'move_same_fs "$@"' _ {}
  else
    # 简单并发
    cnt=0
    for d in "${samefs_list[@]}"; do
      move_same_fs "$d" &
      cnt=$((cnt+1))
      if (( cnt % JOBS == 0 )); then
        wait
      fi
    done
    wait
  fi
fi

# 跨FS部分：空间感知、按体积小→大顺序串行/半自动
if ((${#crossfs_list[@]})); then
  move_cross_fs "${crossfs_list[@]}"
fi

echo "完成。清单：$MANIFEST"
