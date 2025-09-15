#!/usr/bin/env bash
ENV_FAIL_LOG="./conda_install_fail.log"
YML="./conda_environment3.yml"

# 清空旧日志
: > "$ENV_FAIL_LOG"

# 从 yml 中提取 dependencies 列表（只取顶级的那一段）
pkgs=($(sed -n -e '/^dependencies:/,/^prefix:/!d' \
               -e 's/^- //' \
               -e '/^prefix:/d' "$YML"))

for pkg in "${pkgs[@]}"; do
  # 取 package 名称（去掉版本限制部分，例如 asttokens=2.4.1 → asttokens）
  name="${pkg%%=*}"
  # conda list 检查是否已装
  if conda list --name "${CONDA_DEFAULT_ENV}" | grep -qw "^$name"; then
    echo "✔ skip existing: $name"
  else
    echo -n "⏳ installing: $pkg … "
    if conda install "$pkg" -y; then
      echo "ok"
    else
      echo "failed, logging"
      echo "$pkg" >> "$ENV_FAIL_LOG"
    fi
  fi
done

echo
echo "全部完成。失败的包已经记录在 $ENV_FAIL_LOG"
