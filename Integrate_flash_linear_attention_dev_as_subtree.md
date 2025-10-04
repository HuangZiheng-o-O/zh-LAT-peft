```

#!/bin/bash
# =========================================================
# 🧩 功能说明：
# 将子仓库 flash-linear-attention 的 dev 分支完整并入父仓库，
# 不从远程拉取，不访问网络，仅基于本地仓库操作。
#
# 并入方式使用 git subtree，可保留历史记录（或可选压缩为单次提交）。
# =========================================================

# 进入父仓库所在目录
cd /Users/huangziheng/PycharmProjects/code/zh-LAT-peft

# 让脚本遇到错误时立即退出（防止出错继续执行）
set -euo pipefail

# 子目录名（子仓库所在路径）
LOCAL_DIR="flash-linear-attention"

# ---------------------------------------------------------
# 0️⃣ 预检：确保当前仓库状态干净
# ---------------------------------------------------------
# 这里检测父仓库是否有未提交的修改（包括暂存区与工作区）
# 如果有改动，会提示先 commit 或 stash
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "❌ 错误：父仓库存在未提交的更改，请先执行 commit 或 stash。"
  echo "例如： git add -A && git commit -m '保存当前修改'"
  echo "或者： git stash -u"
  exit 1
fi

# 检查子目录是否为独立 git 仓库
if [ ! -d "$LOCAL_DIR/.git" ]; then
  echo "❌ 错误：$LOCAL_DIR 不是一个嵌套 git 仓库（未发现 .git 目录）。"
  exit 1
fi

# ---------------------------------------------------------
# 1️⃣ 指定要并入的子仓库分支
# ---------------------------------------------------------
# 这里强制使用 dev 分支（你希望的版本）
LOCAL_BRANCH="dev"
echo "📦 将使用子仓库分支: $LOCAL_BRANCH"

# ---------------------------------------------------------
# 2️⃣ 创建子仓库的临时裸克隆
# ---------------------------------------------------------
# 裸仓库（bare repo）是没有工作区的纯 Git 数据库，
# 可以当作远程使用（方便 git subtree 拉取历史）
TMP_BARE="/tmp/fla-local.git"
# 删除可能存在的旧临时仓库
rm -rf "$TMP_BARE"

echo "🪣 正在创建临时裸仓库副本 ..."
# 克隆当前子仓库为裸仓库
git clone --bare "$LOCAL_DIR" "$TMP_BARE"

# ---------------------------------------------------------
# 3️⃣ 临时挪走原始目录，为 subtree 添加腾出空间
# ---------------------------------------------------------
echo "📂 备份原始目录到 ${LOCAL_DIR}.BAK ..."
mv "$LOCAL_DIR" "${LOCAL_DIR}.BAK"

# ---------------------------------------------------------
# 4️⃣ 在父仓库中以 subtree 方式导入子仓库
# ---------------------------------------------------------
# 4.1 添加一个临时远端，指向刚才的裸仓库
git remote remove fla-local 2>/dev/null || true
git remote add fla-local "$TMP_BARE"

# 4.2 获取该临时远端（相当于把裸仓库信息导入父仓库）
git fetch fla-local

# 4.3 将子仓库的 dev 分支导入为 subtree
# --prefix 指定导入到哪个子目录（与原路径一致）
# 如果你想把整个历史压缩成一次提交，请在末尾加上 --squash
echo "🌳 正在以 subtree 方式导入 dev 分支 ..."
git subtree add --prefix="$LOCAL_DIR" fla-local "$LOCAL_BRANCH"

# ---------------------------------------------------------
# 5️⃣ 清理临时资源
# ---------------------------------------------------------
# 删除临时远端和裸仓库
git remote remove fla-local
rm -rf "$TMP_BARE"

# ---------------------------------------------------------
# 6️⃣ 验证导入结果
# ---------------------------------------------------------
echo ""
echo "✅ 导入完成！现在父仓库中已有 ${LOCAL_DIR}/ 目录（来自 dev 分支）。"
echo "🔍 你可以执行以下命令检查结果："
echo ""
echo "  git ls-files ${LOCAL_DIR} | head"
echo "  git log --oneline --graph --decorate -- ${LOCAL_DIR} | head"
echo ""
echo "🧾 同时保留了原目录备份： ${LOCAL_DIR}.BAK"
echo "你可以对比两者内容："
echo "  diff -ru ${LOCAL_DIR}.BAK ${LOCAL_DIR} | less"
echo ""
echo "如果确认无误，可以删除备份："
echo "  rm -rf ${LOCAL_DIR}.BAK"
echo ""

# ---------------------------------------------------------
# 7️⃣ 推送父仓库到远程
# ---------------------------------------------------------
# echo "🚀 推送变更到 GitHub（父仓库） ..."
# git push

echo ""
echo "🎉 完成！"
echo "flash-linear-attention 的 dev 分支已作为 subtree 并入外层仓库。"
echo "GitHub 页面将直接显示该目录的所有文件。"
echo ""

```