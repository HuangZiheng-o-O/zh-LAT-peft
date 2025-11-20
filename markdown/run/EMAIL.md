[14 tools called]

- 我已接入 SwanLab 邮件提醒，新增私密 YAML、注册 EmailCallback（不影响现有 SwanLabCallback），并在训练成功/失败/中断三种场景下发通知；也为 shell 的中断 trap 加了只发通知、不 kill 的钩子。以下是你需要放置的 YAML 与启用方法。

### 你需要做的一次性配置
1) 新建私密 YAML（已 gitignore）
- 路径：/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/dangerous/email_notify.yaml
- 内容示例（注意 Gmail App Password 粘贴时去掉空格）：
```yaml
sender_email: "ziheng2701@gmail.com"
receiver_email: "ziheng2701@gmail.com"
password: "16"          # 你的 Google App Password，去掉空格
smtp_server: "smtp.gmail.com"
port: 587
language: "zh"
```
- 模板参考：`mamba-peft/dangerous/email_notify.example.yaml`（已添加）；已更新 `mamba-peft/.gitignore` 忽略真实 YAML。

2) 环境变量（确保 cloud 模式仍开启）
```bash
export SWANLAB_ENABLE=1
export SWANLAB_MODE=cloud
export SWANLAB_PROJECT="你的项目名"
export SWANLAB_EMAIL_YAML=/home/user/mzs_h/code/zh-LAT-peft/mamba-peft/dangerous/email_notify.yaml
# 可选开关（默认开启）
export SWANLAB_EMAIL_ON_START=1
export SWANLAB_EMAIL_ON_FINISH=1
export SWANLAB_EMAIL_ON_INTERRUPT=1
```

3) 快速自测邮件
```bash
cd /home/user/mzs_h/code/zh-LAT-peft/mamba-peft
python -m scripts.utils.email_notify --event STARTED --group smoketest --yaml "$SWANLAB_EMAIL_YAML"
```

### 已做的改动（无需你改代码即可生效）
- 邮件配置模板
  - 新增 `mamba-peft/dangerous/email_notify.example.yaml`（示例）
  - `mamba-peft/.gitignore` 忽略 `dangerous/*.yaml`（保留 example）

- 训练入口（自动注册邮件回调）
  - `mamba-peft/train_gla_only.py`
  - `mamba-peft/train_shared.py`
  - 当 `SWANLAB_ENABLE=1` 且能读到 `SWANLAB_EMAIL_YAML` 时：
    - 注册 `EmailCallback`（基于官方插件）
    - 发送 STARTED（可配）
    - 训练 try/except：
      - 成功后发送 FINISHED（兜底，和 SwanLab on_stop 不冲突）
      - 异常时发送 FAILED（带简要异常信息）

- 中断通知（只发通知、不 kill，可配置开关）
  - `mamba-peft/scripts/train/new/gla_round_clean.sh` 的 `cleanup()` trap 中，当 `SWANLAB_EMAIL_ON_INTERRUPT=1`（默认）时调用：
    - `python -m scripts.utils.email_notify --event INTERRUPTED --group "suite=... round=... data=..." --yaml "${SWANLAB_EMAIL_YAML}"`

- 批量调度透传邮件 YAML
  - `mamba-peft/scripts/train/new/gla_batch_tmux_clean.sh` 已向子进程透传 `SWANLAB_EMAIL_YAML`，无需额外改动。

- 邮件工具（可在任意时刻发自定义邮件）
  - 新增 `mamba-peft/scripts/utils/email_notify.py`
  - 用法：
    - `python -m scripts.utils.email_notify --event STARTED|FINISHED|FAILED|INTERRUPTED --group "<tag>" --yaml "$SWANLAB_EMAIL_YAML" --extra "可选文本"`

### 行为说明
- 训练完成/错误：走 SwanLab 官方 EmailCallback 的 on_stop 通知；并已在训练代码层面加“成功/失败”兜底邮件，避免云端 finish 卡顿时没有邮件。
- 训练中断（Ctrl-C / SIGTERM / tmux 结束等）：当 `SWANLAB_EMAIL_ON_INTERRUPT=1`（默认）时，shell trap 会发送一次 INTERRUPTED 邮件；不会 kill 进程。
- 仍保持 cloud 模式，不关闭 SwanLab；参考官方插件文档用法与回调生命周期说明见文档《邮件通知》[docs.swanlab.cn/plugin/notification-email.html](https://docs.swanlab.cn/plugin/notification-email.html)。

你现在只需把 Gmail App Password（去空格）写入 `dangerous/email_notify.yaml` 并导出 `SWANLAB_EMAIL_YAML`，就能收到 训练开始 / 成功 / 失败 / 中断 四类通知。

- 我添加了 `EmailCallback` 注册，训练成功与失败邮件兜底，以及 `cleanup()` 的中断邮件；批量脚本也透传了 `SWANLAB_EMAIL_YAML`。运行前在 `dangerous/email_notify.yaml` 写入你的 Gmail 信息并导出该路径即可。