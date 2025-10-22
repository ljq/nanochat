#!/bin/bash

# 这个脚本是"100美元能买到的最好的ChatGPT克隆版"，
# 它设计为在8XH100节点上运行约4小时，每小时3美元/GPU。

# 1) 最简单的启动示例：
# bash speedrun.sh
# 2) 在screen会话中启动示例（因为运行需要约4小时）：
# screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
# 3) 使用wandb日志记录的启动示例，但请先设置wandb：
# WANDB_RUN=speedrun screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# 默认中间产物目录在 ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# 使用uv设置Python虚拟环境

# 安装uv（如果尚未安装）
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# 创建本地虚拟环境.venv（如果不存在）
[ -d ".venv" ] || uv venv
# 安装仓库依赖
uv sync
# 激活虚拟环境，使`python`使用项目的虚拟环境而不是系统python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb设置
# 如果您希望使用wandb进行日志记录（很好！推荐）。
# 1) 确保首先登录wandb，例如运行：
#    `wandb login`
# 2) 运行此脚本时设置WANDB_RUN环境变量，例如：
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # 默认使用"dummy"：它作为特殊情况处理，跳过记录到wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# 在运行过程中，我们将把markdown报告写入基础目录中的report/目录。
# 此命令清除它并写入一个头部部分，包含一堆系统信息和标记运行开始的时间戳。
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# 分词器

# 安装Rust / Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# 构建rustbpe分词器
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# 下载前约20亿字符的预训练数据集
# 有关此数据准备方式的详细信息，请参阅dev/repackage_data_reference.py
# 每个数据分片约2.5亿字符
# 因此我们此时下载20亿 / 2.5亿 = 8个数据分片
# 每个分片约100MB文本（压缩后），因此磁盘上约800MB数据
python -m nanochat.dataset -n 8
# 在分词器训练时立即在后台启动下载更多分片
# 请参阅下面的注释了解为什么240是正确的数字
python -m nanochat.dataset -n 240 &
DATASET_DOWNLOAD_PID=$!
# 在约20亿字符数据上训练词汇表大小为2**16 = 65536的分词器
python -m scripts.tok_train --max_chars=2000000000
# 评估分词器（报告压缩比等）
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# 基础模型（预训练）

# 从s3下载eval_bundle以在训练期间评估CORE指标（约162MB）
EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL
    unzip -q eval_bundle.zip
    rm eval_bundle.zip
    mv eval_bundle $NANOCHAT_BASE_DIR
fi

# d20模型有5.61亿参数。
# Chinchilla说#tokens = 20X #params，所以我们需要5.61e6 * 20 = 112亿token。
# 假设我们的分词器是4.8字符/token，这是112亿 * 4.8 ≈ 540亿字符。
# 在2.5亿字符/分片的情况下，这是540亿 / 2.5亿 ≈ 216个分片用于预训练。
# 为安全起见向上取整到240。在约100MB/分片的情况下，这下载约24GB数据到磁盘。
# （整个数据集中可用的分片总数为1822。）
echo "等待数据集下载完成..."
wait $DATASET_DOWNLOAD_PID

# 预训练d20模型
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20 --run=$WANDB_RUN
# 在更大的训练/验证数据块上评估模型并绘制一些样本
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss
# 在CORE任务上评估模型
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval

# -----------------------------------------------------------------------------
# 中期训练（教授模型对话特殊token、工具使用、多项选择）

# 运行中期训练并评估模型
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid

# -----------------------------------------------------------------------------
# 监督微调（领域适应到每个序列本身每行）

# 训练SFT并立即重新评估（应该看到小的提升）
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# 通过CLI与模型聊天！省略-p以交互式聊天
# python -m scripts.chat_cli -p "为什么天空是蓝色的？"

# 更好的是，通过漂亮的WebUI ChatGPT风格与您的模型聊天
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# 强化学习。可选，目前仅在GSM8K上
# （可选）

# 运行强化学习
# torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=$WANDB_RUN
# 仅在GSM8K上评估RL模型
# torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i rl -a GSM8K

# -----------------------------------------------------------------------------
# 通过将所有部分组合在一起生成完整报告
# report.md是输出，为方便起见将复制到当前目录
python -m nanochat.report generate
