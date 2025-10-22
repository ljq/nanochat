# nanochat的1000美元层级
# 设计为在8XH100节点上端到端运行1000美元/24 ≈ 41.6小时
# 注释较少，更多细节请参阅speedrun.sh

# 所有设置内容
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync
source .venv/bin/activate
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi
python -m nanochat.report reset
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL
    unzip -q eval_bundle.zip
    rm eval_bundle.zip
    mv eval_bundle $NANOCHAT_BASE_DIR
fi

# 在约40亿字符上训练分词器并启动其余数据的下载用于预训练
python -m nanochat.dataset -n 16
# 开始下载其余分片，总共800个（请参阅下面为什么是800）
python -m nanochat.dataset -n 800 &
# todo: 下载其余部分
python -m scripts.tok_train --max_chars=4000000000
python -m scripts.tok_eval

# 记录我确定此run1000.sh脚本超参数的过程：
# 我们想要约1000美元 ≈ 41.6小时的8XH100计算预算
# 1) 我猜测此模型的规模约为depth=32
# 2) 确定适合的device_batch_size：
# 运行base_train.py脚本使用--depth=32，我看到--device_batch_size=16
# 内存不足，但--device_batch_size=8适合。在训练期间检查`nvidia-smi`，
# 我看到所有GPU都在约78/80GB VRAM，所以它刚好适合，我们有约50%的良好MFU。
# 因此训练脚本运行正常并显示：
# 词汇表大小：65,536
# 层数：32
# 模型维度：2048
# 头数：16
# KV头数：16
# 每个微批次/rank的token数：8 x 2048 = 16,384
# 每个微批次的token数：131,072
# 总批次大小524,288 => 梯度累积步数：4
# 参数数量：1,879,048,192
# 每个token的估计FLOPs：1.207960e+10
# 从目标数据:参数比计算出的迭代次数：71,680
# 训练token总数：37,580,963,840
# Token : 参数比：20.00
# 总训练FLOPs估计：4.539628e+20
# 步骤 00004/71680 (0.01%) | 损失：8.813754 | 学习率乘数：1.00 | 时间：1571.88ms | token/秒：83,385 | MFU：50.92 | 总时间：0.00m
# 步骤 00005/71680 (0.01%) | 损失：8.488074 | 学习率乘数：1.00 | 时间：1572.76ms | token/秒：83,338 | MFU：50.89 | 总时间：0.00m
# ...
# 3) 验证运行时间是否适合我们的预算：
# 训练脚本使用Chinchilla缩放定律计算最优设置#tokens = 20 * #params。特别是：
# 脚本显示我们将训练71,680步，每步需要1.574秒，所以：
# 估计训练时间：71,680 * 1.574秒 / 60 / 60 = 31.3小时。
# 这没问题，适合我们的预算，并为中期训练、SFT、评估和可能的RL留下约10小时。
# 我们可能甚至适合depth=33或depth=34，但现在让我们继续这个。
# 4) 最后要注意的是运行所需的训练数据量。
# 上面的脚本计算了"训练token总数：37,580,963,840"
# tok_eval.py脚本报告默认分词器设置平均约~4.8字符/token。
# 所以~380亿token # ~4.8字符/token = ~1850亿字符。
# 每个数据分片约2.5亿字符，所以我们需要~1850亿 / 2.5亿 ≈ 740个分片。
# 为安全起见，我将其增加到800个分片，这就是为什么上面在预下载数据集分片时使用了-n 800。
# 如果我们没有足够的数据，训练脚本将循环并对相同数据进行多个epoch，
# 这会降低模型性能。可能2、3个epoch左右~可以，但肯定不理想，在10+个epoch时我们会
# 开始严重过拟合。
# 5) 就是这样，其他所有内容（例如学习率）都由训练脚本自动调整。
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=32 --device_batch_size=8
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval

# 中期训练
# 注意：确保我们在此处使用与基础训练脚本相同的device_batch_size。
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device_batch_size=8 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid

# 监督微调
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# 生成最终报告
python -m nanochat.report generate

# 与它对话
python -m scripts.chat_web
