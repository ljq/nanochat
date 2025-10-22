"""
通过"GRPO"在GSM8K上进行强化学习。

我将GRPO放在引号中，因为我们实际上得到的东西要简单得多，
更类似于普通的REINFORCE：

1) 删除信任区域，因此没有对参考模型的KL正则化
2) 我们是在策略的，因此不需要PPO比率+裁剪。
3) 我们使用GAPO风格的标准化，是token级别的，而不是序列级别的。
4) 不使用z-score标准化(r - mu)/sigma，只使用(r - mu)作为优势。

1个GPU：
python -m scripts.chat_rl

8个GPU：
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=default
"""

import os
import itertools
import re
import wandb
import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K

# RL超参数
run = "dummy" # wandb运行名称
source = "sft" # mid|sft
dtype = "bfloat16"
device_batch_size = 8 # 前向传递不会超过这个值以避免OOM
examples_per_step = 16 # 总共跨所有ranks（注意：示例，不是样本/完成！）
num_samples = 16 # 每个示例（/问题）的样本数量
max_new_tokens = 256
temperature = 1.0
top_k = 50 # TODO: 尝试None？
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.05
num_epochs = 1 # 要在gsm8k上训练的epoch数
save_every = 60 # 每多少步保存模型
eval_every = 60 # 每多少步评估模型的验证pass@k
eval_examples = 400 # 用于评估pass@k的示例数量
# 现在允许通过配置器从CLI覆盖设置
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # 从命令行或配置文件的覆盖
user_config = {k: globals()[k] for k in config_keys} # 对日志记录有用
# -----------------------------------------------------------------------------

# 初始化计算/精度
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0 # 此进程将进行日志记录、检查点等操作
dtype = torch.float32 if dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)

# wandb日志记录初始化
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-rl", name=run, config=user_config)

# 初始化模型和分词器
model, tokenizer, meta = load_model(source, device, phase="eval")
engine = Engine(model, tokenizer) # 用于采样rollout

# -----------------------------------------------------------------------------
# Rollout / 采样生成器循环，为训练生成示例批次

train_task = GSM8K(subset="main", split="train")
val_task = GSM8K(subset="main", split="test")
num_steps = (len(train_task) // examples_per_step) * num_epochs
print0(f"计算步数: {num_steps}")

@torch.no_grad()
def get_batch():
    assistant_end = tokenizer.encode_special("<|assistant_end|>") # 可以使用此token，它仅用于填充，不用于损失。
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size) # 每个rank负责训练数据中的不同示例
    for example_idx in itertools.cycle(rank_indices):

        # 首先获取用户和助手消息的完整对话
        conversation = train_task[example_idx]

        # 标记化对话，删除最后一条助手消息，并为助手准备完成
        # （即保留<|assistant_start|>，但删除其后的所有内容）
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        # 使用批处理生成num_samples个样本，使用循环避免OOM
        model.eval() # 确保模型处于评估模式
        generated_token_sequences = []
        masks = []
        num_sampling_steps = num_samples // device_batch_size # 顺序进行以防止OOM
        for sampling_step in range(num_sampling_steps):
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF # int32的正半部分
            with autocast_ctx:
                generated_token_sequences_batch, masks_batch = engine.generate_batch(
                    tokens,
                    num_samples=device_batch_size,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    seed=seed, # 必须确保为每个采样步骤更改种子
                )
            generated_token_sequences.extend(generated_token_sequences_batch)
            masks.extend(masks_batch)

        # 计算每个样本的奖励
        rewards = []
        for sample_tokens in generated_token_sequences:
            # 仅获取生成的token（提示之后）
            generated_tokens = sample_tokens[prefix_length:]
            # 解码生成的响应
            generated_text = tokenizer.decode(generated_tokens)
            # 计算奖励
            reward = train_task.reward(conversation, generated_text)
            rewards.append(reward)

        # 填充序列以使它们的长度（在时间上）匹配
        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_generated_token_sequences = [seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
        # 将序列和mask堆叠到PyTorch张量中
        ids = torch.tensor(padded_generated_token_sequences, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
        # 为Transformer生成自回归输入和目标
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone() # 克隆以避免就地修改：
        targets[mask_ids[:, 1:] == 0] = -1 # <-- 这里的就地修改。-1是忽略索引
        # 注意，Engine为提示token和工具使用token都返回mask=0。
        # 因此我们将（正确地）最终不训练提示token或工具使用强制token。
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        # 通过简单减去平均值来计算优势（而不是z-score (x-mu)/sigma）
        mu = rewards.mean()
        advantages = rewards - mu
        # 生成输入/目标作为(B, T)的ids和奖励作为(B,)的浮点数
        yield generated_token_sequences, inputs, targets, rewards, advantages

# -----------------------------------------------------------------------------
# GSM8K pass@k的简单评估循环
def run_gsm8k_eval(task, tokenizer, engine,
    max_examples=None,
    num_samples=1,
    max_completion_tokens=256,
    temperature=0.0,
    top_k=50
):
    """
    评估GSM8K任务并返回评估结果的记录列表。
    在分布式设置中，所有ranks协作，但此函数不会
    跨ranks进行归约。这是调用者的责任。
    因为评估可能需要一段时间，此函数将逐个生成记录。
    """
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        # 使用Engine内的批处理生成k个样本
        assert num_samples <= device_batch_size # 通常这是真的。如果不是，我们可以添加循环...
        generated_token_sequences, masks = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k
        )
        # 检查每个样本的正确性
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({
                "is_correct": is_correct
            })
        # 有点臃肿，因为我曾经想进行更复杂的日志记录。
        record = {
            "idx": idx,
            "outcomes": outcomes,
        }
        yield record

# -----------------------------------------------------------------------------
# 训练循环

# 初始化优化器
optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay,
)

# 将初始学习率设置为基本学习率的一部分
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * init_lr_frac
        group["initial_lr"] = group["lr"] # 保存初始学习率以便稍后轻松衰减

# 学习率调度器：在num_steps上简单下降到零
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_steps
    return lrm

# 计算每个rank处理的示例数量以达到所需的examples_per_step
print0(f"每步总序列数: {examples_per_step * num_samples}") # 序列/步的总批次大小
assert examples_per_step % ddp_world_size == 0, "所需的每步示例数必须能被ranks数整除"
examples_per_rank = examples_per_step // ddp_world_size # 每个GPU
print0(f"计算每rank示例数: {examples_per_rank}")

# 启动训练循环
batch_iterator = get_batch()
for step in range(num_steps):

    # 定期评估模型并记录到wandb
    if step % eval_every == 0:
        model.eval()
        passk = torch.zeros(device_batch_size, device=device) # k=1..device_batch_size的pass@k
        with autocast_ctx:
            records_iter = run_gsm8k_eval(val_task, tokenizer, engine, num_samples=device_batch_size, max_examples=eval_examples, temperature=1.0)
            records = list(records_iter) # 收集所有记录
        for k in range(1, device_batch_size + 1):
            passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
        num_records = torch.tensor(len(records), dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
            dist.all_reduce(passk, op=dist.ReduceOp.SUM)
        passk = passk / num_records.item() # 通过总记录数归一化
        print_passk = [f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, device_batch_size + 1)]
        print0(f"步骤 {step} | {', '.join(print_passk)}")
        log_passk = {f"pass@{k}": passk[k - 1].item() for k in range(1, device_batch_size + 1)}
        wandb_run.log({
            "step": step,
            **log_passk,
        })

    # 在数据集中的多个示例上进行前向/后向rollout
    rewards_list = []
    sequence_lengths = []
    for example_step in range(examples_per_rank):
        # 获取与训练数据集中一个示例对应的一个批次
        sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)
        # 评估损失和梯度
        model.train() # 确保模型处于训练模式
        # 我们需要另一个循环，因为我们永远不能超过device_batch_size
        assert inputs_all.size(0) % device_batch_size == 0
        num_passes = inputs_all.size(0) // device_batch_size
        for pass_idx in range(num_passes):
            # 提取此传递的批次
            b0, b1 = pass_idx * device_batch_size, (pass_idx + 1) * device_batch_size
            inputs = inputs_all[b0:b1]
            targets = targets_all[b0:b1]
            rewards = rewards_all[b0:b1]
            advantages = advantages_all[b0:b1]
            # 计算对数概率。注意损失计算NLL = -logp，所以我们取负
            with autocast_ctx:
                logp = -model(inputs, targets, loss_reduction='none').view_as(inputs) # (B, T)
            # 计算PG目标。注意ignore_index=-1确保无效token的损失为0。
            pg_obj = (logp * advantages.unsqueeze(-1)).sum()
            # 通过有效token数、传递数和examples_per_rank归一化
            num_valid = (targets >= 0).sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
            # 注意，不需要添加PPO比率+裁剪，因为我们是在策略的
            # 最后，制定我们想要最小化的损失（而不是我们想要最大化的目标）
            loss = -pg_obj
            loss.backward()
            print0(f"步骤 {step}/{num_steps} | 示例步骤 {example_step} | 传递 {pass_idx} | 损失: {loss.item():.6f} | 平均奖励: {rewards.mean().item()}")
        # 用于日志记录
        rewards_list.append(rewards_all.mean().item())
        sequence_lengths.extend(len(seq) for seq in sequences_all)

    # 大量日志记录此步骤rollout的情况
    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
    if ddp: # 跨ranks聚合
        mean_reward_tensor = torch.tensor(mean_reward, dtype=torch.float, device=device)
        mean_sequence_length_tensor = torch.tensor(mean_sequence_length, dtype=torch.float, device=device)
        dist.all_reduce(mean_reward_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_sequence_length_tensor, op=dist.ReduceOp.AVG)
        mean_reward = mean_reward_tensor.item()
        mean_sequence_length = mean_sequence_length_tensor.item()
    print0(f"步骤 {step}/{num_steps} | 平均奖励: {mean_reward} | 平均序列长度: {mean_sequence_length:.2f}")

    wandb_run.log({
        "step": step,
        "reward": mean_reward,
        "sequence_length": mean_sequence_length,
    })

    # 更新模型参数
    lrm = get_lr_multiplier(step)
    for opt in optimizers: # 首先设置学习率
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    for opt in optimizers: # 然后步进优化器
        opt.step()
    model.zero_grad(set_to_none=True)
    wandb_run.log({
        "step": step,
        "lrm": lrm,
    })

    # 主进程定期保存模型。跳过第一步。保存最后一步。
    if master_process and ((step > 0 and step % save_every == 0) or step == num_steps - 1):
        base_dir = get_base_dir()
        depth = model.config.n_layer
        model_tag = f"d{depth}" # 基于基础模型的深度设置模型标签
        checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", model_tag)
        model_config_kwargs = model.config.__dict__ # 稍微有点调皮，滥用GPTConfig的简单性，TODO 更好
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None, # 注意：我们不费心保存优化器状态
            {
                "model_config": model_config_kwargs,
            }
        )
        print(f"✅ 模型检查点已保存到 {checkpoint_dir}")

# 记录到报告
from nanochat.report import get_report
get_report().log(section="聊天RL", data=[
    user_config, # CLI参数
])

wandb_run.finish() # wandb运行完成
compute_cleanup()
