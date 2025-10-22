"""
对模型进行中期训练。与预训练相同但更简单。
运行方式：

python -m scripts.mid_train

或使用torchrun进行训练：

torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device_batch_size=16
"""

from collections import deque
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import wandb
import torch

from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir
from nanochat.tokenizer import get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.checkpoint_manager import load_model
import torch.distributed as dist

from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk

# -----------------------------------------------------------------------------
run = "dummy" # wandb运行名称默认值（"dummy"是特殊的 - 我们不会记录到wandb）
model_tag = None # 从中加载模型的模型标签（基础模型或中期训练模型）
step = None # 从中加载模型的步骤（基础模型或中期训练模型）
dtype = "bfloat16"
max_seq_len = 2048
device_batch_size = 32
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
init_lr_frac = 1.0 # 初始学习率是基本学习率的这个分数
weight_decay = 0.0
eval_every = 150
eval_tokens = 20*524288
total_batch_size = 524288
dry_run = 0 # dry_run=1用于实验：我们将记录到wandb但不会写入检查点或报告
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # 从命令行或配置文件的覆盖
user_config = {k: globals()[k] for k in config_keys} # 可能对日志记录有用
# -----------------------------------------------------------------------------

# 计算初始化
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0
dtype = torch.float32 if dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)

# wandb日志记录初始化
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-mid", name=run, config=user_config)

# 加载模型和分词器
model, tokenizer, meta = load_model("base", device, phase="train", model_tag=model_tag, step=step)
pretrain_batch_size = meta.get("device_batch_size", None)
if pretrain_batch_size is not None and device_batch_size > pretrain_batch_size:
    print0(f"FOOTGUN警告：基础模型训练使用了device_batch_size {pretrain_batch_size}，您是否向此脚本传递了良好的--device_batch_size？")
orig_model = model
model = torch.compile(model, dynamic=False)
depth = model.config.n_layer
num_flops_per_token = model.estimate_flops()
tokens_per_fwdbwd = device_batch_size * max_seq_len # 单个rank每次迭代的token数
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # 所有ranks每次迭代的总token数
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Token数 / 微批次 / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Token数 / 微批次: {world_tokens_per_fwdbwd:,}")
print0(f"总批次大小 {total_batch_size:,} => 梯度累积步数: {grad_accum_steps}")
token_bytes = get_token_bytes(device=device)

# 初始化优化器（线性层使用Muon，嵌入和lm_head使用AdamW）
optimizers = model.setup_optimizers(unembedding_lr=unembedding_lr, embedding_lr=embedding_lr, matrix_lr=matrix_lr, weight_decay=weight_decay)
adamw_optimizer, muon_optimizer = optimizers
# 将初始学习率覆盖为基本学习率的一部分
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * init_lr_frac
        group["initial_lr"] = group["lr"] # 保存初始学习率以便稍后轻松衰减

# 中期训练数据混合和DataLoader
base_dir = get_base_dir()
train_dataset = TaskMixture([
    SmolTalk(split="train"), # 460K行通用对话
    MMLU(subset="auxiliary_train", split="train"), # 100K行从ARC、MC_TEST、OBQA、RACE抽取的多项选择题
    GSM8K(subset="main", split="train"), # 8K行教授简单数学和（计算器）工具使用
]) # 总计：460K + 100K + 8K = 568K行
val_dataset = TaskMixture([
    SmolTalk(split="test"), # 测试集中的24K行
    MMLU(subset="all", split="test", stop=5200), # 测试集中的14K行，仅使用5.2K以匹配训练比例
    GSM8K(subset="main", split="test", stop=420), # 测试集中的1.32K行，仅使用420以匹配训练比例
]) # 总计：24K + 14K + 1.32K ~= 39K行
# DataLoader在此定义，它发出inputs, targets：形状为(device_batch_size, max_seq_len)的2D张量
# 一个大问题是我们事先不知道最终的迭代次数。因此我们创建
# 这两个全局变量并从数据生成器内部更新它们。
last_step = False # 当我们到达数据集末尾时，我们将切换此值为True
approx_progress = 0.0 # 在epoch过程中将从0到1
def mid_data_generator(split):
    global last_step, approx_progress
    assert split in {"train", "val"}, "split必须是'train'或'val'"
    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)
    assert dataset_size > 0
    needed_tokens = device_batch_size * max_seq_len + 1 # 形成一个训练批次的inputs,targets
    token_buffer = deque()
    scratch = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    cursor = ddp_rank # 每次增加ddp_world_size，因此每个rank处理唯一的文档
    while True:
        # 在生成之前累积足够的token进行一次迭代
        while len(token_buffer) < needed_tokens:
            conversation = dataset[cursor]
            ids, _ = tokenizer.render_conversation(conversation)
            token_buffer.extend(ids)
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor -= dataset_size # 环绕以进行另一个epoch
                if split == "train":
                    last_step = True # 切换last_step为True，这将终止训练循环
        # 构建inputs/targets并生成
        for i in range(needed_tokens):
            scratch[i] = token_buffer.popleft()
        inputs_cpu = scratch[:-1].to(dtype=torch.int32)
        targets_cpu = scratch[1:]
        inputs = inputs_cpu.view(device_batch_size, max_seq_len).to(device=device, dtype=torch.int32, non_blocking=True)
        targets = targets_cpu.view(device_batch_size, max_seq_len).to(device=device, dtype=torch.int64, non_blocking=True)
        if split == "train":
            approx_progress = cursor / dataset_size # 近似进度作为数据集的一部分
        yield inputs, targets

train_loader = mid_data_generator("train")
build_val_loader = lambda: mid_data_generator("val")
progress = 0 # 在epoch过程中将从0到1

# 学习率调度器
def get_lr_multiplier(progress):
    # 训练的前80%：无衰减，然后线性下降到0。
    return 1 if progress < 0.8 else 1 - (progress - 0.8) / 0.2

# Muon优化器的动量调度器
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# -----------------------------------------------------------------------------
# 训练循环
x, y = next(train_loader) # 预取第一批数据
min_val_bpb = float("inf")
smooth_train_loss = 0 # 训练损失的EMA
ema_beta = 0.9 # EMA衰减因子
total_training_time = 0 # 总训练时间（挂钟时间）
step = 0
while True:
    flops_so_far = num_flops_per_token * total_batch_size * step

    # 在所有ranks上同步last_step以避免分布式设置中的挂起
    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item())

    # 定期：评估验证bpb（所有ranks参与）
    if last_step or step % eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"步骤 {step:05d} | 验证bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # 在运行结束时保存检查点（仅主进程）
    if master_process and last_step and not dry_run:
        output_dirname = f"d{depth}" # 例如 d12
        checkpoint_dir = os.path.join(base_dir, "mid_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers], # TODO: 确保跨ranks保存正确完成
            {
                "step": step,
                "val_bpb": val_bpb, # 最后一步的损失
                "model_config": {
                    "sequence_len": max_seq_len,
                    "vocab_size": tokenizer.get_vocab_size(),
                    "n_layer": depth,
                    "n_head": model.config.n_head,
                    "n_kv_head": model.config.n_kv_head,
                    "n_embd": model.config.n_embd,
                },
                "user_config": user_config, # 训练脚本的输入
            }
        )

    if last_step:
        break

    # -------------------------------------------------------------------------
    # 单个训练步骤
    # 评估梯度
    torch.cuda.synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach() # 用于日志记录
        loss = loss / grad_accum_steps # 每个.backward()是梯度求和 => 在此处归一化损失
        loss.backward()
        x, y = next(train_loader) # 在GPU忙于前向/后向时预取下一批次
        progress = max(progress, approx_progress) # 仅单调增加进度
    # 步进优化器
    lrm = get_lr_multiplier(progress)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # 状态
    step += 1

    # 日志记录
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # 训练损失的EMA
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # 去偏EMA
    pct_done = 100 * progress
    tok_per_sec = int(world_tokens_per_fwdbwd / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size # bfloat16 H100 SXM且无2:4稀疏性
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100 # 以%计
    if step > 10:
        total_training_time += dt # 仅计算前10步
    print0(f"步骤 {step:05d} ({pct_done:.2f}%) | 损失: {debiased_smooth_loss:.6f} | 学习率乘数: {lrm:.2f} | 时间: {dt * 1000:.2f}ms | token/秒: {tok_per_sec:,} | MFU: {mfu:.2f} | 总时间: {total_training_time/60:.2f}m")
    if step % 10 == 0:
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        })

# 打印一些额外统计信息
print0(f"峰值内存使用: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f}MiB")
print0(f"总训练时间: {total_training_time/60:.2f}m")
print0(f"最小验证bpb: {min_val_bpb:.4f}")

# 记录到报告
if not dry_run:
    from nanochat.report import get_report
    get_report().log(section="中期训练", data=[
        user_config, # CLI参数
        { # 关于训练设置的统计信息
            "迭代次数": step,
            "DDP世界大小": ddp_world_size,
        },
        { # 关于训练结果的统计信息
            "最小验证bpb": min_val_bpb,
        }
    ])

# 清理
wandb_run.finish() # wandb运行结束
compute_cleanup()
