"""
训练模型。运行方式：

python base_train.py

或分布式运行：

torchrun --nproc_per_node=8 base_train.py
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # 启用可扩展的CUDA内存段
import time
import wandb
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from scripts.base_eval import evaluate_model
print_banner()  # 打印nanochat横幅

# -----------------------------------------------------------------------------
# 用户设置
run = "dummy"  # wandb运行名称默认值（"dummy"是特殊的 - 我们不会记录到wandb）
# 模型架构
depth = 20  # 要训练的Transformer模型的深度，其余参数由此推导
max_seq_len = 2048  # 最大上下文长度
# 训练范围。只有这3个中的一个会被使用，按此优先级顺序。
num_iterations = -1  # 明确的优化步数（-1 = 禁用）
target_flops = -1.0  # 计算达到目标FLOPs所需的步数。用于缩放定律实验（-1 = 禁用）
target_param_data_ratio = 20  # 计算维持固定数据:参数比例所需的步数（Chinchilla=20）（-1 = 禁用）
# 优化
device_batch_size = 32  # 每个设备的批次大小（设置为不OOM）
total_batch_size = 524288  # 期望的总批次大小，以token数计
embedding_lr = 0.2  # 嵌入参数的学习率（Adam）
unembedding_lr = 0.004  # 解嵌入参数的学习率（Adam）
weight_decay = 0.0  # 嵌入/解嵌入参数的权重衰减（Adam）
matrix_lr = 0.02  # 矩阵参数的学习率（Muon）
grad_clip = 1.0  # 梯度裁剪值（0.0 = 禁用）
# 评估
eval_every = 250  # 每多少步评估模型的验证bpb
eval_tokens = 20*524288  # 评估验证损失的token数量
core_metric_every = 2000  # 每多少步评估核心指标
core_metric_max_per_task = 500  # 估计核心指标时每个任务的最大示例数
sample_every = 2000  # 每多少步从模型采样
# 输出
model_tag = ""  # 可选地覆盖输出检查点目录名称的模型标签
# 现在允许通过配置器从CLI覆盖设置
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())  # 从命令行或配置文件覆盖
user_config = {k: globals()[k] for k in config_keys}  # 将用于日志记录
# -----------------------------------------------------------------------------

# 计算初始化
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0  # 此进程将执行日志记录、检查点保存等
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)  # 自动混合精度上下文

# wandb日志记录初始化
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=run, config=user_config)

# 分词器将用于评估，我们还需要词汇表大小
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)  # 获取token字节数用于评估
vocab_size = tokenizer.get_vocab_size()
print0(f"词汇表大小: {vocab_size:,}")

# 模型参数从期望的深度推导
num_layers = depth  # 层数
model_dim = depth * 64  # 宽高比64（通常随着模型大小增加从64变化到128）
num_heads = max(1, (model_dim + 127) // 128)  # 头维度128（这里的除法是向上取整）
num_kv_heads = num_heads  # 1:1 MQA比例
print0(f"层数: {num_layers}")
print0(f"模型维度: {model_dim}")
print0(f"头数: {num_heads}")
print0(f"键值头数: {num_kv_heads}")

# 优化器/数据/训练长度相关的超参数
# 计算达到期望总批次大小所需的梯度累积步数
tokens_per_fwdbwd = device_batch_size * max_seq_len  # 单个rank每次迭代的token数
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size  # 所有rank每次迭代的总token数
assert total_batch_size % world_tokens_per_fwdbwd == 0  # 确保总批次大小能被整除
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd  # 梯度累积步数
print0(f"Token数 / 微批次 / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Token数 / 微批次: {world_tokens_per_fwdbwd:,}")
print0(f"总批次大小 {total_batch_size:,} => 梯度累积步数: {grad_accum_steps}")
# -----------------------------------------------------------------------------
# 初始化模型
model_config_kwargs = dict(sequence_len=max_seq_len, vocab_size=vocab_size, n_layer=num_layers, n_head=num_heads, n_kv_head=num_kv_heads, n_embd=model_dim)
with torch.device("meta"):  # 使用元设备初始化以节省内存
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
model.to_empty(device="cuda")  # 将模型移动到GPU
model.init_weights()  # 初始化权重
orig_model = model  # 原始未编译模型，用于保存原始模型状态字典
model = torch.compile(model, dynamic=False)  # TODO: 考虑dynamic True/False
num_params = sum(p.numel() for p in model.parameters())  # 计算参数数量
print0(f"参数数量: {num_params:,}")
num_flops_per_token = model.estimate_flops()  # 估计每个token的FLOPs
print0(f"估计每个token的FLOPs: {num_flops_per_token:e}")

# 计算迭代次数。要么给定，要么从目标FLOPs计算，要么从目标数据:参数比例计算（按此顺序）
assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
if num_iterations > 0:
    print0(f"使用用户提供的迭代次数: {num_iterations:,}")
elif target_flops > 0:
    # 从目标FLOPs计算迭代次数
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print0(f"从目标FLOPs计算的迭代次数: {num_iterations:,}")
elif target_param_data_ratio > 0:
    # 从目标参数数据比例计算迭代次数
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(f"从目标数据:参数比例计算的迭代次数: {num_iterations:,}")
else:
    raise ValueError("未指定训练范围")
total_tokens = total_batch_size * num_iterations  # 总训练token数
print0(f"总训练token数: {total_tokens:,}")
print0(f"Token数 : 参数数比例: {total_batch_size * num_iterations / num_params:.2f}")  # Chinchilla约为20
print0(f"总训练FLOPs估计: {num_flops_per_token * total_tokens:e}")

# -----------------------------------------------------------------------------
# 初始化优化器（线性层使用Muon，嵌入和lm_head使用AdamW）
optimizers = model.setup_optimizers(unembedding_lr=unembedding_lr, embedding_lr=embedding_lr, matrix_lr=matrix_lr, weight_decay=weight_decay)
adamw_optimizer, muon_optimizer = optimizers

# 初始化训练/验证的DataLoaders
base_dir = get_base_dir()
tokens_dir = os.path.join(base_dir, "tokenized_data")
train_loader = tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="train")
build_val_loader = lambda: tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="val")
x, y = next(train_loader) # 启动加载第一批数据

# -----------------------------------------------------------------------------
# 设置超参数调度器

# 学习率调度器
# TODO: 为AdamW参数实验短预热（期望略有改进）
warmup_ratio = 0.0 # 学习率预热的迭代比例
warmdown_ratio = 0.2 # 学习率下降的迭代比例
final_lr_frac = 0.0 # 最终学习率是初始学习率的这个分数
def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

# Muon优化器的动量调度器
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# -----------------------------------------------------------------------------
# 训练循环
min_val_bpb = float("inf")
smooth_train_loss = 0 # 训练损失的EMA
ema_beta = 0.9 # EMA衰减因子
total_training_time = 0 # 总训练时间（挂钟时间）
# 注意我们运行+1步，以便我们可以在最后评估和保存
for step in range(num_iterations + 1):
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * total_batch_size * step

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

    # 定期：估计CORE指标（所有ranks参与）
    # 使用原始未编译模型，因为输入形状不断变化
    if last_step or (step > 0 and step % core_metric_every == 0):
        model.eval()
        with autocast_ctx:
            results = evaluate_model(orig_model, tokenizer, device, max_per_task=core_metric_max_per_task)
        print0(f"步骤 {step:05d} | CORE指标: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # 定期：从模型采样（仅主进程）
    # 使用原始未编译模型，因为输入形状不断变化
    if master_process and (last_step or (step > 0 and step % sample_every == 0)):
        model.eval()
        prompts = [
            "法国的首都是",
            "金的化学符号是",
            "如果昨天是星期五，那么明天将是",
            "热的反义词是",
            "太阳系的行星是：",
            "我最喜欢的颜色是",
            "如果5*x + 3 = 13，那么x是",
        ]
        engine = Engine(orig_model, tokenizer) # 使用orig_model避免重新编译
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # 在运行结束时保存检查点（仅主进程）
    if master_process and last_step:
        output_dirname = model_tag if model_tag else f"d{depth}" # 例如 d12
        checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers], # TODO: 确保跨ranks保存正确完成
            {
                "step": step,
                "val_bpb": val_bpb, # 最后一步的损失
                "model_config": model_config_kwargs,
                "user_config": user_config, # 训练脚本的输入
                "device_batch_size": device_batch_size,
                "max_seq_len": max_seq_len,
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
    # 梯度裁剪（TODO 可能实验）
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)
    # 步进优化器
    lrm = get_lr_multiplier(step)
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

    # 日志记录
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # 训练损失的EMA
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # 去偏EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(world_tokens_per_fwdbwd / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size # bfloat16 H100 SXM且无2:4稀疏性
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100 # 以%计
    if step > 10:
        total_training_time += dt # 仅计算前10步
    print0(f"步骤 {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | 损失: {debiased_smooth_loss:.6f} | 学习率乘数: {lrm:.2f} | 时间: {dt * 1000:.2f}ms | token/秒: {tok_per_sec:,} | MFU: {mfu:.2f} | 总时间: {total_training_time/60:.2f}m")
    if step % 100 == 0:
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
from nanochat.report import get_report
get_report().log(section="基础模型训练", data=[
    user_config, # CLI参数
    { # 关于训练设置的统计信息
        "参数数量": num_params,
        "每个token的FLOPs数量": f"{num_flops_per_token:e}",
        "计算的迭代次数": num_iterations,
        "训练token数量": total_tokens,
        "Token数 : 参数数比例": total_batch_size * num_iterations / num_params,
        "DDP世界大小": ddp_world_size,
        "warmup_ratio": warmup_ratio,
        "warmdown_ratio": warmdown_ratio,
        "final_lr_frac": final_lr_frac,
    },
    { # 关于训练结果的统计信息
        "最小验证bpb": min_val_bpb,
        "最终验证bpb": val_bpb,
        "CORE指标估计": results["core_metric"],
        "MFU %": f"{mfu:.2f}%",
        "总训练FLOPs": f"{flops_so_far:e}",
        "总训练时间": f"{total_training_time/60:.2f}m",
        "峰值内存使用": f"{torch.cuda.max_memory_allocated() / 1024 / 1024:.2f}MiB",
    }
])

# 清理
wandb_run.finish() # wandb运行结束
compute_cleanup()
