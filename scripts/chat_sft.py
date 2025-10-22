"""
将基础模型微调为聊天模型。
在单个GPU上运行，例如用于调试：

python -m scripts.chat_sft

或使用torchrun进行训练：

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import wandb
import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, get_base_dir, print0, DummyWandb
from nanochat.checkpoint_manager import load_model
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.engine import Engine
from scripts.chat_eval import run_chat_eval

from tasks.common import TaskMixture
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.smoltalk import SmolTalk

# -----------------------------------------------------------------------------
# SFT超参数
run = "dummy" # wandb运行名称默认值（"dummy"是特殊的 - 我们不会记录到wandb）
# 输入模型选项
source = "mid" # base|mid，从哪个检查点加载模型（基础模型或中期训练模型）
model_tag = None # 从哪个模型标签加载模型（基础模型或中期训练模型）
step = None # 从哪个步骤加载模型（基础模型或中期训练模型）
# 计算/精度
dtype = "bfloat16"
device_batch_size = 4 # 最大值以避免内存不足
# 优化
num_epochs = 1
max_iterations = -1 # 覆盖迭代次数（-1 = 使用num_epochs * num_iterations）
target_examples_per_step = 32
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.02
# 评估和日志记录
eval_every = 100
eval_steps = 100
eval_metrics_every = 200
# 现在允许通过配置器覆盖设置
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # 从命令行或配置文件覆盖
user_config = {k: globals()[k] for k in config_keys} # 可能对日志记录有用
# -----------------------------------------------------------------------------

# 计算初始化
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0
dtype = torch.float32 if dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)

# wandb日志记录初始化
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-sft", name=run, config=user_config, save_code=True)

# 加载模型和分词器
model, tokenizer, meta = load_model(source, device, phase="train", model_tag=model_tag, step=step)
orig_model = model # 原始，未编译的模型
# model = torch.compile(model, dynamic=True) # 由于输入长度可变，效果不是很好
engine = Engine(model, tokenizer) # 仅用于内联模型评估

# -----------------------------------------------------------------------------
# 我们将训练的任务数据混合

train_ds = TaskMixture([
    ARC(subset="ARC-Easy", split="train"), # 2.3K行
    ARC(subset="ARC-Challenge", split="train"), # 1.1K行
    GSM8K(subset="main", split="train"), # 8K行
    SmolTalk(split="train", stop=10_000), # 10K行smoltalk
]) # 2.3K + 1.1K + 8K + 10K = 21.4K行
val_ds = SmolTalk(split="test") # 一般对话，24K行（尽管我们实际上不使用全部）

# -----------------------------------------------------------------------------
# 数据加载器

def sft_data_generator(dataset, batch_size):
    pad_token_id = tokenizer.encode_special("<|assistant_end|>") # 使用<|assistant_end|>作为填充token是可以的，这些位置在损失中被掩码
    # 准备一批标记化的对话并生成
    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids, mask in batch) - 1 # n的序列创建n-1的输入/目标
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long) # -1是忽略索引
        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, :n-1] = ids_tensor[:-1]
            # 回忆-1是忽略索引，所以在掩码为0的地方掩码目标
            row_targets = ids_tensor[1:]
            # mask[1:]省略了BOS token的掩码，它目前从来不是目标，所以没问题
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1 # 在掩码为0的地方掩码目标
            targets[i, :n-1] = row_targets
        inputs = inputs.to(device) # 移动到设备
        targets = targets.to(device)
        return inputs, targets
    # 在epoch中迭代数据集，标记化
    batch = []
    while True:
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc)
            batch.append((ids, mask))
            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []

examples_per_step = device_batch_size * ddp_world_size
print0(f"Target examples per step: {target_examples_per_step}")
print0(f"Device batch size: {device_batch_size}")
print0(f"Examples per step is device_batch_size * ddp_world_size: {examples_per_step}")
assert target_examples_per_step % examples_per_step == 0, "Target examples per step must be divisible by examples per step"
grad_accum_steps = target_examples_per_step // examples_per_step
print0(f"=> Setting grad accum steps: {grad_accum_steps}")

num_iterations = (len(train_ds) // target_examples_per_step) * num_epochs
if max_iterations >= 0 and num_iterations > max_iterations:
    print0(f"Number of iterations is too high: {num_iterations}, capping to {max_iterations}")
    num_iterations = max_iterations
train_loader = sft_data_generator(train_ds, batch_size=device_batch_size)
build_val_loader = lambda: sft_data_generator(val_ds, batch_size=device_batch_size)

# -----------------------------------------------------------------------------
# 初始化优化器

optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay,
)
# 将初始学习率设置为基础学习率的一部分
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * init_lr_frac
        group["initial_lr"] = group["lr"] # 保存初始学习率，以便稍后轻松衰减

# -----------------------------------------------------------------------------
# 训练循环

# 学习率调度器
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_iterations
    return lrm

# 开始！
step = 0
train_iter = iter(train_loader)
for step in range(num_iterations):
    last_step = step == num_iterations - 1

    # 评估验证损失
    if last_step or step % eval_every == 0:
        model.eval()
        val_iter = iter(build_val_loader())
        losses = []
        for _ in range(eval_steps):
            val_inputs, val_targets = next(val_iter)
            with torch.no_grad(), autocast_ctx:
                loss = model(val_inputs, val_targets)
            losses.append(loss)
        val_loss = torch.stack(losses).mean() # 在eval_steps上平均
        if ddp:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG) # 在ranks上平均
        val_loss = val_loss.item()
        print0(f"Step {step:05d} | Validation loss: {val_loss:.6f}")
        wandb_run.log({
            "step": step,
            "val_loss": val_loss,
        })
        model.train()

    # 评估多项选择任务的准确性（运行速度快）
    if last_step or (step > 0 and step % eval_metrics_every == 0):
        model.eval()
        metrics = {}
        with torch.no_grad(), autocast_ctx:
            # 注意，由于这些在no_grad内部，我们通常可以负担至少约2X的批次大小
            metrics["mmlu_acc"] = run_chat_eval("MMLU", model, tokenizer, engine, batch_size=device_batch_size*2, max_problems=1024)
            metrics["arc_easy_acc"] = run_chat_eval("ARC-Easy", model, tokenizer, engine, batch_size=device_batch_size*2, max_problems=1024)
        metrics_str = ', '.join(f'{k}: {v:.6f}' for k, v in metrics.items())
        print0(f"Step {step:05d} | {metrics_str}")
        wandb_run.log({
            "step": step,
            **metrics,
        })
        model.train()

    if last_step:
        break

    # 评估梯度
    num_tokens = torch.tensor(0, device=device) # 看到的"活跃"监督token数量
    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets = next(train_iter)
        with autocast_ctx:
            loss = model(train_inputs, train_targets)
        train_loss = loss.detach() # 用于日志记录
        loss = loss / grad_accum_steps # 每个.backward()是一个梯度和 => 在此处归一化损失
        loss.backward() # 累积梯度
        num_tokens += (train_targets >= 0).sum()
    if ddp:
        dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM) # 在ranks上求和

    # 学习率调度器
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm

    # 步进优化器
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    # 日志记录
    train_loss_item = train_loss.item()
    num_tokens_item = num_tokens.item()
    print0(f"Step {step:05d}/{num_iterations:05d} | Training loss: {train_loss_item:.6f}| lrm: {lrm:.6f}| num_tokens: {num_tokens_item:,}")
    wandb_run.log({
        "step": step,
        "lrm": lrm,
        "train_loss": train_loss_item,
        "num_tokens": num_tokens_item,
    })
    step += 1

# 在运行结束时保存模型
if master_process:
    base_dir = get_base_dir()
    depth = model.config.n_layer
    model_tag = f"d{depth}" # base the model tag on the depth of the base model
    checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", model_tag)
    model_config_kwargs = model.config.__dict__ # slightly naughty, abusing the simplicity of GPTConfig, TODO nicer
    save_checkpoint(
        checkpoint_dir,
        step,
        model.state_dict(),
        None, # note: we don't bother to save the optimizer state
        {
            "step": step,
            "val_loss": val_loss,
            **metrics,
            "model_config": model_config_kwargs,
        }
    )
    print(f"✅ Saved model checkpoint to {checkpoint_dir}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Chat SFT", data=[
    user_config, # CLI args
    {
        "Training rows": len(train_ds),
        "Number of iterations": num_iterations,
        "Training loss": train_loss_item,
        "Validation loss": val_loss,
    },
])

# Cleanup
wandb_run.finish()
compute_cleanup()
