"""
加载检查点，并：
- 在更大的训练/验证分割上评估损失
- 从模型采样

运行示例：
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss
"""
import os
import torch
from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_init, print0, compute_cleanup
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.tokenizer import get_token_bytes
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine

# 配置
device_batch_size = 32
split_tokens = 20*524288  # 每个分割要评估的token数量
model_tag = None # 输出目录名称的可选模型标签
model_step = None # 输出目录名称的可选模型步骤
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # 从命令行或配置文件的覆盖

# 加载基础模型和分词器
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=model_tag, step=model_step)
sequence_len = meta["model_config"]["sequence_len"] # 实际上可以是任意的

# 设置我们将运行的精度
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

# 在每个分割上评估损失
tokens_per_step = device_batch_size * sequence_len * ddp_world_size
assert split_tokens % tokens_per_step == 0, "split_tokens必须能被tokens_per_step整除"
steps = split_tokens // tokens_per_step
token_bytes = get_token_bytes(device=device)
bpb_results = {}
for split_name in ["train", "val"]:
    loader = tokenizing_distributed_data_loader(device_batch_size, sequence_len, split_name)
    with autocast_ctx:
        bpb = evaluate_bpb(model, loader, steps, token_bytes)
    print0(f"{split_name} bpb: {bpb:.4f}")
    bpb_results[split_name] = bpb

# 主进程也从模型采样
samples = []
if ddp_rank == 0:
    prompts = [
        "法国的首都是",
        "金的化学符号是",
        "如果昨天是星期五，那么明天将是",
        "热的反义词是",
        "太阳系的行星是：",
        "我最喜欢的颜色是",
        "如果5*x + 3 = 13，那么x是",
    ]
    engine = Engine(model, tokenizer)
    for prompt in prompts:
        tokens = tokenizer(prompt, prepend="<|bos|>")
        with autocast_ctx:
            sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
        sample_str = tokenizer.decode(sample[0])
        print0(sample_str)
        samples.append(sample_str)

# 记录到报告
from nanochat.report import get_report
get_report().log(section="基础模型损失", data=[
    {
        "训练 bpb": bpb_results["train"],
        "验证 bpb": bpb_results["val"],
    },
    {f"样本 {i}": sample for i, sample in enumerate(samples)},
])

# 清理
compute_cleanup()
