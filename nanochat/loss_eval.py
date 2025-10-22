"""
一些帮助评估基础模型的函数。
"""
import math
import torch
import torch.distributed as dist

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """
    与朴素的"平均损失"不同，此函数返回每字节位数（bpb），
    这是一个与分词词汇表大小无关的指标，意味着即使更改词汇表大小，
    您仍然在比较同类事物。其工作原理是，不是像通常那样计算平均损失，
    而是计算总损失，并独立计算总字节数（所有目标token的），然后相除。
    这通过目标token表示的字节数对损失进行归一化。

    增加复杂性的原因是为了：
    1) 所有"正常"token都按其字节长度进行归一化
    2) 特殊token（例如<|bos|>）不包含在指标中 - 它们被掩码掉。
    3) 主动掩码的token（使用ignore_index，例如-1）不包含在指标中。

    除了evaluate_loss，我们还需要token_bytes张量：
    它是一个形状为(vocab_size,)的1D张量，指示每个token id的字节数，
    或者如果token不被计数（例如特殊token）则为0。
    """
    # 记录损失
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        loss2d = model(x, y, loss_reduction='none') # (B, T)
        loss2d = loss2d.view(-1) # 展平
        y = y.view(-1) # 展平
        if (y < 0).any():
            # 如果某些目标token是ignore_index（例如-1），则代码路径稍微复杂一些
            # 任何目标token < 0 都应被忽略：不要用负数索引token_bytes
            valid = y >= 0
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            # 将有效目标映射到它们的字节长度；被忽略的目标贡献0字节
            num_bytes2d = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y, dtype=token_bytes.dtype)
            )
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
        else:
            # 快速路径：没有忽略的目标，可以直接安全索引
            num_bytes2d = token_bytes[y]
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
    # 在所有rank之间求和归约
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    # 将两者移动到cpu，计算bpb并返回
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb
