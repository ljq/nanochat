from collections import deque

import torch

from nanochat.common import get_dist_info
from nanochat.dataset import parquets_iter_batched
from nanochat.tokenizer import get_tokenizer

def tokenizing_distributed_data_loader(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128):
    """从parquet文件流式传输预训练文本，标记化，生成训练批次。"""
    assert split in ["train", "val"], "split必须是'train'或'val'"
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    needed_tokens = B * T + 1 # +1是因为我们还需要最后一个token的目标
    # 获取分词器和bos token
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    # 暂存缓冲区保存一次迭代的token
    token_buffer = deque() # 我们在右侧流式传输token并从左侧弹出
    scratch = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)

    # 文档批次的无限迭代器
    def document_batches():
        while True:
            # 批次将按parquet文件的组大小迭代，通常例如1024行
            for batch in parquets_iter_batched(split=split, start=ddp_rank, step=ddp_world_size):
                # 对于分词器，我们可能希望以通常较小的批次进行，例如128行
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size]
    batches = document_batches()

    batch_index = 0
    while True:
        # 在生成之前累积足够的token用于一次迭代。
        while len(token_buffer) < needed_tokens:
            doc_batch = next(batches)
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)
            batch_index += 1
        # 将token从双端队列移动到暂存缓冲区
        for i in range(needed_tokens):
            scratch[i] = token_buffer.popleft()
        # 创建输入/目标作为1D张量
        inputs_cpu = scratch[:-1].to(dtype=torch.int32)
        targets_cpu = scratch[1:]
        # 重塑为2D并异步移动到GPU
        inputs = inputs_cpu.view(B, T).to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = targets_cpu.view(B, T).to(device="cuda", dtype=torch.int64, non_blocking=True)
        yield inputs, targets
