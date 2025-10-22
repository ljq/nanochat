"""
用于高效推理我们模型的引擎。

一切都围绕token序列工作：
- 用户可以发送token序列到引擎
- 引擎返回下一个token

注意：
- 引擎对标记化一无所知，它纯粹是token id序列。

整个系统尽可能高效地实现。
"""

import torch
import torch.nn.functional as F
import signal
import warnings
from contextlib import contextmanager
from collections import deque
from nanochat.common import compute_init
from nanochat.checkpoint_manager import load_model

# -----------------------------------------------------------------------------
# Calculator tool helpers
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula)
    except Exception as e:
        signal.alarm(0)
        # print(f"Warning: Failed to eval {formula}, exception: {e}") # it's ok ignore wrong calculator usage
        return None

def use_calculator(expr):
    """安全地评估数学表达式。"""
    expr = expr.replace(",", "")
    if any([x not in "0123456789*+-/.() " for x in expr]): # for now disallow non-numeric chars
        return None
    if "**" in expr: # for now disallow power operator, could be very expensive
        return None
    return eval_with_timeout(expr)

# -----------------------------------------------------------------------------
class KVCache:
    """
    与GPT模型协同工作以维护KV缓存。
    注意：.pos在Transformer最后一层插入后自动前进。
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        # Each of K/V is of shape (B, H, T, D) and we have one per layer of the Transformer.
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_cache = None
        self.pos = 0 # current position in time in the cache

    def reset(self):
        self.pos = 0

    def get_pos(self):
        return self.pos

    def prefill(self, other):
        """
        使用另一个KV缓存进行预填充。可选地沿批次维度扩展。
        当我们进行批次1预填充然后想要从中并行生成多个样本时使用。
        """
        # 1) 验证形状
        assert self.kv_cache is None, "无法预填充非空KV缓存"
        assert other.kv_cache is not None, "无法使用None KV缓存进行预填充"
        for ix, (dim1, dim2) in enumerate(zip(self.kv_shape, other.kv_shape)):
            if ix in [0, 1, 3, 5]:
                # num_layers、batch_size、num_heads、head_dim必须匹配
                assert dim1 == dim2, f"批次维度不匹配: {dim1} != {dim2}"
            elif ix == 2:
                # batch_size可以扩展
                assert dim1 == dim2 or dim2 == 1, f"批次维度不匹配: {dim1} != {dim2}"
            elif ix == 4:
                # seq_len：self必须比other长
                assert dim1 >= dim2, f"序列长度不匹配: {dim1} < {dim2}"
        # 2) 初始化缓存
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        # 3) 复制数据
        self.kv_cache[:, :, :, :, :other.pos, :] = other.kv_cache
        # 4) 更新pos
        self.pos = other.pos

    def insert_kv(self, layer_idx, k, v):
        # 在这里延迟初始化缓存，因为我们需要知道dtype/device
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
        # 将新的键/值插入缓存并返回到目前为止的完整缓存
        B, H, T_add, D = k.size()
        t0, t1 = self.pos, self.pos + T_add
        # 如果需要，动态增长缓存
        if t1 > self.kv_cache.size(4):
            t_needed = t1 + 1024 # 我们需要的大小加上1024的缓冲区
            t_needed = (t_needed + 1023) & ~1023 # 然后向上取整到1024的最近倍数
            current_shape = list(self.kv_cache.shape)
            current_shape[4] = t_needed
            self.kv_cache.resize_(current_shape)
        # 将k、v插入缓存
        self.kv_cache[layer_idx, 0, :, :, t0:t1] = k
        self.kv_cache[layer_idx, 1, :, :, t0:t1] = v
        # 返回到当前位置的完整缓存键/值（作为视图）
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1]
        # 在Transformer最后一层处理后递增pos
        if layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1
        return key_view, value_view


# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """从给定形状为(B, vocab_size)的logits中采样单个下一个token。返回(B, 1)。"""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)

# -----------------------------------------------------------------------------

class RowState:
    # 生成期间每行状态跟踪
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or [] # 此行的当前token序列
        self.forced_tokens = deque() # 要强制注入的token队列
        self.in_python_block = False # 我们是否在python块内
        self.python_expr_tokens = [] # 当前python表达式的token
        self.completed = False # 此行是否已完成生成

class Engine:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer # needed for tool use

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """与generate相同，但执行单次预填充然后克隆KV缓存。"""
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # 获取我们需要协调工具使用状态机的特殊token
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>") # 如果采样到，结束行
        bos = self.tokenizer.get_bos_token_id() # 如果采样到，结束行

        # 1) 运行批次1的提示token预填充
        m = self.model.config
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            **kv_model_kwargs,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :]
        next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
        sampled_tokens = next_ids[:, 0].tolist()

        # 2) 为每个样本/行复制KV缓存
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill # 不需要保留此内存

        # 3) 为每个样本初始化状态
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) 主生成循环
        num_generated = 0
        first_iteration = True
        while True:
            # 停止条件：我们已达到最大token数
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # 停止条件：所有行都已完成
            if all(state.completed for state in row_states):
                break

            # 获取采样的token - 要么来自预填充，要么来自前向传递
            if first_iteration:
                # 使用我们已经从预填充中采样的token
                sampled

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        非流式批次生成，只返回最终的token序列。
        返回token序列列表（整数列表的列表）。
        终端token（assistant_end、bos）不包含在结果中。
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # 如果所有行都完成则停止
            if all(completed):
                break
        return results, masks


if __name__ == "__main__":
    """
    快速内联测试以确保naive/slow的model.generate函数
    与此处更快的Engine.generate函数等效。
    """
    import time
    # 初始化计算
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    # 加载模型和分词器
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    # 通用超参数
    kwargs = dict(max_tokens=64, temperature=0.0)
    # 设置起始提示
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
    # 使用model.generate()函数生成参考序列
    generated_tokens = []
    torch.cuda.synchronize()
    t0 = time.time()
    stream = model.generate(prompt_tokens, **kwargs)
    for token in stream:
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"参考时间: {t1 - t0:.2f}s")
    reference_ids = generated_tokens
    # 使用Engine生成token
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs) # 注意：在fp32中运行
    torch.cuda.synchronize()
    t0 = time.time()
    for token_column, token_masks in stream:
        token = token_column[0] # 只打印第一行
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"引擎时间: {t1 - t0:.2f}s")
    # 比较两个序列
    for i in range(len(reference_ids)):
        if reference_ids[i] != generated_tokens[i]:
            print(f"在{i}处不匹配: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"匹配: {reference_ids == generated_tokens}")
