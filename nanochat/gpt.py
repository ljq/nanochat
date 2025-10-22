"""
GPT模型（重写版本，更加简洁）
主要特性：
- 旋转位置编码（无位置嵌入）
- QK归一化
- token嵌入和lm_head使用非绑定权重
- MLP中使用relu^2激活函数
- token嵌入后进行归一化
- rmsnorm中无可学习参数
- 线性层无偏置
- 支持多查询注意力（MQA）以实现更高效的推理
"""

import math
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW

@dataclass
class GPTConfig:
    """GPT模型配置类"""
    sequence_len: int = 1024  # 序列长度
    vocab_size: int = 50304   # 词汇表大小
    n_layer: int = 12         # Transformer层数
    n_head: int = 6           # 查询头数量
    n_kv_head: int = 6        # 键/值头数量（用于MQA）
    n_embd: int = 768         # 嵌入维度


def norm(x):
    """纯函数式rmsnorm，无可学习参数"""
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    """应用旋转位置编码"""
    assert x.ndim == 4  # 多头注意力输入应为4维
    d = x.shape[3] // 2  # 将最后一个维度分成两半
    x1, x2 = x[..., :d], x[..., d:]  # 将最后一个维度分成两半
    y1 = x1 * cos + x2 * sin  # 旋转维度对
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)  # 重新组装
    out = out.to(x.dtype)  # 确保输入输出数据类型匹配
    return out


def repeat_kv(x, n_rep):
    """重复键值头以支持多查询注意力（MQA）"""
    if n_rep == 1:
        return x
    bs, n_kv_heads, slen, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


class CausalSelfAttention(nn.Module):
    """因果自注意力层，支持多查询注意力（MQA）"""
    
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx  # 层索引，用于KV缓存
        self.n_head = config.n_head  # 查询头数量
        self.n_kv_head = config.n_kv_head  # 键值头数量
        self.n_embd = config.n_embd  # 嵌入维度
        self.head_dim = self.n_embd // self.n_head  # 每个头的维度
        assert self.n_embd % self.n_head == 0  # 确保嵌入维度能被头数整除
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0  # MQA约束
        # 线性投影层：查询、键、值、输出投影
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        """前向传播"""
        B, T, C = x.size()  # 批次大小、序列长度、通道数

        # 投影输入以获取查询、键和值
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # 应用旋转位置编码到查询和键以获取相对位置编码
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)  # QK旋转嵌入
        q, k = norm(q), norm(k)  # QK归一化
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # 将头维度变为批次维度：(B, T, H, D) -> (B, H, T, D)

        # 应用KV缓存：将当前k,v插入缓存，获取到目前为止的完整视图
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2)  # 本次前向传播中的查询数量
        Tk = k.size(2)  # 总的键/值数量（缓存中 + 本次前向传播）

        # 应用MQA：为每个查询头复制键/值头
        nrep = self.n_head // self.n_kv_head
        k, v = repeat_kv(k, nrep), repeat_kv(v, nrep)

        # 注意力机制：查询自回归地关注键/值。需要处理几种情况：
        if kv_cache is None or Tq == Tk:
            # 训练期间（无KV缓存），使用因果注意力
            # 即使有KV缓存，当Tq == Tk时也可以使用这个简单版本
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif Tq == 1:
            # 推理期间但本次前向传播只有一个查询：
            # 查询必须关注缓存中的所有键/值
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            # 推理期间且本次前向传播有一组查询：
            # 首先，每个查询关注所有缓存的键/值（即完整前缀）
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)  # True = 保留，False = 掩码
            prefix_len = Tk - Tq
            if prefix_len > 0:  # 不能为负但可能为零
                attn_mask[:, :prefix_len] = True
            # 然后，在这个块内使用因果注意力
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        # 重新组装头并投影回残差流
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """多层感知机（MLP）模块，使用relu^2激活函数"""
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)  # 前向投影
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)  # 反向投影

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # relu^2激活函数
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """Transformer块，包含自注意力层和MLP层"""
    
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)  # 因果自注意力层
        self.mlp = MLP(config)  # 多层感知机层

    def forward(self, x, cos_sin, kv_cache):
        """前向传播，使用残差连接"""
        x = x + self.attn(norm(x), cos_sin, kv_cache)  # 注意力残差连接
        x = x + self.mlp(norm(x))  # MLP残差连接
        return x


class GPT(nn.Module):
    """GPT模型主类"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),  # token嵌入层
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),  # Transformer块
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # 语言模型头
        # 为了支持元设备初始化，我们在这里初始化旋转嵌入，但这是假的
        # 对于rotary_seq_len，这些旋转嵌入在内存中很小/便宜，
        # 所以我们只是过度计算它们，但如果达到该数量则断言失败。
        # 将来我们可以动态增长缓存，现在这样就可以了。
        self.rotary_seq_len = config.sequence_len * 10  # 10倍过度计算应该足够，TODO：使其更优雅？
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)  # persistent=False表示不保存到检查点
        self.register_buffer("sin", sin, persistent=False)
        # 将嵌入从fp32转换为bf16：优化器可以容忍它，并且节省内存：在模型和激活中
        self.transformer.wte.to(dtype=torch.bfloat16)

    def init_weights(self):
        """初始化模型权重"""
        self.apply(self._init_weights)
        # 将分类器权重置零
        torch.nn.init.zeros_(self.lm_head.weight)
        # 将所有块中的c_proj权重置零
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        # 初始化旋转嵌入
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

    def _init_weights(self, module):
        """权重初始化函数"""
        if isinstance(module, nn.Linear):
            # 参考：https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)  # 输出维度
            fan_in = module.weight.size(1)   # 输入维度
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    # TODO: 增加基础theta，例如100K最近更常见
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        """预计算旋转位置编码"""
        # 从模型嵌入自动检测设备
        if device is None:
            device = self.transformer.wte.weight.device
        # 通道步长
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))  # 逆频率
        # 时间步长
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # 计算每个（时间，通道）对的旋转频率
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()  # 保持为bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]  # 添加批次和头维度以便后续广播
        return cos, sin

    def get_device(self):
        """获取模型所在的设备"""
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """返回模型每个token的估计FLOPs。参考：https://arxiv.org/abs/2204.02311"""
        nparams = sum(p.numel() for p in self.parameters())  # 总参数数量
        nparams_embedding = self.transformer.wte.weight.numel()  # 嵌入参数数量
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t  # FLOPs计算公式
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        """设置优化器，使用AdamW和Muon优化器的组合"""
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # 将所有参数分成3组（矩阵、嵌入、lm_head）
        matrix_params = list(self.transformer.h.parameters())  # Transformer块参数
        embedding_params = list(self.transformer.wte.parameters())  # token嵌入参数
        lm_head_params = list(self.lm_head.parameters())  # 语言模型头参数
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)
        # 为嵌入和lm_head创建AdamW优化器
        # 将AdamW参数的LR按∝1/√dmodel缩放（已为768维模型调整LR）
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"将AdamW参数的LR按∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}缩放")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),  # 解嵌入层
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),  # 嵌入层
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # 为线性层创建Muon优化器
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # 将两个优化器组合成一个列表
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]  # 记录初始学习率
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        """前向传播"""
        B, T = idx.size()  # 批次大小、序列长度

        # 获取当前序列长度的旋转嵌入（形状为(1, seq_len, 1, head_dim)）
        assert T <= self.cos.size(1), f"序列长度超过了旋转嵌入缓存：{T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"旋转嵌入和idx在不同的设备上：{idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "旋转嵌入必须为bfloat16"
        # 如果存在KV缓存，我们需要将旋转嵌入偏移到缓存中的当前位置
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]  # 将缓存截断到当前序列长度

        # 前向传播Transformer主干
        x = self.transformer.wte(idx)  # token嵌入
        x = norm(x)  # 归一化
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)  # 通过每个Transformer块
        x = norm(x)  # 最终归一化

        # 前向传播lm_head（计算logits）
        softcap = 15  # logits软上限
        if targets is not None:
            # 训练模式：计算并返回损失
            # TODO：实验Liger Kernels / 分块交叉熵等
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap)  # logits软上限
            logits = logits.float()  # 使用tf32/fp32计算logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # 推理模式：计算并返回logits
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap)  # logits软上限
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        简单的自回归流式推理。
        为了使其超级简单，我们假设：
        - 批次大小为1
        - ids和生成的token是简单的Python列表和整数
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)  # 添加批次维度
        for _ in range(max_tokens):
            logits = self.forward(ids)  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size) 只取最后一个位置的logits
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))  # 获取top-k值
                logits[logits < v[:, [-1]]] = -float('Inf')  # 将非top-k的logits设为负无穷
            if temperature > 0:
                logits = logits / temperature  # 温度缩放
                probs = F.softmax(logits, dim=-1)  # 转换为概率
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)  # 采样
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)  # 贪婪解码
            ids = torch.cat((ids, next_ids), dim=1)  # 将新token添加到序列中
            token = next_ids.item()
            yield token  # 流式生成token
