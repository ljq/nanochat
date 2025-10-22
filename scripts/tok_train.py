"""
使用HuggingFace Tokenizers库训练一个分词器。
采用GPT-4分词器的风格。
"""
import os
import time
import argparse
import torch
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched

# -----------------------------------------------------------------------------
# 解析命令行参数

parser = argparse.ArgumentParser(description='训练一个BPE分词器')
parser.add_argument('--max_chars', type=int, default=10_000_000_000, help='训练的最大字符数（默认：100亿）')
parser.add_argument('--doc_cap', type=int, default=10_000, help='每个文档的最大字符数（默认：10,000）')
parser.add_argument('--vocab_size', type=int, default=65536, help='词汇表大小（默认：65536 = 2^16）')
args = parser.parse_args()
print(f"最大字符数: {args.max_chars:,}")
print(f"文档上限: {args.doc_cap:,}")
print(f"词汇表大小: {args.vocab_size:,}")

# -----------------------------------------------------------------------------
# 文本迭代器

def text_iterator():
    """
    1) 将批次展平为单个迭代器
    2) 将每个文档裁剪到args.doc_cap个字符
    3) 当我们看到args.max_chars个字符时中断
    """
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc
            if len(doc_text) > args.doc_cap:
                doc_text = doc_text[:args.doc_cap]
            nchars += len(doc_text)
            yield doc_text
            if nchars > args.max_chars:
                return
text_iter = text_iterator()

# -----------------------------------------------------------------------------
# 训练分词器
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"训练时间: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# 将分词器保存到磁盘
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)

# -----------------------------------------------------------------------------
# 快速内联完整性检查
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text

# -----------------------------------------------------------------------------
# 还有一件事：我们希望缓存从token id到该token字节数的映射
# 以便高效评估每字节位数。与典型的平均损失不同，这
# 允许我们报告一个不随分词器词汇表大小变化的损失。
# 验证集上的每字节位数是我们关心的主要指标之一。
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id] # 此token的Python字符串表示
    if token_str in special_set:
        token_bytes.append(0) # 特殊字符不计入
    else:
        id_bytes = len(token_str.encode("utf-8")) # 构成此token的字节数
        token_bytes.append(id_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"已保存token_bytes到 {token_bytes_path}")

# 记录到报告
from nanochat.report import get_report
token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
get_report().log(section="分词器训练", data=[
    vars(args), # argparse命令行参数
    {"训练时间": train_time},
    {"特殊token数量": len(special_set)},
    {
        "token字节数最小值": int(token_bytes_nonzero.min().item()),
        "token字节数最大值": int(token_bytes_nonzero.max().item()),
        "token字节数平均值": token_bytes_nonzero.mean().item(),
        "token字节数标准差": token_bytes_nonzero.std().item(),
    }
])
