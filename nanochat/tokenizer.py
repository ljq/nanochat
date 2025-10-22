"""
GPT-4风格的BPE分词器。

提供两种实现：
1) HuggingFace分词器，可以同时进行训练和推理，但非常令人困惑
2) 我们自己的RustBPE分词器用于训练，tiktoken用于高效推理
"""

import os
import copy
from functools import lru_cache

SPECIAL_TOKENS = [
    # 每个文档以序列开始（BOS）token开头，用于分隔文档
    "<|bos|>",
    # 下面的token仅在微调期间使用，用于将对话渲染为token id
    "<|user_start|>", # 用户消息
    "<|user_end|>",
    "<|assistant_start|>", # 助手消息
    "<|assistant_end|>",
    "<|python_start|>", # 助手调用python REPL工具
    "<|python_end|>",
    "<|output_start|>", # python REPL输出回助手
    "<|output_end|>",
]

# 注意：此分割模式与GPT-4不同，我们使用\p{N}{1,2}而不是\p{N}{1,3}
# 我这样做是因为我不想为较小的词汇表大小在数字上"浪费"太多token。
# 我还没有验证这实际上是一个好主意，TODO。
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# Generic GPT-4-style tokenizer based on HuggingFace Tokenizer
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class HuggingFaceTokenizer:
    """围绕HuggingFace分词器的轻量级包装器，提供一些实用功能"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path):
        # init from a HuggingFace pretrained tokenizer (e.g. "gpt2")
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        # init from a local directory on disk (e.g. "out/tokenizer")
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # 从文本迭代器训练
        # 配置HuggingFace分词器
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True, # 需要！
            unk_token=None,
            fuse_unk=False,
        ))
        # 标准化器：无
        tokenizer.normalizer = None
        # 预分词器：GPT-4风格
        # GPT-4使用的正则表达式模式，在BPE之前将文本分割成组
        # 注意：模式从\p{N}{1,3}更改为\p{N}{1,2}，因为我怀疑这对非常小的模型和较小的词汇表大小有害，
        # 因为它在token空间中有点浪费。
        # （但我还没有验证这一点！TODO）
        gpt4_split_regex = Regex(SPLIT_PATTERN) # huggingface要求你将其包装在Regex中！！
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        # 解码器：ByteLevel（它与ByteLevel预分词器配对）
        tokenizer.decoder = decoders.ByteLevel()
        # 后处理器：无
        tokenizer.post_processor = None
        # 训练器：BPE
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0, # 无最小频率
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
        # 开始训练
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)

    def _encode_one(self, text, prepend=None, append=None):
        # 编码单个字符串
        # prepend/append可以是特殊token的字符串或直接是token id。
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids

    def encode_special(self, text):
        # 通过精确匹配编码单个特殊token
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self):
        bos = self.encode_special("<|bos|>")
        return bos

    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        # 将分词器保存到磁盘
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"分词器已保存到 {tokenizer_path}")

# -----------------------------------------------------------------------------
# Tokenizer based on rustbpe + tiktoken combo
import pickle
import rustbpe
import tiktoken

class RustBPETokenizer:
    """围绕tiktoken的轻量级包装器（用于高效推理），但使用rustbpe进行训练"""

    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # 1) 使用rustbpe训练
        tokenizer = rustbpe.Tokenizer()
        # 特殊token稍后在__init__中插入，我们不在这里训练它们
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, f"vocab_size_no_special必须至少为256，得到{vocab_size_no_special}"
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        # 2) 为推理构建关联的tiktoken编码
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks, # dict[bytes, int] (token字节 -> 合并优先级排名)
            special_tokens=special_tokens, # dict[str, int] (特殊token名称 -> token id)
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        # https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
        enc = tiktoken.get_encoding(tiktoken_name)
        # tiktoken将特殊文档分隔符token称为"<|endoftext|>"
        # 是的，这很令人困惑，因为这个token几乎总是被PREPENDED到文档的开头
        # 它最常用于在推理期间向LLM发出新序列开始的信号等。
        # 所以在nanoChat中我们总是使用"<|bos|>"，是"beginning of sequence"的缩写，但历史上它通常被称为"<|endoftext|>"。
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        # text可以是字符串或字符串列表

        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id) # TODO: 这里有点低效？:( 嗯
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id) # TODO: 相同
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"无效输入类型: {type(text)}")

        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.enc.decode(ids)

    def save(self, tokenizer_dir):
        # 将编码对象保存到磁盘
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"分词器编码已保存到 {pickle_path}")

    def render_conversation(self, conversation, max_tokens=2048):
        """
        标记化单个聊天对话（我们在这里称之为"doc"或"document"）。
        返回：
        - ids: list[int] 是此渲染对话的token id列表
        - mask: list[int] 相同长度，mask = 1 表示助手预期要训练的token。
        """
        # 我们将返回的ids、mask和一个帮助构建它们的辅助函数。
        ids, mask = [], []
        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # 有时第一条消息是系统消息...
        # => 只需将其与第二条（用户）消息合并
        if conversation["messages"][0]["role"] == "system":
            # 目前这里需要一些对话手术...
            conversation = copy.deepcopy(conversation) # 避免改变原始对象
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", "系统消息后必须跟着用户消息"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"对话消息少于1条: {messages}"

        # 获取我们需要的所有特殊token
        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        # 现在我们可以标记化对话
        add_tokens(bos, 0)
        for i, message in enumerate(messages):

            # 这里进行一些完整性检查，以防止误用
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, f"消息 {i} 来自 {message['role']} 但应该来自 {must_be_from}"

            # content可以是简单字符串或部分列表（例如包含工具调用）
            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str), "用户消息预期只是字符串"
                value_ids = self.encode(content)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    # 简单字符串 => 只需添加token
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            # 字符串部分 => 只需添加token
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            # python工具调用 => 在<|python_start|>和<|python_end|>内添加token
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            # python输出 => 在<|output_start|>和<|output_end|>内添加token
                            # 这些token都没有监督，因为token在测试时来自Python
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"未知部分类型: {part['type']}")
                else:
                    raise ValueError(f"未知内容类型: {type(content)}")
                add_tokens(assistant_end, 1)

        # 截断到最大max_tokens个token（有助于防止OOM）
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def visualize_tokenization(self, ids, mask):
        """用于调试的小辅助函数：可视化render_conversation的标记化"""
        RED = '\033[91m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
        return '|'.join(tokens)

    def render_for_completion(self, conversation):
        """
        在强化学习期间使用。在该设置中，我们希望
        渲染对话以准备助手进行完成。
        与聊天SFT情况不同，我们不需要返回mask。
        """
        # 我们需要进行一些手术：我们需要弹出最后一条消息（助手的）
        conversation = copy.deepcopy(conversation) # 避免改变原始对象
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "最后一条消息必须来自助手"
        messages.pop() # 原地移除最后一条消息（助手的）

        # 现在标记化对话
        ids, mask = self.render_conversation(conversation)

        # 最后，为准备助手进行完成，附加助手开始token
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids

# -----------------------------------------------------------------------------
# nanochat-specific convenience functions

def get_tokenizer():
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    # return HuggingFaceTokenizer.from_directory(tokenizer_dir)
    return RustBPETokenizer.from_directory(tokenizer_dir)

def get_token_bytes(device="cpu"):
    import torch
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"在{token_bytes_path}找不到token字节？它由tok_train.py写入"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes
