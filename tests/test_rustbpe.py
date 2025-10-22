"""
比较以下分词器的训练：

1. （非常慢）Python参考实现
2. 优化的Python实现
3. HuggingFace分词器训练实现
4. 我们自己的自定义RustBPE训练实现

所有这些都应该计算相同的合并并产生
相同的词汇表和分词结果。

最后，为了效率，我们将使用tiktoken进行推理。
因此，我们希望确保可以将我们的rustbpe分词器
导出到tiktoken，并在推理中使用相同的结果。

运行方式：
python -m pytest tests/test_rustbpe.py -v -s
-v 是详细模式，-s 是显示打印
"""

import regex as re
from collections import Counter, defaultdict
import time
import rustbpe
import tiktoken
import pytest

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# Reference tokenizer, pretty much copy pasted and pruned a bit from minbpe

def get_stats(ids, counts=None):
    """
    给定一个整数列表，返回连续对的计数字典
    示例：[1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    可选地允许更新现有的计数字典
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """
    在整数列表（ids）中，将所有连续出现的
    对替换为新的整数token idx
    示例：ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

class RegexTokenizer:
    """基于正则表达式的分词器参考实现"""

    def __init__(self, pattern=None):
        """
        - pattern: 可选字符串，用于覆盖默认值（GPT-4分割模式）
        - special_tokens: str -> int 特殊token字典
          示例：{'<|endoftext|>': 100257}
        """
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.merges = {} # (int, int) -> int
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.vocab = self._build_vocab()

    def _build_vocab(self):
        """词汇表简单且确定性地从合并中派生"""
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def train(self, text, vocab_size, verbose=False):
        """训练分词器"""
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # 跟踪训练期间合并是否在任何时候是模糊的（对的计数不唯一）
        ambiguous = False

        # 将文本分割成文本块
        text_chunks = re.findall(self.compiled_pattern, text)

        # 输入文本预处理
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        # 迭代合并最常见的对以创建新token
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
        for i in range(num_merges):
            # 计算每个连续对出现的次数
            stats = {}
            for chunk_ids in ids:
                # 传入stats将在原地更新它，累加计数
                get_stats(chunk_ids, stats)
            # 找到计数最高的对
            pair = max(stats, key=stats.get)
            # 检查合并是否模糊 - 即最大值不唯一
            pair_count = stats[pair]
            pairs_with_max_count = [pair for pair, count in stats.items() if count == pair_count]
            if len(pairs_with_max_count) > 1:
                # 打印前10对及其计数
                # print(f"{i} Merge is ambiguous! {pair} has {pair_count} occurrences")
                # for print_pair, print_count in sorted(stats.items(), key=lambda x: x[1], reverse=True)[:10]:
                #     print(f"{print_pair}: {print_count}")
                ambiguous = True
            # 创建一个新token：分配下一个可用的id
            idx = 256 + i
            # 将ids中所有出现的对替换为idx
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            # 保存合并
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # 打印
            if verbose:
                print(f"合并 {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) 有 {stats[pair]} 次出现")

        # 保存类变量
        self.merges = merges # 在encode()中使用
        self.vocab = vocab   # 在decode()中使用
        return ambiguous

    def _encode_chunk(self, text_bytes):
        """返回token id"""
        # 开始。首先，将所有字节转换为0..255范围内的整数
        ids = list(text_bytes)
        while len(ids) >= 2:
            # 找到合并索引最低的对
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # 微妙之处：如果没有更多可用的合并，键将为
            # 每个对产生inf，而min将是
            # 列表中的第一个对，任意地
            # 我们可以通过成员资格检查来检测这个终止情况
            if pair not in self.merges:
                break # 没有其他可以合并的了
            # 否则让我们合并最佳对（最低合并索引）
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """忽略任何特殊token的编码。"""
        # 根据正则表达式模式定义的类别将文本分割成文本块
        text_chunks = re.findall(self.compiled_pattern, text)
        # 所有文本块分别编码，然后结果连接
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # 原始字节
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

# -----------------------------------------------------------------------------
# Faster Python tokenizer, optimized version of the reference tokenizer

def fast_merge_inplace(ids, pair, idx):
    """
    在整数列表（ids）中，将所有连续出现的
    对替换为新的整数token idx（原地操作）
    示例：ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    # Find all positions where the pair occurs
    i = 0
    while i < len(ids) - 1:
        if ids[i] == pair[0] and ids[i+1] == pair[1]:
            ids[i] = idx
            ids.pop(i+1)
        else:
            i += 1
    return ids


class FastRegexTokenizer:
    """优化的基于正则表达式的分词器"""

    def __init__(self, pattern=None):
        """
        - pattern: 可选字符串，用于覆盖默认值（GPT-4分割模式）
        - special_tokens: str -> int 特殊token字典
          示例：{'<|endoftext|>': 100257}
        """
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.merges = {}
        self.vocab = self._build_vocab()

    def _build_vocab(self):
        """词汇表简单且确定性地从合并中派生"""
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def train(self, text, vocab_size, verbose=False):
        """
        引入了许多优化：
        - 通过内联函数删除函数调用开销
        - 使用.pop()原地修改ids列表而不是创建新列表
        - 将相同的块折叠为唯一的块
        - 更聪明地更新计数 - 仅围绕受影响的块
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # many, many chunks are identical, so we can "collapse" them to just the unique ones
        counts = Counter(text_chunks)
        unique_chunks = [ch for ch, count in counts.items()]
        chunk_counts = [count for ch, count in counts.items()]

        # input text preprocessing
        ids = [list(ch.encode("utf-8")) for ch in unique_chunks]
        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes

        # Initial count: build stats and position tracking
        stats = defaultdict(int)
        positions = defaultdict(set)  # pair -> set of chunk indices that contain this pair

        for chunk_idx, (chunk_ids, count) in enumerate(zip(ids, chunk_counts)):
            for pair in zip(chunk_ids, chunk_ids[1:]):
                stats[pair] += count
                positions[pair].add(chunk_idx)

        for i in range(num_merges):
            if not stats:
                break

            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i

            # Get chunks that contain this pair
            affected_chunks = positions[pair]

            # Track count changes for incremental update
            count_changes = defaultdict(int)

            # Replace all occurrences of pair in affected chunks only
            for chunk_idx in affected_chunks:
                chunk_ids = ids[chunk_idx]
                chunk_count = chunk_counts[chunk_idx]
                ix = 0
                while ix < len(chunk_ids) - 1:
                    if chunk_ids[ix] == pair[0] and chunk_ids[ix+1] == pair[1]:
                        # Track what pairs are being removed/added
                        # Remove: (prev, A), (A, B), (B, next)
                        if ix > 0:
                            old_left = (chunk_ids[ix-1], chunk_ids[ix])
                            count_changes[old_left] -= chunk_count

                        # The merged pair disappears
                        count_changes[pair] -= chunk_count

                        if ix + 2 < len(chunk_ids):
                            old_right = (chunk_ids[ix+1], chunk_ids[ix+2])
                            count_changes[old_right] -= chunk_count

                        # Apply the merge
                        chunk_ids[ix] = idx
                        chunk_ids.pop(ix+1)

                        # Add: (prev, C), (C, next)
                        if ix > 0:
                            new_left = (chunk_ids[ix-1], chunk_ids[ix])
                            count_changes[new_left] += chunk_count

                        if ix + 1 < len(chunk_ids):
                            new_right = (chunk_ids[ix], chunk_ids[ix+1])
                            count_changes[new_right] += chunk_count
                    else:
                        ix += 1

            # Apply incremental changes to stats and positions
            for changed_pair, delta in count_changes.items():
                if changed_pair == pair:
                    # The merged pair should disappear completely
                    continue

                stats[changed_pair] += delta

                # Update positions for changed pairs - only check affected chunks
                for chunk_idx in affected_chunks:
                    chunk_ids = ids[chunk_idx]
                    contains_pair = any((chunk_ids[j], chunk_ids[j+1]) == changed_pair
                                      for j in range(len(chunk_ids) - 1))
                    if contains_pair:
                        positions[changed_pair].add(chunk_idx)
                    else:
                        positions[changed_pair].discard(chunk_idx)

            # Remove the merged pair completely
            del stats[pair]
            del positions[pair]

            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def register_special_tokens(self, special_tokens):
        """注册特殊token"""
        # special_tokens 是一个 str -> int 的字典
        # 示例：{"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        """给定ids（整数列表），返回Python字符串"""
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes):
        """返回token id"""
        # 开始。首先，将所有字节转换为0..255范围内的整数
        ids = list(text_bytes)
        while len(ids) >= 2:
            # 找到合并索引最低的对
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # 微妙之处：如果没有更多可用的合并，键将为
            # 每个对产生inf，而min将是
            # 列表中的第一个对，任意地
            # 我们可以通过成员资格检查来检测这个终止情况
            if pair not in self.merges:
                break # 没有其他可以合并的了
            # 否则让我们合并最佳对（最低合并索引）
            idx = self.merges[pair]
            ids = fast_merge_inplace(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """忽略任何特殊token的编码。"""
        # 根据正则表达式模式定义的类别将文本分割成文本块
        text_chunks = re.findall(self.compiled_pattern, text)
        # 所有文本块分别编码，然后结果连接
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # 原始字节
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

# -----------------------------------------------------------------------------
# HuggingFace tokenizer
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class HuggingFaceTokenizer:
    """HuggingFace分词器的轻量包装器，用于一些实用功能"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        """从文本迭代器训练"""
        # 配置HuggingFace分词器
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True, # 需要！
            unk_token=None,
            fuse_unk=False,
        ))
        # 标准化器：无
        tokenizer.normalizer = None
        # 预分词器：GPT-4风格
        gpt4_split_regex = Regex(GPT4_SPLIT_PATTERN) # huggingface要求你将其包装在Regex中！！
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
            special_tokens=[], # 无特殊token
        )
        # 开始训练
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def encode_ordinary(self, text):
        """忽略任何特殊token的编码。"""
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        return ids

# -----------------------------------------------------------------------------
# Test all of the above

@pytest.fixture(scope="module")
def enwik8_path():
    """下载并缓存enwik8数据集的fixture。"""
    import os
    import zipfile
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    # download and unzip enwik8 to .cache directory
    enwik8_url = "https://mattmahoney.net/dc/enwik8.zip"
    enwik8_local_path = os.path.join(base_dir, "enwik8")
    enwik8_local_path_zip = os.path.join(base_dir, "enwik8.zip")
    if not os.path.exists(enwik8_local_path):
        print(f"Downloading enwik8 to {enwik8_local_path_zip}")
        import requests
        response = requests.get(enwik8_url)
        with open(enwik8_local_path_zip, "wb") as f:
            f.write(response.content)
        with zipfile.ZipFile(enwik8_local_path_zip, "r") as zip_ref:
            zip_ref.extractall(base_dir)
        print(f"Unzipped enwik8 to {enwik8_local_path}")
        os.remove(enwik8_local_path_zip)
        print(f"Removed {enwik8_local_path_zip}")
    else:
        print(f"Using existing enwik8 at {enwik8_local_path}")
    return enwik8_local_path


@pytest.fixture(scope="module")
def enwik8_small(enwik8_path):
    """提供100KB enwik8用于快速测试的fixture。"""
    with open(enwik8_path, "r") as f:
        return f.read(100_000)

@pytest.fixture(scope="module")
def enwik8_large(enwik8_path):
    """提供10MB enwik8用于性能测试的fixture。"""
    with open(enwik8_path, "r") as f:
        return f.read(10**7)

def time_function(func, *args, **kwargs):
    """计时函数调用并返回结果和经过的时间"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed = end_time - start_time
    return result, elapsed

def test_correctness(enwik8_small):
    """测试所有分词器实现产生相同的结果。"""
    text = enwik8_small
    encode_text = text
    vocab_size = 256 + 20  # 20 merges

    # Train slow reference
    print("\nTraining slow reference...")
    slow_reference_tokenizer = RegexTokenizer()
    ambiguous_flag, slow_reference_train_time = time_function(slow_reference_tokenizer.train, text, vocab_size)
    slow_reference_ids, slow_reference_encode_time = time_function(slow_reference_tokenizer.encode_ordinary, encode_text)
    print(f"Slow reference train time: {slow_reference_train_time:.4f}s")
    print(f"Slow reference encode time: {slow_reference_encode_time:.4f}s")
    print(slow_reference_ids[:20])

    if ambiguous_flag:
        print("‼️ WARNING: merge order was detected to be ambiguous given current text and vocab size")
        print("The implementation could be correct but we might see different results below")
    else:
        print("✅ Merge order is NOT ambiguous")

    # Train fast reference
    print("\nTraining fast reference...")
    fast_reference_tokenizer = FastRegexTokenizer()
    _, fast_reference_train_time = time_function(fast_reference_tokenizer.train, text, vocab_size)
    fast_reference_ids, fast_reference_encode_time = time_function(fast_reference_tokenizer.encode_ordinary, encode_text)
    print(f"Fast reference train time: {fast_reference_train_time:.4f}s")
    print(f"Fast reference encode time: {fast_reference_encode_time:.4f}s")
    print(fast_reference_ids[:20])

    # Assert fast equals slow
    assert fast_reference_ids == slow_reference_ids, "Fast reference should match slow reference"
    print("✅ Fast == Slow")

    # Train HuggingFace
    print("\nTraining HuggingFace...")
    hf_tokenizer, hf_train_time = time_function(HuggingFaceTokenizer.train_from_iterator, [text], vocab_size)
    hf_ids, hf_encode_time = time_function(hf_tokenizer.encode_ordinary, encode_text)
    print(f"HuggingFace train time: {hf_train_time:.4f}s")
    print(f"HuggingFace encode time: {hf_encode_time:.4f}s")
    print(hf_ids[:20])

    # HuggingFace有不同的字节顺序，所以我们需要自定义匹配
    def custom_match(ids1, ids2):
        """自定义匹配函数，处理HuggingFace的字节顺序差异"""
        perm = {}
        for x, y in zip(ids1, ids2):
            if x < 256:
                if x in perm:
                    if perm[x] != y:
                        return False
                perm[x] = y
            if x >= 256 and x != y:
                return False
        return True

    assert custom_match(hf_ids, fast_reference_ids), "HuggingFace should match fast reference"
    print("✅ HuggingFace == Fast")

    # Finally use our own Rust implementation
    print("\nTraining rustbpe...")
    rustbpe_tokenizer = rustbpe.Tokenizer()
    _, rustbpe_train_time = time_function(rustbpe_tokenizer.train_from_iterator, [text], vocab_size)
    rustbpe_ids, rustbpe_encode_time = time_function(rustbpe_tokenizer.encode, encode_text)
    print(f"RustBPE train time: {rustbpe_train_time:.4f}s")
    print(f"RustBPE encode time: {rustbpe_encode_time:.4f}s")
    print(rustbpe_ids[:20])

    assert rustbpe_ids == fast_reference_ids, "RustBPE should match fast reference"
    print("✅ RustBPE == Fast")

    # Now export rustbpe to tiktoken for more efficient inference
    print("\nTesting tiktoken export...")
    pattern = rustbpe_tokenizer.get_pattern()
    mergeable_ranks_list = rustbpe_tokenizer.get_mergeable_ranks()
    mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens={},
    )
    tiktoken_ids, tiktoken_encode_time = time_function(enc.encode, encode_text)
    print(f"Tiktoken encode time: {tiktoken_encode_time:.4f}s")
    print(tiktoken_ids[:20])

    assert tiktoken_ids == rustbpe_ids, "Tiktoken should match RustBPE"
    print("✅ Tiktoken == RustBPE")


@pytest.mark.slow
def test_training_performance(enwik8_large):
    """使用更大的数据集比较优化分词器（Python、Rust、HuggingFace）的训练速度。"""
    text = enwik8_large
    vocab_size = 2048
    print(f"\nText length: {len(text)}")

    # Commenting out because it's just way too slow to matter
    # Train optimized python version
    # print("Training optimized python version...")
    # optimized_python_tokenizer = FastRegexTokenizer()
    # _, optimized_python_train_time = time_function(optimized_python_tokenizer.train, text, vocab_size)
    # print(f"Optimized python train time: {optimized_python_train_time:.4f}s")

    # Train rustbpe
    print("\nTraining rustbpe...")
    rustbpe_tokenizer = rustbpe.Tokenizer()
    _, rustbpe_train_time = time_function(rustbpe_tokenizer.train_from_iterator, [text], vocab_size)
    print(f"RustBPE train time: {rustbpe_train_time:.4f}s")
    assert rustbpe_train_time > 0, "Training should take some time"

    # Train HuggingFace
    print("\nTraining HuggingFace...")
    hf_tokenizer, hf_train_time = time_function(HuggingFaceTokenizer.train_from_iterator, [text], vocab_size)
    print(f"HuggingFace train time: {hf_train_time:.4f}s")
    assert hf_train_time > 0, "Training should take some time"

    # Print comparison
    print(f"\n📊 Performance comparison:")
    print(f"   RustBPE: {rustbpe_train_time:.4f}s")
    print(f"   HuggingFace: {hf_train_time:.4f}s")
    print(f"   Speedup: {hf_train_time/rustbpe_train_time:.2f}x")

def test_interface(enwik8_small):
    """测试RustBPETokenizer接口的训练、编码、解码和序列化功能。"""
    import tempfile
    from nanochat.tokenizer import RustBPETokenizer

    # Simple train test
    vocab_size = 300
    tok = RustBPETokenizer.train_from_iterator([enwik8_small], vocab_size)
    assert tok.get_vocab_size() == vocab_size, f"Expected vocab size {vocab_size}, got {tok.get_vocab_size()}"
    print(f"✅ Trained tokenizer with vocab size {vocab_size}")

    # Encode/decode text
    encode_text = "Hello world! How are you? 🙃"
    ids = tok.encode(encode_text)
    print(f"\nInput text: {encode_text}")
    print(f"IDs: {ids}")
    decoded = tok.decode(ids)
    print(f"Decoded: {decoded}")
    assert decoded == encode_text, f"Decoded text doesn't match: {decoded} != {encode_text}"
    print("✅ Encode/decode test passed")

    # Encode batch test
    ids_new = tok.encode([encode_text, encode_text])
    assert all(x == ids for x in ids_new), "Batch encoding should produce identical results"
    print("✅ Encode batch OK")

    # append/prepend functionality
    ids_special = tok.encode(encode_text, prepend="<|bos|>", append="<|bos|>")
    bos_token_id = tok.encode_special("<|bos|>")
    assert ids_special == [bos_token_id] + ids + [bos_token_id], "Special tokens not correctly added"
    print("✅ append/prepend OK")

    # Save/load test through a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tok.save(tmp_dir)
        tok_reloaded = RustBPETokenizer.from_directory(tmp_dir)
        ids_reloaded = tok_reloaded.encode(encode_text)
        assert ids_reloaded == ids, "Reloaded tokenizer should produce same results"
        print("✅ Save/load through temporary directory OK")
