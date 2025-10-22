"""
将FinewebEdu-100B数据集重新打包成分片：

- 每个分片大小约为100MB（经过zstd压缩后）
- parquet文件以1000行的行组大小写入
- 对数据集进行洗牌

这将被上传到HuggingFace进行托管。
重要的是，我们的DataLoader将能够流式传输
数据并在磁盘上缓存，减少训练延迟。

注意：此文件仅作为数据集准备的参考/文档，
在项目运行时不会使用。
"""
import os
import time

from datasets import load_dataset
import pyarrow.parquet as pq
import pyarrow as pa

# 源数据集
dataset_kwargs = {
    "path": "HuggingFaceFW/fineweb-edu",
    "split": "train",
    "name": "sample-100BT", # 约100B GPT-2 token，按~3字符/token计算 => 约300B字符总数
}
ds = load_dataset(**dataset_kwargs)

# 洗牌以打乱顺序
ds = ds.shuffle(seed=42)
ndocs = len(ds) # 要处理的总文档数
print(f"Total number of documents: {ndocs}")

# 重新打包成parquet文件
output_dir = "/home/ubuntu/.cache/nanochat/base_data"
os.makedirs(output_dir, exist_ok=True)

# 写入parquet文件
chars_per_shard = 250_000_000  # 每个分片的字符数
row_group_size = 1024 # HuggingFace使用1000，但我们使用2的倍数，对分布式数据加载器更友好
shard_docs = []  # 当前分片的文档列表
shard_index = 0   # 分片索引
shard_characters = 0  # 当前分片的字符数
total_docs_processed = 0  # 已处理的总文档数
total_time_spent = 0  # 总花费时间
t0 = time.time()  # 开始时间
for doc in ds:
    text = doc['text']
    shard_docs.append(text)
    shard_characters += len(text)
    collected_enough_chars = shard_characters >= chars_per_shard  # 是否收集了足够的字符
    docs_multiple_of_row_group_size = len(shard_docs) % row_group_size == 0  # 文档数是否是行组大小的倍数
    if collected_enough_chars and docs_multiple_of_row_group_size: # 导致约100MB的文本（压缩后）
        shard_path = os.path.join(output_dir, f"shard_{shard_index:05d}.parquet")
        shard_table = pa.Table.from_pydict({"text": shard_docs})
        pq.write_table(
            shard_table,
            shard_path,
            row_group_size=row_group_size,
            use_dictionary=False, # 这通常用于分类数据
            compression="zstd", # 有效值：{‘NONE’, ‘SNAPPY’, ‘GZIP’, ‘BROTLI’, ‘LZ4’, ‘ZSTD’}
            compression_level=3,
            write_statistics=False, # 文本不需要统计信息
        )
        t1 = time.time()
        dt = t1 - t0 # 仅针对这个分片的时间
        t0 = t1
        total_docs_processed += len(shard_docs)
        total_time_spent += dt
        remaining_docs = ndocs - total_docs_processed  # 剩余文档数
        avg_time_per_doc = total_time_spent / total_docs_processed  # 平均每个文档的处理时间
        remaining_time = remaining_docs * avg_time_per_doc  # 预计剩余时间
        remaining_time_hours = remaining_time / 3600  # 转换为小时
        print(f"写入 {shard_path}. 文档数: {len(shard_docs)} | 字符数: {shard_characters} | 时间: {dt:.2f}s | 剩余时间: {remaining_time_hours:.2f}h")
        shard_docs = []
        shard_characters = 0
        shard_index += 1

# 演示数据后来如何上传到HuggingFace
def upload():
    """将数据集上传到HuggingFace Hub"""
    import os
    from huggingface_hub import HfApi
    token = os.getenv("HF_TOKEN")  # 从环境变量获取HuggingFace token
    api = HfApi(token=token)
    api.upload_large_folder(
        folder_path=output_dir,
        repo_id="karpathy/fineweb-edu-100b-shuffle",
        repo_type="dataset",
    )
# upload()  # 注释掉，实际使用时取消注释
