"""
基础/预训练数据集是一组parquet文件。
此文件包含用于以下功能的实用程序：
- 迭代parquet文件并从中生成文档
- 如果文件不在磁盘上，则按需下载文件

有关数据集准备方式的详细信息，请参阅`repackage_data_reference.py`。
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# 当前预训练数据集的具体信息

# 数据在互联网上托管并按需下载的URL
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822 # 最后一个数据分片是shard_01822.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet" # 文件名的格式
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 这些函数对其他模块是有用的实用程序，可以/应该被导入

def list_parquet_files(data_dir=None):
    """查看数据目录并返回所有parquet文件的完整路径。"""
    data_dir = DATA_DIR if data_dir is None else data_dir
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def parquets_iter_batched(split, start=0, step=1):
    """
    高效地迭代数据集，以底层行组为批次。
    - split可以是"train"或"val"。最后一个parquet文件将是val。
    - start/step对于在DDP中跳过行很有用。例如：start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

# -----------------------------------------------------------------------------
def download_single_file(index):
    """下载单个文件索引，带有一些退避重试机制"""

    # 构造此文件的本地文件路径，如果已存在则跳过
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"跳过 {filepath}（已存在）")
        return True

    # 构造此文件的远程URL
    url = f"{BASE_URL}/{filename}"
    print(f"下载 {filename}...")

    # 带重试的下载
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # 首先写入临时文件
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB块
                    if chunk:
                        f.write(chunk)
            # 将临时文件移动到最终位置
            os.rename(temp_path, filepath)
            print(f"成功下载 {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"尝试 {attempt}/{max_attempts} 失败，文件 {filename}: {e}")
            # 清理任何部分文件
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # 尝试几次，使用指数退避：2^attempt秒
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"在 {max_attempts} 次尝试后下载 {filename} 失败")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载FineWeb-Edu 100BT数据集分片")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="要下载的分片数量（默认：-1），-1 = 禁用")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="并行下载工作进程数（默认：4）")
    args = parser.parse_args()

    num = MAX_SHARD + 1 if args.num_files == -1 else min(args.num_files, MAX_SHARD + 1)
    ids_to_download = list(range(num))
    print(f"使用 {args.num_workers} 个工作进程下载 {len(ids_to_download)} 个分片...")
    print(f"目标目录: {DATA_DIR}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    # 报告结果
    successful = sum(1 for success in results if success)
    print(f"完成！已下载: {successful}/{len(ids_to_download)} 个分片到 {DATA_DIR}")
