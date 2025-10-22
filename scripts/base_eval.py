"""
评估给定模型的CORE指标。

在单个GPU上运行：
python base_eval.py

使用torchrun在例如8个GPU上运行：
torchrun --nproc_per_node=8 base_eval.py

脚本将在控制台打印CORE指标。
"""
import os
import sys
import time
import json
import random
import yaml

import pandas as pd
import torch

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir
from nanochat.tokenizer import HuggingFaceTokenizer
from nanochat.checkpoint_manager import load_model
from nanochat.core_eval import evaluate_task

# -----------------------------------------------------------------------------
# nanoChat specific function dealing with I/O etc.

def evaluate_model(model, tokenizer, device, max_per_task=-1):
    """
    在CORE基准测试上评估基础模型。
    - max_per_task: 为测试将每个任务的数据裁剪为此数量的示例（-1 = 禁用）
    TODO: 清理此函数，删除对所有文件的需求，删除pandas依赖等。
    """
    # 加载配置和任务元数据
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']
    eval_metadata = pd.read_csv(eval_meta_data)

    # 评估每个任务
    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task['label']
        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }
        print0(f"评估: {label} ({task_meta['num_fewshot']}-shot, 类型: {task_meta['task_type']})... ", end='')

        # 加载此任务的数据
        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r') as f:
            data = [json.loads(line.strip()) for line in f]

        # 洗牌数据，因为在许多情况下它看起来是有序的，但我们希望
        # 能够只运行数据的子集以进行调试等目的。
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        # 运行此任务的评估
        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)

        results[label] = accuracy
        row = eval_metadata[eval_metadata["Eval Task"] == label]
        random_baseline = row["Random baseline"].values[0]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        end_time = time.time()
        print0(f"准确率: {accuracy:.4f} | 中心化: {centered_result:.4f} | 时间: {end_time - start_time:.2f}s")

    core_metric = sum(centered_results.values()) / len(centered_results)
    out = {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric
    }
    return out

# -----------------------------------------------------------------------------
# HuggingFace loading utilities and light wrappers for a model

class ModelWrapper:
    """HuggingFace模型的轻量级包装器"""
    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids):
        outputs = self.model(input_ids)
        logits = outputs.logits
        return logits

def load_hf_model(hf_path: str, device):
    print0(f"从以下位置加载模型: {hf_path}")
    # 加载模型
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(hf_path)
    model.to(device)
    model.eval()
    max_seq_len = 1024 if "openai-community/gpt2" in hf_path else None
    model = ModelWrapper(model, max_seq_len=max_seq_len)
    # 加载分词器
    tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)
    return model, tokenizer

# -----------------------------------------------------------------------------
def main():
    assert len(sys.argv) in [1, 2], "用法: python base_eval.py [hf_path]"

    # 分布式/精度设置
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    # 从命令行或文件系统加载模型和分词器
    if len(sys.argv) >= 2:
        # 目前假设如果给出路径，它是huggingface模型路径
        hf_path = sys.argv[1]
        print0(f"从以下位置加载huggingface模型: {hf_path}")
        model, tokenizer = load_hf_model(hf_path, device)
        model_name = hf_path # 仅用于日志记录
        model_slug = hf_path.replace("/", "-") # 用于输出csv文件
    else:
        # 从文件系统加载本地模型
        model, tokenizer, meta = load_model("base", device, phase="eval")
        model_name = f"base_model (步骤 {meta['step']})" # 仅用于日志记录
        model_slug = f"base_model_{meta['step']:06d}" # 用于输出csv文件

    # 评估模型
    with autocast_ctx:
        out = evaluate_model(model, tokenizer, device)

    # 将结果写入csv文件
    core_metric = None
    centered_results = {}
    if ddp_rank == 0:
        base_dir = get_base_dir()
        output_csv_path = os.path.join(base_dir, "base_eval", f"{model_slug}.csv")
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        results = out["results"]
        centered_results = out["centered_results"]
        core_metric = out["core_metric"]
        with open(output_csv_path, 'w') as f:
            f.write(f"{'任务':<35}, {'准确率':<10}, {'中心化':<10}\n")
            for label in results:
                f.write(f"{label:<35}, {results[label]:<10.6f}, {centered_results[label]:<10.6f}\n")
            f.write(f"{'CORE':<35}, {'':<10}, {core_metric:<10.6f}\n")
        # 也将csv文件的内容打印到控制台
        print0("="*80)
        print0(f"模型: {model_name}")
        print0("="*80)
        with open(output_csv_path, 'r') as f:
            print0(f.read())

    # 记录到报告
    from nanochat.report import get_report
    get_report().log(section="基础模型评估", data=[
        {
            "模型": model_name,
            "CORE指标": core_metric,
        },
        centered_results, # 完整表格
    ])

    compute_cleanup()

if __name__ == "__main__":
    main()
