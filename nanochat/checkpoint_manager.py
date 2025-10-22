"""
用于保存和加载模型/优化器/状态检查点的工具。
"""
import os
import re
import glob
import json
import logging
import torch

from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.common import setup_default_logging

# Set up logging
setup_default_logging()
logger = logging.getLogger(__name__)
def log0(message):
    """只在DDP rank 0上记录日志，避免重复输出"""
    if int(os.environ.get('RANK', 0)) == 0:
        logger.info(message)

def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data):
    """保存检查点，包括模型状态、优化器状态和元数据"""
    assert int(os.environ.get('RANK', 0)) == 0 # 目前防止误操作
    os.makedirs(checkpoint_dir, exist_ok=True)
    # 保存模型状态（参数）
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    torch.save(model_data, model_path)
    log0(f"保存模型文件到: {model_path}")
    # 保存优化器状态（对于SFT或任何其他微调很有用）
    if optimizer_data is not None:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}.pt")
        torch.save(optimizer_data, optimizer_path)
        log0(f"保存优化器文件到: {optimizer_path}")
    # 将元数据字典保存为json
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "w") as f:
        json.dump(meta_data, f, indent=2)
    log0(f"保存元数据文件到: {meta_path}")


def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False):
    """加载检查点，包括模型状态、优化器状态和元数据"""
    # 加载模型状态
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    # 如果请求，加载优化器状态
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}.pt")
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # 加载元数据
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r") as f:
        meta_data = json.load(f)
    return model_data, optimizer_data, meta_data


def build_model(checkpoint_dir, step, device, phase):
    """
    从给定检查点构建模型的一堆重复代码。
    返回：
    - 基础模型 - 未编译，未包装在DDP中
    - 分词器
    - 基础模型训练期间保存的元数据
    """
    assert phase in ["train", "eval"], f"无效阶段: {phase}"
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    # 修复：修复torch编译问题，该问题在所有键前添加_orig_mod.
    model_data = {k.lstrip("_orig_mod."): v for k, v in model_data.items()}
    model_config_kwargs = meta_data["model_config"]
    log0(f"使用配置构建模型: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    with torch.device("meta"):
        model = GPT(model_config)
    # 加载模型状态
    model.to_empty(device=device)
    model.init_weights() # 注意：这很愚蠢，但我们需要初始化旋转嵌入。TODO：修复模型重新初始化
    model.load_state_dict(model_data, strict=True, assign=True)
    # 将模型置于正确的训练阶段/模式
    if phase == "eval":
        model.eval()
    else:
        model.train()
    # 加载分词器
    tokenizer = get_tokenizer()
    # 健全性检查：模型和分词器之间的兼容性
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"]
    return model, tokenizer, meta_data


def find_largest_model(checkpoint_dir):
    """尝试猜测模型标签：取可用的最大模型"""
    model_tags = [f for f in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, f))]
    if not model_tags:
        raise FileNotFoundError(f"在 {checkpoint_dir} 中未找到检查点")
    # 1) 通常所有模型标签都是d<数字>的形式，首先尝试这个：
    candidates = []
    for model_tag in model_tags:
        match = re.match(r"d(\d+)", model_tag)
        if match:
            model_depth = int(match.group(1))
            candidates.append((model_depth, model_tag))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    # 2) 如果失败，取最近更新的模型：
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    return model_tags[0]


def find_last_step(checkpoint_dir):
    """查看checkpoint_dir并找到具有最高步数的model_<step>.pt"""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"在 {checkpoint_dir} 中未找到检查点")
    last_step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
    return last_step

# -----------------------------------------------------------------------------
# 考虑nanochat目录结构的便利函数

def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    """从目录加载模型，自动猜测模型标签和步数"""
    if model_tag is None:
        # 通过默认为最大模型来猜测模型标签
        model_tag = find_largest_model(checkpoints_dir)
        log0(f"未提供模型标签，猜测模型标签: {model_tag}")
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    if step is None:
        # 通过默认为最后一步来猜测步数
        step = find_last_step(checkpoint_dir)
    assert step is not None, f"在 {checkpoint_dir} 中未找到检查点"
    # 构建模型
    log0(f"从 {checkpoint_dir} 加载模型，步数 {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data

def load_model(source, *args, **kwargs):
    """根据源类型加载模型（基础、中期、SFT、RL）"""
    model_dir = {
        "base": "base_checkpoints",
        "mid": "mid_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)
