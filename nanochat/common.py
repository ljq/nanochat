"""
nanochat的通用工具函数。
"""

import os
import re
import logging
import torch
import torch.distributed as dist

class ColoredFormatter(logging.Formatter):
    """自定义格式化器，为日志消息添加颜色。"""
    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 洋红色
    }
    RESET = '\033[0m'  # 重置颜色
    BOLD = '\033[1m'   # 粗体
    
    def format(self, record):
        # 为级别名称添加颜色
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        # 格式化消息
        message = super().format(record)
        # 为消息的特定部分添加颜色
        if levelname == 'INFO':
            # 高亮数字和百分比
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        return message

def setup_default_logging():
    """设置默认日志配置"""
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )

setup_default_logging()
logger = logging.getLogger(__name__)  # 全局日志记录器

def get_base_dir():
    """获取nanochat基础目录，默认位于~/.cache/nanochat"""
    # 将nanochat中间文件与其他缓存数据放在~/.cache中（默认）
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir

def print0(s="",**kwargs):
    """只在DDP rank 0上打印，避免重复输出"""
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

def print_banner():
    """打印nanochat ASCII横幅"""
    # 使用https://manytools.org/hacker-tools/ascii-banner/制作的酷DOS Rebel字体ASCII横幅
    banner = """
                                                   █████                 █████
                                                  ░░███                 ░░███
 ████████    ██████   ████████    ██████   ██████  ░███████    ██████   ███████
░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███ ░░░███░
 ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████   ░███
 ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███   ░███ ███
 ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░████████  ░░█████
░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░    ░░░░░
"""
    print0(banner)

def is_ddp():
    """检查是否在分布式数据并行（DDP）环境中运行"""
    # TODO 是否有更合适的方法
    return int(os.environ.get('RANK', -1)) != -1

def get_dist_info():
    """获取分布式训练信息"""
    if is_ddp():
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])  # 全局rank
        ddp_local_rank = int(os.environ['LOCAL_RANK'])  # 本地rank
        ddp_world_size = int(os.environ['WORLD_SIZE'])  # 世界大小
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1  # 非DDP模式

def compute_init():
    """基本初始化函数，我们反复使用，所以做成通用函数。"""

    # 目前需要CUDA
    assert torch.cuda.is_available(), "分布式运行目前需要CUDA"

    # 可重现性
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # 暂时跳过完全可重现性，可能稍后调查性能下降
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # 精度设置
    torch.set_float32_matmul_precision("high")  # 使用tf32代替fp32进行矩阵乘法

    # 分布式设置：分布式数据并行（DDP），可选
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp:
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)  # 使"cuda"默认指向此设备
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()  # 同步所有进程
    else:
        device = torch.device("cuda")

    if ddp_rank == 0:
        logger.info(f"分布式世界大小: {ddp_world_size}")

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device

def compute_cleanup():
    """compute_init的配套函数，在脚本退出前清理资源"""
    if is_ddp():
        dist.destroy_process_group()  # 销毁进程组

class DummyWandb:
    """如果我们希望不使用wandb但具有所有相同的签名，这个类很有用"""
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        """模拟wandb的log方法，什么也不做"""
        pass
    def finish(self):
        """模拟wandb的finish方法，什么也不做"""
        pass
