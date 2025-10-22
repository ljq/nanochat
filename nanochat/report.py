"""
用于生成训练报告卡的工具。比通常的代码更混乱，将修复。
"""

import os
import re
import shutil
import subprocess
import socket
import datetime
import platform
import psutil
import torch

def run_command(cmd):
    """运行shell命令并返回输出，如果失败则返回None。"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except:
        return None

def get_git_info():
    """获取当前git提交、分支和脏状态。"""
    info = {}
    info['commit'] = run_command("git rev-parse --short HEAD") or "unknown"
    info['branch'] = run_command("git rev-parse --abbrev-ref HEAD") or "unknown"

    # Check if repo is dirty (has uncommitted changes)
    status = run_command("git status --porcelain")
    info['dirty'] = bool(status) if status is not None else False

    # Get commit message
    info['message'] = run_command("git log -1 --pretty=%B") or ""
    info['message'] = info['message'].split('\n')[0][:80]  # First line, truncated

    return info

def get_gpu_info():
    """获取GPU信息。"""
    if not torch.cuda.is_available():
        return {"available": False}

    num_devices = torch.cuda.device_count()
    info = {
        "available": True,
        "count": num_devices,
        "names": [],
        "memory_gb": []
    }

    for i in range(num_devices):
        props = torch.cuda.get_device_properties(i)
        info["names"].append(props.name)
        info["memory_gb"].append(props.total_memory / (1024**3))

    # Get CUDA version
    info["cuda_version"] = torch.version.cuda or "unknown"

    return info

def get_system_info():
    """获取系统信息。"""
    info = {}

    # Basic system info
    info['hostname'] = socket.gethostname()
    info['platform'] = platform.system()
    info['python_version'] = platform.python_version()
    info['torch_version'] = torch.__version__

    # CPU and memory
    info['cpu_count'] = psutil.cpu_count(logical=False)
    info['cpu_count_logical'] = psutil.cpu_count(logical=True)
    info['memory_gb'] = psutil.virtual_memory().total / (1024**3)

    # User and environment
    info['user'] = os.environ.get('USER', 'unknown')
    info['nanochat_base_dir'] = os.environ.get('NANOCHAT_BASE_DIR', 'out')
    info['working_dir'] = os.getcwd()

    return info

def estimate_cost(gpu_info, runtime_hours=None):
    """根据GPU类型和运行时间估算训练成本。"""

    # Rough pricing, from Lambda Cloud
    default_rate = 2.0
    gpu_hourly_rates = {
        "H100": 3.00,
        "A100": 1.79,
        "V100": 0.55,
    }

    if not gpu_info.get("available"):
        return None

    # Try to identify GPU type from name
    hourly_rate = None
    gpu_name = gpu_info["names"][0] if gpu_info["names"] else "unknown"
    for gpu_type, rate in gpu_hourly_rates.items():
        if gpu_type in gpu_name:
            hourly_rate = rate * gpu_info["count"]
            break

    if hourly_rate is None:
        hourly_rate = default_rate * gpu_info["count"]  # Default estimate

    return {
        "hourly_rate": hourly_rate,
        "gpu_type": gpu_name,
        "estimated_total": hourly_rate * runtime_hours if runtime_hours else None
    }

def generate_header():
    """为训练报告生成头部。"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    git_info = get_git_info()
    gpu_info = get_gpu_info()
    sys_info = get_system_info()
    cost_info = estimate_cost(gpu_info)

    header = f"""# nanochat training report

Generated: {timestamp}

## Environment

### Git Information
- Branch: {git_info['branch']}
- Commit: {git_info['commit']} {"(dirty)" if git_info['dirty'] else "(clean)"}
- Message: {git_info['message']}

### Hardware
- Platform: {sys_info['platform']}
- CPUs: {sys_info['cpu_count']} cores ({sys_info['cpu_count_logical']} logical)
- Memory: {sys_info['memory_gb']:.1f} GB
"""

    if gpu_info.get("available"):
        gpu_names = ", ".join(set(gpu_info["names"]))
        total_vram = sum(gpu_info["memory_gb"])
        header += f"""- GPUs: {gpu_info['count']}x {gpu_names}
- GPU Memory: {total_vram:.1f} GB total
- CUDA Version: {gpu_info['cuda_version']}
"""
    else:
        header += "- GPUs: None available\n"

    if cost_info and cost_info["hourly_rate"] > 0:
        header += f"""- Hourly Rate: ${cost_info['hourly_rate']:.2f}/hour\n"""

    header += f"""
### Software
- Python: {sys_info['python_version']}
- PyTorch: {sys_info['torch_version']}

"""

    # 膨胀指标：打包所有源代码并评估其权重
    packaged = run_command('files-to-prompt . -e py -e md -e rs -e html -e toml -e sh --ignore "*target*" --cxml')
    num_chars = len(packaged)
    num_lines = len(packaged.split('\n'))
    num_files = len([x for x in packaged.split('\n') if x.startswith('<source>')])
    num_tokens = num_chars // 4 # assume approximately 4 chars per token

    # count dependencies via uv.lock
    uv_lock_lines = 0
    if os.path.exists('uv.lock'):
        with open('uv.lock', 'r') as f:
            uv_lock_lines = len(f.readlines())

    header += f"""
### Bloat
- Characters: {num_chars:,}
- Lines: {num_lines:,}
- Files: {num_files:,}
- Tokens (approx): {num_tokens:,}
- Dependencies (uv.lock lines): {uv_lock_lines:,}

"""
    return header

# -----------------------------------------------------------------------------

def slugify(text):
    """将文本字符串转换为slug格式。"""
    return text.lower().replace(" ", "-")

# 预期的文件及其顺序
EXPECTED_FILES = [
    "tokenizer-training.md",
    "tokenizer-evaluation.md",
    "base-model-training.md",
    "base-model-loss.md",
    "base-model-evaluation.md",
    "midtraining.md",
    "chat-evaluation-mid.md",
    "chat-sft.md",
    "chat-evaluation-sft.md",
    "chat-rl.md",
    "chat-evaluation-rl.md",
]
# 我们当前感兴趣的指标
chat_metrics = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "ChatCORE"]

def extract(section, keys):
    """从部分中提取单个键的简单函数"""
    if not isinstance(keys, list):
        keys = [keys] # convenience
    out = {}
    for line in section.split("\n"):
        for key in keys:
            if key in line:
                out[key] = line.split(":")[1].strip()
    return out

def extract_timestamp(content, prefix):
    """从具有给定前缀的内容中提取时间戳。"""
    for line in content.split('\n'):
        if line.startswith(prefix):
            time_str = line.split(":", 1)[1].strip()
            try:
                return datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            except:
                pass
    return None

class Report:
    """维护一堆日志，生成最终的markdown报告。"""

    def __init__(self, report_dir):
        os.makedirs(report_dir, exist_ok=True)
        self.report_dir = report_dir

    def log(self, section, data):
        """将数据部分记录到报告中。"""
        slug = slugify(section)
        file_name = f"{slug}.md"
        file_path = os.path.join(self.report_dir, file_name)
        with open(file_path, "w") as f:
            f.write(f"## {section}\n")
            f.write(f"timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for item in data:
                if not item:
                    # skip falsy values like None or empty dict etc.
                    continue
                if isinstance(item, str):
                    # directly write the string
                    f.write(item)
                else:
                    # render a dict
                    for k, v in item.items():
                        if isinstance(v, float):
                            vstr = f"{v:.4f}"
                        elif isinstance(v, int) and v >= 10000:
                            vstr = f"{v:,.0f}"
                        else:
                            vstr = str(v)
                        f.write(f"- {k}: {vstr}\n")
            f.write("\n")
        return file_path

    def generate(self):
        """生成最终报告。"""
        report_dir = self.report_dir
        report_file = os.path.join(report_dir, "report.md")
        print(f"生成报告到 {report_file}")
        final_metrics = {} # 最重要的最终指标，我们将在最后添加为表格
        start_time = None
        end_time = None
        with open(report_file, "w") as out_file:
            # 首先写入头部
            header_file = os.path.join(report_dir, "header.md")
            if os.path.exists(header_file):
                with open(header_file, "r") as f:
                    header_content = f.read()
                    out_file.write(header_content)
                    start_time = extract_timestamp(header_content, "Run started:")
                    # 捕获膨胀数据以供稍后摘要（Bloat头部之后直到\n\n的内容）
                    bloat_data = re.search(r"### Bloat\n(.*?)\n\n", header_content, re.DOTALL)
                    bloat_data = bloat_data.group(1) if bloat_data else ""
            # 处理所有单独的部分
            for file_name in EXPECTED_FILES:
                section_file = os.path.join(report_dir, file_name)
                if not os.path.exists(section_file):
                    print(f"警告: {section_file} 不存在，跳过")
                    continue
                with open(section_file, "r") as in_file:
                    section = in_file.read()
                # 从此部分提取时间戳（最后一部分的时间戳将"粘住"作为end_time）
                if "rl" not in file_name:
                    # 跳过RL部分进行end_time计算，因为RL是实验性的
                    end_time = extract_timestamp(section, "timestamp:")
                # 从部分中提取最重要的指标
                if file_name == "base-model-evaluation.md":
                    final_metrics["base"] = extract(section, "CORE")
                if file_name == "chat-evaluation-mid.md":
                    final_metrics["mid"] = extract(section, chat_metrics)
                if file_name == "chat-evaluation-sft.md":
                    final_metrics["sft"] = extract(section, chat_metrics)
                if file_name == "chat-evaluation-rl.md":
                    final_metrics["rl"] = extract(section, "GSM8K") # RL仅评估GSM8K
                # 附加此报告部分
                out_file.write(section)
                out_file.write("\n")
            # 添加最终指标表格
            out_file.write("## 摘要\n\n")
            # 从头部复制膨胀指标
            out_file.write(bloat_data)
            out_file.write("\n\n")
            # 收集所有唯一的指标名称
            all_metrics = set()
            for stage_metrics in final_metrics.values():
                all_metrics.update(stage_metrics.keys())
            # 自定义排序：CORE第一，ChatCORE最后，其余在中间
            all_metrics = sorted(all_metrics, key=lambda x: (x != "CORE", x == "ChatCORE", x))
            # 固定列宽
            stages = ["base", "mid", "sft", "rl"]
            metric_width = 15
            value_width = 8
            # 写入表格头部
            header = f"| {'指标'.ljust(metric_width)} |"
            for stage in stages:
                header += f" {stage.upper().ljust(value_width)} |"
            out_file.write(header + "\n")
            # 写入分隔符
            separator = f"|{'-' * (metric_width + 2)}|"
            for stage in stages:
                separator += f"{'-' * (value_width + 2)}|"
            out_file.write(separator + "\n")
            # 写入表格行
            for metric in all_metrics:
                row = f"| {metric.ljust(metric_width)} |"
                for stage in stages:
                    value = final_metrics.get(stage, {}).get(metric, "-")
                    row += f" {str(value).ljust(value_width)} |"
                out_file.write(row + "\n")
            out_file.write("\n")
            # 计算并写入总挂钟时间
            if start_time and end_time:
                duration = end_time - start_time
                total_seconds = int(duration.total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                out_file.write(f"总挂钟时间: {hours}h{minutes}m\n")
            else:
                out_file.write("总挂钟时间: 未知\n")
        # 也将report.md文件复制到当前目录
        print(f"为方便起见，将report.md复制到当前目录")
        shutil.copy(report_file, "report.md")
        return report_file

    def reset(self):
        """重置报告。"""
        # Remove section files
        for file_name in EXPECTED_FILES:
            file_path = os.path.join(self.report_dir, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
        # Remove report.md if it exists
        report_file = os.path.join(self.report_dir, "report.md")
        if os.path.exists(report_file):
            os.remove(report_file)
        # Generate and write the header section with start timestamp
        header_file = os.path.join(self.report_dir, "header.md")
        header = generate_header()
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(header_file, "w") as f:
            f.write(header)
            f.write(f"Run started: {start_time}\n\n---\n\n")
        print(f"Reset report and wrote header to {header_file}")

# -----------------------------------------------------------------------------
# nanochat特定的便利函数

class DummyReport:
    """虚拟报告类，用于非rank 0进程"""
    def log(self, *args, **kwargs):
        pass
    def reset(self, *args, **kwargs):
        pass

def get_report():
    """获取报告实例，仅rank 0记录到报告"""
    from nanochat.common import get_base_dir, get_dist_info
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp_rank == 0:
        report_dir = os.path.join(get_base_dir(), "report")
        return Report(report_dir)
    else:
        return DummyReport()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="生成或重置nanochat训练报告。")
    parser.add_argument("command", nargs="?", default="generate", choices=["generate", "reset"], help="要执行的操作（默认：generate）")
    args = parser.parse_args()
    if args.command == "generate":
        get_report().generate()
    elif args.command == "reset":
        get_report().reset()
