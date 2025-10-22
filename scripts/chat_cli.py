"""
新的和升级的聊天模式，因为自上一个版本以来很多代码已经改变。

目前仅设计为在单个GPU上运行：
python -m scripts.chat_cli -i mid
"""
import argparse
import torch
from nanochat.common import compute_init
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

parser = argparse.ArgumentParser(description='与模型聊天')
parser.add_argument('-i', '--source', type=str, default="sft", help="模型来源: sft|mid|rl")
parser.add_argument('-g', '--model-tag', type=str, default=None, help='要加载的模型标签')
parser.add_argument('-s', '--step', type=int, default=None, help='要加载的步骤')
parser.add_argument('-p', '--prompt', type=str, default='', help='提示模型，获取单个响应')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='生成的温度')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k采样参数')
args = parser.parse_args()

# 初始化模型和分词器
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)

# 聊天状态机的特殊token
bos = tokenizer.get_bos_token_id()
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

# 创建Engine以进行高效生成
engine = Engine(model, tokenizer)

print("\nNanoChat交互模式")
print("-" * 50)
print("输入'quit'或'exit'结束对话")
print("输入'clear'开始新对话")
print("-" * 50)

conversation_tokens = [bos]

while True:

    if args.prompt:
        # 从启动命令获取提示
        user_input = args.prompt
    else:
        # 从控制台交互式获取提示
        try:
            user_input = input("\n用户: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

    # 处理特殊命令
    if user_input.lower() in ['quit', 'exit']:
        print("再见！")
        break

    if user_input.lower() == 'clear':
        conversation_tokens = [bos]
        print("对话已清除。")
        continue

    if not user_input:
        continue

    # 将用户消息添加到对话
    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(user_input))
    conversation_tokens.append(user_end)

    # 启动助手
    conversation_tokens.append(assistant_start)
    generate_kwargs = {
        "num_samples": 1,
        "max_tokens": 256,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }
    response_tokens = []
    print("\n助手: ", end="", flush=True)
    with autocast_ctx:
        for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
            token = token_column[0] # 弹出批次维度（num_samples=1）
            response_tokens.append(token)
            token_text = tokenizer.decode([token])
            print(token_text, end="", flush=True)
    print()
    # 我们必须确保助手结束token是最后一个token
    # 所以即使生成因最大token数而结束，我们也必须将其附加到末尾
    if response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)
    conversation_tokens.extend(response_tokens)

    # 在提示模式下，我们只想要单个响应并退出
    if args.prompt:
        break
