"""
评估聊天模型。
所有通用代码都在这里，所有评估特定的代码都在nanochat目录中并从这里导入。

运行示例：
python -m scripts.chat_eval -a ARC-Easy
torchrun --nproc_per_node=8 -m scripts.chat_eval -- -a ARC-Easy
"""

import argparse
from functools import partial

import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, get_dist_info, print0
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.arc import ARC
from tasks.gsm8k import GSM8K

# -----------------------------------------------------------------------------
# 生成式评估循环（我们逐个问题处理，采样，评估）

def run_generative_eval(task_object, tokenizer, model, engine, num_samples, max_new_tokens, temperature, top_k, max_problems=None):

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    # 运行评估
    num_passed, total = 0, 0
    for i in range(ddp_rank, num_problems, ddp_world_size):
        conversation = task_object[i]

        # 标记化提示
        encoded_prompt = tokenizer.render_for_completion(conversation)
        # 获取完成
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        # 将完成解码为文本
        prefix_length = len(encoded_prompt)
        completions = [tokenizer.decode(result_tokens[prefix_length:]) for result_tokens in results]
        # 评估成功标准
        outcomes = [task_object.evaluate(conversation, completion) for completion in completions]
        passed = any(outcomes)

        # 保持统计
        total += 1
        num_passed += int(passed)

        # 日志记录（在控制台中覆盖同一行）
        print(f"\r\033[KRank {ddp_rank} | {num_passed}/{total} ({100*num_passed/total:.2f}%)", end='', flush=True)

    # 在最终摘要之前用换行符完成原地进度行
    print()

    # 在所有ranks上聚合结果
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    print0("=" * 50)
    print0(f"最终: {num_passed}/{total} ({100*num_passed/total:.2f}%)")

    # 返回准确率
    return num_passed/total

# -----------------------------------------------------------------------------
# 分类评估循环
# 更容易，因为我们不需要采样。因此，我们可以一次处理批次，只检查正确答案选择的logits。

def run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=None):

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()
    bos = tokenizer.get_bos_token_id() # use BOS as pad token is ok, these positions are ignored

    # 我们将一次处理独立问题的批次，因为不需要采样
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    ceil_div = lambda x, y: -(-x // y)
    num_batches = ceil_div(num_problems, batch_size)

    # 运行评估
    letter_to_id_cache = {} # 许多字母会经常重复，让我们为分词器节省一些工作
    num_passed, total = 0, 0
    for i in range(ddp_rank, num_batches, ddp_world_size):
        i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)

        # 准备问题批次。它们可能长度都不同，所以我们填充/整理它们。
        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids = [tokenizer.render_for_completion(conversation) for conversation in conversations] # TODO: 重做这个工作方式
        max_length = max(len(ids) for ids in prompt_ids)
        answer_time_positions = [len(ids) - 1 for ids in prompt_ids] # 最后一个token的位置（和预测的答案）
        padded_prompt_ids = [ids + [bos] * (max_length - len(ids)) for ids in prompt_ids]
        prompt_ids = torch.tensor(padded_prompt_ids, dtype=torch.long, device=device)

        # 并行获取整个对话批次的logits（这里效率提升）
        with torch.no_grad():
            logits = model(prompt_ids) # (B, T, V)

        # 专注于可用答案，仅关注与选择对应的字母
        # 注意这大大帮助了评估，因为它特别将焦点缩小到仅可用的字母
        # 更难的替代方案是从Assistant生成并检查它是否用正确的字母响应
        # （例如A、B、C、D），但评估通常以这种方式使任务更容易。
        for idx, conversation in enumerate(conversations):
            # 获取此问题所有可用字母的token id
            letters = conversation['letters']
            letter_ids = []
            for letter in letters:
                if not letter in letter_to_id_cache:
                    encoded_letter = tokenizer.encode(letter)
                    assert len(encoded_letter) == 1, "每个字母必须是单个token"
                    letter_to_id_cache[letter] = encoded_letter[0]
                letter_ids.append(letter_to_id_cache[letter])
            # 将logits聚焦到答案位置和答案的可用字母
            answer_pos = answer_time_positions[idx]
            focus_logits = logits[idx, answer_pos, letter_ids]
            # 获取argmax字母（预测的答案）
            argmax_letter_id = focus_logits.argmax(dim=-1).item()
            predicted_letter = letters[argmax_letter_id]
            # 评估结果
            outcome = task_object.evaluate(conversation, predicted_letter)
            num_passed += int(outcome)
            total += 1

    # 在所有ranks上聚合结果
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    average = num_passed/total
    print0(f"最终: {num_passed}/{total} ({100*average:.2f}%)")
    return average

# -----------------------------------------------------------------------------

def run_chat_eval(task_name, model, tokenizer, engine,
                   batch_size=1, num_samples=1, max_new_tokens=512, temperature=0.0, top_k=50,
                   max_problems=None):
    # 创建评估对象
    task_module = {
        'HumanEval': HumanEval,
        'MMLU': partial(MMLU, subset="all", split="test"),
        'ARC-Easy': partial(ARC, subset="ARC-Easy", split="test"),
        'ARC-Challenge': partial(ARC, subset="ARC-Challenge", split="test"),
        'GSM8K': partial(GSM8K, subset="main", split="test"),
    }[task_name]
    task_object = task_module()
    # 运行评估
    if task_object.eval_type == 'generative':
        acc = run_generative_eval(task_object, tokenizer, model, engine, num_samples, max_new_tokens, temperature, top_k, max_problems=max_problems)
    elif task_object.eval_type == 'categorical':
        acc = run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=max_problems)
    else:
        raise ValueError(f"不支持的任务评估类型: {task_object.eval_type}")
    return acc

# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--source', type=str, required=True, help="模型来源: sft|mid|rl")
    parser.add_argument('-a', '--task-name', type=str, default=None, help="任务名称。默认=所有任务。使用|分割多个任务。")
    parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
    parser.add_argument('-t', '--temperature', type=float, default=0.0)
    parser.add_argument('-m', '--max-new-tokens', type=int, default=512)
    parser.add_argument('-n', '--num-samples', type=int, default=1)
    parser.add_argument('-k', '--top-k', type=int, default=50)
    parser.add_argument('-b', '--batch-size', type=int, default=8, help='分类评估的批次大小')
    parser.add_argument('-g', '--model-tag', type=str, default=None, help='要加载的模型标签')
    parser.add_argument('-s', '--step', type=int, default=None, help='要加载的步骤')
    parser.add_argument('-x', '--max-problems', type=int, default=None, help='要评估的最大问题数')
    args = parser.parse_args()

    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)

    # 获取要评估的任务
    all_tasks = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval']
    baseline_accuracies = {
        'ARC-Easy': 0.25, # 多项选择4选1 => 25%
        'ARC-Challenge': 0.25, # 多项选择4选1 => 25%
        'MMLU': 0.25, # 多项选择4选1 => 25%
        'GSM8K': 0.0, # 开放式 => 0%
        'HumanEval': 0.0, # 开放式 => 0%
    }
    task_names = all_tasks if args.task_name is None else args.task_name.split('|')

    # 顺序运行所有任务评估
    results = {}
    for task_name in task_names:
        with autocast_ctx:
            acc = run_chat_eval(
                task_name,
                model, tokenizer, engine,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                max_problems=args.max_problems,
            )
            results[task_name] = acc
            print0(f"{task_name} 准确率: {100 * acc:.2f}%")

    # 记录到报告
    from nanochat.report import get_report
    all_tasks_were_evaluated = all(task_name in results for task_name in all_tasks)
    # 如果可以，计算ChatCORE指标（类似于CORE，它是平均中心化准确率）
    # 这样，ChatCORE范围从0（随机基线）到1（峰值性能）
    chatcore_metric_dict = {}
    if all_tasks_were_evaluated:
        centered_mean = 0
        for task_name, acc in results.items():
            baseline_acc = baseline_accuracies.get(task_name, 0.0)
            centered_acc = (acc - baseline_acc) / (1.0 - baseline_acc)
            centered_mean += centered_acc
        chatcore_metric = centered_mean / len(results)
        chatcore_metric_dict = {"ChatCORE 指标": chatcore_metric}
    get_report().log(section="聊天评估 " + args.source, data=[
        vars(args), # 命令行参数
        results,
        chatcore_metric_dict,
    ])

    compute_cleanup()
