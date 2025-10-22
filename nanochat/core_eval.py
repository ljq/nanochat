"""
用于评估CORE指标的函数，如DCLM论文中所述。
https://arxiv.org/abs/2406.11794

待办事项：
- 除了squad之外，所有任务都大致匹配。我们得到31%，参考是37%。找出原因。
"""
import random

from jinja2 import Template
import torch
import torch.distributed as dist

# -----------------------------------------------------------------------------
# Prompt rendering utilities

def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """为多项选择题渲染完整的提示"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(choice=choice, **context) for choice in item['choices']]
    return prompts


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """为模式问题渲染完整的提示"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(context=context_option, **context)
               for context_option in item['context_options']]
    return prompts


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """
    为语言建模任务渲染完整的提示。
    注意我们在模板中手动修剪上下文，
    因为在某些数据集中似乎有尾随空格（我们不想要）。
    """
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    # 返回两个提示：不带延续和带延续
    prompt_without = template.render(include_continuation=False, **context)
    prompt_with = template.render(include_continuation=True, **context)
    # 由于数据存储的方式，我认为在LM情况下需要在这里进行strip。
    # 否则我们可能在prompt_without中得到尾随空格（这些空格会被吸收到
    # prompt_with中的下一个token中），意味着我们在token空间中得不到
    # 一个干净的前缀来检测最终的延续。分词器...
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction='left'):
    """
    查找token序列之间的公共前缀或后缀的长度
    - direction: 'left'表示前缀，'right'表示后缀
    """
    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        'left': range(min_len),
        'right': range(-1, -min_len-1, -1)
    }[direction]
    # 查找token序列开始不同的第一个位置
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id):
    """堆叠token序列列表，在右侧填充到最长"""
    bsz, seq_len = len(tokens), max(len(x) for x in tokens)
    input_ids = torch.full((bsz, seq_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(tokens):
        input_ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return input_ids


def batch_sequences_mc(tokenizer, prompts):
    # 在多项选择中，上下文相同但延续不同（公共前缀）
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # 找出每个延续的开始和结束位置
    answer_start_idx = find_common_length(tokens, direction='left')
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts):
    # 在模式任务中，上下文不同但延续相同（公共后缀）
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # 找出每个上下文的开始和结束位置
    suffix_length = find_common_length(tokens, direction='right')
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    # 在LM任务中，我们有两个提示：不带延续和带延续
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    assert start_idx < end_idx, "不带延续的提示应该是带延续提示的前缀"
    assert tokens_without == tokens_with[:start_idx], "不带延续的提示应该是带延续提示的前缀"
    # 在LM任务中我们只需要带延续的提示，即批次大小为1
    return [tokens_with], [start_idx], [end_idx]


@torch.no_grad()
def forward_model(model, input_ids):
    """
    获取BxT的token id张量，返回BxT的损失和argmax预测张量。
    损失的最后一列设置为nan，因为我们在那里没有自回归目标。
    """
    batch_size, seq_len = input_ids.size()
    outputs = model(input_ids)
    # 将张量向左滚动一个位置以获取（自回归）目标id
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    # 在所有位置计算交叉熵
    losses = torch.nn.functional.cross_entropy(
        outputs.view(batch_size * seq_len, -1),
        target_ids.view(batch_size * seq_len),
        reduction='none'
    ).view(batch_size, seq_len)
    # 将最后一列设置为nan，因为那里没有自回归损失
    losses[:, -1] = float('nan')
    # 获取每个位置的argmax预测
    predictions = outputs.argmax(dim=-1)
    return losses, predictions


@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta):
    """评估单个示例，如果正确返回True，否则返回False"""
    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']

    # 采样few-shot示例（排除当前项）
    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    # 根据任务类型渲染提示并批处理序列
    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == 'schema':
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == 'language_modeling':
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # 有些模型不能前向传播超过一定长度的序列（例如GPT-2）
    # 在这些情况下，我们必须将序列截断到最大长度并调整索引
    if hasattr(model, 'max_seq_len') and model.max_seq_len is not None:
        max_tokens = model.max_seq_len
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_tokens:
                num_to_crop = len(t) - max_tokens
                new_tokens.append(t[-max_tokens:]) # 取最后max_tokens个token
                new_start_idxs.append(s - num_to_crop) # 向下移动索引
                new_end_idxs.append(e - num_to_crop)
                assert s - num_to_crop >= 0, "这应该永远不会发生，对吧？"
                assert e - num_to_crop >= 0, "这应该永远不会发生，对吧？"
            else:
                new_tokens.append(t) # 保持不变
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

    # 将所有序列堆叠成一个批次
    pad_token_id = tokenizer.get_bos_token_id() # 使用BOS作为填充token是可以的
    input_ids = stack_sequences(tokens, pad_token_id)
    input_ids = input_ids.to(device)

    # 前向传播模型，获取每个token的自回归损失和argmax预测
    losses, predictions = forward_model(model, input_ids)

    # 检查损失/预测是否正确
    if task_type == 'language_modeling':
        # 语言建模任务目前总是批次大小为1
        si = start_idxs[0]
        ei = end_idxs[0]
        # predictions[i] 自回归地预测 input_ids[i+1]
        predicted_tokens = predictions[0, si-1:ei-1]
        actual_tokens = input_ids[0, si:ei]
        is_correct = torch.all(predicted_tokens == actual_tokens).item()
    elif task_type in ['multiple_choice', 'schema']:
        # 对于MC/模式：找到平均损失最低的选项
        mean_losses = [losses[i, si-1:ei-1].mean().item()
                        for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))]
        pred_idx = mean_losses.index(min(mean_losses))
        is_correct = pred_idx == item['gold']
    else:
        raise ValueError(f"不支持的任务类型: {task_type}")

    return is_correct


def evaluate_task(model, tokenizer, data, device, task_meta):
    """
    此函数负责跨多个示例评估一个任务。
    如果脚本使用torchrun运行，它还处理到所有进程的调度。
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    # 将示例跨每个rank进行步进分配
    for idx in range(rank, len(data), world_size):
        is_correct = evaluate_example(idx, model, tokenizer, data, device, task_meta)
        correct[idx] = float(is_correct)
    # 如果运行分布式，在所有进程之间同步结果
    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    # 计算平均值
    mean_correct = correct.mean().item()
    return mean_correct
