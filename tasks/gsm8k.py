"""
GSM8K评估。
https://huggingface.co/datasets/openai/gsm8k

问题实例示例：

问题：
Weng每小时照看孩子赚12美元。昨天，她只照看了50分钟。她赚了多少钱？
答案：
Weng每分钟赚12/60 = $<<12/60=0.2>>0.2。
工作50分钟，她赚了0.2 x 50 = $<<0.2*50=10>>10。
#### 10

注意GSM8K在<< >>标签内使用工具调用。
"""

import re
from datasets import load_dataset
from tasks.common import Task


GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
def extract_answer(completion):
    """
    提取####标记后的数值答案。
    遵循官方代码进行标准化：
    https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/dataset.py#L28
    """
    match = GSM_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None


class GSM8K(Task):

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["main", "socratic"], "GSM8K subset must be main|socratic"
        assert split in ["train", "test"], "GSM8K split must be train|test"
        self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        """ 从数据集中获取单个问题。 """
        row = self.ds[index]
        question = row['question'] # 问题提示的字符串
        answer = row['answer'] # 完整解决方案和####标记后答案的字符串
        # 创建并返回Conversation对象
        # 这很棘手，因为GSM8K使用工具调用，我们需要在这里解析。
        assistant_message_parts = []
        parts = re.split(r'(<<[^>]+>>)', answer)
        for part in parts:
            if part.startswith('<<') and part.endswith('>>'):
                # 这是一个计算器工具调用
                inner = part[2:-2]  # 移除<< >>
                # 在=上分割以获取表达式和结果
                if '=' in inner:
                    expr, result = inner.rsplit('=', 1)
                else:
                    expr, result = inner, ""
                # 添加工具调用作为一部分
                assistant_message_parts.append({"type": "python", "text": expr})
                # 添加结果作为一部分
                assistant_message_parts.append({"type": "python_output", "text": result})
            else:
                # 工具调用之间的常规文本
                assistant_message_parts.append({"type": "text", "text": part})
        # 现在将所有内容放在一起
        messages = [
            {"role": "user", "content": question}, # 注意：简单字符串
            {"role": "assistant", "content": assistant_message_parts}, # 注意：部分列表（作为字典）
        ]
        conversation = {
            "messages": messages,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        给定（对话，完成），返回评估结果（0 = 错误，1 = 正确）
        注意：
        - 对话包含用户和助手消息（包含真实答案）
        - assistant_response通常是通过采样获得的替代助手消息

        TODO: 技术上，assistant_response应该是一个Message（要么是字符串，要么是部分列表）
              我们可能稍后处理这个问题。现在只假设字符串。
        """
        assert isinstance(assistant_response, str), "目前假设简单字符串响应"
        # 首先提取真实答案
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "最后一条消息必须来自助手"
        assert isinstance(assistant_message['content'], list), "这预期是一个部分列表"
        last_text_part = assistant_message['content'][-1]['text'] # 这包含GSM8K中的最终答案
        # 提取真实答案和预测答案
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        # 比较并返回成功作为整数
        is_correct = int(pred_num == ref_num)
        return is_correct

    def reward(self, conversation, assistant_response):
        """
        在RL期间使用。为保持简单，只需重用上面的评估。
        稍后可以使其更复杂（例如格式匹配等）。
        """
        is_correct = self.evaluate(conversation, assistant_response)
        is_correct_float = float(is_correct)
        return is_correct_float
