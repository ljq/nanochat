"""
来自Allen AI的ARC数据集。
https://huggingface.co/datasets/allenai/ai2_arc
"""

from datasets import load_dataset
from tasks.common import Task, render_mc

class ARC(Task):

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["ARC-Easy", "ARC-Challenge"], "ARC subset must be ARC-Easy or ARC-Challenge"
        assert split in ["train", "validation", "test"], "ARC split must be train|validation|test"
        self.ds = load_dataset("allenai/ai2_arc", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"] # 问题文本
        choices = row["choices"]["text"] # 每个选择的文本
        answer_string = row["answerKey"] # 例如 "A", "B", "C", "D"
        letters = row["choices"]["label"] # 例如 ["A", "B", "C", "D"]
        assert answer_string in letters, f"ARC答案 {answer_string} 必须是 {letters} 之一" # 完整性检查
        # 创建并返回Conversation对象
        user_message = render_mc(question, letters, choices)
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer_string}
        ]
        conversation = {
            "messages": messages,
            "letters": letters, # 在评估期间有用，因此我们可以将助手预测缩小并限制为字母之一
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        # 严格来说这里的断言不是必需的，但目前我们评估的方式，我们期望这是真的
        # 我将保留此断言以防止误用，但将来可能会删除它。
        assert assistant_response in conversation['letters'], f"ARC答案 {assistant_response} 预期是 {conversation['letters']} 之一"
        assistant_message = conversation['messages'][-1]['content'] # 例如 "A"
        return assistant_response == assistant_message
