"""
MMLU数据集。
https://huggingface.co/datasets/cais/mmlu
"""

from datasets import load_dataset
from tasks.common import Task, render_mc

class MMLU(Task):

    letters = ('A', 'B', 'C', 'D')
    groups = ('abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions')

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["all", "auxiliary_train"], f"subset {subset} must be all|auxiliary_train"
        assert split in ["train", "validation", "dev", "test"], f"split {split} must be train|validation|dev|test"
        if subset == "auxiliary_train":
            assert split == "train", "auxiliary_train must be split into train"
        self.subset = subset
        self.split = split
        self.ds = load_dataset("cais/mmlu", subset, split=split).shuffle(seed=42)
        if subset == "auxiliary_train":
            # I don't understand why but the auxiliary_train rows have some weird additional 'train' wrapper
            self.ds = self.ds.map(lambda row: row['train'], remove_columns=['train'])

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"] # 问题文本
        choices = row["choices"] # 每个选择的文本
        answer = row["answer"] # 答案的索引，例如0,1,2,3（对应A,B,C,D）
        subject = row["subject"] # 例如 "college_biology", "college_chemistry"等
        assert len(choices) == 4, "MMLU应该有4个选择"
        # 创建并返回Conversation对象
        user_message = render_mc(question, self.letters, choices)
        assistant_message = self.letters[answer]
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
        conversation = {
            "messages": messages,
            "subject": subject, # 稍后可能对按主题分组指标有用
            "letters": self.letters, # 在评估期间有用，因此我们可以将助手预测缩小并限制为字母之一
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        # 严格来说这里的断言不是必需的，但目前我们评估的方式，我们期望这是真的
        # 我将保留此断言以防止误用，但将来可能会删除它。
        assert assistant_response in self.letters, f"MMLU答案 {assistant_response} 预期是 {self.letters} 之一"
        assistant_message = conversation['messages'][-1]['content'] # 例如 "A"
        return assistant_response == assistant_message
