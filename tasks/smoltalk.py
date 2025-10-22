"""
HuggingFace的SmolTalk。良好的"通用"对话数据集。
https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
我们使用"smol"版本，更适合较小的模型。
"""

from datasets import load_dataset
from tasks.common import Task

class SmolTalk(Task):
    """ smol-smoltalk数据集。训练集460K行，测试集24K行。 """

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SmolTalk split must be train|test"
        self.ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = row["messages"]
        # ---------------------------------------------------------------------
        # 完整性检查断言在这里
        # TODO: 我们稍后可以移除这些断言，现在只是不想有任何误用
        # 开头有一个可选的系统消息
        assert len(messages) >= 1
        first_message = messages[0]
        if first_message["role"] == "system":
            rest_messages = messages[1:] # 可选的系统消息是可以的
        else:
            rest_messages = messages
        assert len(rest_messages) >= 2, "SmolTalk消息必须至少有2条消息"
        for i, message in enumerate(rest_messages):
            # 用户和助手交替作为user,assistant,user,assistant,...
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role, f"消息 {i} 有角色 {message['role']} 但应该是 {expected_role}"
            assert isinstance(message["content"], str), "内容必须是字符串"
        # ---------------------------------------------------------------------
        # 创建并返回Conversation对象（也可以发出系统消息）
        conversation = {
            "messages": messages,
        }
        return conversation
