"""
穷人的配置器。可能是个糟糕的主意。示例用法：
$ python train.py config/override_file.py --batch_size=32
这将首先运行config/override_file.py，然后将batch_size覆盖为32

这个文件中的代码将从例如train.py中按以下方式运行：
>>> exec(open('configurator.py').read())

所以这不是一个Python模块，只是将这些代码从train.py中移开
这个脚本中的代码然后会覆盖globals()

我知道人们不会喜欢这个，我只是真的不喜欢配置
复杂性以及必须在每个变量前加上config.。如果有人
提出更好的简单Python解决方案，我洗耳恭听。
"""

import os
import sys
from ast import literal_eval

def print0(s="",**kwargs):
    """只在DDP rank 0上打印，避免重复输出"""
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

for arg in sys.argv[1:]:
    if '=' not in arg:
        # 假设是配置文件的名称
        assert not arg.startswith('--')
        config_file = arg
        print0(f"使用 {config_file} 覆盖配置:")
        with open(config_file) as f:
            print0(f.read())
        exec(open(config_file).read())
    else:
        # 假设是--key=value参数
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                # 尝试评估它（例如如果是布尔值、数字等）
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # 如果出错，就使用字符串
                attempt = val
            # 确保类型匹配
            if globals()[key] is not None:
                attempt_type = type(attempt)
                default_type = type(globals()[key])
                assert attempt_type == default_type, f"类型不匹配: {attempt_type} != {default_type}"
            # 祈祷一切顺利
            print0(f"覆盖: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"未知配置键: {key}")
