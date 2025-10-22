# nanochat

![nanochat logo](dev/nanochat.png)

> 100美元能买到的最好的ChatGPT。

这个仓库是一个完整的类ChatGPT大语言模型（LLM）的全栈实现，采用单一、简洁、最小化、可定制、依赖轻量的代码库。nanochat设计为通过像[speedrun.sh](speedrun.sh)这样的脚本在单个8XH100节点上运行，从开始到结束运行整个流程。这包括分词、预训练、微调、评估、推理以及通过简单UI提供Web服务，让你可以像使用ChatGPT一样与你自己的LLM对话。nanochat将成为Eureka Labs正在开发的LLM101n课程的顶点项目。

## 与它对话

为了了解这个仓库的最终目标，你目前可以在[nanochat.karpathy.ai](https://nanochat.karpathy.ai/)上找到托管的[nanochat d32](https://github.com/karpathy/nanochat/discussions/8)。"d32"表示这个模型在Transformer神经网络中有32层。这个模型有19亿参数，通过简单地运行单个脚本[run1000.sh](run1000.sh)在380亿token上训练，总训练成本约为800美元（在8XH100 GPU节点上约33小时训练时间）。虽然今天这足以超越2019年的GPT-2，但它与现代大语言模型如GPT-5相比仍有巨大差距。与这些微型模型对话时，你会看到它们犯很多错误，有点天真和愚蠢，会产生大量幻觉，有点像孩子。这有点有趣。但nanochat的独特之处在于它完全属于你 - 完全可配置、可调整、可定制，并由你从头到尾训练。要训练并与你自己的模型对话，我们转向...

## 快速开始

感受魔力的最快方式是运行speedrun脚本[speedrun.sh](speedrun.sh)，它训练并推理100美元级别的nanochat。在8XH100节点上每小时24美元，总运行时间约为4小时。从你喜欢的提供商启动一个新的8XH100 GPU盒子（例如我使用并喜欢[Lambda](https://lambda.ai/service/gpu-cloud)），然后启动训练脚本：

```bash
bash speedrun.sh
```

或者，由于脚本运行4小时，我喜欢在一个新的screen会话`speedrun`中这样启动（并将输出记录到`speedrun.log`）：

```bash
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
```

如果你不太熟悉，请查看[screen速查表](https://gist.github.com/jctosta/af918e1618682638aa82)。你可以在screen会话中观看进度，或者用`Ctrl-a d`分离并用`tail speedrun.log`查看进度。现在等待4小时。完成后，你可以通过类似ChatGPT的Web UI与你的LLM对话。确保你的本地uv虚拟环境已激活（运行`source .venv/bin/activate`），然后启动服务：

```bash
python -m scripts.chat_web
```

然后访问显示的URL。确保正确访问，例如在Lambda上使用你所在节点的公共IP，后跟端口，例如[http://209.20.xxx.xxx:8000/](http://209.20.xxx.xxx:8000/)等。然后像通常与ChatGPT对话一样与你的LLM对话！让它写故事或诗歌。问它你是谁以看到幻觉。问它为什么天空是蓝色的。或者为什么是绿色的。speedrun是一个4e19 FLOPs能力的模型，所以有点像与幼儿园小朋友对话:)。

---

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

你也可以`cat report.md`文件，它出现在项目目录中，包含运行的"成绩单"，即一堆评估和指标。在最后，你会看到一个汇总表格，例如：

---

- 字符数: 333,989
- 行数: 8,304
- 文件数: 44
- Token数（约）: 83,497
- 依赖项（uv.lock行数）: 2,004

| 指标          | BASE     | MID      | SFT      | RL       |
|---------------|----------|----------|----------|----------|
| CORE          | 0.2219   | -        | -        | -        |
| ARC-Challenge | -        | 0.2875   | 0.2807   | -        |
| ARC-Easy      | -        | 0.3561   | 0.3876   | -        |
| GSM8K         | -        | 0.0250   | 0.0455   | 0.0758   |
| HumanEval     | -        | 0.0671   | 0.0854   | -        |
| MMLU          | -        | 0.3111   | 0.3151   | -        |
| ChatCORE      | -        | 0.0730   | 0.0884   | -        |

总挂钟时间: 3h51m

---

（你的表格可能默认缺少RL数字）。关于speedrun脚本以及要寻找和期望的更多信息，请参考我在仓库讨论区发布的演练：["介绍nanochat：100美元能买到的最好的ChatGPT"](https://github.com/karpathy/nanochat/discussions/1)。

## 更大的模型

不出所料，100美元不足以训练一个高性能的ChatGPT克隆。事实上，LLM以其数百万美元的资本支出而闻名。对于我们的目的，我认为还有两个更有趣的规模。首先是约300美元的d26模型（即深度=26），训练约12小时，略微超越GPT-2 CORE分数。其次是1000美元级别（约41.6小时），只是因为这是一个不错的整数。但这两者尚未完全支持，因此尚未附加到主分支中。

也就是说，为了给出一个概念，训练GPT-2级别模型d26所需的[speedrun.sh](speedrun.sh)文件示例更改仅涉及三个更改：

```bash
...
# 你需要下载更多用于预训练的数据分片
# 获取参数数量，乘以20得到token数，乘以4.8得到字符数，
# 除以2.5亿得到分片数量。待办：需要改进这个...
python -m nanochat.dataset -n 450 &
...
# 使用--depth增加模型大小。为了避免内存不足，将设备批大小减半32 -> 16：
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=26 --device_batch_size=16
...
# 确保在中期训练期间使用相同的设置：
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device_batch_size=16
```

就是这样！最需要注意的事情是确保你有足够的数据分片进行训练（否则代码将循环并在相同的训练集上做更多轮次，稍微降低学习速度），以及管理你的内存/VRAM，主要通过减少`device_batch_size`直到适合（脚本通过增加梯度累积循环次数自动补偿，简单地将并行计算转换为顺序计算）。

关于运行nanochat的计算环境的更多信息：

- 代码在Ampere 8XA100 GPU节点上也能正常运行，但会慢一些。
- 所有代码甚至可以在单个GPU上通过省略`torchrun`正常运行，并产生几乎相同的结果（代码将自动切换到梯度累积），但你必须等待8倍时间。
- 如果你的GPU(s)少于80GB，你必须调整一些超参数，否则会出现OOM / VRAM不足。在脚本中查找`--device_batch_size`并减少它直到适合。例如从32（默认）到16、8、4、2，甚至1。少于这个你需要更了解你在做什么并更有创意。
- 大部分代码是相当标准的PyTorch，所以它应该在任何支持PyTorch的环境中运行 - xpu、mps等，但我没有开箱即用地实现这个，所以可能需要一些调整。

## 在CPU / MPS上运行

如果你想在Macbook或CPU机器上调整nanochat，这里有一个进行中的[CPU|MPS PR](https://github.com/karpathy/nanochat/pull/88)。如果你在Macbook上，在运行`base_train.py`时使用`--device_type=mps`。有关更多信息，请参阅PR及其差异。没有GPU节点你不会走得太远，但至少你将能够运行代码，并可能通过一些耐心训练一个非常小的LLM。

## 问题

nanochat设计为简短而甜美。这样做的一个大优势是我们可以将所有文件打包在一起，并复制粘贴到你喜欢的LLM中询问任意问题。例如，我喜欢使用[files-to-prompt](https://github.com/simonw/files-to-prompt)实用程序像这样打包仓库：

```bash
files-to-prompt . -e py -e md -e rs -e html -e toml -e sh --ignore "*target*" --cxml > packaged.txt
```

这包括所有py、rs、html、toml、sh文件，排除`rustbpe/target`文件夹，并选择cxml输出格式。所有内容都写入`packaged.txt`文件，目前测量约330KB（即远低于最先进LLM的约10万token），以及约8K行代码在45个文件中。

或者，我推荐使用[DeepWiki](https://deepwiki.com/)来自Devin/Cognition来询问这个仓库的问题。在这个仓库的URL中，只需将github.com更改为deepwiki.com，你就可以开始了。

## 测试

我在这里投入不多，但存在一些测试，特别是对于分词器。运行例如：

```bash
python -m pytest tests/test_rustbpe.py -v -s
```

## 贡献

nanochat远未完成。目标是改进在<1000美元预算下可端到端工作的微型模型的最新技术水平。可访问性是关于总体成本，也是关于认知复杂性 - nanochat不是一个详尽可配置的LLM"框架"；代码库中不会有巨大的配置对象、模型工厂或if-then-else怪物。它是一个单一、连贯、最小化、可读、可定制、最大可复制的"强基线"代码库，设计为从头到尾运行并产生具体的ChatGPT克隆及其成绩单。

## 致谢

- 名称（nanochat）源自我的早期项目[nanoGPT](https://github.com/karpathy/nanoGPT)，它只涵盖预训练。
- nanochat也受到[modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt)的启发，它通过清晰的指标和排行榜将nanoGPT仓库游戏化，并借用了它的许多想法和一些预训练实现。
- 感谢[HuggingFace](https://huggingface.co/)提供fineweb和smoltalk。
- 感谢[Lambda](https://lambda.ai/service/gpu-cloud)提供用于开发此项目的计算资源。
- 感谢首席LLM专家🧙‍♂️ Alec Radford的建议/指导。

## 引用

如果你发现nanochat对你的研究有帮助，请引用为：

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that $100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## 文件结构说明

以下是nanochat项目的主要文件及其用途：

### 核心模块 (nanochat/)
- **nanochat/gpt.py** - GPT模型架构实现，包含Transformer层、注意力机制等
- **nanochat/adamw.py** - AdamW优化器实现
- **nanochat/muon.py** - Muon优化器实现，用于线性层训练
- **nanochat/checkpoint_manager.py** - 模型检查点保存和加载管理
- **nanochat/common.py** - 通用工具函数，包括分布式训练初始化
- **nanochat/configurator.py** - 配置参数管理，支持命令行覆盖
- **nanochat/core_eval.py** - 核心评估指标计算
- **nanochat/dataloader.py** - 数据加载器实现，支持分布式训练
- **nanochat/dataset.py** - 数据集处理和下载
- **nanochat/engine.py** - 模型推理引擎，支持批量生成
- **nanochat/execution.py** - 执行上下文管理
- **nanochat/loss_eval.py** - 损失评估函数
- **nanochat/report.py** - 训练报告生成
- **nanochat/tokenizer.py** - 分词器接口和实现
- **nanochat/ui.html** - Web聊天界面

### 训练脚本 (scripts/)
- **scripts/base_train.py** - 基础模型预训练脚本
- **scripts/mid_train.py** - 中期训练脚本，在预训练基础上继续训练
- **scripts/chat_sft.py** - 监督微调训练脚本
- **scripts/chat_rl.py** - 强化学习训练脚本
- **scripts/tok_train.py** - 分词器训练脚本
- **scripts/base_eval.py** - 基础模型评估脚本
- **scripts/base_loss.py** - 基础损失评估脚本
- **scripts/chat_eval.py** - 聊天模型评估脚本
- **scripts/tok_eval.py** - 分词器评估脚本
- **scripts/chat_cli.py** - 命令行聊天界面
- **scripts/chat_web.py** - Web聊天服务器

### 任务模块 (tasks/)
- **tasks/common.py** - 任务混合和数据加载
- **tasks/arc.py** - ARC问答任务实现
- **tasks/gsm8k.py** - GSM8K数学推理任务实现
- **tasks/humaneval.py** - HumanEval代码生成任务实现
- **tasks/mmlu.py** - MMLU多任务语言理解任务实现
- **tasks/smoltalk.py** - SmolTalk对话数据集处理

### 分词器 (rustbpe/)
- **rustbpe/src/lib.rs** - Rust实现的BPE分词器核心逻辑
- **rustbpe/Cargo.toml** - Rust项目配置
- **rustbpe/README.md** - 分词器文档

### 开发工具 (dev/)
- **dev/generate_logo.html** - 项目logo生成工具
- **dev/nanochat.png** - 项目logo图片
- **dev/repackage_data_reference.py** - 数据重新打包参考脚本

### 运行脚本
- **speedrun.sh** - 快速运行脚本（约4小时训练）
- **run1000.sh** - 1000美元级别训练脚本
- **uv.lock** - Python依赖锁定文件
- **pyproject.toml** - Python项目配置

## 许可证

MIT
