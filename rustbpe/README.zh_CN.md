# rustbpe

> 缺失的tiktoken训练代码

一个非常轻量级的Rust库，用于训练GPT分词器。问题是推理库[tiktoken](https://github.com/openai/tiktoken)很棒，但只做推理。另一方面，huggingface的[tokenizers](https://github.com/huggingface/tokenizers)库做训练，但它相当臃肿，而且真的很难导航，因为它必须支持多年来人们处理分词器的所有不同历史包袱。最近，我还写了[minbpe](https://github.com/karpathy/minbpe)库，它既做训练又做推理，但只在低效的Python中。基本上我真正想要的是一个不花哨、超级简单但仍然相对高效的GPT分词器训练代码（比minbpe更高效，比tokenizers更干净/简单），然后导出训练好的词汇表用于tiktoken推理。这有道理吗？所以我们在这里。这里有更多的优化机会，我只是提前停止了一点，因为与之前的minbpe不同，rustbpe现在足够简单和快速，并且不是nanochat的显著瓶颈。
