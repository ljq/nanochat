"""
来自Keller等人的Muon优化器。
也从modded-nanogpt借用了很多想法。
"""
import torch
from torch import Tensor
import torch.distributed as dist

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz迭代来计算G的零次幂/正交化。我们选择使用
    五次迭代，其系数被选择为在零点处最大化斜率。为了最小化步数，
    经验证明，即使在迭代不再在整个区间上收敛到1的点之后继续增加
    零点处的斜率也是有效的。因此，这个迭代不会产生UV^T，而是产生类似US'V^T的东西，
    其中S'是对角矩阵，S_{ii}' ~ Uniform(0.5, 1.5)，结果证明相对于UV^T，
    这完全不会损害模型性能，其中USV^T = G是SVD。
    """
    assert G.ndim >= 2 # 由@scottjmaddox实现的批处理Muon，并由@YouJiacheng在实践中应用
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # 确保谱范数最多为1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # 执行NS迭代
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # 五次计算策略改编自@jxbz、@leloykun和@YouJiacheng的建议
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - 通过Newton-schulz正交化的动量

    https://kellerjordan.github.io/posts/muon/

    Muon内部运行标准的SGD-momentum，然后执行正交化后处理步骤，
    其中每个2D参数的更新被替换为最近的正交矩阵。为了有效地正交化每个更新，
    我们使用Newton-Schulz迭代，其优点是可以在GPU上稳定地在bfloat16中运行。

    一些警告：
    - 此优化器不应用于嵌入层、最终全连接层或任何{0,1}维参数；
      这些都应该通过标准方法（例如AdamW）进行优化。
    - 要将其与4D卷积滤波器一起使用，只需将其最后3个维度展平即可。

    参数：
        lr: 内部SGD使用的学习率。
        momentum: 内部SGD使用的动量。
        nesterov: 是否在内部SGD中使用Nesterov风格的动量。（推荐）
        ns_steps: 使用的Newton-Schulz迭代步数。
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        """初始化Muon优化器"""
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            group = dict(params=[p for p in params if p.numel() == size])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        """执行优化步骤"""
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for p in params:
                g = p.grad
                assert g is not None
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf: Tensor = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g, alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)


class DistMuon(torch.optim.Optimizer):
    """
    Muon: SGD-momentum +（可选）Nesterov，然后通过Newton–Schulz正交化2D更新，
    最后应用纵横比缩放步长。执行自己的分布式同步：
      - reduce_scatter(AVG)用于梯度平均
      - all_gather用于复制更新后的权重

    注意：
      * 专为2D参数设计（例如，重塑为2D的线性/卷积核）。不要用于0D/1D
        参数，如嵌入或标量。
      * 动量缓冲区仅在每个参数的"所有者"rank上维护（rank由下面的
        块循环分配选择）。如果在单个rank上检查点优化器状态，
        请事先合并状态。

    参数：
        params: 张量的可迭代对象
        lr: 学习率
        momentum: 动量系数在[0,1)范围内
        nesterov: 如果为True，使用Nesterov风格更新（g <- lerp(g, buf, momentum)）；否则使用buf
        ns_steps: 用于正交化的Newton–Schulz迭代次数
    """
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5):
        """初始化分布式Muon优化器"""
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params = list(params)
        assert all(p.ndim == 2 for p in params), "Muon期望仅2D参数"
        rank = dist.get_rank()
        # 按形状对所有参数进行分组
        shapes = sorted({p.shape for p in params}) # 排序以确保一致/确定性排序
        param_groups = []
        for shape in shapes:
            group_params = [p for p in params if p.shape == shape]
            device, dtype = group_params[0].device, group_params[0].dtype
            assert all(p.device == device for p in group_params)
            assert all(p.dtype == dtype for p in group_params)
            if rank == 0:
                print(f"Muon: 分组 {len(group_params)} 个形状为 {shape} 的参数，设备 {device}，数据类型 {dtype}")
            param_groups.append(dict(params=group_params, zero_buffer=torch.zeros_like(group_params[0])))
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        """执行分布式优化步骤"""
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # 确保所有梯度都存在
        assert all(p.grad is not None for group in self.param_groups for p in group["params"]), "所有参数都必须有梯度"

        # 启动所有reduce scatter操作以在所有rank之间平均梯度
        all_reduce_futures = []
        for group in self.param_groups:
            params = group["params"]
            zero_buffer = group["zero_buffer"]
            # 以world_size为组遍历参数
            for base_i in range(0, len(params), world_size):
                # 每个参数的计算所有者是rank i % world_size
                owner_idx = base_i + rank
                # 每个rank将其world_size参数的块堆叠到列表中
                rs_input = [p.grad for p in params[base_i:base_i + world_size]]
                # 用零缓冲区填充rs_input以完成组
                rs_input.extend([zero_buffer] * (world_size - len(rs_input)))
                # 输出缓冲区根据rank在组中跨步
                rs_output = params[owner_idx].grad if owner_idx < len(params) else torch.empty_like(zero_buffer)
                # 在此world_size参数组内reduce scatter梯度
                work = dist.reduce_scatter(rs_output, rs_input, op=dist.ReduceOp.AVG, async_op=True).get_future()
                all_reduce_futures.append(work)

        # 现在每个rank计算更新并收集
        future_idx = 0
        all_gather_futures = []
        for group in self.param_groups:
            params = group["params"]
            zero_buffer = group["zero_buffer"]
            # 以world_size为组遍历参数
            for base_i in range(0, len(params), world_size):
                # 每个参数的计算所有者是rank i % world_size
                owner_idx = base_i + rank # 计算此rank拥有的参数索引
                # 等待reduce scatter完成
                all_reduce_futures[future_idx].wait() # 可能稍后我们可以使用wait_any轮询代替
                future_idx += 1
                # 所有者计算Muon更新，结果在其参数中
                if owner_idx < len(params):
                    p = params[owner_idx]
                    g = p.grad  # 现在已在所有rank之间平均
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1.0 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                    scale = (max(1.0, p.size(-2) / p.size(-1)) ** 0.5)
                    p.add_(g, alpha=-group["lr"] * scale)
                # 将更新后的参数复制到所有rank
                ag_input = params[owner_idx] if owner_idx < len(params) else zero_buffer
                ag_output = params[base_i:base_i + world_size]
                ag_output.extend([torch.empty_like(zero_buffer) for _ in range(world_size - len(ag_output))]) # 填充
                work = dist.all_gather(ag_output, ag_input, async_op=True).get_future()
                all_gather_futures.append(work)

        # 等待所有工作完成
        torch.futures.collect_all(all_gather_futures).wait()
