import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, kl_divergence


def kl_divergence_manual(p_logits, q_logits):
    """
    手动计算KL散度 KL(p||q)
    参数:
        p_logits: 第一个分布的logits (batch_size, num_classes)
        q_logits: 第二个分布的logits (batch_size, num_classes)
    返回:
        KL散度值 (batch_size,)
    """
    # 将logits转换为概率分布
    p_probs = F.softmax(p_logits, dim=-1)
    q_probs = F.softmax(q_logits, dim=-1)

    # 计算KL散度: sum(p * log(p/q))
    kl = p_probs * (torch.log(p_probs + 1e-10) - torch.log(q_probs + 1e-10))
    return kl.sum(dim=-1)


def kl_divergence_categorical(p_dist, q_dist):
    """
    使用PyTorch分布计算离散分布的KL散度
    参数:
        p_dist: Categorical分布
        q_dist: Categorical分布
    返回:
        KL散度值
    """
    return kl_divergence(p_dist, q_dist)


def kl_divergence_normal(p_mean, p_std, q_mean, q_std):
    """
    计算正态分布的KL散度 KL(N(p_mean, p_std)||N(q_mean, q_std))
    参数:
        p_mean: 第一个正态分布的均值
        p_std: 第一个正态分布的标准差
        q_mean: 第二个正态分布的均值
        q_std: 第二个正态分布的标准差
    返回:
        KL散度值
    """
    p_dist = Normal(p_mean, p_std)
    q_dist = Normal(q_mean, q_std)
    return kl_divergence(p_dist, q_dist)


def kl_divergence_loss(p_logits, q_logits, reduction='mean'):
    """
    KL散度损失函数，常用于变分自编码器(VAE)等
    参数:
        p_logits: 预测分布的logits
        q_logits: 目标分布的logits
        reduction: 损失归约方式 ('none', 'mean', 'sum')
    返回:
        KL散度损失
    """
    kl = kl_divergence_manual(p_logits, q_logits)

    if reduction == 'mean':
        return kl.mean()
    elif reduction == 'sum':
        return kl.sum()
    elif reduction == 'none':
        return kl
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


# 示例用法
if __name__ == "__main__":
    # 设置随机种子以便重现结果
    torch.manual_seed(42)

    print("KL散度示例")
    print("=" * 50)

    # 示例1: 离散分布的KL散度
    print("\n1. 离散分布KL散度:")
    batch_size, num_classes = 2, 3

    # 创建两个概率分布
    p_logits = torch.randn(batch_size, num_classes)
    q_logits = torch.randn(batch_size, num_classes)

    print(f"p_logits: {p_logits}")
    print(f"q_logits: {q_logits}")

    # 手动计算KL散度
    kl_manual = kl_divergence_manual(p_logits, q_logits)
    print(f"手动计算KL(p||q): {kl_manual}")

    # 使用PyTorch分布计算
    p_probs = F.softmax(p_logits, dim=-1)
    q_probs = F.softmax(q_logits, dim=-1)
    p_dist = Categorical(probs=p_probs)
    q_dist = Categorical(probs=q_probs)
    kl_pytorch = kl_divergence(p_dist, q_dist)
    print(f"PyTorch分布KL(p||q): {kl_pytorch}")

    # 示例2: 正态分布的KL散度
    print("\n2. 正态分布KL散度:")
    p_mean = torch.tensor([0.0, 1.0])
    p_std = torch.tensor([1.0, 0.5])
    q_mean = torch.tensor([0.5, 1.5])
    q_std = torch.tensor([1.5, 0.8])

    kl_normal = kl_divergence_normal(p_mean, p_std, q_mean, q_std)
    print(f"KL(N({p_mean},{p_std})||N({q_mean},{q_std})): {kl_normal}")

    # 示例3: KL散度损失
    print("\n3. KL散度损失:")
    loss = kl_divergence_loss(p_logits, q_logits, reduction='mean')
    print(f"平均KL损失: {loss}")

    print("\nKL散度计算完成!")