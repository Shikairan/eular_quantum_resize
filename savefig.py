import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---- 自适应读入 ----
npy_files = list(Path('.').glob('*.npy'))
if len(npy_files) != 1:
    raise FileNotFoundError('当前目录必须且只能有 1 个 .npy 文件')
file_path = npy_files[0]
E = np.load(file_path)          # 形状 (S,M,N)
S, M, N = E.shape
print(f'已加载 {file_path.name}  ->  shape={E.shape}')

# ---- 统计 ----
meanMN = E.mean(axis=0)         # (M,N)
stdMN  = E.std(axis=0)          # (M,N)

# 时间步维度（对 N 平均）
mean_step = meanMN.mean(axis=1)          # (M,)
std_step  = np.sqrt((stdMN**2).mean(axis=1) / N)

# 环境维度（对 M 平均）
mean_env = meanMN.mean(axis=0)           # (N,)
std_env  = np.sqrt((stdMN**2).mean(axis=0) / M)

# ---- 画图 ----
fig, ax = plt.subplots(1, 2, figsize=(13, 4), sharey=True)

x_step = np.arange(M)
ax[0].plot(x_step, mean_step, label='mean')
ax[0].fill_between(x_step, mean_step-std_step, mean_step+std_step,
                   alpha=.25, label='±1 std')
ax[0].set_title('Average error per time-step')
ax[0].set_xlabel('time-step')
ax[0].set_ylabel('error')
ax[0].grid(True)
ax[0].legend()

x_env = np.arange(N)
ax[1].plot(x_env, mean_env, label='mean')
ax[1].fill_between(x_env, mean_env-std_env, mean_env+std_env,
                   alpha=.25, label='±1 std')
ax[1].set_title('Average error per environment')
ax[1].set_xlabel('environment id')
ax[1].grid(True)
ax[1].legend()

plt.tight_layout()
plt.savefig("1.png")
