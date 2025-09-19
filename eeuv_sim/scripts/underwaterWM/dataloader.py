# dataloader.py
# 从 HDF5 读取 episode，生成 (x, u, x_next, mask) 序列样本
from typing import List, Dict, Tuple, Optional, Iterable
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import Standardizer, quat_normalize_np

# -----------------------------
# 工具：列出 episode 名称（按字典序）
# -----------------------------
def list_episodes(file_path: str) -> List[str]:
    with h5py.File(file_path, 'r') as f:
        eps = sorted(list(f.keys()))
    return eps

# -----------------------------
# 统计均值/方差（可选 episode 白名单）
# -----------------------------
def compute_stats_from_hdf5(
    file_path: str,
    episode_whitelist: Optional[Iterable[str]] = None
) -> Tuple[Standardizer, int]:
    """
    统计 x/u 的 mean/std，并返回推力维度 u_dim
    只对 episode_whitelist 指定的子集做统计（None = 全部）
    """
    x_accum, u_accum = [], []
    u_dim: Optional[int] = None

    with h5py.File(file_path, 'r') as f:
        eps = (episode_whitelist if episode_whitelist is not None else f.keys())
        for ep in eps:
            g = f[ep]
            pos = np.array(g["position"])                 # [N,3]
            ori = quat_normalize_np(np.array(g["orientation"]))  # [N,4]
            lv  = np.array(g["linear_velocity"])          # [N,3]
            av  = np.array(g["angular_velocity"])         # [N,3]
            thr = np.array(g["thrusts"])                  # [N, M]

            if u_dim is None:
                u_dim = thr.shape[1]
            else:
                assert thr.shape[1] == u_dim, "不同 episode 的 thrust 维度不一致"

            N = min(pos.shape[0], ori.shape[0], lv.shape[0], av.shape[0], thr.shape[0])
            if N < 2:
                continue

            x = np.concatenate([pos[:N], ori[:N], lv[:N], av[:N]], axis=-1)   # [N,13]
            x_accum.append(x)
            u_accum.append(thr[:N])

    if not x_accum or u_dim is None:
        raise RuntimeError("HDF5 中没有有效 episode 或 thrust 维度不可识别。")

    X = np.concatenate(x_accum, axis=0)
    U = np.concatenate(u_accum, axis=0)

    x_mean, x_std = X.mean(axis=0), X.std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    u_mean, u_std = U.mean(axis=0), U.std(axis=0)
    u_std[u_std < 1e-6] = 1.0

    std = Standardizer(x_mean=x_mean, x_std=x_std, u_mean=u_mean, u_std=u_std)
    return std, int(u_dim)

# -----------------------------
# 数据集（可选 episode 白名单）
# -----------------------------
class ROVHDF5SequenceDataset(Dataset):
    """
    产出窗口序列 (x, u, x_next, mask)
      x:      (T,13)
      u:      (T,u_dim)
      x_next: (T,13)  对齐：x_next[t] = x[t+1]
      mask:   (T,)    1 有效 / 0 padding
    """
    def __init__(self,
                 file_path: str,
                 seq_len: int = 32,
                 stride: int = 16,
                 pad_last: bool = True,
                 standardizer: Optional[Standardizer] = None,
                 episode_whitelist: Optional[Iterable[str]] = None):
        self.file_path = file_path
        self.seq_len = seq_len
        self.stride = stride
        self.pad_last = pad_last
        self.std = standardizer

        self.episodes: List[Dict[str, np.ndarray]] = []
        self.u_dim: Optional[int] = None

        with h5py.File(file_path, 'r') as f:
            eps = (episode_whitelist if episode_whitelist is not None else f.keys())
            for ep in eps:
                g = f[ep]
                pos = np.array(g["position"])
                ori = quat_normalize_np(np.array(g["orientation"]))
                lv  = np.array(g["linear_velocity"])
                av  = np.array(g["angular_velocity"])
                thr = np.array(g["thrusts"])

                if self.u_dim is None:
                    self.u_dim = thr.shape[1]
                else:
                    assert thr.shape[1] == self.u_dim, "不同 episode 的 thrust 维度不一致"

                N = min(pos.shape[0], ori.shape[0], lv.shape[0], av.shape[0], thr.shape[0])
                if N < 2:
                    continue

                pos, ori, lv, av, thr = pos[:N], ori[:N], lv[:N], av[:N], thr[:N]
                x = np.concatenate([pos, ori, lv, av], axis=-1)  # [N,13]
                u = thr                                          # [N,u_dim]
                self.episodes.append({"x": x, "u": u})

        if not self.episodes or self.u_dim is None:
            raise RuntimeError("未读取到有效 episode。")

        # 建索引 (ep_idx, start, valid_len)
        self.index: List[Tuple[int, int, int]] = []
        for e_idx, ep in enumerate(self.episodes):
            N = ep["x"].shape[0]
            # 有效转移步数 N-1；窗口最后一个 t 对应 next[t]=x[t+1]
            max_start = (N - 1) - 1
            if max_start < 0:
                continue
            s = 0
            while s <= max_start:
                valid = min(self.seq_len, (N - 1) - s)
                self.index.append((e_idx, s, valid))
                s += self.stride
            # 尾窗（覆盖到序列末尾）
            if self.pad_last and (len(self.index) == 0 or self.index[-1][0] != e_idx or self.index[-1][1] + self.index[-1][2] < (N - 1)):
                s = max(0, (N - 1) - self.seq_len)
                valid = (N - 1) - s
                if valid > 0:
                    if len(self.index) == 0 or self.index[-1] != (e_idx, s, valid):
                        self.index.append((e_idx, s, valid))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        e_idx, s, valid = self.index[idx]
        ep = self.episodes[e_idx]
        x_all = ep["x"]  # [N,13]
        u_all = ep["u"]  # [N,u_dim]

        x_win = x_all[s : s + valid + 1]
        u_win = u_all[s : s + valid]
        x_t   = x_win[:-1]
        x_tp1 = x_win[1:]

        if self.std is not None:
            x_t   = self.std.apply_x_np(x_t)
            x_tp1 = self.std.apply_x_np(x_tp1)
            u_win = self.std.apply_u_np(u_win)
        else:
            x_t[:, 3:7]   = quat_normalize_np(x_t[:, 3:7])
            x_tp1[:, 3:7] = quat_normalize_np(x_tp1[:, 3:7])

        T  = self.seq_len
        xd = np.zeros((T, 13), dtype=np.float32)
        ud = np.zeros((T, self.u_dim), dtype=np.float32)
        xn = np.zeros((T, 13), dtype=np.float32)
        m  = np.zeros((T,), dtype=np.float32)

        xd[:valid] = x_t.astype(np.float32)
        ud[:valid] = u_win.astype(np.float32)
        xn[:valid] = x_tp1.astype(np.float32)
        m[:valid]  = 1.0

        return {
            "x": torch.from_numpy(xd),
            "u": torch.from_numpy(ud),
            "x_next": torch.from_numpy(xn),
            "mask": torch.from_numpy(m),
        }

# -----------------------------
# DataLoader 构造（可选 episode 白名单）
# -----------------------------
def make_dataloader_from_hdf5(
    file_path: str,
    seq_len: int = 32,
    stride: int = 16,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pad_last: bool = True,
    use_standardizer: bool = True,
    episode_whitelist: Optional[Iterable[str]] = None,
) -> Tuple[DataLoader, Optional[Standardizer], int]:
    std, u_dim_from_stats = (None, None)
    if use_standardizer:
        std, u_dim_from_stats = compute_stats_from_hdf5(file_path, episode_whitelist)
    ds = ROVHDF5SequenceDataset(file_path=file_path,
                                seq_len=seq_len,
                                stride=stride,
                                pad_last=pad_last,
                                standardizer=std,
                                episode_whitelist=episode_whitelist)
    u_dim = ds.u_dim if u_dim_from_stats is None else u_dim_from_stats
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                    num_workers=num_workers, drop_last=False)
    return dl, std, u_dim
