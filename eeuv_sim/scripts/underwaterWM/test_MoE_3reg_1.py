#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_MoE_3reg.py
评测已训练的 MoE 世界模型（3 专家）：
1) 单步预测结果评估（1-step）
2) rollout 20 步（可配）结果评估
3) 工况（regime）预测性能（门控分类）

注意：
- 默认使用训练阶段保存的 standardizer（npz）对状态/动作做标准化。
- 若未找到 standardizer，可加 --no_standardize（但**不推荐**，与训练分布不一致）。
- 评测指标默认在“原始物理量单位”上统计（即先反标准化再计算误差）。

用法示例：
python test_MoE_3reg.py \
  --file_path /path/to/test.hdf5 \
  --ckpt /path/to/checkpoints/moe3_epoch120.pt \
  --save_dir ./eval_moe3 \
  --rollout_h 20
"""
import os, argparse, json, math, numpy as np, torch, h5py
from typing import Optional, Dict
from torch.utils.data import Dataset, DataLoader
from worldmodel_MoE_3reg_1 import WMConfigMoE3, MoEWorldModel3, rollout as moe_rollout

# --------------------------- 标准化工具（与训练匹配） ---------------------------
class SimpleStandardizer:
    def __init__(self):
        self.mean=None; self.std=None; self.u_mean=None; self.u_std=None

    def load(self, path:str):
        arr = np.load(path)
        self.mean = torch.tensor(arr["mean"], dtype=torch.float32)
        self.std  = torch.tensor(arr["std"],  dtype=torch.float32).clamp_min(1e-6)
        self.u_mean = torch.tensor(arr["u_mean"], dtype=torch.float32)
        self.u_std  = torch.tensor(arr["u_std"],  dtype=torch.float32).clamp_min(1e-6)

    def transform_states(self, X: torch.Tensor) -> torch.Tensor:
        # 把参数移到 X 所在 device，既兼容 CPU 也兼容 CUDA
        mean = self.mean.to(X.device)
        std  = self.std.to(X.device)
        return (X - mean) / std

    def inverse_transform_states(self, Xn: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(Xn.device)
        std  = self.std.to(Xn.device)
        return Xn * std + mean

    def transform_actions(self, U: torch.Tensor) -> torch.Tensor:
        u_mean = self.u_mean.to(U.device)
        u_std  = self.u_std.to(U.device)
        return (U - u_mean) / u_std

    def inverse_transform_actions(self, Un: torch.Tensor) -> torch.Tensor:
        u_mean = self.u_mean.to(Un.device)
        u_std  = self.u_std.to(Un.device)
        return Un * u_std + u_mean

# --------------------------- 评测数据集（读取+应用已保存的 standardizer） ---------------------------
class H5SeqDatasetEval(Dataset):
    """
    从 HDF5 读取序列片段；保留 regime_id ∈ {1,2,3}；
    返回 (x, x_next, u, regime_idx) 均为“已标准化”的张量（如启用标准化）。
    另外也返回未标准化的副本用于统计真实量纲误差。
    """
    def __init__(self, file_path:str, seq_len:int=64, stride:int=32,
                 standardizer: Optional[SimpleStandardizer]=None,
                 use_standardizer:bool=True):
        self.fp=file_path; self.seq_len=seq_len; self.stride=stride
        self.std=standardizer if (use_standardizer and standardizer is not None) else None
        self.episodes=[]; self.keys=[]; self.ep_meta={}
        with h5py.File(self.fp,"r") as f:
            all_eps=sorted([k for k in f.keys() if k.startswith("episode_")], key=lambda s:int(s.split("_")[1]))
            for k in all_eps:
                reg=int(f[k]["regime_id"][0]) if len(f[k]["regime_id"])>0 else -1
                if reg in (1,2,3):
                    self.episodes.append(k)
            for e,k in enumerate(self.episodes):
                N=len(f[k]["time"])
                if N<seq_len+1: continue
                for s in range(0, N-seq_len, stride):
                    self.keys.append((e,s))

    def __len__(self): return len(self.keys)

    def __getitem__(self, idx):
        ep_idx,s=self.keys[idx]
        with h5py.File(self.fp,"r") as f:
            k=self.episodes[ep_idx]
            pos=f[k]["position"][s:s+self.seq_len+1].astype(np.float32)
            ori=f[k]["orientation"][s:s+self.seq_len+1].astype(np.float32)
            lin=f[k]["linear_velocity"][s:s+self.seq_len+1].astype(np.float32)
            ang=f[k]["angular_velocity"][s:s+self.seq_len+1].astype(np.float32)
            x=np.concatenate([pos,ori,lin,ang],axis=1).astype(np.float32) # (T+1,13)

            u_app=f[k]["thrusts_applied"][s:s+self.seq_len].astype(np.float32)  # (T,8)
            # regime labels
            regime_step = None
            if "labels" in f[k] and "regime_step" in f[k]["labels"]:
                regime_step = f[k]["labels"]["regime_step"][s:s+self.seq_len].astype(np.int64)
            if regime_step is None:
                reg_val=int(f[k]["regime_id"][0]) if len(f[k]["regime_id"])>0 else 1
                regime_step = np.full((self.seq_len,), reg_val, dtype=np.int64)
            regime_idx = np.clip(regime_step,1,3) - 1  # ∈{0,1,2}

        # 标准化
        x_t  = x[:-1]   # (T,13)
        x_t1 = x[1:]    # (T,13)
        if self.std is not None:
            xt_n  = self.std.transform_states(torch.from_numpy(x_t))
            xt1_n = self.std.transform_states(torch.from_numpy(x_t1))
            u_n   = self.std.transform_actions(torch.from_numpy(u_app))
        else:
            xt_n  = torch.from_numpy(x_t)
            xt1_n = torch.from_numpy(x_t1)
            u_n   = torch.from_numpy(u_app)

        return {
            "x": xt_n.float(), "x_next": xt1_n.float(), "u": u_n.float(),
            "x_raw": torch.from_numpy(x_t).float(), "x_next_raw": torch.from_numpy(x_t1).float(),
            "u_raw": torch.from_numpy(u_app).float(),
            "regime_idx": torch.from_numpy(regime_idx).long()
        }

# --------------------------- 误差计算辅助 ---------------------------
@torch.no_grad()
def quat_angle_deg(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """q1,q2 shape (...,4); 返回弧度->度的相对旋转角（[0, 180]）。"""
    def _norm(q): return q / (q.norm(dim=-1, keepdim=True).clamp_min(1e-9))
    q1n=_norm(q1); q2n=_norm(q2)
    dot = (q1n*q2n).sum(dim=-1).abs().clamp(-1.0,1.0)
    ang = 2.0*torch.arccos(dot).clamp(0.0, math.pi)
    return ang * (180.0/math.pi)

def add_batch_stats(acc:Dict[str, float], cnt:Dict[str, int],
                    pred_next_raw: torch.Tensor, gt_next_raw: torch.Tensor):
    """在原始量纲上累计误差。pred/gt shape: (B,T,13)"""
    p_pred, q_pred = pred_next_raw[...,0:3], pred_next_raw[...,3:7]
    v_pred, w_pred = pred_next_raw[...,7:10], pred_next_raw[...,10:13]
    p_gt, q_gt = gt_next_raw[...,0:3], gt_next_raw[...,3:7]
    v_gt, w_gt = gt_next_raw[...,7:10], gt_next_raw[...,10:13]

    # RMSE 累计（向量维度先均方后平均）
    def _rmse_add(name, a, b):
        se = (a - b)**2
        se = se.mean(dim=-1)  # 每个样本/时间步的向量均方
        acc[name] = acc.get(name, 0.0) + float(se.sum().cpu().item())
        cnt[name] = cnt.get(name, 0) + int(se.numel())

    _rmse_add("pos_rmse", p_pred, p_gt)
    _rmse_add("linvel_rmse", v_pred, v_gt)
    _rmse_add("angvel_rmse", w_pred, w_gt)

    # 四元数角度误差（度）
    ang = quat_angle_deg(q_pred.reshape(-1,4), q_gt.reshape(-1,4))
    acc["quat_deg_sum"] = acc.get("quat_deg_sum", 0.0) + float(ang.sum().cpu().item())
    cnt["quat_deg_sum"] = cnt.get("quat_deg_sum", 0) + int(ang.numel())

# --------------------------- 单步评估 ---------------------------
@torch.no_grad()
def eval_single_step(model: MoEWorldModel3, loader: DataLoader, std: Optional[SimpleStandardizer], device="cpu"):
    model.eval()
    acc, cnt = {}, {}
    for batch in loader:
        x = batch["x"].to(device)             # (B,T,13) normalized (若 std 非空)
        u = batch["u"].to(device)             # (B,T,8) normalized (若 std 非空)
        x1_gt = batch["x_next"].to(device)    # (B,T,13) normalized
        # forward（MoE软路由平均/或根据 cfg.use_hard_routing 硬路由）
        out = model.forward_moe(x, u, train=False)
        D, w = out["deltas"], out["w"]         # (B,T,K,12), (B,T,K)
        if not model.cfg.use_hard_routing:
            delta = (D * w.unsqueeze(-1)).sum(dim=-2)  # (B,T,12)
        else:
            idx = w.argmax(dim=-1)  # (B,T)
            B,T,_ = D.shape[:3]
            delta = D[torch.arange(B)[:,None], torch.arange(T)[None,:], idx]  # (B,T,12)

        # 组合得到 x_next 预测（仍在“标准化域”）
        x_next_pred = model.compose_next(x.reshape(-1,13), delta.reshape(-1,12)).reshape(x.shape).to(device)
        # 反标准化到原始物理域
        if std is not None:
            pred_raw = std.inverse_transform_states(x_next_pred)
            gt_raw   = std.inverse_transform_states(x1_gt)
        else:
            pred_raw = x_next_pred
            gt_raw   = x1_gt
        add_batch_stats(acc, cnt, pred_raw, gt_raw)

    # 聚合
    eps = 1e-12
    metrics = {
        "1step/pos_RMSE": math.sqrt(acc.get("pos_rmse",0.0) / max(1,cnt.get("pos_rmse",0)) + eps),
        "1step/linvel_RMSE": math.sqrt(acc.get("linvel_rmse",0.0) / max(1,cnt.get("linvel_rmse",0)) + eps),
        "1step/angvel_RMSE": math.sqrt(acc.get("angvel_rmse",0.0) / max(1,cnt.get("angvel_rmse",0)) + eps),
        "1step/quat_deg": (acc.get("quat_deg_sum",0.0) / max(1,cnt.get("quat_deg_sum",0))),
    }
    return metrics

# --------------------------- Rollout 评估 ---------------------------
@torch.no_grad()
def eval_rollout(model: MoEWorldModel3, loader: DataLoader, std: Optional[SimpleStandardizer],
                 device="cpu", horizon:int=20):
    model.eval()
    acc, cnt = {}, {}
    for batch in loader:
        x = batch["x"].to(device)      # (B,T,13) normalized
        u = batch["u"].to(device)      # (B,T,8) normalized
        T = x.size(1)
        H = min(horizon, T-1)
        # 从 x[:,0] 开始，rollout H 步；注意 moe_rollout 接受 (B,13) 与 (B,H,8)
        x0 = x[:,0]
        u_seq = u[:,:H]
        x_hat = moe_rollout(model, x0, u_seq)     # (B,H+1,13) normalized
        # 与 GT 对齐：x 的第 1..H 步是 GT
        x_hat_tail = x_hat[:,1:]                  # (B,H,13)
        x_gt_tail  = x[:,1:1+H]                   # (B,H,13)
        # 反标准化
        if std is not None:
            pred_raw = std.inverse_transform_states(x_hat_tail)
            gt_raw   = std.inverse_transform_states(x_gt_tail)
        else:
            pred_raw = x_hat_tail
            gt_raw   = x_gt_tail
        add_batch_stats(acc, cnt, pred_raw, gt_raw)

    eps = 1e-12
    metrics = {
        f"roll{horizon}/pos_RMSE": math.sqrt(acc.get("pos_rmse",0.0) / max(1,cnt.get("pos_rmse",0)) + eps),
        f"roll{horizon}/linvel_RMSE": math.sqrt(acc.get("linvel_rmse",0.0) / max(1,cnt.get("linvel_rmse",0)) + eps),
        f"roll{horizon}/angvel_RMSE": math.sqrt(acc.get("angvel_rmse",0.0) / max(1,cnt.get("angvel_rmse",0)) + eps),
        f"roll{horizon}/quat_deg": (acc.get("quat_deg_sum",0.0) / max(1,cnt.get("quat_deg_sum",0))),
    }
    return metrics

# --------------------------- 工况（regime）预测评估 ---------------------------
@torch.no_grad()
def eval_regime_pred(model: MoEWorldModel3, loader: DataLoader, device="cpu"):
    model.eval()
    total=0; correct=0
    cm = torch.zeros(3,3, dtype=torch.long)  # [gt, pred]
    for batch in loader:
        x = batch["x"].to(device)             # normalized or raw 都可（只喂入门控）
        u = batch["u"].to(device)
        y = batch["regime_idx"].to(device)    # (B,T) ∈{0,1,2}
        out = model.forward_moe(x, u, train=False)
        logits = out["logits"]                # (B,T,3)
        yhat = logits.argmax(dim=-1)          # (B,T)
        total += y.numel()
        correct += int((yhat==y).sum().cpu().item())
        # 混淆矩阵
        for i in range(3):
            for j in range(3):
                cm[i,j] += int(((y==i) & (yhat==j)).sum().cpu().item())
    acc = correct / max(1,total)
    # 每类召回
    recall = {}
    for i, name in enumerate(["reg1","reg2","reg3"]):
        denom = int(cm[i,:].sum().cpu().item())
        recall[name] = (int(cm[i,i].cpu().item())/max(1,denom))
    return {"reg/acc": acc, "reg/recall_reg1": recall["reg1"], "reg/recall_reg2": recall["reg2"], "reg/recall_reg3": recall["reg3"],
            "reg/confusion": cm.tolist()}

# --------------------------- 加载 ckpt / 标准化文件 ---------------------------
def load_ckpt_make_model(ckpt_path:str, device:str="cpu") -> MoEWorldModel3:
    ckpt = torch.load(ckpt_path, map_location=device)
    # 从 ckpt 复原 config
    if "cfg" in ckpt and isinstance(ckpt["cfg"], dict):
        cfg = WMConfigMoE3(**ckpt["cfg"])
        cfg.device = device
    else:
        cfg = WMConfigMoE3(device=device)
    model = MoEWorldModel3(cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model

def find_standardizer(std_path: Optional[str], ckpt_path: str) -> Optional[str]:
    if std_path and os.path.isfile(std_path):
        return std_path
    # 尝试在 ckpt 同目录下寻找常见命名
    cand = []
    if ckpt_path:
        d = os.path.dirname(os.path.abspath(ckpt_path))
        cand += [os.path.join(d, n) for n in ["std_stats.npz", "std_state.npz", "standardizer_moe.npz"]]
    for p in cand:
        if os.path.isfile(p): return p
    return None

# --------------------------- 主流程 ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--file_path", type=str, required=True, help="测试集 HDF5 文件路径")
    p.add_argument("--ckpt", type=str, required=True, help="训练得到的 checkpoint (.pt)")
    p.add_argument("--std_path", type=str, default=None, help="standardizer npz 路径；默认自动在 ckpt 目录下查找")
    p.add_argument("--save_dir", type=str, default="./eval_moe3", help="评测日志保存目录（将写入 metrics.json）")
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--rollout_h", type=int, default=20)
    p.add_argument("--no_standardize", action="store_true", help="不做标准化（不推荐）")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = args.device

    # 1) 加载模型
    model = load_ckpt_make_model(args.ckpt, device=device)

    # 2) 装载 standardizer（优先使用提供路径，否则在 ckpt 同目录下自动查找）
    std_file = None if args.no_standardize else find_standardizer(args.std_path, args.ckpt)
    std = None
    if std_file is not None:
        try:
            std = SimpleStandardizer(); std.load(std_file)
            print(f"[Info] Loaded standardizer: {std_file}")
        except Exception as e:
            print(f"[Warn] Failed to load standardizer from {std_file}: {e}. 将以未标准化模式评测。")
            std = None
    else:
        if not args.no_standardize:
            print("[Warn] 未提供或未找到 standardizer npz，模型若以标准化训练，评测会失真。可手动指定 --std_path 或使用 --no_standardize 显式关闭。")

    # 3) 数据集 / DataLoader
    dataset = H5SeqDatasetEval(args.file_path, seq_len=args.seq_len, stride=args.stride,
                               standardizer=std, use_standardizer=(std is not None))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    # 4) 评测：单步 / rollout / 工况分类
    m1 = eval_single_step(model, loader, std, device=device)
    m2 = eval_rollout(model, loader, std, device=device, horizon=args.rollout_h)
    m3 = eval_regime_pred(model, loader, device=device)
    metrics = {}
    metrics.update(m1); metrics.update(m2); metrics.update(m3)

    # 打印与保存
    print("\n===== Evaluation Metrics =====")
    for k in sorted(metrics.keys()):
        v = metrics[k]
        if isinstance(v, float):
            print(f"{k:>24s}: {v:.6f}")
        else:
            print(f"{k:>24s}: {v}")
    out_json = os.path.join(args.save_dir, "metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] Saved metrics to: {out_json}")

if __name__ == "__main__":
    main()
