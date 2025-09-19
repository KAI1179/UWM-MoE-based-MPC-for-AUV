#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_MoE.py — MoE 世界模型评测（单步 + rollout）
特性：
- 读取评测集（与训练一致）
- 标准化严格对齐训练（四元数不标准化；x/u 用 standardizer_moe.npz）
- 单步评测：pos/lin/ang RMSE、quat geodesic 角误差（deg）、FDI（BCE/F1）、工况分类准确率
- 额外统计：逐路 FDI 最佳阈值（最大 F1）、阈值0.5下逐路 P/R/F1、工况混淆矩阵、各工况 RMSE
- rollout 评测：内置“最小驻留 + 滞回”专家路由策略，贴近推理期配置
"""

import os, re, json, math, argparse
from glob import glob
from typing import Dict, Any, Optional

import h5py
import numpy as np
import torch

from worldmodel_MoE import WMConfigMoE, MoEWorldModel  # 依赖你的模型文件

# -------------------- 数学工具 --------------------
def quat_angle_deg(qp: np.ndarray, qt: np.ndarray) -> np.ndarray:
    """四元数 geodesic 角误差（度），输入 wxyz。"""
    qp = qp / (np.linalg.norm(qp, axis=-1, keepdims=True) + 1e-8)
    qt = qt / (np.linalg.norm(qt, axis=-1, keepdims=True) + 1e-8)
    dot = np.clip(np.abs(np.sum(qp * qt, axis=-1)), -1.0, 1.0)
    return 2.0 * np.arccos(dot) * 180.0 / math.pi

def rmse(a: np.ndarray, b: np.ndarray, eps: float=1e-12) -> float:
    d = (a - b).reshape(-1)
    return float(np.sqrt(np.mean(d * d + eps)))

def bce_np(pred: np.ndarray, target: np.ndarray, eps: float=1e-9) -> float:
    """NaN/Inf 安全的 BCE。"""
    p = np.nan_to_num(pred, nan=0.5, posinf=1.0 - eps, neginf=eps)
    t = np.nan_to_num(target, nan=0.0)
    p = np.clip(p, eps, 1.0 - eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        loss = -(t * np.log(p) + (1.0 - t) * np.log(1.0 - p))
    valid = np.isfinite(loss)
    if not valid.any():
        return float("nan")
    return float(np.mean(loss[valid]))

def f1_bin_np(pred: np.ndarray, target: np.ndarray, thr: float=0.5, eps: float=1e-9) -> float:
    p = np.nan_to_num(pred, nan=0.5); t = np.nan_to_num(target, nan=0.0)
    y = (p >= thr).astype(np.int32); g = (t >= 0.5).astype(np.int32)
    tp = ((y==1) & (g==1)).sum(); fp = ((y==1) & (g==0)).sum(); fn = ((y==0) & (g==1)).sum()
    prec = tp / (tp + fp + eps); rec = tp / (tp + fn + eps)
    return float(2 * prec * rec / (prec + rec + eps))

def thr_prf1_at_threshold(p: np.ndarray, g: np.ndarray, thr: float=0.5):
    """单路 P/R/F1（阈值 thr）"""
    y = (np.nan_to_num(p, nan=0.5) >= thr).astype(np.int32)
    g = (np.nan_to_num(g, nan=0.0) >= 0.5).astype(np.int32)
    tp = ((y==1) & (g==1)).sum(); fp = ((y==1) & (g==0)).sum(); fn = ((y==0) & (g==1)).sum()
    prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
    f1 = 2*prec*rec/(prec+rec+1e-9)
    return float(prec), float(rec), float(f1)

def best_thresholds_per_thr(p: np.ndarray, g: np.ndarray, grid=None):
    """逐路网格搜索最佳阈值（最大 F1）。返回 [(thr,f1,prec,rec,pos_rate), ...]*8"""
    if grid is None:
        grid = np.linspace(0.1, 0.9, 17)
    out = []
    for j in range(p.shape[1]):
        pj = np.nan_to_num(p[:,j], nan=0.5)
        gj = np.nan_to_num(g[:,j], nan=0.0)
        mask = np.isfinite(pj) & np.isfinite(gj)
        if mask.sum() == 0:
            out.append((0.5, float("nan"), float("nan"), float("nan"), 0.0)); continue
        pj = pj[mask]; gj = gj[mask]
        best = (-1.0, 0.5, 0.0, 0.0)  # (f1,thr,prec,rec)
        for thr in grid:
            pr, rc, f1 = thr_prf1_at_threshold(pj, gj, thr)
            if f1 > best[0]:
                best = (f1, thr, pr, rc)
        pos_rate = float((gj>=0.5).mean())
        out.append((best[1], best[0], best[2], best[3], pos_rate))
    return out

# -------------------- 标准化器（训练对齐） --------------------
class StdFromNPZ:
    def __init__(self, npz_path: str):
        d = np.load(npz_path)
        self.mean   = torch.tensor(d["mean"],   dtype=torch.float32)
        self.std    = torch.tensor(d["std"],    dtype=torch.float32)
        self.u_mean = torch.tensor(d["u_mean"], dtype=torch.float32)
        self.u_std  = torch.tensor(d["u_std"],  dtype=torch.float32)

    def x_to_norm(self, X: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=X.device, dtype=X.dtype)
        std  = self.std.to(device=X.device, dtype=X.dtype)
        return (X - mean) / (std + 1e-8)

    def x_from_norm(self, Xn: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=Xn.device, dtype=Xn.dtype)
        std  = self.std.to(device=Xn.device, dtype=Xn.dtype)
        return Xn * (std + 1e-8) + mean

    def u_to_norm(self, U: torch.Tensor) -> torch.Tensor:
        u_mean = self.u_mean.to(device=U.device, dtype=U.dtype)
        u_std  = self.u_std.to(device=U.device, dtype=U.dtype)
        return (U - u_mean) / (u_std + 1e-8)

def load_standardizer(ckpt_dir: str) -> Optional[StdFromNPZ]:
    for name in ["standardizer_moe.npz", "std_stats.npz"]:
        p = os.path.join(ckpt_dir, name)
        if os.path.isfile(p):
            try:
                s = StdFromNPZ(p)
                print(f"[Info] Loaded standardizer: {p}")
                return s
            except Exception as e:
                print(f"[Warn] Failed loading {p}: {e}")
    print("[Warn] Standardizer not found. 评测将退化。")
    return None

# -------------------- 数据读取（materialize 为 numpy） --------------------
def read_episode(grp: h5py.Group) -> Dict[str, np.ndarray]:
    pos = grp["position"][()].astype(np.float32)
    ori = grp["orientation"][()].astype(np.float32)
    lin = grp["linear_velocity"][()].astype(np.float32)
    ang = grp["angular_velocity"][()].astype(np.float32)
    x = np.concatenate([pos, ori, lin, ang], axis=1).astype(np.float32)  # (T,13)

    # 控制（applied）
    if "thrusts_applied" in grp:
        u_app = grp["thrusts_applied"][()].astype(np.float32)
    else:
        found = None
        for k in ["applied_thrust", "u_applied", "thrusts", "u"]:
            if k in grp:
                found = grp[k][()].astype(np.float32); break
        if found is None:
            raise KeyError("未找到 thrusts_applied（或兼容字段）")
        u_app = found

    # 工况标签
    if "labels" in grp and "regime_step" in grp["labels"]:
        regime = grp["labels"]["regime_step"][()].astype(np.int32)
    elif "regime_id" in grp:
        regime = grp["regime_id"][()].astype(np.int32)
    else:
        regime = np.full((len(x),), 1, dtype=np.int32)

    # 健康掩码
    mask = None
    if "labels" in grp:
        lab = grp["labels"]
        if "health_mask_gt_hys" in lab:
            mask = lab["health_mask_gt_hys"][()].astype(np.float32)
        elif "health_mask_gt" in lab:
            mask = lab["health_mask_gt"][()].astype(np.float32)
    if mask is None and "health_mask" in grp:
        mask = grp["health_mask"][()].astype(np.float32)

    # 对齐长度
    T = min(len(x), len(u_app), len(regime))
    if mask is not None:
        T = min(T, len(mask))
        mask = mask[:T]
    return dict(x=x[:T], u_app=u_app[:T], regime=regime[:T], mask=mask)

def load_h5(path: str) -> Dict[str, Dict[str, np.ndarray]]:
    episodes = {}
    with h5py.File(path, "r") as f:
        keys = sorted([k for k in f.keys() if k.startswith("episode_")],
                      key=lambda s: int(re.findall(r"\d+", s)[0]))
        if not keys:
            raise RuntimeError("未找到 episode_* 分组")
        for k in keys:
            episodes[k] = read_episode(f[k])
    return episodes

# -------------------- 模型加载 --------------------
def latest_ckpt(ckpt_dir: str) -> str:
    cands = glob(os.path.join(ckpt_dir, "moe_epoch*.pt"))
    if not cands:
        raise FileNotFoundError(f"未在 {ckpt_dir} 找到 checkpoint")
    def ep_of(p):
        m = re.search(r"epoch(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else -1
    cands.sort(key=ep_of)
    return cands[-1]

def build_model(ckpt: str, device: str="cpu") -> MoEWorldModel:
    obj = torch.load(ckpt, map_location=device)
    cfg = WMConfigMoE(**obj["cfg"]) if "cfg" in obj else WMConfigMoE()
    model = MoEWorldModel(cfg).to(device)
    model.load_state_dict(obj["model_state"], strict=True)
    model.eval()
    return model

# -------------------- rollout（含最小驻留 + 滞回） --------------------
@torch.no_grad()
def rollout_predict(model: MoEWorldModel, x0: torch.Tensor, u_seq: torch.Tensor, H: int,
                    min_dwell_steps: int = 5, hysteresis_margin: float = 0.15) -> torch.Tensor:
    """
    x0: (B,13), u_seq: (B,H,8) —— 均已标准化；返回 (B,H+1,13)（包含 x0）
    策略：
      - 专家路由硬切，但最小驻留 min_dwell_steps（默认 0.5s/dt=0.1）
      - 切换仅当 新专家权重大于当前专家 + hysteresis_margin
    """
    B = x0.size(0)
    x_hat = torch.empty(B, H+1, x0.size(-1), device=x0.device, dtype=x0.dtype)
    x_hat[:, 0] = x0
    x_t = x0

    last_idx = None
    dwell_cnt = torch.zeros(B, dtype=torch.long, device=x0.device)

    for t in range(H):
        u_t = u_seq[:, t]  # (B,8)
        out = model.forward(x_t, u_t, train=False)  # 期待 out["deltas"]:(B,K,12), out["w"]:(B,K)
        D = out["deltas"]; w = out["w"]
        if D.dim() == 4:  # 兼容 (B=1,T,K,12)
            D = D[:, 0]; w = w[:, 0]
        idx_new = w.argmax(dim=-1)  # (B,)
        w_new = w.gather(1, idx_new.unsqueeze(1)).squeeze(1)

        if last_idx is None:
            use_idx = idx_new
            dwell_cnt = torch.ones_like(dwell_cnt)
        else:
            w_last = w.gather(1, last_idx.unsqueeze(1)).squeeze(1)
            better = (w_new - w_last) > hysteresis_margin
            allow  = (dwell_cnt >= min_dwell_steps) & better
            use_idx = torch.where(allow, idx_new, last_idx)
            dwell_cnt = torch.where(use_idx == last_idx, dwell_cnt + 1, torch.ones_like(dwell_cnt))

        last_idx = use_idx
        delta_t = D[torch.arange(B, device=D.device), use_idx]  # (B,12)
        x_t = model.compose_next(x_t, delta_t)
        x_hat[:, t+1] = x_t
    return x_hat

# -------------------- 单步评测 --------------------
@torch.no_grad()
def eval_one_step(model: MoEWorldModel, std, episodes: Dict[str, Dict[str, np.ndarray]], device="cpu") -> Dict[str, Any]:
    pos_err, lin_err, ang_err, quat_deg = [], [], [], []
    fdi_bces, fdi_f1s = [], []
    reg_acc_n, reg_acc_d = 0, 0

    # 额外统计
    K = int(model.cfg.num_experts) if hasattr(model, "cfg") and hasattr(model.cfg, "num_experts") else 4
    cm = np.zeros((K, K), dtype=np.int64)  # 混淆矩阵：GT 行（0..K-1），Pred 列
    reg_err = {rid: {"pos":[], "lin":[], "ang":[], "quat":[]} for rid in range(1, K+1)}
    m_pred_collect, m_gt_collect = [], []

    use_hard = getattr(model.cfg, "use_hard_routing", False)

    for name, ep in episodes.items():
        x = ep["x"]; u = ep["u_app"]
        T = min(len(x), len(u))
        if T < 2: continue

        x_in = torch.from_numpy(x[:-1]).to(device).float()
        u_in = torch.from_numpy(u[:-1]).to(device).float()
        if std is not None:
            x_in = std.x_to_norm(x_in)
            u_in = std.u_to_norm(u_in)

        out = model.forward(x_in, u_in, train=False)
        D = out["deltas"]; w = out["w"]; logits = out["logits"]
        if D.dim() == 4:  # (B=1,T-1,K,12) 兼容
            D = D[0]; w = w[0]; logits = logits[0]  # -> (T-1,K,12)/(T-1,K)

        if use_hard:
            idx = w.argmax(dim=-1)                                        # (T-1,)
            delta = D[torch.arange(D.size(0), device=D.device), idx]      # (T-1,12)
        else:
            delta = (D * w.unsqueeze(-1)).sum(dim=-2)                     # (T-1,12)

        x_next_hat_n = model.compose_next(x_in, delta)                    # (T-1,13)

        x_next_gt = torch.from_numpy(x[1:]).to(device).float()
        x_next_hat = std.x_from_norm(x_next_hat_n) if std is not None else x_next_hat_n
        x_next_hat = x_next_hat.detach().cpu().numpy()
        x_next_gt  = x_next_gt.detach().cpu().numpy()

        # 误差
        pos_err.append(rmse(x_next_hat[..., :3],    x_next_gt[..., :3]))
        lin_err.append(rmse(x_next_hat[..., 7:10],  x_next_gt[..., 7:10]))
        ang_err.append(rmse(x_next_hat[..., 10:13], x_next_gt[..., 10:13]))
        quat_deg.append(np.mean(quat_angle_deg(x_next_hat[..., 3:7], x_next_gt[..., 3:7])))

        # 工况混淆 & 分工况误差
        reg_seq = ep["regime"][1:1+logits.shape[0]]
        pred_reg0 = logits.argmax(dim=-1).detach().cpu().numpy()            # 0..K-1
        gt_reg0   = np.clip(reg_seq - 1, 0, K-1)                            # 0..K-1
        for a,b in zip(gt_reg0.tolist(), pred_reg0.tolist()):
            cm[a, b] += 1
        err_pos = np.sqrt(np.mean((x_next_hat[..., :3]    - x_next_gt[..., :3])**2, axis=-1))
        err_lin = np.sqrt(np.mean((x_next_hat[..., 7:10]  - x_next_gt[..., 7:10])**2, axis=-1))
        err_ang = np.sqrt(np.mean((x_next_hat[..., 10:13] - x_next_gt[...,10:13])**2, axis=-1))
        err_qua = quat_angle_deg(x_next_hat[..., 3:7], x_next_gt[..., 3:7])
        for rid in range(1, K+1):
            sel = (reg_seq == rid)
            if np.any(sel):
                reg_err[rid]["pos"].append(float(err_pos[sel].mean()))
                reg_err[rid]["lin"].append(float(err_lin[sel].mean()))
                reg_err[rid]["ang"].append(float(err_ang[sel].mean()))
                reg_err[rid]["quat"].append(float(err_qua[sel].mean()))

        # FDI
        if ep.get("mask", None) is not None:
            m_hat = model.fdi_forward(x_in.unsqueeze(0), u_in.unsqueeze(0))[0].detach().cpu().numpy()  # (T-1,8)
            m_gt  = ep["mask"][1:1+m_hat.shape[0]]
            fdi_bces.append(bce_np(m_hat, m_gt))
            fdi_f1s.append(f1_bin_np(m_hat, m_gt, 0.5))
            m_pred_collect.append(m_hat); m_gt_collect.append(m_gt)

        # Regime acc
        reg_acc_n += int(np.sum(pred_reg0 == gt_reg0))
        reg_acc_d += int(len(gt_reg0))

    # 汇总
    res = dict(
        step_pos_rmse=float(np.mean(pos_err)) if pos_err else None,
        step_lin_rmse=float(np.mean(lin_err)) if lin_err else None,
        step_ang_rmse=float(np.mean(ang_err)) if ang_err else None,
        step_quat_deg=float(np.mean(quat_deg)) if quat_deg else None,
    )
    if fdi_bces:
        res["fdi_bce"] = float(np.mean(fdi_bces))
        res["fdi_f1"]  = float(np.mean(fdi_f1s))
    if reg_acc_d > 0:
        res["regime_acc"] = float(reg_acc_n / max(1, reg_acc_d))

    # 工况混淆矩阵 & 分工况 RMSE
    res["regime_confusion"] = cm.tolist()
    for rid in range(1, K+1):
        for k in ["pos","lin","ang","quat"]:
            if reg_err[rid][k]:
                res[f"step_{k}_rmse_reg{rid}"] = float(np.mean(reg_err[rid][k]))

    # 逐路 FDI 最佳阈值 & 阈值0.5下指标
    if m_pred_collect:
        P = np.concatenate(m_pred_collect, axis=0)
        G = np.concatenate(m_gt_collect,   axis=0)
        best = best_thresholds_per_thr(P, G)
        res["fdi_thr_best"] = [
            {"thr": float(t), "f1": float(f1), "prec": float(pr), "rec": float(rc), "pos_rate": float(px)}
            for (t, f1, pr, rc, px) in best
        ]
        prc_rec_f1 = [thr_prf1_at_threshold(P[:,j], G[:,j], 0.5) for j in range(8)]
        res["fdi_thr_metrics@0.5"] = [{"prec": float(pr), "rec": float(rc), "f1": float(f1)} for (pr, rc, f1) in prc_rec_f1]
        # 也保留均值（与此前 fdi_f1 对应）
        res["fdi_f1_per_thr_mean"] = (np.mean(np.array([x["f1"] for x in res["fdi_thr_metrics@0.5"]])).item()
                                      if res.get("fdi_thr_metrics@0.5") else None)
    return res

# -------------------- Rollout 评测 --------------------
@torch.no_grad()
def eval_rollout(model: MoEWorldModel, std, episodes: Dict[str, Dict[str, np.ndarray]],
                 H: int=20, device="cpu", min_dwell_steps: int=5, hysteresis_margin: float=0.15) -> Dict[str, Any]:
    pos_all, lin_all, ang_all, quat_all = [], [], [], []
    per_step = dict(pos=[], lin=[], ang=[], quat=[])

    for name, ep in episodes.items():
        x = ep["x"]; u = ep["u_app"]
        T = min(len(x), len(u))
        if T <= H: continue

        for t0 in range(0, T - H - 1):
            x0 = torch.from_numpy(x[t0]).to(device).float().unsqueeze(0)     # (1,13)
            uH = torch.from_numpy(u[t0:t0+H]).to(device).float().unsqueeze(0)  # (1,H,8)
            if std is not None:
                x0_n = std.x_to_norm(x0); uH_n = std.u_to_norm(uH)
            else:
                x0_n, uH_n = x0, uH

            x_hat_n = rollout_predict(model, x0_n, uH_n, H=H,
                                      min_dwell_steps=min_dwell_steps,
                                      hysteresis_margin=hysteresis_margin)[0]  # (H+1,13)
            x_hat = std.x_from_norm(x_hat_n) if std is not None else x_hat_n
            x_hat = x_hat.detach().cpu().numpy(); x_gt = x[t0:t0+H+1]

            pos_all.append(rmse(x_hat[1:, :3],    x_gt[1:, :3]))
            lin_all.append(rmse(x_hat[1:, 7:10],  x_gt[1:, 7:10]))
            ang_all.append(rmse(x_hat[1:, 10:13], x_gt[1:, 10:13]))
            quat_all.append(float(quat_angle_deg(x_hat[1:, 3:7], x_gt[1:, 3:7]).mean()))

            per_step["pos"].append([rmse(x_hat[t, :3],   x_gt[t, :3])   for t in range(1, H+1)])
            per_step["lin"].append([rmse(x_hat[t, 7:10], x_gt[t, 7:10]) for t in range(1, H+1)])
            per_step["ang"].append([rmse(x_hat[t,10:13], x_gt[t,10:13]) for t in range(1, H+1)])
            per_step["quat"].append([float(quat_angle_deg(x_hat[t,3:7][None], x_gt[t,3:7][None])[0])
                                      for t in range(1, H+1)])

    res = dict(
        roll_pos_rmse=float(np.mean(pos_all)) if pos_all else None,
        roll_lin_rmse=float(np.mean(lin_all)) if lin_all else None,
        roll_ang_rmse=float(np.mean(ang_all)) if ang_all else None,
        roll_quat_deg=float(np.mean(quat_all)) if quat_all else None,
    )
    if per_step["pos"]:
        for k in ["pos","lin","ang","quat"]:
            res[f"per_step_{k}"] = (np.mean(np.array(per_step[k]), axis=0)).tolist()
    return res

# -------------------- 主函数 --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--ckpt_dir", type=str, default="./checkpoints_moe")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--out", type=str, default="./eval_out_moe.json")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--min_dwell_steps", type=int, default=5, help="专家最小驻留步数（dt=0.1→5=0.5s）")
    ap.add_argument("--hysteresis_margin", type=float, default=0.15, help="切换滞回裕度")
    args = ap.parse_args()

    device = args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    if device != args.device:
        print("[Warn] CUDA 不可用，退回 CPU。")

    print(f"[Info] Loading dataset: {args.data}")
    episodes = load_h5(args.data)
    steps = sum(len(e["x"]) for e in episodes.values())
    print(f"[Info] Episodes: {len(episodes)} | Total steps: {steps}")

    ckpt_path = args.ckpt if args.ckpt else latest_ckpt(args.ckpt_dir)
    print(f"[Info] Using checkpoint: {ckpt_path}")
    model = build_model(ckpt_path, device=device)
    std = load_standardizer(os.path.dirname(ckpt_path))

    # 单步
    print("[Eval] One-step ...")
    m1 = eval_one_step(model, std, episodes, device=device)
    for k,v in m1.items():
        if isinstance(v, float) and v is not None:
            print(f"  {k}: {v:.6f}")
        elif v is not None:
            print(f"  {k}: {v}")

    # Rollout
    H = args.horizon
    print(f"[Eval] Rollout (H={H}) ...")
    m2 = eval_rollout(model, std, episodes, H=H, device=device,
                      min_dwell_steps=args.min_dwell_steps,
                      hysteresis_margin=args.hysteresis_margin)
    for k,v in m2.items():
        if isinstance(v, float) and v is not None:
            print(f"  {k}: {v:.6f}")
        elif v is not None:
            print(f"  {k}: len={len(v)}")

    out = dict(
        data=args.data,
        ckpt=ckpt_path,
        device=device,
        horizon=H,
        one_step=m1,
        rollout=m2,
        min_dwell_steps=args.min_dwell_steps,
        hysteresis_margin=args.hysteresis_margin,
    )
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] Results saved to: {args.out}")

if __name__ == "__main__":
    main()
