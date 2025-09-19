# test_MoE.py
import os, re, json, math, argparse, numpy as np
from collections import defaultdict

import torch
import torch.nn.functional as F

# 复用训练期的数据管道与模型与标准化器
from run_MoE import H5SeqDataset, SimpleStandardizer
from worldmodel_MoE import WMConfigMoE, MoEWorldModel, rollout as rollout_fn

# ========== 实用函数 ==========
def find_latest_ckpt(save_dir: str):
    if not os.path.isdir(save_dir):
        return None
    cands = [f for f in os.listdir(save_dir) if re.match(r"moe_epoch\d+\.pt$", f)]
    if not cands:
        return None
    cands.sort(key=lambda s: int(re.search(r"(\d+)", s).group(1)))
    return os.path.join(save_dir, cands[-1])

def load_model(ckpt: str, device: str = "cuda"):
    blob = torch.load(ckpt, map_location=device)
    cfgd = blob.get("cfg", None)
    if cfgd is None:
        cfg = WMConfigMoE(device=device)
    else:
        cfg = WMConfigMoE(**{**cfgd, "device": device})
    model = MoEWorldModel(cfg).to(device)
    model.load_state_dict(blob["model_state"], strict=True)
    model.eval()
    return model, cfg, blob.get("epoch", None)

def load_standardizer(npz_path: str) -> SimpleStandardizer:
    """
    从训练保存的 standardizer_moe.npz 载入统计，避免测试分布漂移。
    """
    if (npz_path is None) or (not os.path.exists(npz_path)):
        return None
    blob = np.load(npz_path)
    std = SimpleStandardizer()
    std.mean   = torch.tensor(blob["mean"], dtype=torch.float32)
    std.std    = torch.tensor(blob["std"], dtype=torch.float32)
    std.u_mean = torch.tensor(blob["u_mean"], dtype=torch.float32)
    std.u_std  = torch.tensor(blob["u_std"], dtype=torch.float32)
    return std

def quat_angle_deg(q_pred: torch.Tensor, q_true: torch.Tensor) -> torch.Tensor:
    """四元数测地线角误差（度）。输入 (...,4)。"""
    qp = q_pred / (q_pred.norm(dim=-1, keepdim=True) + 1e-8)
    qt = q_true / (q_true.norm(dim=-1, keepdim=True) + 1e-8)
    dot = torch.clamp((qp * qt).sum(dim=-1).abs(), -1.0, 1.0)
    ang = 2.0 * torch.arccos(dot)  # [rad]
    return ang * (180.0 / math.pi)

def state_errors(x_hat: torch.Tensor, x_gt: torch.Tensor):
    """
    输入形状 (...,13)。返回 dict:
    - pos_mae, lin_mae, ang_mae, ori_deg
    """
    pos_mae = (x_hat[..., 0:3]  - x_gt[..., 0:3]).abs().mean()
    lin_mae = (x_hat[..., 7:10] - x_gt[..., 7:10]).abs().mean()
    ang_mae = (x_hat[...,10:13] - x_gt[...,10:13]).abs().mean()
    ori_deg = quat_angle_deg(x_hat[..., 3:7], x_gt[..., 3:7]).mean()
    return {"pos_mae": pos_mae, "ori_deg": ori_deg, "lin_mae": lin_mae, "ang_mae": ang_mae}

def fuse_delta(D: torch.Tensor, w: torch.Tensor, hard: bool = False):
    """
    D: (B,T,K,12)  w: (B,T,K)
    返回 (B,T,12)
    """
    if hard:
        idx = w.argmax(dim=-1)                             # (B,T)
        B, T, K, Ddim = D.shape
        out = D[torch.arange(B)[:,None], torch.arange(T)[None,:], idx]  # (B,T,12)
        return out
    else:
        return (D * w.unsqueeze(-1)).sum(dim=-2)

def thresh(x: torch.Tensor, t=0.5):  # -> {0,1}
    return (x >= t).to(torch.int32)

def f1_prec_rec(tp, fp, fn, eps=1e-8):
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    f1   = 2 * prec * rec / (prec + rec + eps)
    return f1, prec, rec

def macro_f1_from_scores(scores_fault: np.ndarray, gt_fault: np.ndarray, thr: float) -> (float, list):
    """
    scores_fault: (N,8)  模型输出的“故障概率”
    gt_fault:     (N,8)  GT 故障标签（故障=1，健康=0）
    thr: 阈值
    返回：macro_f1, per_channel_f1_list
    """
    ypred = (scores_fault > thr).astype(np.int32)
    f1s = []
    for ch in range(scores_fault.shape[1]):
        y = gt_fault[:, ch]
        p = ypred[:, ch]
        tp = int(((p == 1) & (y == 1)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        f1, _, _ = f1_prec_rec(tp, fp, fn)
        f1s.append(float(f1))
    macro = float(np.mean(f1s)) if f1s else 0.0
    return macro, f1s

# ========== 评测器 ==========
class Evaluator:
    def __init__(self, model: MoEWorldModel, device: str, use_hard_route: bool = False,
                 roll_h: int = 20, roll_stride: int = 2, only_postfail_for_reg4: bool = True):
        self.model = model
        self.device = device
        self.use_hard = use_hard_route
        self.roll_h = roll_h
        self.roll_stride = roll_stride
        self.only_postfail_for_reg4 = only_postfail_for_reg4

        # 汇总器
        self.step_err_sum = defaultdict(float)
        self.step_n = 0

        self.roll_err_sum = defaultdict(float)
        self.roll_n = 0

        # 分工况（0..3）
        self.step_err_sum_reg = [defaultdict(float) for _ in range(4)]
        self.step_n_reg = [0,0,0,0]
        self.roll_err_sum_reg = [defaultdict(float) for _ in range(4)]
        self.roll_n_reg = [0,0,0,0]

        # 工况分类（按工况掩码计分；reg4 仅 post-fail）
        self.cls_tp = np.zeros((4,), dtype=np.int64)
        self.cls_total = np.zeros((4,), dtype=np.int64)

        # FDI per-channel（thr@0.5累计计数）
        self.fdi_counts = {i: {"tp":0, "fp":0, "fn":0} for i in range(8)}
        self.fdi_counts_reg = [[{"tp":0, "fp":0, "fn":0} for _ in range(8)] for __ in range(4)]

        # FDI 阈值搜索：收集全体分数与GT
        self.fdi_scores_list = []  # 每批 (B*T,8) 的“故障概率”
        self.fdi_gt_list = []      # 每批 (B*T,8) 的 GT 故障标签

    @torch.no_grad()
    def feed_batch(self, batch):
        x   = batch["x"].to(self.device)         # (B,T,13)
        x1  = batch["x_next"].to(self.device)    # (B,T,13)
        u   = batch["u"].to(self.device)         # (B,T,8)
        reg = batch.get("regime_idx").to(self.device)  # (B,T) in {0..3}
        rw  = batch.get("regime_weight", None)   # (B,T) float
        if rw is not None: rw = rw.to(self.device)

        # ---- 单步：状态预测 + 工况分类 + FDI ----
        out = self.model.forward(x, u, train=False)           # logits,w,D
        D, w, logits = out["deltas"], out["w"], out["logits"] # (B,T,K,12),(B,T,K),(B,T,K)
        delta_hat = fuse_delta(D, w, hard=self.use_hard)      # (B,T,12)
        x_hat = self.model.compose_next(x, delta_hat)         # (B,T,13)

        # 状态误差
        errs = state_errors(x_hat, x1)
        for k,v in errs.items():
            self.step_err_sum[k] += float(v.detach().cpu())
        self.step_n += 1

        # 分工况 mask（注意：工况4可选择仅统计post-fail）
        if self.only_postfail_for_reg4:
            if rw is not None:
                mask_reg4 = (reg == 3) & (rw > 0.5)   # 只算post-fail步
            elif "health_gt" in batch:
                mask_reg4 = (reg == 3) & ((batch["health_gt"].to(self.device).min(dim=-1).values < 0.5))
            else:
                mask_reg4 = (reg == 3)
        else:
            mask_reg4 = (reg == 3)

        mask_by_reg = [
            (reg == 0),
            (reg == 1),
            (reg == 2),
            mask_reg4
        ]

        # 分工况误差
        for ridx in range(4):
            if mask_by_reg[ridx].any():
                m = mask_by_reg[ridx].unsqueeze(-1)  # (B,T,1)
                e_reg = state_errors(x_hat[m.expand_as(x_hat)].view(-1,13),
                                     x1[m.expand_as(x1)].view(-1,13))
                for k,v in e_reg.items():
                    self.step_err_sum_reg[ridx][k] += float(v.detach().cpu())
                self.step_n_reg[ridx] += 1

        # 工况分类（门控 argmax），工况4仅在 post-fail 计分
        y_pred = w.argmax(dim=-1)  # (B,T)
        for c in range(4):
            m = mask_by_reg[c].view(-1)
            total = int(m.sum().item())
            if total > 0:
                self.cls_total[c] += total
                self.cls_tp[c]    += int((y_pred.view(-1)[m] == c).sum().item())

        # FDI：m_hat∈[0,1] 健康；故障概率 = 1 - m_hat；thr@0.5计数，同时收集全体分数做阈值搜索
        if "health_gt" in batch:
            m_hat = self.model.fdi_forward(x, u)             # (B,T,8), 健康概率
            fault_prob = 1.0 - m_hat                         # 故障概率
            ypred  = (fault_prob >= 0.5).int()
            ytrue  = (1.0 - batch["health_gt"].to(self.device)).int()  # 故障=1

            # 用于阈值搜索的缓存
            self.fdi_scores_list.append(fault_prob.detach().cpu().numpy().reshape(-1, 8))
            self.fdi_gt_list.append(ytrue.detach().cpu().numpy().reshape(-1, 8))

            # 全局逐路（thr@0.5）
            for ch in range(8):
                tp = int(((ypred[...,ch]==1) & (ytrue[...,ch]==1)).sum().item())
                fp = int(((ypred[...,ch]==1) & (ytrue[...,ch]==0)).sum().item())
                fn = int(((ypred[...,ch]==0) & (ytrue[...,ch]==1)).sum().item())
                self.fdi_counts[ch]["tp"] += tp
                self.fdi_counts[ch]["fp"] += fp
                self.fdi_counts[ch]["fn"] += fn
            # 分工况
            for ridx in range(4):
                mask = mask_by_reg[ridx]
                if mask.any():
                    mh = ypred[mask]
                    yt = ytrue[mask]
                    for ch in range(8):
                        tp = int(((mh[...,ch]==1) & (yt[...,ch]==1)).sum().item())
                        fp = int(((mh[...,ch]==1) & (yt[...,ch]==0)).sum().item())
                        fn = int(((mh[...,ch]==0) & (yt[...,ch]==1)).sum().item())
                        self.fdi_counts_reg[ridx][ch]["tp"] += tp
                        self.fdi_counts_reg[ridx][ch]["fp"] += fp
                        self.fdi_counts_reg[ridx][ch]["fn"] += fn

        # ---- rollout ----
        B, T, _ = u.shape
        H = min(self.roll_h, T-1)
        if H <= 0:
            return
        starts = list(range(0, T-1-H+1, self.roll_stride))
        if not starts:
            return

        x0_list, u_list, y_list, reg0_list, rw0_list = [], [], [], [], []
        for b in range(B):
            for t0 in starts:
                x0_list.append(x[b, t0])                 # (13,)
                u_list.append(u[b, t0:t0+H])             # (H,8)
                y_list.append(x[b, t0+1:t0+H+1])         # (H,13)
                reg0_list.append(int(reg[b, t0].item()))
                if self.only_postfail_for_reg4:
                    if rw is not None:
                        rw0_list.append(float(rw[b, t0].item()))
                    elif "health_gt" in batch:
                        rw0_list.append(float((batch["health_gt"][b, t0].min().item() < 0.5)))
                    else:
                        rw0_list.append(1.0)
                else:
                    rw0_list.append(1.0)

        if not x0_list:
            return

        x0_b = torch.stack(x0_list, dim=0).to(self.device)     # (N,13)
        u_b  = torch.stack(u_list,  dim=0).to(self.device)     # (N,H,8)
        y_b  = torch.stack(y_list,  dim=0).to(self.device)     # (N,H,13)
        # 使用提供的 rollout()（内部按一步一步选择/融合专家）
        xhat_seq = rollout_fn(self.model, x0_b, u_b)           # (N,H+1,13)

        # 只对 1..H 的预测与真值做误差
        roll_err = state_errors(xhat_seq[:,1:], y_b)
        for k,v in roll_err.items():
            self.roll_err_sum[k] += float(v.detach().cpu())
        self.roll_n += 1

        # 分工况（按起点的工况归属；reg4 仅post-fail起点）
        for ridx in range(4):
            idx = [i for i,rg in enumerate(reg0_list) if rg == ridx]
            if ridx == 3 and self.only_postfail_for_reg4:
                idx = [i for i in idx if rw0_list[i] > 0.5]  # 仅post-fail的起点
            if idx:
                xb = xhat_seq[idx, 1:].contiguous().view(-1,13)
                yb = y_b[idx].contiguous().view(-1,13)
                e = state_errors(xb, yb)
                for k,v in e.items():
                    self.roll_err_sum_reg[ridx][k] += float(v.detach().cpu())
                self.roll_n_reg[ridx] += 1

    def summarize(self):
        step = {k: v/self.step_n for k,v in self.step_err_sum.items()} if self.step_n>0 else {}
        roll = {k: v/self.roll_n for k,v in self.roll_err_sum.items()} if self.roll_n>0 else {}

        # 分工况
        step_reg = []
        roll_reg = []
        for r in range(4):
            step_reg.append({k: (v/self.step_n_reg[r]) if self.step_n_reg[r]>0 else None
                             for k,v in self.step_err_sum_reg[r].items()})
            roll_reg.append({k: (v/self.roll_n_reg[r]) if self.roll_n_reg[r]>0 else None
                             for k,v in self.roll_err_sum_reg[r].items()})

        # 工况分类（overall 同样按“reg4只统计post-fail”的规则）
        cls = {}
        total_all = int(self.cls_total.sum())
        hit_all   = int(self.cls_tp.sum())
        for c in range(4):
            total = int(self.cls_total[c])
            hit   = int(self.cls_tp[c])
            cls[f"reg{c+1}_acc"] = (hit/total) if total>0 else None
        cls["overall_acc"] = (hit_all/total_all) if total_all>0 else None

        # FDI per-channel（thr@0.5计数）
        fdi = {}
        fdi_macro_f1 = []
        for ch in range(8):
            tp,fp,fn = self.fdi_counts[ch]["tp"], self.fdi_counts[ch]["fp"], self.fdi_counts[ch]["fn"]
            f1,pr,re = f1_prec_rec(tp,fp,fn)
            fdi[f"ch{ch}_f1"]   = float(f1)
            fdi[f"ch{ch}_prec"] = float(pr)
            fdi[f"ch{ch}_rec"]  = float(re)
            fdi_macro_f1.append(float(f1))
        fdi["macro_f1@0.5"] = float(np.mean(fdi_macro_f1)) if fdi_macro_f1 else None

        # FDI 分工况（thr@0.5）
        fdi_reg = []
        for r in range(4):
            fr = {}
            macro = []
            for ch in range(8):
                tp = self.fdi_counts_reg[r][ch]["tp"]
                fp = self.fdi_counts_reg[r][ch]["fp"]
                fn = self.fdi_counts_reg[r][ch]["fn"]
                f1,pr,re = f1_prec_rec(tp,fp,fn)
                fr[f"ch{ch}_f1"]   = float(f1)
                fr[f"ch{ch}_prec"] = float(pr)
                fr[f"ch{ch}_rec"]  = float(re)
                macro.append(float(f1))
            fr["macro_f1@0.5"] = float(np.mean(macro)) if macro else None
            fdi_reg.append(fr)

        # FDI 阈值搜索（全局）
        thr_search = {"best_thr": None, "macro_f1": None, "per_ch_f1": None}
        if self.fdi_scores_list and self.fdi_gt_list:
            scores_all = np.concatenate(self.fdi_scores_list, axis=0)  # (N,8) 故障概率
            gt_all     = np.concatenate(self.fdi_gt_list, axis=0)      # (N,8) 故障标签
            best_thr, best_macro, best_per = None, -1.0, None
            for thr in np.linspace(0.05, 0.95, 19):
                macro, per = macro_f1_from_scores(scores_all, gt_all, thr)
                if macro > best_macro:
                    best_macro = macro
                    best_thr   = float(thr)
                    best_per   = per
            thr_search = {"best_thr": best_thr, "macro_f1": best_macro, "per_ch_f1": best_per}

        return {
            "one_step": step,
            "rollout": roll,
            "one_step_by_regime": step_reg,
            "rollout_by_regime": roll_reg,
            "regime_cls": cls,
            "fdi": fdi,
            "fdi_by_regime": fdi_reg,
            "fdi_thr_search": thr_search,
            "counters": {
                "step_n": self.step_n,
                "roll_n": self.roll_n,
                "step_n_reg": self.step_n_reg,
                "roll_n_reg": self.roll_n_reg
            }
        }

# ========== 主流程 ==========
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str,
                   # default="/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data_regimes_thrust1data10_1.hdf5")
                   default="/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data_regimes_thrust1data10_1_labeled.hdf5")
    p.add_argument("--ckpt", type=str, default=None, help="权重路径（优先）。")
    p.add_argument("--save_dir", type=str, default="./checkpoints_moe", help="若未给 --ckpt，则在此目录中取最新。")
    p.add_argument("--std_path", type=str, default=None, help="标准化统计(npz)，默认会从 save_dir/standardizer_moe.npz 读取。")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use_hard_route", action="store_true", help="单步融合采用硬路由（默认软融合）。")
    p.add_argument("--roll_h", type=int, default=20, help="rollout 步数（默认 20）。")
    p.add_argument("--roll_stride", type=int, default=2, help="rollout 起点步距，越大评测越快。")
    p.add_argument("--only_postfail_reg4", action="store_true",
                   help="工况4仅统计 post-fail（默认开启）。")
    p.add_argument("--no_only_postfail_reg4", dest="only_postfail_reg4", action="store_false")
    p.set_defaults(only_postfail_reg4=True)
    p.add_argument("--out", type=str, default="eval_results.json")
    args = p.parse_args()

    ckpt_path = args.ckpt or find_latest_ckpt(args.save_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"未找到权重。请用 --ckpt 指定，或把训练产物放在 {args.save_dir} 下（moe_epoch*.pt）。")
    print(f"[Info] Using checkpoint: {ckpt_path}")

    model, cfg, ep = load_model(ckpt_path, device=args.device)
    print(f"[Info] Loaded epoch={ep}  device={args.device}  n_experts={cfg.n_experts}")

    # 解析标准化统计路径
    std_path = args.std_path
    if std_path is None:
        # 优先 checkpoints_moe/standardizer_moe.npz；若不存在，再尝试 std_stats.npz
        cand1 = os.path.join(args.save_dir, "standardizer_moe.npz")
        cand2 = os.path.join(args.save_dir, "std_stats.npz")
        std_path = cand1 if os.path.exists(cand1) else (cand2 if os.path.exists(cand2) else None)
    if std_path and (not os.path.exists(std_path)):
        print(f"[Warn] std_path not found: {std_path}")
        std_path = None

    # 数据（与训练一致）：seq_len 要 >= rollout_h + 1
    seq_len = max(args.seq_len, args.roll_h + 1)
    ds = H5SeqDataset(args.data, seq_len=seq_len, stride=seq_len//2, use_standardizer=True)

    # 关键：加载训练时期的标准化器，覆盖测试集重算的统计，避免分布漂移
    std_loaded = load_standardizer(std_path) if std_path else None
    if std_loaded is not None:
        ds.std = std_loaded
        print(f"[Info] Loaded standardizer from: {std_path}")
    else:
        print("[Warn] No standardizer loaded; using TEST-set-fitted statistics (results may drift).")

    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0,
        collate_fn=lambda b: {k: torch.stack([x[k] for x in b]) for k in b[0]}
    )

    ev = Evaluator(model, device=args.device, use_hard_route=args.use_hard_route,
                   roll_h=args.roll_h, roll_stride=args.roll_stride,
                   only_postfail_for_reg4=args.only_postfail_reg4)

    with torch.no_grad():
        for batch in loader:
            ev.feed_batch(batch)

    results = ev.summarize()

    # 打印简表
    def fmt(d, keys=("pos_mae","ori_deg","lin_mae","ang_mae")):
        return " ".join([f"{k}:{(d.get(k,None) if d else None):.4f}" if (d and d.get(k) is not None) else f"{k}:None" for k in keys])

    print("\n==== One-step (overall) ====")
    print(fmt(results["one_step"]))
    print("==== Rollout (overall) ====")
    print(fmt(results["rollout"]))

    names = ["①悬停/微速","②低速巡航","③急转/高攻角","④故障"]
    print("\n==== One-step by Regime ====")
    for i, r in enumerate(results["one_step_by_regime"]):
        print(f"Reg{i+1}({names[i]}): {fmt(r)}")
    print("\n==== Rollout by Regime ====")
    for i, r in enumerate(results["rollout_by_regime"]):
        print(f"Reg{i+1}({names[i]}): {fmt(r)}")

    print("\n==== Regime Classification Acc ====")
    for k,v in results["regime_cls"].items():
        if k != "overall_acc":
            print(f"{k}: {None if v is None else round(v,4)}")
    print(f"overall_acc: {None if results['regime_cls']['overall_acc'] is None else round(results['regime_cls']['overall_acc'],4)}")

    print("\n==== FDI per-channel (thr=0.5) ====")
    fdi = results["fdi"]
    for ch in range(8):
        print(f"ch{ch}: f1={fdi[f'ch{ch}_f1']:.4f} prec={fdi[f'ch{ch}_prec']:.4f} rec={fdi[f'ch{ch}_rec']:.4f}")
    print(f"FDI macro-F1@0.5: {fdi['macro_f1@0.5']:.4f}")

    if results.get("fdi_thr_search", {}).get("best_thr", None) is not None:
        thr_best = results["fdi_thr_search"]["best_thr"]
        macro_best = results["fdi_thr_search"]["macro_f1"]
        print(f"Best FDI thr={thr_best:.2f}, macro-F1={macro_best:.4f}")

    # 保存 JSON
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {args.out}")

if __name__ == "__main__":
    main()
