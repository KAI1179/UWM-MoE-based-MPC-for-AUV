# test.py
import os
import math
import argparse
import torch

from dataloader import make_dataloader_from_hdf5, list_episodes
from utils import Standardizer, build_delta_targets_seq, quat_normalize
from worldModel import WMConfig, ROVGRUModel, rollout

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_file", type=str, required=True, help="测试集 HDF5")
    p.add_argument("--ckpt", type=str, required=True, help="训练好的 checkpoint 路径（.pt）")
    p.add_argument("--std_path", type=str, default=None, help="standardizer.npz 路径（默认取 ckpt 同目录）")
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--k_rollout", type=int, default=10)
    p.add_argument("--episodes", type=str, nargs="*", default=None,
                   help="仅评估这些 episode（默认：全部）")
    p.add_argument("--no_attitude_rmse", action="store_true", help="不计算姿态角 RMSE")
    return p.parse_args()

@torch.no_grad()
def eval_loader(model, loader, cfg, k_rollout=10, calc_attitude=True):
    model.eval()
    nll_sum, nll_cnt = 0.0, 0
    pos_rmse_num, pos_rmse_den = 0.0, 0
    att_rmse_num, att_rmse_den = 0.0, 0  # 姿态角（弧度）

    for batch in loader:
        x = batch["x"].to(cfg.device)         # (B,T,13)
        u = batch["u"].to(cfg.device)         # (B,T,u_dim)
        x_next = batch["x_next"].to(cfg.device)
        mask = batch["mask"].to(cfg.device)

        # 1) 一步 NLL（teacher forcing）
        delta_gt = build_delta_targets_seq(x, x_next)
        pred = model(x, u)
        mu, logv = pred["mu"], pred["logvar"]
        inv_var = torch.exp(-logv)
        nll = 0.5 * (((delta_gt - mu)**2) * inv_var + logv).sum(dim=-1)  # (B,T)
        nll_sum += (nll * mask).sum().item()
        nll_cnt += mask.sum().item()

        # 2) k 步滚动
        B, T, _ = u.shape
        K = min(k_rollout, T-1)
        if K >= 1:
            x0 = x[:, 0]
            u_k = u[:, :K]
            x_hat = rollout(model, x0, u_k)      # (B,K+1,13)
            x_gt  = x[:, :K+1]                   # 对齐真值
            m_k   = mask[:, :K+1]

            # 位置 RMSE
            e_pos = (x_hat[..., 0:3] - x_gt[..., 0:3])**2  # (B,K+1,3)
            pos_rmse_num += (e_pos.sum(-1) * m_k).sum().item()
            pos_rmse_den += m_k.sum().item()

            if calc_attitude:
                # 姿态角误差（通过四元数误差的角度）
                qh, qg = x_hat[..., 3:7], x_gt[..., 3:7]
                qh = quat_normalize(qh); qg = quat_normalize(qg)
                # 角度 = 2*acos(|w|)；先算误差四元数 (qh ~= dq * qg => dq = qh * qg^{-1})
                qg_inv = torch.cat([qg[..., :1], -qg[..., 1:]], dim=-1)
                dq = torch.stack([
                    qh[...,0]*qg_inv[...,0] - qh[...,1]*qg_inv[...,1] - qh[...,2]*qg_inv[...,2] - qh[...,3]*qg_inv[...,3],
                    qh[...,0]*qg_inv[...,1] + qh[...,1]*qg_inv[...,0] + qh[...,2]*qg_inv[...,3] - qh[...,3]*qg_inv[...,2],
                    qh[...,0]*qg_inv[...,2] - qh[...,1]*qg_inv[...,3] + qh[...,2]*qg_inv[...,0] + qh[...,3]*qg_inv[...,1],
                    qh[...,0]*qg_inv[...,3] + qh[...,1]*qg_inv[...,2] - qh[...,2]*qg_inv[...,1] + qh[...,3]*qg_inv[...,0],
                ], dim=-1)
                w = torch.clamp(dq[..., 0].abs(), 0.0, 1.0)
                theta = 2.0 * torch.acos(w)  # (B,K+1)
                att_rmse_num += ((theta**2) * m_k).sum().item()
                att_rmse_den += m_k.sum().item()

    nll_step = nll_sum / max(1, nll_cnt)
    pos_rmse = math.sqrt(pos_rmse_num / max(1, pos_rmse_den)) if pos_rmse_den > 0 else float("nan")
    if calc_attitude:
        att_rmse = math.sqrt(att_rmse_num / max(1, att_rmse_den)) if att_rmse_den > 0 else float("nan")
    else:
        att_rmse = float("nan")
    return {"nll_step": nll_step, "pos_rmse@K": pos_rmse, "att_rmse@K(rad)": att_rmse}

def main():
    args = parse_args()
    # 1) 载入 ckpt & cfg
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg_dict = ckpt.get("cfg", None)
    if cfg_dict is None:
        raise RuntimeError("checkpoint 中缺少 cfg 字段")
    cfg = WMConfig(**cfg_dict)
    cfg.device = args.device

    # 2) 载入 standardizer（用训练集统计）
    std_path = args.std_path
    if std_path is None:
        std_path = os.path.join(os.path.dirname(args.ckpt), "standardizer.npz")
    if not os.path.exists(std_path):
        raise FileNotFoundError(f"未找到 standardizer: {std_path}")
    std = Standardizer.load(std_path)

    # 3) 测试 DataLoader（只应用训练的 std，不重新计算）
    #    注意：如果只想评估部分 episodes，传 --episodes ep1 ep2 ...
    dl_test, _, u_dim = make_dataloader_from_hdf5(
        file_path=args.test_file,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pad_last=True,
        use_standardizer=False,
        episode_whitelist=args.episodes
    )
    dl_test.dataset.std = std  # 应用训练统计

    # 4) 构建模型并载入权重
    model = ROVGRUModel(cfg).to(cfg.device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    # 5) 评估
    stats = eval_loader(model, dl_test, cfg, k_rollout=args.k_rollout, calc_attitude=(not args.no_attitude_rmse))
    print(f"[TEST] NLL(step): {stats['nll_step']:.4f} | pos RMSE@{args.k_rollout}: {stats['pos_rmse@K']:.4f}"
          + ("" if args.no_attitude_rmse else f" | att RMSE@{args.k_rollout}(rad): {stats['att_rmse@K(rad)']:.4f}"))

if __name__ == "__main__":
    main()
