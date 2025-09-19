import argparse, math
import numpy as np
import torch
import matplotlib.pyplot as plt

from worldModel import ROVGRUModel, WMConfig

from utils import Standardizer, quat_normalize_np
from dataloader import make_dataloader_from_hdf5

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--episode", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save_fig", default="./logvar.png")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    # 1) 加载模型
    cfg = WMConfig()                 # 若你的 ckpt 里保存了 cfg，就从 ckpt 读出来
    model = ROVGRUModel(cfg).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])

    # 2) 载入 episode
    reader = make_dataloader_from_hdf5(args.file, args.episode)  # 依你项目 API 调整
    x0, u_seq, x_seq = reader.get()  # x0 初始状态，u_seq (T,udim)，x_seq (T+1,xdim)
    dt = getattr(reader, "dt", None)

    # 3) rollout 并记录 logvar
    h = None
    x_pred = torch.from_numpy(x0[None, None, :]).float().to(device)
    logvars = []
    names = [
        "Δp_x","Δp_y","Δp_z",
        "Δv_x","Δv_y","Δv_z",
        "Δω_x","Δω_y","Δω_z",
        "Δθ_x","Δθ_y","Δθ_z"
    ]

    for t in range(u_seq.shape[0]):
        u_t = torch.from_numpy(u_seq[t:t+1, None, :]).float().to(device)
        with torch.no_grad():
            out = model(x_pred, u_t, h0=h)
            mu, lv = out["mu"], out["logvar"]           # (1,1,12)
            logvars.append(lv.squeeze(0).squeeze(0).cpu().numpy())
            # 用 mu 更新下一状态（你工程里现有的 compose_next）
            x_pred = compose_next(x_pred, u_t, mu)      # 按你项目函数签名替换
            h = out.get("h", None)

    logvars = np.stack(logvars, axis=0)  # (T,12)
    sigmas = np.exp(0.5 * logvars)

    # 4) 画图
    t_axis = np.arange(logvars.shape[0]) * (dt if dt is not None else 1.0)

    plt.figure(figsize=(10,6))
    for i in range(12):
        plt.plot(t_axis, logvars[:, i], label=names[i])
    plt.xlabel("time (s)" if dt is not None else "step")
    plt.ylabel("log variance")
    plt.title(f"logvar over time — {args.episode}")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(args.save_fig.replace(".png","_logvar.png"), dpi=200)

    plt.figure(figsize=(10,6))
    for i in range(12):
        plt.plot(t_axis, sigmas[:, i], label=names[i])
    plt.xlabel("time (s)" if dt is not None else "step")
    plt.ylabel("sigma (std dev)")
    plt.title(f"sigma over time — {args.episode}")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(args.save_fig.replace(".png","_sigma.png"), dpi=200)

    np.savez_compressed(args.save_fig.replace(".png","_uncert.npz"),
                        logvars=logvars, sigmas=sigmas, names=names, t=t_axis)

if __name__ == "__main__":
    main()
