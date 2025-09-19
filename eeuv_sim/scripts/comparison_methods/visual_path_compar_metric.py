import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import h5py

# ---------------------------
# 轨迹工具函数
# ---------------------------

def generate_smooth_3d_trajectory(waypoints, num_points=800):
    waypoints = np.array(waypoints, dtype=float)
    t = np.linspace(0, 1, len(waypoints))
    cs_x = CubicSpline(t, waypoints[:, 0])
    cs_y = CubicSpline(t, waypoints[:, 1])
    cs_z = CubicSpline(t, waypoints[:, 2])
    t_smooth = np.linspace(0, 1, num_points)
    x_smooth = cs_x(t_smooth)
    y_smooth = cs_y(t_smooth)
    z_smooth = cs_z(t_smooth)
    return np.vstack((x_smooth, y_smooth, z_smooth)).T

def read_actual_path(h5_file):
    """
    读取实际轨迹；若存在时间/姿态/推力等字段则一并读出。
    期望键名：
      - position: (N,3)
      - time 或 t: (N,)
      - yaw/pitch/roll: (N,) 可选
      - thruster 或 u: (N, m) 可选  用于控制代价
    """
    out = {"position": None, "time": None, "yaw": None, "pitch": None, "roll": None, "u": None}
    with h5py.File(h5_file, 'r') as f:
        out["position"] = np.array(f["position"])
        # 容错式读取
        if "time" in f: out["time"] = np.array(f["time"])
        elif "t" in f: out["time"] = np.array(f["t"])
        for k in ["yaw", "pitch", "roll"]:
            if k in f: out[k] = np.array(f[k])
        for k in ["thruster", "u", "input"]:
            if k in f:
                out["u"] = np.array(f[k])
                break
    return out

def path_length(P):
    d = np.diff(P, axis=0)
    return float(np.sum(np.linalg.norm(d, axis=1)))

def _project_point_to_segment(p, a, b):
    """返回投影点、到线段距离、归一化投影参数 tau∈[0,1]"""
    ab = b - a
    ab2 = np.dot(ab, ab)
    if ab2 <= 1e-12:
        return a, np.linalg.norm(p - a), 0.0
    tau = np.clip(np.dot(p - a, ab) / ab2, 0.0, 1.0)
    proj = a + tau * ab
    return proj, np.linalg.norm(p - proj), tau

def compute_cross_track_errors(traj_actual, traj_ref):
    """
    用近邻+线段投影计算每个实际点到参考曲线的横向误差（3D 欧氏距离）。
    同时给出对应参考弧长 s_ref，用于后续统计/对齐。
    """
    ref = np.asarray(traj_ref)
    act = np.asarray(traj_actual)
    # 参考轨迹弧长
    seg = np.diff(ref, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    s_cum = np.concatenate([[0.0], np.cumsum(seg_len)])  # len = M
    # KDTree 加速近邻定位
    tree = cKDTree(ref)
    dists = np.empty(len(act))
    s_hit  = np.empty(len(act))
    for i, p in enumerate(act):
        idx = tree.query(p, k=1)[1]
        # 在 idx-1--idx 与 idx--idx+1 两段上试投影
        candidates = []
        for j0, j1 in [(idx-1, idx), (idx, idx+1)]:
            if 0 <= j0 < len(ref)-1 and 0 <= j1 < len(ref):
                proj, dist, tau = _project_point_to_segment(p, ref[j0], ref[j1])
                s_here = s_cum[j0] + tau * seg_len[j0]
                candidates.append((dist, s_here))
        if not candidates:  # 端点
            dist = np.linalg.norm(p - ref[idx])
            s_here = s_cum[idx]
        else:
            dist, s_here = min(candidates, key=lambda x: x[0])
        dists[i] = dist
        s_hit[i] = s_here
    return dists, s_hit, s_cum[-1]

def curvature_smoothness(P):
    """
    离散平滑度指标：
      - avg_turn_deg_per_m：相邻段夹角（度）的路径长度归一化均值
      - ISC：积分平方曲率（简化版，用角度代替曲率近似）
    """
    P = np.asarray(P)
    v1 = P[1:-1] - P[:-2]
    v2 = P[2:] - P[1:-1]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    mask = (n1 > 1e-9) & (n2 > 1e-9)
    v1u = v1[mask] / n1[mask, None]
    v2u = v2[mask] / n2[mask, None]
    cosang = np.clip(np.einsum('ij,ij->i', v1u, v2u), -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))  # 每个折点的转角（度）
    L = path_length(P)
    avg_turn_deg_per_m = float(np.mean(ang) / max(L, 1e-9))
    ISC = float(np.sum((np.radians(ang))**2))  # 积分平方“曲率”近似
    return {
        "avg_turn_deg_per_m": avg_turn_deg_per_m,
        "ISC": ISC,
    }

def percent_within_threshold(errors, thr):
    if len(errors) == 0:
        return 0.0
    return float(np.mean(errors <= thr) * 100.0)

def evaluate_metrics(traj_ref, traj_act, time=None, yaw=None, pitch=None, roll=None, u=None,
                     cte_band=0.5):
    """
    返回一个指标字典，含主流论文常用的 CTE 统计、路径效率、平滑度；
    若可用则附加时间/姿态/控制指标。
    """
    # CTE 及弧长
    cte, s_hit, s_total = compute_cross_track_errors(traj_act, traj_ref)
    # 几何误差统计
    stats = {
        "CTE_MAE": float(np.mean(np.abs(cte))),
        "CTE_RMSE": float(np.sqrt(np.mean(cte**2))),
        "CTE_MAX": float(np.max(cte)),
        "CTE_P95": float(np.percentile(cte, 95.0)),
        "CTE_within_{:.2f}m_%".format(cte_band): percent_within_threshold(cte, cte_band)
    }
    # 路径效率
    L_ref = path_length(traj_ref)
    L_act = path_length(traj_act)
    stats.update({
        "PathLength_ref": L_ref,
        "PathLength_act": L_act,
        "PathLength_ratio_act_over_ref": float(L_act / max(L_ref, 1e-9))
    })
    # 平滑度（对实际轨迹）
    stats.update({f"Smooth_{k}": v for k, v in curvature_smoothness(traj_act).items()})
    # 可用就加：时间、姿态、控制
    if time is not None and len(time) == len(traj_act):
        stats.update({
            "Duration": float(time[-1] - time[0]),
            "Mean_speed_est": float(path_length(traj_act) / max(time[-1] - time[0], 1e-9))
        })
    # 姿态误差（若参考姿态不可得，则只报告变化平滑性可选；此处简化：不计算姿态误差）
    # 控制代价（示例：L2 总能量）
    if u is not None:
        u = np.asarray(u)
        stats.update({
            "Control_energy_L2": float(np.sum(np.linalg.norm(u, axis=1)**2))
        })
    return stats, cte, s_hit, s_total

# ---------------------------
# 配置：参考 & 实际
# ---------------------------

waypoints = [
    [5,   0,  -10],
    [12,  10, -20],
    [20, -10, -5],
    [28,   5, -18],
    [35,   0,  -8]
]

trajectory_ref = generate_smooth_3d_trajectory(waypoints, num_points=1000)
way_x, way_y, way_z = zip(*waypoints)

# === 你的 h5 日志 ===
# h5_path = "..."
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/underwaterWM/logs/wmpc_MoE_rov_log_0902_2100.h5"
# h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/underwaterWM/logs/wmpc_MoE_rov_log_0904_1340.h5" ## WMPC-UWM-MoE
h5_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/underwaterWM/logs/wmpc_rov_log_0825_1950.h5"  ## baseline UWM-MPC

log = read_actual_path(h5_path)
trajectory_actual = log["position"]
time_actual = log["time"]
yaw_actual = log["yaw"]
pitch_actual = log["pitch"]
roll_actual = log["roll"]
u_actual = log["u"]

# ---------------------------
# 计算指标
# ---------------------------
metrics, cte, s_hit, s_total = evaluate_metrics(
    trajectory_ref, trajectory_actual,
    time=time_actual, yaw=yaw_actual, pitch=pitch_actual, roll=roll_actual, u=u_actual,
    cte_band=0.5  # 你可改为 1.0 等
)

# 控制台友好打印
print("\n=== Quantitative Metrics (AUV/ROV Path-Following) ===")
for k, v in metrics.items():
    print(f"{k:36s}: {v:.6f}" if isinstance(v, float) else f"{k:36s}: {v}")

# ---------------------------
# 可视化（左：3D 轨迹；右：CTE 曲线）
# ---------------------------

fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type":"scene"}, {"type":"xy"}]],
    subplot_titles=("Reference vs Actual ROV Trajectory", "Cross-Track Error (CTE)")
)

# 参考轨迹
fig.add_trace(go.Scatter3d(
    x=trajectory_ref[:, 0], y=trajectory_ref[:, 1], z=trajectory_ref[:, 2],
    mode='lines', name='Reference Trajectory',
    line=dict(color='blue', width=4)
), row=1, col=1)

# 参考轨迹关键点
fig.add_trace(go.Scatter3d(
    x=way_x, y=way_y, z=way_z,
    mode='markers+text', name='Waypoints',
    marker=dict(size=6, color='red'),
    text=[f'P{i}' for i in range(len(waypoints))],
    textposition="top center"
), row=1, col=1)

# 实际轨迹
fig.add_trace(go.Scatter3d(
    x=trajectory_actual[:, 0], y=trajectory_actual[:, 1], z=trajectory_actual[:, 2],
    mode='lines+markers', name='Actual Trajectory',
    line=dict(color='green', width=3),
    marker=dict(size=3, color='green')
), row=1, col=1)

# CTE 曲线（按样本序号画；也可以按 s_hit 归一化到弧长）
fig.add_trace(go.Scatter(
    x=np.arange(len(cte)), y=cte,
    mode='lines', name='CTE (m)'
), row=1, col=2)
# 容差带
cte_band = 0.5
# fig.add_hline(y=cte_band, line=dict(color='gray', dash='dot'), row=1, col=2)
# fig.add_hline(y=0.0, line=dict(color='black', dash='dash'), row=1, col=2)
# fig.add_hline(y=-cte_band, line=dict(color='gray', dash='dot'), row=1, col=2)

def _hline_on_col2(fig, y, line):
    # 在第2列(CTE子图)画横线；x 跨满整张图（0~1 domain）
    fig.add_shape(
        type="line",
        x0=0, x1=1, xref="x2 domain",
        y0=y, y1=y, yref="y2",
        line=line,
        row=1, col=2
    )

_hline_on_col2(fig,  cte_band,  dict(color='gray',  dash='dot'))
_hline_on_col2(fig,  0.0,       dict(color='black', dash='dash'))
_hline_on_col2(fig, -cte_band,  dict(color='gray',  dash='dot'))


# 摘要文字框
summary_text = (
    f"CTE MAE: {metrics['CTE_MAE']:.3f} m<br>"
    f"CTE RMSE: {metrics['CTE_RMSE']:.3f} m<br>"
    f"CTE P95: {metrics['CTE_P95']:.3f} m<br>"
    f"CTE max: {metrics['CTE_MAX']:.3f} m<br>"
    f"Within ±{cte_band} m: {metrics[f'CTE_within_{cte_band:.2f}m_%']:.1f}%<br>"
    f"L_act/L_ref: {metrics['PathLength_ratio_act_over_ref']:.3f}<br>"
    f"Smooth avg_turn(deg/m): {metrics['Smooth_avg_turn_deg_per_m']:.4f}"
)
fig.add_annotation(text=summary_text, xref="paper", yref="paper",
                   x=0.98, y=0.02, showarrow=False, align="right",
                   bordercolor="black", borderwidth=1, bgcolor="white", opacity=0.9)

fig.update_scenes(
    xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
    row=1, col=1
)

fig.update_xaxes(title_text="Sample Index", row=1, col=2)
fig.update_yaxes(title_text="CTE (m)", row=1, col=2)

fig.update_layout(
    margin=dict(l=0, r=0, b=0, t=40),
    showlegend=True
)

fig.show()
