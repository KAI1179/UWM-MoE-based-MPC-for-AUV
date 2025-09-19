#!/usr/bin/env python3
import os
import sys
import h5py
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List

def robust_constant_from_series(arr: np.ndarray, valid_fn=None, fallback=-1):
    """
    从一段应该“常量”的逐步数组中，取鲁棒代表值：
    - 可选 valid_fn 过滤非法值（例如 fault_index 中的 <0）
    - 取过滤后数值的中位数作为代表；若为空则返回 fallback
    """
    if arr is None or len(arr) == 0:
        return fallback
    x = np.asarray(arr).astype(float)
    if valid_fn is not None:
        x = x[valid_fn(x)]
    if x.size == 0:
        return fallback
    return int(np.round(np.median(x)))

def main(h5_path: str):
    if not os.path.exists(h5_path):
        print(f"[ERR] File not found: {h5_path}", file=sys.stderr)
        sys.exit(1)

    with h5py.File(h5_path, "r") as f:
        # 找到所有 episode 组
        ep_keys: List[str] = sorted(
            [k for k in f.keys() if k.startswith("episode_")],
            key=lambda s: int(s.split("_")[1])
        )
        if not ep_keys:
            print("[WARN] No episodes found.")
            return

        regime_counter: Counter = Counter()
        fault_counter: Counter = Counter()
        bad_fault_eps: List[str] = []  # regime 4 里没有合法 fault_index 的 episode
        regime_to_eps: Dict[int, List[str]] = defaultdict(list)

        for k in ep_keys:
            g = f[k]
            # 读取逐步的 regime_id/fault_index，并鲁棒地提取“episode 级常量”
            reg_series = g["regime_id"][()] if "regime_id" in g else np.array([])
            fidx_series = g["fault_index"][()] if "fault_index" in g else np.array([])

            regime_id = robust_constant_from_series(reg_series, fallback=-1)
            regime_counter[regime_id] += 1
            regime_to_eps[regime_id].append(k)

            if regime_id == 4:
                # 有效 fault_index: 0..7
                fault_idx = robust_constant_from_series(
                    fidx_series, valid_fn=lambda x: (x >= 0) & (x <= 7), fallback=-1
                )
                if 0 <= fault_idx <= 7:
                    fault_counter[fault_idx] += 1
                else:
                    bad_fault_eps.append(k)

        # -------- 输出统计 --------
        total_eps = len(ep_keys)
        print(f"\n[OK] File: {h5_path}")
        print(f"[INFO] Total episodes: {total_eps}\n")

        # 1) 各 regime 的 episode 数量
        print("Per-regime episode counts:")
        for reg in sorted(regime_counter.keys()):
            print(f"  - regime {reg}: {regime_counter[reg]}")

        # 2) regime 4 的 fault_index 分布（0~7），包含 0 计数
        print("\nRegime 4 fault_index distribution (by episode):")
        for i in range(8):
            print(f"  - fault_index {i}: {fault_counter.get(i, 0)}")

        # 附加：异常情况提示
        if bad_fault_eps:
            print("\n[WARN] Episodes in regime 4 without a valid fault_index (expected 0..7):")
            for k in bad_fault_eps:
                print(f"  * {k}")

        # （可选）打印各 regime 示例 episode 范围
        # print("\nDebug: sample episodes per regime (up to 5 each):")
        # for reg, eps in sorted(regime_to_eps.items()):
        #     print(f"  regime {reg}: {eps[:5]}{' ...' if len(eps) > 5 else ''}")

if __name__ == "__main__":
    # 默认使用你给定的路径；也可在命令行传入其他文件路径
    default_path = "/home/xukai/ros2_ws/src/eeuv_sim/scripts/data_collector/data/rov_data_regimes_thrust1data10_10.hdf5"
    h5_path = sys.argv[1] if len(sys.argv) > 1 else default_path
    main(h5_path)
