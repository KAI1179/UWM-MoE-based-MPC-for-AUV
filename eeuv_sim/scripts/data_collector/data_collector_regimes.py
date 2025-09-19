import time
import numpy as np
import h5py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from gazebo_msgs.msg import EntityState
from eeuv_sim.srv import ResetToPose

import random

# =========================
# Regime-specific sampler
# =========================

class RegimeActionSampler:
    """
    Regime-aware thrust sampler using tanh-squashed AR(1) latent.
    Generates commanded thrusts u_cmd in [-Tmax, Tmax]^8.
    Supports four regimes:
      1: hover/micro-speed
      2: low-speed cruise
      3: aggressive turns/high AoA
      4: single-thruster failure (random i, t_fail ~ U[5,20]s)
    """
    def __init__(self, regime_id:int, thrust_limit:float=20.0, seed=None):
        self.regime_id = regime_id
        self.Tmax = thrust_limit
        self.rng = np.random.default_rng(seed)
        self.z = np.zeros(8)      # latent state for AR(1)
        self.u_prev = np.zeros(8) # previous commanded
        self.mask = np.ones(8)    # health mask for applied thrusts
        # Fault params (only used for regime 4)
        self.fault_idx = -1
        self.t_fail = -1.0
        self.failed = False

        # regime hyperparams
        if regime_id == 1:
            # hover: tiny amplitude, very smooth
            self.rho = 0.97
            self.gain = 0.3
            self.bias = np.zeros(8)
        elif regime_id == 2:
            # cruise: medium amplitude, smooth
            self.rho = 0.92
            self.gain = 0.8
            self.bias = self.rng.normal(0.0, 0.2, size=8)
        elif regime_id == 3:
            # aggressive: large amplitude, faster variation
            self.rho = 0.7
            self.gain = 1.5
            self.bias = self.rng.normal(0.0, 0.3, size=8)
        elif regime_id == 4:
            # failure: start as cruise-like, then one thruster fails
            self.rho = 0.9
            self.gain = 1.0
            self.bias = self.rng.normal(0.0, 0.2, size=8)
            self.fault_idx = int(self.rng.integers(0, 8))
            # self.t_fail = float(self.rng.uniform(5.0, 20.0))
            self.t_fail = float(self.rng.choice(range(5, 21)))
            self.mask[self.fault_idx] = 1.0  # initially healthy
        else:
            raise ValueError(f"Unknown regime_id {regime_id}")

    def _ar1_tanh(self):
        eps = self.rng.normal(0.0, 1.0, size=8)
        self.z = self.rho * self.z + np.sqrt(1 - self.rho**2) * eps
        u = self.Tmax * np.tanh(self.gain * self.z + self.bias)
        return u

    def step(self, t_now: float):
        """
        Return commanded and applied thrusts at time t_now (seconds from episode start).
        For regime 4, if t_now >= t_fail, one thruster's applied thrust is forced to zero
        (commanded remains unchanged so datasets can capture mismatch).
        """
        u_cmd = self._ar1_tanh()

        if self.regime_id == 4 and (not self.failed) and (t_now >= self.t_fail):
            self.mask[self.fault_idx] = 0.0
            self.failed = True

        u_applied = u_cmd * self.mask  # simulate actuator health mask
        self.u_prev = u_cmd
        return u_cmd, u_applied, self.mask.copy(), self.fault_idx, self.t_fail


# =========================
# Collector (reuses original structure)
# =========================

class ROVDataCollectorRegimes(Node):
    def __init__(self, thrust_hz=1.0, data_hz=10.0):
        super().__init__('rov_data_collector_regimes')
        self.thrust_hz = float(thrust_hz)
        self.data_hz = float(data_hz)
        self.thrust_dt = 1.0 / self.thrust_hz
        self.data_dt = 1.0 / self.data_hz
        self.thrust_to_data_per = max(1, int(round(self.data_hz / self.thrust_hz)))

        # Topics (keep same as example for compatibility)
        self.publisher = self.create_publisher(Float32MultiArray, '/ucat/thruster_cmd', 1)
        self.state_sub = self.create_subscription(EntityState, '/ucat/state', self.state_callback, 1)
        self.state = None
        self.current_state_time = 0.0
        self.last_state_time = 0.0

        # Timing helpers copied from example
        self.fast_forward = 1.0
        self.time_optimize_value = 0.0
        self.dt_optimize_gain = 0.05

        # Reset via /reset_to_pose (as required)
        self.reset_client = self.create_client(ResetToPose, '/reset_to_pose')
        if not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('/reset_to_pose service not available!')

    def state_callback(self, msg):
        self.state = msg
        self.current_state_time = time.time()

    def _wait_new_state(self, timeout_sec=0.1):
        last_seen = self.current_state_time
        end_time = time.time() + timeout_sec
        while time.time() < end_time:
            rclpy.spin_once(self, timeout_sec=0.0)
            if self.current_state_time != last_seen:
                return True
        return False

    def wait_time_optimizer(self, start_time, end_time):
        dt_error = (self.data_dt / self.fast_forward) - (end_time - start_time)
        if dt_error < 0.1:
            self.time_optimize_value += self.dt_optimize_gain * dt_error
            self.time_optimize_value = float(np.clip(self.time_optimize_value, -0.1, 0.1))

    def reset_environment(self):
        req = ResetToPose.Request()
        # Random pose within legal bounds (similar to example)
        x = random.uniform(10, 30)
        y = random.uniform(-10, 10)
        z = random.uniform(20, 5)    # NOTE: coordinate convention from example

        import tf_transformations
        yaw = random.uniform(-3.14, 3.14)

        req.x = float(x); req.y = float(y); req.z = float(z)
        req.roll = 0.0; req.pitch = 0.0; req.yaw = float(yaw)

        future = self.reset_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        self.state = None
        ok = self._wait_new_state(timeout_sec=1.0)
        if not ok:
            self.get_logger().warn("No updated state received after reset!")
        return True

    def step(self, action):
        msg = Float32MultiArray(data=action.tolist())
        self.publisher.publish(msg)

    def collect_episode(self, regime_id:int, max_steps=600, bounds=None):
        _ = self.reset_environment()
        sampler = RegimeActionSampler(regime_id=regime_id, thrust_limit=20.0,
                                      seed=int(time.time()*1e6)%2**32)

        data = {
            "time": [],
            "position": [],
            "orientation": [],
            "linear_velocity": [],
            "angular_velocity": [],
            "thrusts_cmd": [],
            "thrusts_applied": [],
            "health_mask": [],
            "regime_id": [],
            "fault_index": [],
            "t_fail": [],
        }

        step = 0
        start_wall = time.perf_counter()
        u_cmd = np.zeros(8)
        u_applied = np.zeros(8)
        mask = np.ones(8)

        self._wait_new_state(timeout_sec=0.2)

        while rclpy.ok() and step < max_steps:
            got_new = self._wait_new_state(timeout_sec=0.05)
            if not got_new or self.state is None:
                continue

            pos = self.state.pose.position
            ori = self.state.pose.orientation
            lin = self.state.twist.linear
            ang = self.state.twist.angular

            if bounds is not None:
                x, y, z = pos.x, pos.y, pos.z
                if not (bounds['x'][0] <= x <= bounds['x'][1] and
                        bounds['y'][0] <= y <= bounds['y'][1] and
                        bounds['z'][0] <= z <= bounds['z'][1]):
                    self.get_logger().info(f'Boundary hit: x={x:.2f}, y={y:.2f}, z={z:.2f}')
                    break

            step += 1
            t_now = time.perf_counter() - start_wall

            if step % self.thrust_to_data_per == 1:
                u_cmd, u_applied, mask, fault_idx, t_fail = sampler.step(t_now)

            data["time"].append(t_now)
            data["position"].append([pos.x, pos.y, pos.z])
            data["orientation"].append([ori.w, ori.x, ori.y, ori.z])
            data["linear_velocity"].append([lin.x, lin.y, lin.z])
            data["angular_velocity"].append([ang.x, ang.y, ang.z])
            data["thrusts_cmd"].append(u_cmd.tolist())
            data["thrusts_applied"].append(u_applied.tolist())
            data["health_mask"].append(mask.tolist())
            data["regime_id"].append(int(regime_id))
            if sampler.regime_id == 4:
                data["fault_index"].append(int(sampler.fault_idx))
                data["t_fail"].append(float(sampler.t_fail))
            else:
                data["fault_index"].append(-1)
                data["t_fail"].append(-1.0)

            self.step(u_applied)

            start_tick = time.perf_counter()
            try:
                time.sleep((self.data_dt / self.fast_forward) + self.time_optimize_value)
            except Exception:
                time.sleep(self.data_dt / self.fast_forward)
            end_tick = time.perf_counter()
            self.wait_time_optimizer(start_tick, end_tick)

        return data

    def save_to_hdf5(self, data, h5file, episode_idx):
        grp = h5file.create_group(f"episode_{episode_idx}")
        for k, v in data.items():
            grp.create_dataset(k, data=np.array(v), compression="gzip")
