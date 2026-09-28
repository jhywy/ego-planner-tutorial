"""
EGO-Planner 教学版完整演示

功能：
1. 使用栅格 A* 生成无碰撞初始路径，避免从穿过障碍物的直线开始优化。
2. 使用 ESDF（欧氏距离场）计算障碍物距离和梯度。
3. 使用三次均匀 B-spline 表示连续平滑轨迹。
4. 优化 B-spline 控制点：安全性、平滑性、目标约束。
5. 对最终高密度轨迹进行连续碰撞检查。
6. 显示轨迹、控制点、ESDF、安全边界和代价收敛曲线。

安装依赖：
    pip install numpy matplotlib scipy

说明：
本程序是用于理解 EGO-Planner 思想的教学实现，不是官方 ROS 工程实现。
官方实现还包含真实传感器、局部地图、动力学约束和更复杂的优化过程。
"""

import heapq
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class EGOPlannerDemo:
    """二维 EGO-Planner 教学版。"""

    def __init__(
        self,
        start,
        goal,
        obstacles,
        world_size=(10.0, 10.0),
        resolution=0.1,
        clearance=0.45,
    ):
        self.start = np.asarray(start, dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        self.obstacles = list(obstacles)  # [(cx, cy, radius), ...]
        self.world_size = np.asarray(world_size, dtype=float)
        self.resolution = float(resolution)
        self.clearance = float(clearance)

        self.x_min = -self.world_size[0] / 2.0
        self.y_min = -self.world_size[1] / 2.0
        self.nx = int(round(self.world_size[0] / self.resolution)) + 1
        self.ny = int(round(self.world_size[1] / self.resolution)) + 1

        self.x_coords = np.linspace(
            self.x_min, self.x_min + (self.nx - 1) * self.resolution, self.nx
        )
        self.y_coords = np.linspace(
            self.y_min, self.y_min + (self.ny - 1) * self.resolution, self.ny
        )
        self.X, self.Y = np.meshgrid(self.x_coords, self.y_coords)
        self.esdf = self.build_esdf()

    # ------------------------------------------------------------------
    # 地图和 ESDF
    # ------------------------------------------------------------------
    def build_esdf(self):
        """计算每个栅格到最近圆形障碍物边界的有符号距离。"""
        distance = np.full_like(self.X, np.inf, dtype=float)

        for cx, cy, radius in self.obstacles:
            d = np.sqrt((self.X - cx) ** 2 + (self.Y - cy) ** 2) - radius
            distance = np.minimum(distance, d)

        return distance

    def esdf_value(self, point):
        """用双线性插值查询任意世界坐标点的 ESDF 值。"""
        x, y = np.asarray(point, dtype=float)
        gx = (x - self.x_min) / self.resolution
        gy = (y - self.y_min) / self.resolution

        gx = np.clip(gx, 0.0, self.nx - 1.000001)
        gy = np.clip(gy, 0.0, self.ny - 1.000001)
        ix = int(np.floor(gx))
        iy = int(np.floor(gy))
        fx = gx - ix
        fy = gy - iy

        v00 = self.esdf[iy, ix]
        v10 = self.esdf[iy, min(ix + 1, self.nx - 1)]
        v01 = self.esdf[min(iy + 1, self.ny - 1), ix]
        v11 = self.esdf[min(iy + 1, self.ny - 1), min(ix + 1, self.nx - 1)]

        return (
            (1 - fx) * (1 - fy) * v00
            + fx * (1 - fy) * v10
            + (1 - fx) * fy * v01
            + fx * fy * v11
        )

    def esdf_gradient(self, point, eps=None):
        """使用中心差分计算 ESDF 梯度。"""
        if eps is None:
            eps = self.resolution
        p = np.asarray(point, dtype=float)
        dx = np.array([eps, 0.0])
        dy = np.array([0.0, eps])
        return np.array([
            (self.esdf_value(p + dx) - self.esdf_value(p - dx)) / (2 * eps),
            (self.esdf_value(p + dy) - self.esdf_value(p - dy)) / (2 * eps),
        ])

    # ------------------------------------------------------------------
    # 栅格 A*：只用于生成安全初值，不是最终轨迹
    # ------------------------------------------------------------------
    def world_to_grid(self, point):
        x, y = np.asarray(point, dtype=float)
        ix = int(round((x - self.x_min) / self.resolution))
        iy = int(round((y - self.y_min) / self.resolution))
        return ix, iy

    def grid_to_world(self, node):
        ix, iy = node
        return np.array([
            self.x_min + ix * self.resolution,
            self.y_min + iy * self.resolution,
        ])

    def grid_is_free(self, node, margin=None):
        ix, iy = node
        if ix < 0 or ix >= self.nx or iy < 0 or iy >= self.ny:
            return False
        if margin is None:
            margin = self.clearance
        return self.esdf[iy, ix] >= margin

    def astar_initial_path(self):
        """在膨胀障碍物地图中运行 8 邻域 A*。"""
        start = self.world_to_grid(self.start)
        goal = self.world_to_grid(self.goal)

        # 将起点、终点吸附到最近的安全栅格
        if not self.grid_is_free(start):
            start = self.nearest_free_grid(start)
        if not self.grid_is_free(goal):
            goal = self.nearest_free_grid(goal)

        motions = [
            (-1, -1, np.sqrt(2)), (-1, 0, 1), (-1, 1, np.sqrt(2)),
            (0, -1, 1), (0, 1, 1),
            (1, -1, np.sqrt(2)), (1, 0, 1), (1, 1, np.sqrt(2)),
        ]

        def heuristic(a, b):
            return np.hypot(a[0] - b[0], a[1] - b[1])

        open_heap = [(heuristic(start, goal), 0.0, start)]
        parent = {start: None}
        g_cost = {start: 0.0}
        visited = set()

        while open_heap:
            _, current_g, current = heapq.heappop(open_heap)
            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                path = []
                node = current
                while node is not None:
                    path.append(self.grid_to_world(node))
                    node = parent[node]
                return np.asarray(path[::-1])

            for dx, dy, motion_cost in motions:
                nxt = (current[0] + dx, current[1] + dy)
                if not self.grid_is_free(nxt):
                    continue

                # 禁止对角穿越障碍物的尖角
                if dx != 0 and dy != 0:
                    if not self.grid_is_free((current[0] + dx, current[1])):
                        continue
                    if not self.grid_is_free((current[0], current[1] + dy)):
                        continue

                new_g = current_g + motion_cost
                if new_g < g_cost.get(nxt, np.inf):
                    g_cost[nxt] = new_g
                    parent[nxt] = current
                    f = new_g + heuristic(nxt, goal)
                    heapq.heappush(open_heap, (f, new_g, nxt))

        raise RuntimeError("A* 没有找到安全初始路径，请扩大地图或减小 clearance。")

    def nearest_free_grid(self, node):
        """寻找距离给定栅格最近的安全栅格。"""
        ix, iy = node
        candidates = []
        for radius in range(max(self.nx, self.ny)):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    p = (ix + dx, iy + dy)
                    if self.grid_is_free(p):
                        candidates.append(p)
            if candidates:
                return min(candidates, key=lambda p: (p[0] - ix) ** 2 + (p[1] - iy) ** 2)
        raise RuntimeError("地图中没有可用的安全栅格。")

    @staticmethod
    def simplify_polyline(path, max_points=8):
        """均匀抽取 A* 路径点作为 B-spline 控制点初值。"""
        if len(path) <= max_points:
            return path.copy()
        indices = np.linspace(0, len(path) - 1, max_points).round().astype(int)
        return path[np.unique(indices)]

    # ------------------------------------------------------------------
    # B-spline 表示
    # ------------------------------------------------------------------
    @staticmethod
    def bspline_trajectory(control_points, n_samples=1000):
        """用开放均匀节点向量生成真正的三次 B-spline。"""
        cp = np.asarray(control_points, dtype=float)
        degree = 3
        if len(cp) < degree + 1:
            raise ValueError("三次 B-spline 至少需要 4 个控制点。")

        # n 个控制点、degree 次数需要 n + degree + 1 个节点
        interior = np.linspace(0.0, 1.0, len(cp) - degree + 1)
        knots = np.concatenate([
            np.zeros(degree),
            interior,
            np.ones(degree),
        ])
        t = np.linspace(0.0, 1.0, n_samples)
        sx = BSpline(knots, cp[:, 0], degree)
        sy = BSpline(knots, cp[:, 1], degree)
        return np.column_stack((sx(t), sy(t)))

    # ------------------------------------------------------------------
    # 代价和优化
    # ------------------------------------------------------------------
    def trajectory_cost(self, control_points):
        trajectory = self.bspline_trajectory(control_points, 500)
        distances = np.array([self.esdf_value(p) for p in trajectory])

        # 障碍物和安全距离代价
        obstacle_cost = np.sum(np.where(
            distances < 0.0,
            100000.0 * (1.0 - distances) ** 2,
            np.where(
                distances < self.clearance,
                1000.0 * (self.clearance - distances) ** 2,
                0.0,
            ),
        ))

        # 离散二阶差分近似曲率/加速度，鼓励平滑
        second_diff = (
            trajectory[2:] - 2.0 * trajectory[1:-1] + trajectory[:-2]
        )
        smooth_cost = 20.0 * np.sum(second_diff ** 2)

        # 控制点长度正则，避免不必要的大幅折返
        first_diff = np.diff(control_points, axis=0)
        length_cost = 0.05 * np.sum(first_diff ** 2)

        # 固定起点和终点约束
        endpoint_cost = 10000.0 * (
            np.sum((control_points[0] - self.start) ** 2)
            + np.sum((control_points[-1] - self.goal) ** 2)
        )

        return float(obstacle_cost + smooth_cost + length_cost + endpoint_cost)

    def optimize(self, steps=60, learning_rate=0.01):
        """使用有限差分优化 B-spline 中间控制点。"""
        initial_path = self.astar_initial_path()
        initial_path = self.simplify_polyline(initial_path, max_points=8)

        # 保证至少有 4 个控制点
        if len(initial_path) < 4:
            t = np.linspace(0.0, 1.0, 4)
            control_points = self.start + t[:, None] * (self.goal - self.start)
        else:
            control_points = initial_path.copy()

        control_points[0] = self.start
        control_points[-1] = self.goal

        history = [control_points.copy()]
        cost_history = []
        epsilon = 0.02
        lr = learning_rate

        best_cp = control_points.copy()
        best_cost = self.trajectory_cost(control_points)

        for _ in range(steps):
            current_cost = self.trajectory_cost(control_points)
            cost_history.append(current_cost)

            if current_cost < best_cost:
                best_cost = current_cost
                best_cp = control_points.copy()

            gradient = np.zeros_like(control_points)

            # 只优化中间控制点，起点和终点固定
            for i in range(1, len(control_points) - 1):
                for dim in range(2):
                    plus = control_points.copy()
                    minus = control_points.copy()
                    plus[i, dim] += epsilon
                    minus[i, dim] -= epsilon
                    gradient[i, dim] = (
                        self.trajectory_cost(plus)
                        - self.trajectory_cost(minus)
                    ) / (2.0 * epsilon)

            norm = np.linalg.norm(gradient[1:-1])
            if norm > 20.0:
                gradient[1:-1] *= 20.0 / norm

            candidate = control_points.copy()
            candidate[1:-1] -= lr * gradient[1:-1]
            candidate[0] = self.start
            candidate[-1] = self.goal
            candidate_cost = self.trajectory_cost(candidate)

            if candidate_cost < current_cost:
                control_points = candidate
                lr = min(lr * 1.05, 0.08)
            else:
                lr *= 0.5

            history.append(control_points.copy())

            if lr < 1e-6:
                break

        # 选择整个优化过程中代价最低的轨迹
        return best_cp, history, cost_history, initial_path

    # ------------------------------------------------------------------
    # 检查和绘图
    # ------------------------------------------------------------------
    def collision_report(self, trajectory):
        distances = np.array([self.esdf_value(p) for p in trajectory])
        return {
            "minimum_distance": float(np.min(distances)),
            "collision_points": int(np.sum(distances < 0.0)),
            "unsafe_points": int(np.sum(distances < self.clearance)),
            "collision_free": bool(np.all(distances >= 0.0)),
            "clearance_satisfied": bool(np.all(distances >= self.clearance)),
        }

    def plot(self, control_points, history, cost_history, initial_path):
        final_trajectory = self.bspline_trajectory(control_points, 1600)

        fig = plt.figure(figsize=(19, 7), constrained_layout=True)
        fig.suptitle("EGO-Planner Educational Demo", fontsize=16, fontweight="bold")

        # 图 1：B-spline 轨迹
        ax1 = fig.add_subplot(1, 3, 1)
        self.draw_obstacles(ax1)
        ax1.plot(initial_path[:, 0], initial_path[:, 1], "k:", linewidth=1.5, label="A* initial path")

        # 显示少量重规划过程
        selected = np.linspace(0, len(history) - 1, min(6, len(history))).astype(int)
        for index in selected:
            temp = self.bspline_trajectory(history[index], 300)
            ax1.plot(temp[:, 0], temp[:, 1], "--", color="orange", alpha=0.35)

        ax1.plot(
            final_trajectory[:, 0], final_trajectory[:, 1],
            "b-", linewidth=3, label="Final cubic B-spline"
        )
        ax1.scatter(
            control_points[:, 0], control_points[:, 1],
            c="purple", marker="s", s=70, edgecolors="black",
            label="B-spline control points", zorder=5
        )
        ax1.scatter(*self.start, c="green", s=150, label="Start", zorder=6)
        ax1.scatter(*self.goal, c="red", marker="*", s=220, label="Goal", zorder=6)
        ax1.set_title("B-spline Trajectory")
        ax1.set_xlabel("X (meters)")
        ax1.set_ylabel("Y (meters)")
        ax1.legend(fontsize=8, loc="best")
        ax1.grid(alpha=0.3)
        ax1.set_aspect("equal")

        # 图 2：ESDF
        ax2 = fig.add_subplot(1, 3, 2)
        esdf_visual = np.clip(self.esdf, -1.0, 3.0)
        image = ax2.contourf(self.X, self.Y, esdf_visual, levels=25, cmap="RdYlGn")
        fig.colorbar(image, ax=ax2, label="Distance to obstacle (m)")
        ax2.contour(self.X, self.Y, self.esdf, levels=[0.0], colors="black", linewidths=2)
        ax2.contour(
            self.X, self.Y, self.esdf,
            levels=[self.clearance], colors="red", linestyles="--", linewidths=2
        )
        ax2.plot(final_trajectory[:, 0], final_trajectory[:, 1], "b-", linewidth=2)
        ax2.scatter(*self.start, c="green", s=100)
        ax2.scatter(*self.goal, c="red", marker="*", s=150)
        ax2.set_title("ESDF and Safety Boundary")
        ax2.set_xlabel("X (meters)")
        ax2.set_ylabel("Y (meters)")
        ax2.set_aspect("equal")

        # 图 3：代价收敛
        ax3 = fig.add_subplot(1, 3, 3)
        if cost_history:
            ax3.plot(np.arange(1, len(cost_history) + 1), cost_history, "b-o", markersize=3)
        ax3.set_title("Optimization Cost")
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Total cost")
        ax3.grid(alpha=0.3)

        plt.show()

    def draw_obstacles(self, ax):
        for index, (cx, cy, radius) in enumerate(self.obstacles):
            circle = plt.Circle(
                (cx, cy), radius, color="gray", alpha=0.7,
                label="Obstacle" if index == 0 else None,
            )
            ax.add_patch(circle)
        ax.set_xlim(self.x_min, self.x_min + self.world_size[0])
        ax.set_ylim(self.y_min, self.y_min + self.world_size[1])


def main():
    start = (-4.0, -4.0)
    goal = (4.0, 4.0)
    obstacles = [
        (0.0, 0.0, 1.2),
        (2.5, 1.2, 1.0),
        (-1.0, 2.8, 1.2),
        (0.0, -2.2, 1.1),
    ]

    planner = EGOPlannerDemo(
        start=start,
        goal=goal,
        obstacles=obstacles,
        world_size=(10.0, 10.0),
        resolution=0.1,
        clearance=0.45,
    )

    print("=" * 70)
    print("EGO-Planner Educational Demo")
    print("=" * 70)
    print("Generating A* collision-free initial path...")

    control_points, history, cost_history, initial_path = planner.optimize(
        steps=60,
        learning_rate=0.01,
    )

    final_trajectory = planner.bspline_trajectory(control_points, 2000)
    report = planner.collision_report(final_trajectory)

    print("\nFinal trajectory report:")
    print(f"  Minimum distance:       {report['minimum_distance']:.3f} m")
    print(f"  Collision points:       {report['collision_points']}")
    print(f"  Unsafe points:          {report['unsafe_points']}")
    print(f"  Collision free:         {report['collision_free']}")
    print(f"  Clearance satisfied:    {report['clearance_satisfied']}")
    print(f"  Optimization iterations: {len(cost_history)}")

    planner.plot(control_points, history, cost_history, initial_path)


if __name__ == "__main__":
    main()
