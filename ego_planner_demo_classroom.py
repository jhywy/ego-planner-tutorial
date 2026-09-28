"""
EGO-Planner 课堂讲解版：从地图到平滑安全轨迹

运行：
    pip install numpy matplotlib scipy
    python ego_planner_classroom.py

课堂主线：
    障碍物地图 -> ESDF 距离场 -> A* 安全初值 -> B-spline 轨迹
    -> 控制点优化 -> 最终碰撞检查

这不是官方 EGO-Planner 的 ROS 实现，而是帮助初学者理解算法思想的二维演示。
真实系统还需要传感器、局部地图、动力学约束和实时重规划。
"""

import heapq
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from matplotlib import font_manager

warnings.filterwarnings("ignore", category=UserWarning)


def configure_chinese_font():
    """选择系统中可用的中文字体，避免 Glyph missing 警告。"""
    candidates = [
        "Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
        "WenQuanYi Zen Hei", "Arial Unicode MS"
    ]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            break
    else:
        # 找不到中文字体时使用英文标签，避免图中出现方框。
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


configure_chinese_font()


class EGOPlannerClassroom:
    """二维教学版规划器。每个方法都对应课堂中的一个算法概念。"""

    def __init__(self, start, goal, obstacles, world_size=(10, 10), resolution=0.1,
                 clearance=0.45):
        self.start = np.asarray(start, dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        self.obstacles = obstacles  # 每个障碍物为 (圆心x, 圆心y, 半径)
        self.world_size = np.asarray(world_size, dtype=float)
        self.resolution = resolution
        self.clearance = clearance

        # 建立规则栅格坐标。ESDF 的每个元素表示该位置到障碍物的距离。
        self.x_min = -self.world_size[0] / 2
        self.y_min = -self.world_size[1] / 2
        self.nx = int(round(self.world_size[0] / resolution)) + 1
        self.ny = int(round(self.world_size[1] / resolution)) + 1
        x = np.linspace(self.x_min, self.x_min + (self.nx - 1) * resolution, self.nx)
        y = np.linspace(self.y_min, self.y_min + (self.ny - 1) * resolution, self.ny)
        self.X, self.Y = np.meshgrid(x, y)
        self.esdf = self.build_esdf()

    # ==================== 第一部分：ESDF ====================
    def build_esdf(self):
        """计算 ESDF：内部为负，边界为零，自由空间为正。"""
        distance = np.full_like(self.X, np.inf, dtype=float)
        for cx, cy, radius in self.obstacles:
            d = np.hypot(self.X - cx, self.Y - cy) - radius
            distance = np.minimum(distance, d)
        return distance

    def esdf_value(self, point):
        """对任意连续坐标查询 ESDF；这里使用双线性插值。"""
        x, y = np.asarray(point, dtype=float)
        gx = np.clip((x - self.x_min) / self.resolution, 0, self.nx - 1.000001)
        gy = np.clip((y - self.y_min) / self.resolution, 0, self.ny - 1.000001)
        ix, iy = int(np.floor(gx)), int(np.floor(gy))
        fx, fy = gx - ix, gy - iy
        v00 = self.esdf[iy, ix]
        v10 = self.esdf[iy, min(ix + 1, self.nx - 1)]
        v01 = self.esdf[min(iy + 1, self.ny - 1), ix]
        v11 = self.esdf[min(iy + 1, self.ny - 1), min(ix + 1, self.nx - 1)]
        return ((1 - fx) * (1 - fy) * v00 + fx * (1 - fy) * v10
                + (1 - fx) * fy * v01 + fx * fy * v11)

    def esdf_gradient(self, point):
        """ESDF 梯度指向距离增大的方向，也就是更安全的方向。"""
        eps = self.resolution
        p = np.asarray(point, dtype=float)
        dx, dy = np.array([eps, 0]), np.array([0, eps])
        return np.array([
            (self.esdf_value(p + dx) - self.esdf_value(p - dx)) / (2 * eps),
            (self.esdf_value(p + dy) - self.esdf_value(p - dy)) / (2 * eps),
        ])

    # ==================== 第二部分：A* 安全初值 ====================
    def world_to_grid(self, p):
        return tuple(np.rint((np.asarray(p) - [self.x_min, self.y_min])
                            / self.resolution).astype(int))

    def grid_to_world(self, node):
        return np.array([self.x_min + node[0] * self.resolution,
                         self.y_min + node[1] * self.resolution])

    def grid_is_free(self, node):
        ix, iy = node
        return (0 <= ix < self.nx and 0 <= iy < self.ny
                and self.esdf[iy, ix] >= self.clearance)

    def astar(self):
        """在膨胀后的障碍物地图中寻找一条安全离散路径。"""
        start, goal = self.world_to_grid(self.start), self.world_to_grid(self.goal)
        moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1),
                 (1, -1), (1, 0), (1, 1)]

        def h(a):
            return np.hypot(a[0] - goal[0], a[1] - goal[1])

        queue = [(h(start), 0.0, start)]
        parent, cost = {start: None}, {start: 0.0}
        visited = set()
        while queue:
            _, g, current = heapq.heappop(queue)
            if current in visited:
                continue
            visited.add(current)
            if current == goal:
                path = []
                while current is not None:
                    path.append(self.grid_to_world(current))
                    current = parent[current]
                return np.asarray(path[::-1])
            for dx, dy in moves:
                nxt = (current[0] + dx, current[1] + dy)
                if not self.grid_is_free(nxt):
                    continue
                # 禁止斜着穿过两个相邻障碍栅格。
                if dx and dy and (not self.grid_is_free((current[0] + dx, current[1]))
                                   or not self.grid_is_free((current[0], current[1] + dy))):
                    continue
                step = np.hypot(dx, dy)
                new_cost = g + step
                if new_cost < cost.get(nxt, np.inf):
                    cost[nxt] = new_cost
                    parent[nxt] = current
                    heapq.heappush(queue, (new_cost + h(nxt), new_cost, nxt))
        raise RuntimeError("A* 没有找到路径：请减小安全距离或调整地图。")

    @staticmethod
    def resample_path(path, count=8):
        """将较长的 A* 折线均匀抽样，作为 B-spline 控制点初值。"""
        if len(path) <= count:
            return path.copy()
        ids = np.linspace(0, len(path) - 1, count).round().astype(int)
        return path[np.unique(ids)]

    # ==================== 第三部分：B-spline ====================
    @staticmethod
    def bspline(control_points, samples=1000):
        """用开放均匀节点向量生成三次 B-spline。控制点不必在曲线上。"""
        cp = np.asarray(control_points, dtype=float)
        degree = 3
        if len(cp) < 4:
            raise ValueError("三次 B-spline 至少需要 4 个控制点。")
        knots = np.r_[np.zeros(degree), np.linspace(0, 1, len(cp) - degree + 1),
                      np.ones(degree)]
        t = np.linspace(0, 1, samples)
        sx = BSpline(knots, cp[:, 0], degree)
        sy = BSpline(knots, cp[:, 1], degree)
        return np.c_[sx(t), sy(t)]

    # ==================== 第四部分：轨迹优化 ====================
    def cost_function(self, cp):
        """总代价 = 障碍物代价 + 平滑代价 + 控制点长度代价。"""
        trajectory = self.bspline(cp, 500)
        distance = np.array([self.esdf_value(p) for p in trajectory])
        obstacle_cost = np.sum(np.where(
            distance < 0, 100000 * (1 - distance) ** 2,
            np.where(distance < self.clearance,
                     1000 * (self.clearance - distance) ** 2, 0)))
        # 二阶差分近似加速度，越小代表越平滑。
        second = trajectory[2:] - 2 * trajectory[1:-1] + trajectory[:-2]
        smooth_cost = 20 * np.sum(second ** 2)
        length_cost = 0.05 * np.sum(np.diff(cp, axis=0) ** 2)
        return float(obstacle_cost + smooth_cost + length_cost)

    def optimize(self, iterations=60, learning_rate=0.01):
        """用有限差分估计梯度，只移动中间控制点，起点终点保持固定。"""
        initial_path = self.resample_path(self.astar(), 8)
        cp = initial_path if len(initial_path) >= 4 else np.linspace(self.start, self.goal, 4)
        cp = cp.astype(float)
        cp[0], cp[-1] = self.start, self.goal
        history, costs = [cp.copy()], []
        best_cp, best_cost = cp.copy(), self.cost_function(cp)
        lr, epsilon = learning_rate, 0.02

        for _ in range(iterations):
            old_cost = self.cost_function(cp)
            costs.append(old_cost)
            if old_cost < best_cost:
                best_cp, best_cost = cp.copy(), old_cost
            gradient = np.zeros_like(cp)
            for i in range(1, len(cp) - 1):
                for dim in range(2):
                    plus, minus = cp.copy(), cp.copy()
                    plus[i, dim] += epsilon
                    minus[i, dim] -= epsilon
                    gradient[i, dim] = (self.cost_function(plus)
                                        - self.cost_function(minus)) / (2 * epsilon)
            norm = np.linalg.norm(gradient[1:-1])
            if norm > 20:
                gradient[1:-1] *= 20 / norm
            candidate = cp.copy()
            candidate[1:-1] -= lr * gradient[1:-1]
            new_cost = self.cost_function(candidate)
            if new_cost < old_cost:
                cp, lr = candidate, min(lr * 1.05, 0.08)
            else:
                lr *= 0.5
            history.append(cp.copy())
            if lr < 1e-6:
                break
        return best_cp, history, costs, initial_path

    # ==================== 第五部分：检查和显示 ====================
    def report(self, trajectory):
        distances = np.array([self.esdf_value(p) for p in trajectory])
        return (float(distances.min()), int(np.sum(distances < 0)),
                int(np.sum(distances < self.clearance)))

    def draw(self, cp, history, costs, initial_path):
        """课堂可视化：左图轨迹，中图 ESDF，右图优化曲线。"""
        trajectory = self.bspline(cp, 1600)
        fig = plt.figure(figsize=(19, 7), constrained_layout=True)
        fig.suptitle("EGO-Planner 课堂演示：局部规划、ESDF 与 B-spline",
                     fontsize=16, fontweight="bold")

        ax1 = fig.add_subplot(131)
        self.draw_obstacles(ax1)
        ax1.plot(initial_path[:, 0], initial_path[:, 1], "k:", label="A* 初始路径")
        for i in np.linspace(0, len(history) - 1, min(6, len(history))).astype(int):
            p = self.bspline(history[i], 250)
            ax1.plot(p[:, 0], p[:, 1], "--", color="orange", alpha=.35)
        ax1.plot(trajectory[:, 0], trajectory[:, 1], "b", lw=3, label="最终 B-spline")
        ax1.scatter(cp[:, 0], cp[:, 1], c="purple", marker="s", label="控制点", zorder=5)
        ax1.scatter(*self.start, c="green", s=130, label="起点", zorder=6)
        ax1.scatter(*self.goal, c="red", marker="*", s=200, label="终点", zorder=6)
        ax1.set_title("1. 初始路径与平滑轨迹")
        ax1.set_xlabel("X / 米"); ax1.set_ylabel("Y / 米")
        ax1.legend(fontsize=8); ax1.grid(alpha=.3); ax1.set_aspect("equal")

        ax2 = fig.add_subplot(132)
        image = ax2.contourf(self.X, self.Y, np.clip(self.esdf, -1, 3), levels=25, cmap="RdYlGn")
        fig.colorbar(image, ax=ax2, label="到障碍物距离 / 米")
        ax2.contour(self.X, self.Y, self.esdf, levels=[0], colors="black", linewidths=2)
        ax2.contour(self.X, self.Y, self.esdf, levels=[self.clearance],
                    colors="red", linestyles="--", linewidths=2)
        ax2.plot(trajectory[:, 0], trajectory[:, 1], "b", lw=2)
        ax2.set_title("2. ESDF 与安全距离边界")
        ax2.set_xlabel("X / 米"); ax2.set_ylabel("Y / 米"); ax2.set_aspect("equal")

        ax3 = fig.add_subplot(133)
        ax3.plot(np.arange(1, len(costs) + 1), costs, "b-o", ms=3)
        ax3.set_title("3. 优化代价变化")
        ax3.set_xlabel("迭代次数"); ax3.set_ylabel("总代价"); ax3.grid(alpha=.3)
        plt.show()

    def draw_obstacles(self, ax):
        for i, (cx, cy, radius) in enumerate(self.obstacles):
            ax.add_patch(plt.Circle((cx, cy), radius, color="gray", alpha=.7,
                                    label="障碍物" if i == 0 else None))
        ax.set_xlim(self.x_min, self.x_min + self.world_size[0])
        ax.set_ylim(self.y_min, self.y_min + self.world_size[1])


def main():
    """课堂演示入口：建议按下面的 1～5 步讲解。"""
    start, goal = (-4, -4), (4, 4)
    obstacles = [(0, 0, 1.2), (2.5, 1.2, 1.0), (-1, 2.8, 1.2), (0, -2.2, 1.1)]
    planner = EGOPlannerClassroom(start, goal, obstacles, clearance=.45)

    print("=" * 70)
    print("EGO-Planner 课堂讲解版")
    print("1. A* 生成安全初始路径")
    print("2. ESDF 告诉我们每个位置离障碍物有多远")
    print("3. B-spline 把控制点变成连续平滑轨迹")
    print("4. 优化控制点，兼顾安全与平滑")
    print("5. 对最终轨迹做高密度碰撞检查")

    cp, history, costs, initial_path = planner.optimize()
    trajectory = planner.bspline(cp, 2000)
    minimum, collision_count, unsafe_count = planner.report(trajectory)
    print("\n最终检查结果：")
    print(f"最小障碍物距离：{minimum:.3f} 米")
    print(f"碰撞采样点数：{collision_count}")
    print(f"小于安全距离的采样点数：{unsafe_count}")
    print(f"优化迭代次数：{len(costs)}")
    planner.draw(cp, history, costs, initial_path)


if __name__ == "__main__":
    main()
