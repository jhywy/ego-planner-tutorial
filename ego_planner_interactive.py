"""
EGO-Planner 交互式演示版

特点：
1. 鼠标点击选择起点和终点
2. 动态显示 A* 搜索过程
3. 动态显示 B-spline 优化过程
4. 实时显示轨迹、ESDF、代价函数变化
5. 支持暂停、继续、重置操作

操作说明：
- 在左图上点击选择"起点"（绿色），再点击选择"终点"（红色）
- 点击"开始规划"按钮开始算法
- 点击"暂停/继续"可以暂停或继续演示
- 点击"重置"可以清除当前规划结果

安装依赖：
    pip install numpy matplotlib scipy
"""

import heapq
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from scipy.interpolate import BSpline

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class InteractiveEGOPlanner:
    """交互式 EGO-Planner 演示：支持鼠标交互和动画演示。"""

    def __init__(self, obstacles, world_size=(10, 10), resolution=0.1, clearance=0.45):
        """
        初始化规划器。

        参数：
            obstacles: 障碍物列表，每个为 (cx, cy, radius)
            world_size: 世界大小 (width, height)
            resolution: 栅格分辨率
            clearance: 安全距离
        """
        self.obstacles = obstacles
        self.world_size = np.asarray(world_size, dtype=float)
        self.resolution = resolution
        self.clearance = clearance

        # 栅格初始化
        self.x_min = -self.world_size[0] / 2
        self.y_min = -self.world_size[1] / 2
        self.nx = int(round(self.world_size[0] / resolution)) + 1
        self.ny = int(round(self.world_size[1] / resolution)) + 1
        x = np.linspace(self.x_min, self.x_min + (self.nx - 1) * resolution, self.nx)
        y = np.linspace(self.y_min, self.y_min + (self.ny - 1) * resolution, self.ny)
        self.X, self.Y = np.meshgrid(x, y)
        self.esdf = self._build_esdf()

        # 规划参数
        self.start = None
        self.goal = None
        self.initial_path = None
        self.control_points = None
        self.final_trajectory = None

        # 动画状态
        self.astar_trace = []  # A* 搜索轨迹
        self.optimization_history = []  # 优化过程历史
        self.cost_history = []
        self.is_running = False
        self.current_step = 0
        self.total_steps = 0

    # ==================== 地图与 ESDF ====================
    def _build_esdf(self):
        """计算 ESDF 距离场。"""
        distance = np.full_like(self.X, np.inf, dtype=float)
        for cx, cy, radius in self.obstacles:
            d = np.hypot(self.X - cx, self.Y - cy) - radius
            distance = np.minimum(distance, d)
        return distance

    def esdf_value(self, point):
        """查询点的 ESDF 值（双线性插值）。"""
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
        """计算 ESDF 梯度。"""
        eps = self.resolution
        p = np.asarray(point, dtype=float)
        dx = np.array([eps, 0])
        dy = np.array([0, eps])
        return np.array([
            (self.esdf_value(p + dx) - self.esdf_value(p - dx)) / (2 * eps),
            (self.esdf_value(p + dy) - self.esdf_value(p - dy)) / (2 * eps),
        ])

    # ==================== 坐标转换 ====================
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

    # ==================== A* 规划 ====================
    def astar_with_trace(self):
        """
        执行 A*，返回路径和搜索轨迹（用于动画）。
        astar_trace 是一个列表，每个元素是 (visited_set, current_node)，
        代表 A* 每一步扩展的节点。
        """
        if self.start is None or self.goal is None:
            return None, []

        start = self.world_to_grid(self.start)
        goal = self.world_to_grid(self.goal)
        moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1),
                 (1, -1), (1, 0), (1, 1)]

        def h(a):
            return np.hypot(a[0] - goal[0], a[1] - goal[1])

        queue = [(h(start), 0.0, start)]
        parent, cost = {start: None}, {start: 0.0}
        visited = set()
        trace = []  # 记录搜索过程

        while queue:
            _, g, current = heapq.heappop(queue)
            if current in visited:
                continue
            visited.add(current)
            trace.append((visited.copy(), current))  # 记录当前访问的节点

            if current == goal:
                path = []
                while current is not None:
                    path.append(self.grid_to_world(current))
                    current = parent[current]
                return np.asarray(path[::-1]), trace

            for dx, dy in moves:
                nxt = (current[0] + dx, current[1] + dy)
                if not self.grid_is_free(nxt):
                    continue
                if dx and dy and (not self.grid_is_free((current[0] + dx, current[1]))
                                   or not self.grid_is_free((current[0], current[1] + dy))):
                    continue
                step = np.hypot(dx, dy)
                new_cost = g + step
                if new_cost < cost.get(nxt, np.inf):
                    cost[nxt] = new_cost
                    parent[nxt] = current
                    heapq.heappush(queue, (new_cost + h(nxt), new_cost, nxt))

        raise RuntimeError("A* 无解")

    @staticmethod
    def resample_path(path, count=8):
        """从 A* 路径抽样控制点。"""
        if len(path) <= count:
            return path.copy()
        ids = np.linspace(0, len(path) - 1, count).round().astype(int)
        return path[np.unique(ids)]

    # ==================== B-spline ====================
    @staticmethod
    def bspline(control_points, samples=1000):
        """生成三次 B-spline 轨迹。"""
        cp = np.asarray(control_points, dtype=float)
        degree = 3
        if len(cp) < 4:
            raise ValueError("至少需要 4 个控制点")
        knots = np.r_[np.zeros(degree), np.linspace(0, 1, len(cp) - degree + 1),
                      np.ones(degree)]
        t = np.linspace(0, 1, samples)
        sx = BSpline(knots, cp[:, 0], degree)
        sy = BSpline(knots, cp[:, 1], degree)
        return np.c_[sx(t), sy(t)]

    # ==================== 轨迹优化 ====================
    def cost_function(self, cp):
        """计算总代价。"""
        trajectory = self.bspline(cp, 300)
        distance = np.array([self.esdf_value(p) for p in trajectory])
        obstacle_cost = np.sum(np.where(
            distance < 0, 100000 * (1 - distance) ** 2,
            np.where(distance < self.clearance,
                     1000 * (self.clearance - distance) ** 2, 0)))
        second = trajectory[2:] - 2 * trajectory[1:-1] + trajectory[:-2]
        smooth_cost = 20 * np.sum(second ** 2)
        length_cost = 0.05 * np.sum(np.diff(cp, axis=0) ** 2)
        return float(obstacle_cost + smooth_cost + length_cost)

    def optimize_with_history(self, iterations=60, learning_rate=0.01):
        """
        优化，记录每一步的控制点和代价（用于动画）。
        返回 (final_cp, history, costs)
        """
        if self.initial_path is None:
            return None, [], []

        cp = self.resample_path(self.initial_path, 8)
        if len(cp) < 4:
            cp = np.linspace(self.start, self.goal, 4)
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

        return best_cp, history, costs

    # ==================== 交互式演示 ====================
    def run_planning(self):
        """执行规划过程，保存所有动画状态。"""
        if self.start is None or self.goal is None:
            print("请先用鼠标选择起点和终点")
            return False

        print("正在执行 A* 搜索...")
        self.initial_path, self.astar_trace = self.astar_with_trace()
        print(f"A* 找到路径，共 {len(self.initial_path)} 个离散点")

        print("正在优化 B-spline 控制点...")
        self.control_points, self.optimization_history, self.cost_history = \
            self.optimize_with_history(iterations=60, learning_rate=0.01)
        print(f"优化完成，共 {len(self.cost_history)} 次迭代")

        self.final_trajectory = self.bspline(self.control_points, 1600)
        self.total_steps = len(self.astar_trace) + len(self.cost_history)
        self.current_step = 0
        self.is_running = True
        return True


class InteractiveVisualization:
    """交互式可视化界面。"""

    def __init__(self, planner):
        self.planner = planner
        self.fig = plt.figure(figsize=(18, 10), constrained_layout=True)
        self.fig.suptitle("EGO-Planner 交互式演示：鼠标选择起点→终点，点击\"开始规划\"",
                         fontsize=16, fontweight="bold")

        # 四个子图
        self.ax_map = self.fig.add_subplot(2, 2, 1)
        self.ax_esdf = self.fig.add_subplot(2, 2, 2)
        self.ax_traj = self.fig.add_subplot(2, 2, 3)
        self.ax_cost = self.fig.add_subplot(2, 2, 4)

        # 鼠标事件
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        # 按钮区域
        ax_start = plt.axes([0.05, 0.05, 0.08, 0.04])
        ax_pause = plt.axes([0.15, 0.05, 0.08, 0.04])
        ax_reset = plt.axes([0.25, 0.05, 0.08, 0.04])
        self.btn_start = Button(ax_start, 'Start Plan')
        self.btn_pause = Button(ax_pause, 'Pause/Resume')
        self.btn_reset = Button(ax_reset, 'Reset')

        self.btn_start.on_clicked(lambda event: self._on_start_click())
        self.btn_pause.on_clicked(lambda event: self._on_pause_click())
        self.btn_reset.on_clicked(lambda event: self._on_reset_click())

        self.ani = None
        self._draw_initial()

    def _draw_initial(self):
        """绘制初始界面。"""
        self.ax_map.clear()
        self._draw_obstacles(self.ax_map)
        self.ax_map.set_title("1. Select Start & Goal (click on map)")
        self.ax_map.set_xlabel("X (m)"); self.ax_map.set_ylabel("Y (m)")
        self.ax_map.grid(alpha=0.3); self.ax_map.set_aspect("equal")

        self.ax_esdf.clear()
        image = self.ax_esdf.contourf(self.planner.X, self.planner.Y,
                                       np.clip(self.planner.esdf, -1, 3),
                                       levels=25, cmap="RdYlGn")
        plt.colorbar(image, ax=self.ax_esdf, label="Distance (m)")
        self.ax_esdf.set_title("2. ESDF Heatmap")
        self.ax_esdf.set_aspect("equal")

        self.ax_traj.clear()
        self._draw_obstacles(self.ax_traj)
        self.ax_traj.set_title("3. A* & B-spline Trajectory")
        self.ax_traj.set_xlabel("X (m)"); self.ax_traj.set_ylabel("Y (m)")
        self.ax_traj.grid(alpha=0.3); self.ax_traj.set_aspect("equal")

        self.ax_cost.clear()
        self.ax_cost.set_title("4. Optimization Cost")
        self.ax_cost.set_xlabel("Iteration"); self.ax_cost.set_ylabel("Cost")
        self.ax_cost.grid(alpha=0.3)

        plt.draw()

    def _draw_obstacles(self, ax):
        """绘制障碍物。"""
        for i, (cx, cy, r) in enumerate(self.planner.obstacles):
            ax.add_patch(plt.Circle((cx, cy), r, color="gray", alpha=0.7,
                                    label="Obstacle" if i == 0 else None))
        ax.set_xlim(self.planner.x_min, self.planner.x_min + self.planner.world_size[0])
        ax.set_ylim(self.planner.y_min, self.planner.y_min + self.planner.world_size[1])

    def _on_click(self, event):
        """处理鼠标点击事件。"""
        if event.inaxes != self.ax_map or self.planner.is_running:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        if self.planner.start is None:
            self.planner.start = np.array([x, y])
            print(f"起点已设置：{self.planner.start}")
        elif self.planner.goal is None:
            self.planner.goal = np.array([x, y])
            print(f"终点已设置：{self.planner.goal}")
            self._draw_initial()  # 更新显示
            self._show_selected_points()

    def _show_selected_points(self):
        """显示已选择的起点和终点。"""
        if self.planner.start is not None:
            self.ax_map.scatter(*self.planner.start, c="green", s=150,
                               label="Start", marker="o", zorder=5)
        if self.planner.goal is not None:
            self.ax_map.scatter(*self.planner.goal, c="red", s=200,
                               label="Goal", marker="*", zorder=5)
        if self.planner.start is not None or self.planner.goal is not None:
            self.ax_map.legend()
        plt.draw()

    def _on_start_click(self):
        """开始规划。"""
        if self.planner.is_running or self.planner.astar_trace:
            return
        if not self.planner.run_planning():
            return
        # 启动动画
        self.ani = FuncAnimation(self.fig, self._animate, frames=self.planner.total_steps,
                                 interval=50, repeat=False)
        plt.draw()

    def _on_pause_click(self):
        """暂停/继续。"""
        if self.ani is not None:
            if self.planner.is_running:
                self.ani.event_source.stop()
                self.planner.is_running = False
                print("已暂停")
            else:
                self.ani.event_source.start()
                self.planner.is_running = True
                print("已继续")

    def _on_reset_click(self):
        """重置。"""
        if self.ani is not None:
            self.ani.event_source.stop()
        self.planner.start = None
        self.planner.goal = None
        self.planner.astar_trace = []
        self.planner.optimization_history = []
        self.planner.cost_history = []
        self.planner.is_running = False
        self.planner.current_step = 0
        self._draw_initial()
        print("已重置")

    def _animate(self, frame):
        """动画帧回调。"""
        self.planner.current_step = frame
        astar_frames = len(self.planner.astar_trace)

        self.ax_map.clear()
        self._draw_obstacles(self.ax_map)
        self.ax_map.set_title("1. A* Search & Path")
        self.ax_map.set_xlabel("X (m)"); self.ax_map.set_ylabel("Y (m)")
        self.ax_map.grid(alpha=0.3); self.ax_map.set_aspect("equal")

        # A* 搜索阶段
        if frame < astar_frames:
            visited, current = self.planner.astar_trace[frame]
            visited_world = [self.planner.grid_to_world(node) for node in visited]
            if visited_world:
                visited_world = np.array(visited_world)
                self.ax_map.scatter(visited_world[:, 0], visited_world[:, 1],
                                   c="cyan", s=10, alpha=0.5, label="Visited")
            current_world = self.planner.grid_to_world(current)
            self.ax_map.scatter(*current_world, c="blue", s=30, marker="x", label="Current")
            self.ax_map.set_title(f"1. A* Search (step {frame}/{astar_frames})")

        # B-spline 优化阶段
        else:
            opt_frame = frame - astar_frames
            if self.planner.initial_path is not None:
                self.ax_map.plot(self.planner.initial_path[:, 0],
                                self.planner.initial_path[:, 1], "k:", label="A* Path")
            if opt_frame < len(self.planner.optimization_history):
                cp = self.planner.optimization_history[opt_frame]
                traj = self.planner.bspline(cp, 500)
                self.ax_map.plot(traj[:, 0], traj[:, 1], "b-", lw=2, label="B-spline")
                self.ax_map.scatter(cp[:, 0], cp[:, 1], c="purple", marker="s",
                                   label="Control Points", zorder=5)
            self.ax_map.set_title(f"2. B-spline Optimization (step {opt_frame}/"
                                 f"{len(self.planner.optimization_history)})")

        if self.planner.start is not None:
            self.ax_map.scatter(*self.planner.start, c="green", s=150, label="Start", zorder=6)
        if self.planner.goal is not None:
            self.ax_map.scatter(*self.planner.goal, c="red", s=200, marker="*", label="Goal", zorder=6)
        self.ax_map.legend(fontsize=8)

        # 轨迹和代价曲线
        self.ax_traj.clear()
        self._draw_obstacles(self.ax_traj)
        if self.planner.final_trajectory is not None:
            self.ax_traj.plot(self.planner.final_trajectory[:, 0],
                             self.planner.final_trajectory[:, 1], "b-", lw=3)
        self.ax_traj.set_title("3. Final Trajectory")
        if self.planner.start is not None:
            self.ax_traj.scatter(*self.planner.start, c="green", s=150, zorder=6)
        if self.planner.goal is not None:
            self.ax_traj.scatter(*self.planner.goal, c="red", s=200, marker="*", zorder=6)
        self.ax_traj.set_aspect("equal")

        self.ax_cost.clear()
        if self.planner.cost_history:
            self.ax_cost.plot(np.arange(1, len(self.planner.cost_history) + 1),
                             self.planner.cost_history, "b-o", ms=3)
            self.ax_cost.set_xlabel("Iteration"); self.ax_cost.set_ylabel("Cost")
        self.ax_cost.grid(alpha=0.3)
        self.ax_cost.set_title("4. Cost Convergence")

        plt.draw()

    def show(self):
        plt.show()


def main():
    """主程序入口。"""
    obstacles = [
        (0, 0, 1.2),
        (2.5, 1.2, 1.0),
        (-1, 2.8, 1.2),
        (0, -2.2, 1.1),
    ]

    planner = InteractiveEGOPlanner(obstacles, world_size=(10, 10), resolution=0.1, clearance=0.45)
    viz = InteractiveVisualization(planner)
    viz.show()


if __name__ == "__main__":
    main()
