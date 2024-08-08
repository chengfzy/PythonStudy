"""
Some Example about Catmull-Rom Spline

Note:
- Basic, Simple, Simplest几个的推导都可以参考[1], MonoLaneMapping对应alpha参考[4]
- Basic是基础的版本, 没有考虑参数tao, 只有alpha, 没有找到添加tao时对应的参考文献, [1]中没有写
- 注意Basic, Simple, Simplest和MonoLaneMapping的tao表达含义不一样, 代码中分开进行了表达
- 当alpha=0, tao=0, tao_mono_lane_mapping=0.5, 几个计算结果是完全一样的
- Simple/Simplest有考虑参考tao和alpha, 但与MonoLaneMapping中的tao对应不上, 意义不太一样
MonoLaneMapping对应alpha=0的情况, 没有参数考虑其他情况
-  

Ref:
[1] https://qroph.github.io/2018/07/30/smooth-paths-using-catmull-rom-splines.html
[2] https://en.wikipedia.org/wiki/Centripetal_Catmull%E2%80%93Rom_spline
[3] https://github.com/qroph/qroph.github.io/blob/master/assets/interactive/catmullrom.js
[4] Qiao et al., “Online Monocular Lane Mapping Using Catmull-Rom Spline.”
"""

from enum import Enum, auto
import numpy as np
import matplotlib.pyplot as plt


class CalculateMethod(Enum):
    Basic = auto()  # the basic method
    Simple = auto()  # the simple method
    Simplest = auto()  # the simplest method
    MonoLaneMapping = auto()  # paper from MonoLaneMapping, Ref [4]


class CatmullRomSpline:
    def __init__(self, tao=0.5, tao_mono_lane_mapping=0.5, alpha=0.5, cal_method=CalculateMethod.Simple):
        self.tao = tao  # tao for Simple/Simplest
        self.tao_mono_lane_mapping = tao_mono_lane_mapping  # tao for MonoLaneMapping
        self.alpha = alpha  # alpha, 0: uniform, 0.5: centripetal, 1: chordal
        self.cal_method = cal_method  # calculate method

    def compute_points(self, ctrl_points: np.ndarray, num_points=20):
        """
        Compute the points in spline segment

        Args:
            ctrl_points (np.ndarray): control points, Nx2, [P0, P1, ..., Pn]
            num_points (int, optional): The number of points to include in the resulting curve segment. Defaults to 20.
        """
        # check ctrl points num
        ctrl_points_num = len(ctrl_points)
        if ctrl_points_num < 4:
            return None

        all_points = []
        if self.cal_method == CalculateMethod.Basic:
            for n in range(ctrl_points_num - 3):
                all_points.extend(self.__compute_single_points_basic(ctrl_points[n : n + 4, :], num_points=num_points))
        elif self.cal_method == CalculateMethod.Simple:
            for n in range(ctrl_points_num - 3):
                all_points.extend(self.__compute_single_points_simple(ctrl_points[n : n + 4, :], num_points=num_points))
        elif self.cal_method == CalculateMethod.Simplest:
            for n in range(ctrl_points_num - 3):
                all_points.extend(
                    self.__compute_single_points_simplest(ctrl_points[n : n + 4, :], num_points=num_points)
                )
        elif self.cal_method == CalculateMethod.MonoLaneMapping:
            for n in range(ctrl_points_num - 3):
                all_points.extend(self.__compute_mono_lane_mapping(ctrl_points[n : n + 4, :], num_points=num_points))
        return np.array(all_points)

    def __compute_single_points_basic(self, ctrl_points: np.ndarray, num_points=20):
        """
        Compute the points in single spline segment, basic method, see Ref[2]

        Args:
            ctrl_points (np.ndarray): control points, [P0,P1,P2,P3]
            num_points (int, optional): The number of points to include in the resulting curve segment. Defaults to 20.
        """
        p0, p1, p2, p3 = ctrl_points[0], ctrl_points[1], ctrl_points[2], ctrl_points[3]
        t0 = 0.0
        t1 = self.__tj(t0, p0, p1)
        t2 = self.__tj(t1, p1, p2)
        t3 = self.__tj(t2, p2, p3)
        t = np.linspace(t1, t2, num_points).reshape(num_points, 1)
        A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
        B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
        # NOTE: how to add tao, no document or reference, below is not correct
        # return (1 - self.tao) * (t2 - t) / (t2 - t1) * B1 + (1 - self.tao) * (t - t1) / (t2 - t1) * B2
        return (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2

    def __compute_single_points_simple(self, ctrl_points: np.ndarray, num_points=20):
        """
        Compute the points in single spline segment, simple method, see Ref[1], [3]

        Args:
            ctrl_points (np.ndarray): control points, [P0,P1,P2,P3]
            num_points (int, optional): The number of points to include in the resulting curve segment. Defaults to 20.
        """
        p0, p1, p2, p3 = ctrl_points[0], ctrl_points[1], ctrl_points[2], ctrl_points[3]
        t0 = 0.0
        t1 = self.__tj(t0, p0, p1)
        t2 = self.__tj(t1, p1, p2)
        t3 = self.__tj(t2, p2, p3)

        m1 = (1 - self.tao) * (t2 - t1) * ((p1 - p0) / (t1 - t0) - (p2 - p0) / (t2 - t0) + (p2 - p1) / (t2 - t1))
        m2 = (1 - self.tao) * (t2 - t1) * ((p2 - p1) / (t2 - t1) - (p3 - p1) / (t3 - t1) + (p3 - p2) / (t3 - t2))
        a = 2 * (p1 - p2) + m1 + m2
        b = -3 * (p1 - p2) - 2 * m1 - m2
        c = m1
        d = p1

        t = np.linspace(0, 1, num_points).reshape(num_points, 1)
        return a * t**3 + b * t**2 + c * t + d

    def __compute_single_points_simplest(self, ctrl_points: np.ndarray, num_points=20):
        """
        Compute the points in single spline segment, simplest method, see Ref[1]

        Args:
            ctrl_points (np.ndarray): control points, [P0,P1,P2,P3]
            num_points (int, optional): The number of points to include in the resulting curve segment. Defaults to 20.
        """
        p0, p1, p2, p3 = ctrl_points[0], ctrl_points[1], ctrl_points[2], ctrl_points[3]
        t01 = np.linalg.norm(p0 - p1) ** self.alpha
        t12 = np.linalg.norm(p1 - p2) ** self.alpha
        t23 = np.linalg.norm(p2 - p3) ** self.alpha

        m1 = (1 - self.tao) * (p2 - p1 + t12 * ((p1 - p0) / t01 - (p2 - p0) / (t01 + t12)))
        m2 = (1 - self.tao) * (p2 - p1 + t12 * ((p3 - p2) / t23 - (p3 - p1) / (t12 + t23)))
        a = 2 * (p1 - p2) + m1 + m2
        b = -3 * (p1 - p2) - 2 * m1 - m2
        c = m1
        d = p1

        t = np.linspace(0, 1, num_points).reshape(num_points, 1)
        return a * t**3 + b * t**2 + c * t + d

    def __compute_mono_lane_mapping(self, ctrl_points: np.ndarray, num_points=20):
        """
        Compute the points in single spline segment, see Ref[4]

        Args:
            ctrl_points (np.ndarray): control points, [P0,P1,P2,P3]
            num_points (int, optional): The number of points to include in the resulting curve segment. Defaults to 20.
        """
        # 论文中的操作, uniform形式, 不考虑alpha
        all_u = np.linspace(0, 1, num_points).reshape(num_points, 1)
        M = np.array(
            [
                [0, 1, 0, 0],
                [-self.tao_mono_lane_mapping, 0, self.tao_mono_lane_mapping, 0],
                [
                    2 * self.tao_mono_lane_mapping,
                    self.tao_mono_lane_mapping - 3,
                    3 - 2 * self.tao_mono_lane_mapping,
                    -self.tao_mono_lane_mapping,
                ],
                [
                    -self.tao_mono_lane_mapping,
                    2 - self.tao_mono_lane_mapping,
                    self.tao_mono_lane_mapping - 2,
                    self.tao_mono_lane_mapping,
                ],
            ]
        )

        points = []
        for v in all_u:
            u = v[0]
            U = np.array([1, u, u**2, u**3])
            points.append(U.transpose() @ M @ ctrl_points)

        return np.array(points)

    def __tj(self, ti: float, pi: np.ndarray, pj: np.ndarray) -> float:
        """calculate tj"""
        return np.linalg.norm(pi - pj) ** self.alpha + ti


if __name__ == '__main__':
    # common config
    ctrl_points = np.array([[0, 1.5], [2, 2], [3, 1], [3.8, 0.55], [4, 0.5], [5, 1], [6, 2], [7, 3]])
    tao, tao_mono_lane_mapping = 0, 0.5
    # plot config
    colors = ['b', 'r', 'k', 'm']

    # calculate spline with different method
    methods = [CalculateMethod.Basic, CalculateMethod.Simple, CalculateMethod.Simplest, CalculateMethod.MonoLaneMapping]
    # methods = [CalculateMethod.Basic, CalculateMethod.MonoLaneMapping]
    all_points = []
    for method in methods:
        # calculate catmull-rom spline points
        spline = CatmullRomSpline(tao=tao, tao_mono_lane_mapping=tao_mono_lane_mapping, alpha=0, cal_method=method)
        all_points.append(spline.compute_points(ctrl_points))

    # plot
    fig = plt.figure('Catmull-Rom Spline with Different Calculate Method', figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(ctrl_points[:, 0], ctrl_points[:, 1], 'ro', markersize=10, label='Control Points')
    for n, (method, points) in enumerate(zip(methods, all_points)):
        ax.plot(points[:, 0], points[:, 1], f'.-{colors[n]}', label=method.name)
    ax.set_title('Catmull-Rom Spline with Different Calculate Method')
    ax.grid()
    ax.legend()

    # plot with different alpha
    alphas = [0, 0.5, 1.0]
    all_points = []
    for alpha in alphas:
        # calculate catmull-rom spline points
        spline = CatmullRomSpline(tao=tao, alpha=alpha, cal_method=CalculateMethod.Basic)
        all_points.append(spline.compute_points(ctrl_points))

    # plot
    fig = plt.figure('Catmull-Rom Spline with Different Alpha', figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(ctrl_points[:, 0], ctrl_points[:, 1], 'ro', markersize=10, label='Control Points')
    for n, (alpha, points) in enumerate(zip(alphas, all_points)):
        ax.plot(points[:, 0], points[:, 1], f'.-{colors[n]}', label=rf'$\alpha: {alpha:0.3f}$')
    ax.set_title('Catmull-Rom Spline with Different Alpha')
    ax.grid()
    ax.legend()

    plt.show(block=True)
