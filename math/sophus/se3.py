from __future__ import annotations
import numpy as np
from . import SO3
import typing


class SE3:
    """
    SE(3)
    """

    def __init__(self, so3: SO3, t: np.ndarray):
        """
        Initialize SE3 with a rotation(SO3) and translation(t). and se3 is represented as [t, phi]

        Args:
            so3 (SO3): SO3 rotation
            t (np.ndarray): Translation, 3x1 numpy array

        Raises:
            TypeError: translation part should be a vector with 3 elements
        """
        if t.shape != (3,):
            raise TypeError('translation part should be a vector with 3 elements')
        self.__so3 = so3
        self.__t = t.astype(dtype=np.float)

    def __repr__(self):
        return f'SE3: [{repr(self.__so3)}, t: {repr(self.__t)}]'

    def __str__(self):
        phi = self.__so3.log() * 180.0 / np.pi
        return f'phi: [{phi[0]: .5f}, {phi[1]: .5f}, {phi[2]: .5f}] deg, t: [{self.__t[0]: .5f}, {self.__t[1]: .5f}, {self.__t[2]: .5f}] m'

    def __getitem__(self, key):
        assert 0 <= key <= 7
        if key < 4:
            return self.__so3[key]
        else:
            return self.__t[key - 4]

    def __mul__(self, right):
        """
        SE3 * SE3 or SE3 * point

        Args:
            right (SE3 or 3x1/4x1 np.ndarray): SE3 or 3x1 point, or 4x1 point

        Returns:
            _type_: SE3 or point
        """
        if isinstance(right, SE3):
            # SE3 * SE3, T =  T1 * T2 = [R1 * R2, t1 + R1*t2]
            return SE3(self.__so3 * right.so3, self.__t + self.__so3 * right.t)
        elif isinstance(right, np.ndarray):
            # SE3 * p, write point as a 4x1 vector [0, 1], then p1 = T * p = R * p + t
            if right.shape == (3,):
                return self.__t + self.__so3 * right
            if right.shape == (4,) and right[-1] == 1:
                return np.hstack((self.__t + self.__so3 * right[:-1], 1))
            else:
                raise TypeError('right point should be a vector with 3 elements')
        else:
            raise TypeError(f'unsupported type {type(right)}')

    def __eq__(self, other):
        """Check SE3 is equal to other"""
        if isinstance(other, SE3):
            return self.__so3 == other.so3 and (self.__t == other.t).all()
        return False

    @property
    def so3(self) -> SO3:
        """
        Get rotation with SO3

        Returns:
            SO3: rotation
        """
        return self.__so3

    @property
    def t(self) -> np.ndarray:
        """
        Get translation

        Returns:
            np.ndarray: translation
        """
        return self.__t

    @staticmethod
    def hat(v: np.ndarray) -> np.ndarray:
        """
        hat operator

        Args:
            v (np.ndarray): 6x1 se3 vector, [t, phi]

        Raises:
            TypeError: v should be se3 of 6x1 vector

        Returns:
            np.ndarray: 4x4 matrix
        """
        if v.shape != (6,):
            raise TypeError('v should be se3 of 6x1 vector')
        t = v[:3]
        phi = v[3:]
        return np.vstack((np.hstack((SO3.hat(phi), np.array([t]).transpose())), np.zeros((1, 4))))

    @staticmethod
    def vee(mat: np.ndarray) -> np.ndarray:
        """
        vee operator

        Args:
            mat (np.ndarray): 4x4 matrix

        Raises:
            TypeError: mat should be a 4x4 matrix

        Returns:
            np.ndarray: 6x1 se3 vector, [t, phi]
        """
        if mat.shape != (4, 4):
            raise TypeError('mat should be a 4x4 matrix')
        return np.hstack((mat[:-1, -1], np.array([mat[2, 1], mat[0, 2], mat[1, 0]])))

    @staticmethod
    def exp(v: np.ndarray) -> SE3:
        """
        Exponential map

        Args:
            v (np.ndarray): [t, phi] = [translation, so3], 6x1 vector

        Raises:
            TypeError: input v should be se3 of 6x1 vector

        Returns:
            SE3: Transformation represented by SE3
        """
        if v.shape != (6,):
            raise TypeError('input v should be se3 of 6x1 vector')
        t = v[:3]
        phi = v[3:]
        theta = np.linalg.norm(phi)
        so3 = SO3.exp(phi)
        phi_hat = SO3.hat(phi)
        phi_hat2 = phi_hat.dot(phi_hat)

        if theta < np.finfo(float).eps:
            v = so3.mat()
        else:
            v = (
                np.identity(3)
                + (1 - np.cos(theta)) / theta**2 * phi_hat
                + (theta - np.sin(theta)) / theta**3 * phi_hat2
            )

        return SE3(so3, v.dot(t))

    def log(self) -> np.ndarray:
        """
        Logarithmic map

        Returns:
            np.ndarray: se3, [t, phi] = [translation, so3]
        """
        phi = self.__so3.log()
        theta = np.linalg.norm(phi)
        phi_hat = SO3.hat(phi)

        if np.abs(theta) < np.finfo(float).eps:
            v_inv = np.identity(3) - 0.5 * phi_hat + 1.0 / 12.0 * phi_hat.dot(phi_hat)
        else:
            half_theta = 0.5 * theta
            v_inv = (
                np.identity(3)
                - 0.5 * phi_hat
                + (1.0 - theta * np.cos(half_theta) / (2.0 * np.sin(half_theta))) / theta**2 * phi_hat.dot(phi_hat)
            )

        return np.hstack((v_inv.dot(self.__t), phi))

    def inverse(self) -> SE3:
        """
        Calculate the inverse of SE3

        Returns:
            SE3: Inverse of SE3
        """
        inv_r = self.__so3.inverse()
        return SE3(inv_r, inv_r * -self.__t)

    def mat(self) -> np.ndarray:
        """
        Get the transformation matrix of SE3

        Returns:
            np.ndarray: 4x4 transformation matrix
        """
        return np.vstack((self.mat3x4(), np.array([[0, 0, 0, 1]])))

    def mat3x4(self) -> np.ndarray:
        """
        Get the transformation matrix of SE3

        Returns:
            np.ndarray: 3x4 transformation matrix
        """
        return np.hstack((self.__so3.mat(), np.array([self.__t]).transpose()))

    @staticmethod
    def from_mat(mat: np.ndarray) -> SE3:
        """
        Construct SE3 with 4x4 or 3x4 transformation mat

        Args:
            mat (np.ndarray): 4x4 or 3x4 transformation mat

        Returns:
            SE3: Transformation represented by SE3
        """
        if mat.shape == (3, 4):
            return SE3(SO3(mat[:3, :3]), mat[:3, -1])
        elif mat.shape == (4, 4):
            if mat[-1, 0] == 0 and mat[-1, 1] == 0 and mat[-1, 2] == 0 and mat[-1, 3] == 1:
                return SE3(SO3(mat[:3, :3]), mat[:3, -1])
            else:
                raise TypeError(f'the last row of 4x4 transformation mat {mat[-1, :]} not equal [0, 0, 0, 1]')
        else:
            raise TypeError('input mat should be 4x4 or 3x4 transformation matrix')

    @staticmethod
    def average(poses: typing.List[SE3], max_iter_num=30, param_tolerance=1e-10) -> SE3:
        """
        Calculate the average of list of poses

        Args:
            poses (List[SE3]): list of poses represented by SE3
            max_iter_num: max iteration number
            param_tolerance: parameter tolerance

        Returns:
            SE3: The average pose represented by SE3
        """
        if len(poses) == 0:
            raise RuntimeError('cannot input empty poses')

        ret = poses[0]
        for n in range(max_iter_num):
            avg = sum([(ret.inverse() * v).log() for v in poses]) / len(poses)
            ret_new = ret * SE3.exp(avg)
            if np.linalg.norm((ret_new.inverse() * ret).log()) < param_tolerance:
                return ret_new
            # print(f'[{n}] avg = {avg}, ret = {ret}, ret_new = {ret_new}\n')
            ret = ret_new
        return ret
