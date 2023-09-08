from __future__ import annotations
import numpy as np
from . import Quaternion
import typing


class SO3:
    """
    SO(3)
    """

    def __init__(self, q: typing.Union[Quaternion, np.ndarray]):
        """
        Initialize with quaternion or rotation matrix

        Args:
            q (Quaternion, or 3x3 np.ndarray): Quaternion or 3x3 rotation matrix

        Raises:
            TypeError: could only initialized with Quaternion or 3x3 rotation matrix
        """
        if isinstance(q, Quaternion):
            self.__q = q
            self.__q.normalize()
        elif isinstance(q, np.ndarray) and q.shape == (3, 3):
            self.__q = Quaternion.from_rotation_mat(q)
            self.__q.normalize()
        else:
            raise TypeError('could only initialized with Quaternion or 3x3 rotation matrix')

    def __repr__(self):
        return f"SO3: {repr(self.__q)}"

    def __str__(self):
        phi = self.log() * 180.0 / np.pi
        return f'[{phi[0]: .5f}, {phi[1]: .5f}, {phi[2]: .5f}] deg'

    def __getitem__(self, key):
        return self.__q[key]

    def __mul__(self, right):
        """
        SO3 * SO3  or SO3 * 3x1 point

        Args:
            right (SO3, or 3x1 np.ndarray): SO3 or 3x1 point

        Returns:
            SO3, or 3x1 np.ndarray: SO3 or point
        """
        if isinstance(right, SO3):
            # SO3 * SO3
            return SO3(self.__q * right.q)
        elif isinstance(right, np.ndarray):
            # SO3 * p, write p1 as a quaternion [0, p], then p' = q * p1 * q*
            if right.shape != (3,):
                raise TypeError('right point should be a vector with 3 elements')
            return (self.__q * Quaternion(0, right) * self.__q.conj()).img
        else:
            raise TypeError(f'unsupported type {type(right)}')

    def __eq__(self, other):
        """Check SO3 is equal to other"""
        if isinstance(other, SO3):
            return self.__q == other.q
        return False

    @property
    def q(self) -> Quaternion:
        """
        Get internal Quaternion

        Returns:
            Quaternion: internal Quaternion
        """
        return self.__q

    @staticmethod
    def hat(v: np.ndarray) -> np.ndarray:
        """
        hat operator

        Args:
            v (np.ndarray): 3x1 so3 vector, phi

        Raises:
            TypeError: v should be so3 of 3x1 vector

        Returns:
            np.ndarray: 3x3 matrix
        """
        if v.shape != (3,):
            raise TypeError('v should be so3 of 3x1 vector')
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    @staticmethod
    def vee(mat: np.ndarray) -> np.ndarray:
        """
        vee operator

        Args:
            mat (np.ndarray): 3x3 matrix and mat should be the following structure
                                [ 0, -z,  y]
                                [ z,  0, -x]
                                [-y,  a, 0 ]

        Raises:
            TypeError: mat should be a 3x3 matrix

        Returns:
            np.ndarray: 3x1 so3 vector, phi
        """
        if mat.shape != (3, 3):
            raise TypeError('mat should be a 3x3 matrix')
        return np.array([mat[2, 1], mat[0, 2], mat[1, 0]])

    @staticmethod
    def exp(v: np.ndarray) -> SO3:
        """
        Exponential map

        Args:
            v (np.ndarray): Axis angle

        Raises:
            TypeError: input v should be so3 of 3x1 vector

        Returns:
            SO3: Rotation represented by SO3
        """
        if v.shape != (3,):
            raise TypeError('input v should be so3 of 3x1 vector')

        theta = np.linalg.norm(v)
        if theta < np.finfo(float).eps:
            theta2 = theta**2
            theta4 = theta2**2
            real = 1.0 - 1.0 / 8.0 * theta2 + 1.0 / 384.0 * theta4
            img = 0.5 - 1.0 / 48.0 * theta2 + 1.0 / 3840.0 * theta4
            return SO3(Quaternion(real, img * v))
        else:
            return SO3(Quaternion(np.cos(0.5 * theta), np.sin(0.5 * theta) * v / theta))

    def log(self) -> np.ndarray:
        """
        Logarithmic map

        Raises:
            ArithmeticError: _description_

        Returns:
            np.ndarray: so3 angle
        """
        w = self.__q.w
        n2 = self.__q.img.dot(self.__q.img)
        n = n2**0.5

        if n < np.finfo(float).eps:
            # if quaternion is normalized and n=0, then w should be 1, w=0 should never happen here
            if abs(w) < np.finfo(float).eps:
                raise ArithmeticError(f'quaternion {self.__q} should be normalized')
            v = 2.0 / w - 2.0 / 3.0 * n2 / w**3
        else:
            if abs(w) < np.finfo(float).eps:
                if w > 0.0:
                    v = np.pi / n
                else:
                    v = -np.pi / n
            else:
                v = 2.0 * np.arctan(n / w) / n

        return v * self.__q.img

    def inverse(self) -> SO3:
        """
        Calculate the inverse of SO3

        Returns:
            SO3: Inverse of SO3
        """
        return SO3(self.__q.conj())

    def mat(self) -> np.ndarray:
        """
        Get the rotation matrix of SO3

        Returns:
            np.ndarray: 3x3 rotation matrix
        """
        return self.__q.rotation_mat()

    @staticmethod
    def average(rots: typing.List[SO3], max_iter_num=30, param_tolerance=1e-10) -> SO3:
        """
        Calculate the average of list of rotations

        Args:
            rots (List[SO3]): list of rotations represented by SO3
            max_iter_num: max iteration number
            param_tolerance: parameter tolerance

        Returns:
            SO3: The average rotation represented by SO3
        """
        if len(rots) == 0:
            raise RuntimeError('cannot input empty rotations')

        ret = rots[0]
        for _ in range(max_iter_num):
            avg = sum([(ret.inverse() * v).log() for v in rots]) / len(rots)
            ret_new = ret * SO3.exp(avg)
            if np.linalg.norm((ret_new.inverse() * ret).log()) < param_tolerance:
                return ret_new
            ret = ret_new
        return ret
