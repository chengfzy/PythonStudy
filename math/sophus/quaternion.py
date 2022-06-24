from __future__ import annotations
import numpy as np
import typing


class Quaternion:
    """
    Quaternion, represent as an imaginary 3x1 vector and a real scalar
        q = qv + qw = x * i + y * j + z * k + w, where qv = [x, y, z], qw = w

    It's often written as q = [qw, qv]^T
    """

    def __init__(self, real: float, img: np.ndarray):
        """
        Construct with a real scalar and an imaginary 3x1 vector

        Args:
            real (float): Real part
            img (np.ndarray): Imaginary part, with size of 3x1

        Raises:
            TypeError: Imaginary part should be a vector with 3 elements
        """
        if img.shape != (3,):
            raise TypeError('imaginary part should be a vector with 3 elements')
        self.__real = real  # real part, qw
        self.__img = img.astype(dtype=np.float)  # imaginary part, qv = [x, y, z]^T

    def __add__(self, other):
        """ Quaternion addition """
        return Quaternion(self.__real + other.real, self.__img + other.img)

    def __sub__(self, other):
        """ Quaternion subtraction """
        return Quaternion(self.__real - other.real, self.__img - other.img)

    def __neg__(self):
        """ Quaternion negative """
        return Quaternion(-self.__real, -self.__img)

    def __mul__(self, other):
        """ Quaternion multiplication """
        if not isinstance(other, Quaternion):
            raise TypeError('should input Quaternion class')
        return Quaternion(self.__real * other.__real - self.__img.dot(other.__img),
                          self.__real * other.__img + other.__real * self.__img + np.cross(self.__img, other.__img))

    def __truediv__(self, scalar: float):
        """ Quaternion division with an scalar """
        if not (isinstance(scalar, float) or isinstance(scalar, int)):
            raise TypeError('scalar must be an scalar type (integer or float')
        return Quaternion(self.__real / scalar, self.__img / scalar)

    def __abs__(self):
        """ Quaternion absolute value """
        return self.norm()

    def __eq__(self, other):
        """ Check quaternion is equal to other """
        if isinstance(other, Quaternion):
            return self.__real == other.__real and (self.__img == other.__img).all()
        return False

    def __getitem__(self, key):
        """ Use following convention [qv, qw] = [x, y, z, w] """
        assert 0 <= key <= 3
        if key == 3:
            return self.__real
        else:
            return self.__img[key]

    def __repr__(self):
        """ Get the representation (w, x, y, z) """
        return f"({self.w():.5f} + {self.x():.5f}i + {self.y():.5f}j + {self.z():.5f}k)"

    @property
    def real(self) -> float:
        """ Get real part """
        return self.__real

    @property
    def img(self) -> float:
        """ Get imaginary party """
        return self.__img

    @property
    def x(self) -> float:
        """ Get x value """
        return self.__img[0]

    @property
    def y(self) -> float:
        """ Get y value """
        return self.__img[1]

    @property
    def z(self) -> float:
        """ Get z value """
        return self.__img[2]

    @property
    def w(self) -> float:
        """ Get w value """
        return self.__real

    @staticmethod
    def identity() -> Quaternion:
        """
        Get the identity quaternion, qI = [1, 0, 0, 0], qI * q = q * qI = q 

        Returns:
            Quaternion: Identity quaternion
        """
        return Quaternion(1, np.array([0, 0, 0]))

    @staticmethod
    def zero() -> Quaternion:
        """
        Get the zero quaternion, q0 = [0, 0, 0, 0]

        Returns:
            Quaternion: Zero quaternion
        """
        return Quaternion(0, np.array([0, 0, 0]))

    def conj(self) -> Quaternion:
        """
        Get the conjugated Quaternion, q* = [qw, -qv] 

        Returns:
            Quaternion: Conjugated quaternion
        """
        return Quaternion(self.__real, -self.__img)

    def squared_norm(self) -> float:
        """
        Calculate the squared norm, ||q||^2 = q * q* = q* * q = x^2 + y^2 + z^2 + w^2

        Returns:
            float: Squared norm
        """
        return self.__real**2 + self.__img[0]**2 + self.__img[1]**2 + self.__img[2]**2

    def norm(self) -> float:
        """
        Calculate the quaternion norm, ||q|| = sqrt(q * q*) = sqrt(q* * q) = sqrt(x^2 + y^2 + z^2 + w^2)

        Returns:
            float: Quaternion norm
        """
        return self.squared_norm()**0.5

    def normalize(self) -> typing.NoReturn:
        """
        Normalize the quaternion
        """
        squared_norm = self.squared_norm()
        if squared_norm > np.finfo(float).eps:
            norm = squared_norm**0.5
            self.__real /= norm
            self.__img /= norm

    def inv(self) -> Quaternion:
        """
        Calculate inverse quaternion
        q * q^-1 = q^-1 * q = qI, q^-1 = q* / ||q||^2 

        Returns:
            Quaternion: Inverse quaternion
        """
        return self.conj() / self.squared_norm()

    def rotation_mat(self) -> np.ndarray:
        """
        Get rotation matrix of this quaternion

        Returns:
            np.ndarray: Rotation matrix of 3x3
        """
        rot = np.zeros((3, 3))

        txx = 2 * self.x * self.x
        tyy = 2 * self.y * self.y
        tzz = 2 * self.z * self.z
        twx = 2 * self.w * self.x
        twy = 2 * self.w * self.y
        twz = 2 * self.w * self.z
        txy = 2 * self.x * self.y
        txz = 2 * self.x * self.z
        tyz = 2 * self.y * self.z

        rot[0, 0] = 1. - tyy - tzz
        rot[0, 1] = txy - twz
        rot[0, 2] = txz + twy
        rot[1, 0] = txy + twz
        rot[1, 1] = 1. - txx - tzz
        rot[1, 2] = tyz - twx
        rot[2, 0] = txz - twy
        rot[2, 1] = tyz + twx
        rot[2, 2] = 1. - txx - tyy

        return rot

    @staticmethod
    def from_rotation_mat(rot: np.ndarray) -> Quaternion:
        """
        Construct quaternion from rotation matrix

        Args:
            rot (np.ndarray): 3x3 rotation matrix

        Raises:
            TypeError: input rot should be a 3x3 matrix

        Returns:
            Quaternion: Quaternion
        """
        if rot.shape != (3, 3):
            raise TypeError('input rot should be a 3x3 matrix')

        t = rot.trace()
        if t > 0:
            t = np.sqrt(t + 1.0)
            w = 0.5 * t
            t = 0.5 / t
            x = (rot[2, 1] - rot[1, 2]) * t
            y = (rot[0, 2] - rot[2, 0]) * t
            z = (rot[1, 0] - rot[0, 1]) * t
            return Quaternion(w, np.array([x, y, z]))
        else:
            i = 0
            if rot[1, 1] > rot[0, 0]:
                i = 1
            if rot[2, 2] > rot[i, i]:
                i = 2
            j = (i + 1) % 3
            k = (j + 1) % 3

            data = np.zeros(4)  # quaternion item [x, y, z, w]
            t = np.sqrt(rot[i, i] - rot[j, j] - rot[k, k] + 1.0)
            data[i] = 0.5 * t
            t = 0.5 / t
            data[-1] = (rot[k, j] - rot[j, k]) * t  # w
            data[j] = (rot[j, i] + rot[i, j]) * t
            data[k] = (rot[k, i] + rot[i, k]) * t
            return Quaternion(data[-1], data[:3])
