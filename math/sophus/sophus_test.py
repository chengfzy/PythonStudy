import os, sys

sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))

import unittest
from sophus import Quaternion, SO3, SE3
import random
import numpy as np
import typing


class TestQuaternion(unittest.TestCase):
    """ Unit test for Quaternion """

    def setUp(self) -> typing.NoReturn:
        self.q = Quaternion(random.random(), np.random.rand(3))
        self.q /= self.q.norm()

    def test_inverse(self) -> typing.NoReturn:
        product = self.q * self.q.inv()
        self.assertAlmostEqual(product, Quaternion.identity())

    def test_rotation_mat(self) -> typing.NoReturn:
        # case1: tr(rot) > 0
        rot1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        q1 = Quaternion(2, np.array([0.25, -0.5, 0.25]))
        self.assertAlmostEqual(Quaternion.from_rotation_mat(rot1), q1)

        # case2. tr(rot) < 0
        rot2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, -9]])
        q2 = Quaternion(-0.534522483824849, np.array([0.801783725737273, 1.870828693386971, 1.870828693386971]))
        self.assertAlmostEqual(Quaternion.from_rotation_mat(rot2), q2)

        # case3: random test
        self.assertAlmostEqual(Quaternion.from_rotation_mat(self.q.rotation_mat()), self.q)


class TestSO3(unittest.TestCase):
    """ Unit test for SO3 """

    def setUp(self) -> typing.NoReturn:
        self.v = np.random.rand(3)
        self.r = SO3.exp(self.v)

    def test_exp_log(self) -> typing.NoReturn:
        np.testing.assert_almost_equal(self.v, self.r.log())

        # certain case with precise result
        v1 = np.array([1., 2., 3.])
        r1 = np.array([[-0.694920557641312, 0.713520990527788, 0.089292858861912],
                       [-0.192006972791999, -0.303785044339471, 0.933192353823647],
                       [0.692978167741770, 0.631349699383718, 0.348107477830265]])
        np.testing.assert_almost_equal(SO3.exp(v1).mat(), r1)

        # zero angle
        v2 = np.array([0, 0, 0])
        r2 = np.identity(3)
        np.testing.assert_equal(SO3.exp(v2).mat(), r2)

    def test_hat_vee(self) -> typing.NoReturn:
        mat = np.array([[0, -self.v[2], self.v[1]], [self.v[2], 0, -self.v[0]], [-self.v[1], self.v[0], 0]])
        np.testing.assert_almost_equal(SO3.hat(self.v), mat)
        np.testing.assert_almost_equal(SO3.vee(mat), self.v)

    def test_inverse(self) -> typing.NoReturn:
        inv_r = self.r.inverse()
        r1 = self.r * inv_r
        r2 = SO3(Quaternion.identity())
        self.assertAlmostEqual(r1.q, r2.q)

    def test_average(self) -> typing.NoReturn:
        # almost the same if the rotations are close to each other
        rots = [SO3.exp(-np.pi + np.random.random(3) * 0.2 * np.pi) for n in range(10)]
        for n, v in enumerate(rots):
            print(f'[{n}] r = {v.log() * 180. / np.pi}')
        logs = np.array([v.log() for v in rots])
        m1a = np.mean(logs, axis=0)
        m1b = SO3.average(rots).log()
        print(f'[1] average log = {m1a * 180. / np.pi}')
        print(f'[1] geometry average = {m1b * 180. / np.pi}\n')

        # if multiple with another rotation, the average isn't the same, bi-invariance fail for mean log
        # bi-invariance: R is the average of {Ri}, but R also is the average of {Rl * Ri * R}
        mul_rot = SO3.exp(-np.pi + np.random.random(3) * 2 * np.pi)
        print(f'multipled rotation = {mul_rot}')
        rots2 = [r * mul_rot for r in rots]
        for n, v in enumerate(rots2):
            print(f'[{n}] r = {v.log() * 180. / np.pi}')
        logs = np.array([v.log() for v in rots2])
        m3a = np.mean(logs, axis=0)
        m3b = SO3.average(rots).log()
        print(f'[3] average log = {m3a * 180. / np.pi}')
        print(f'[3] geometry average = {m3b * 180. / np.pi}\n')

        # don't correct if the rotations are not close to each other
        rots = [SO3.exp(-np.pi + np.random.random(3) * 2 * np.pi) for n in range(10)]
        for n, v in enumerate(rots):
            print(f'[{n}] r = {v.log() * 180. / np.pi}')
        logs = np.array([v.log() for v in rots])
        m4a = np.average(logs, axis=0)
        m4b = SO3.average(rots).log()
        print(f'[4] average log = {m4a * 180. / np.pi}')
        print(f'[4] geometry average = {m4b * 180. / np.pi}')


class TestSE3(unittest.TestCase):
    """ Unit test for SE3 """

    def setUp(self) -> typing.NoReturn:
        self.v = np.random.rand(6)  # [t, phi]
        self.t = SE3.exp(self.v)

    def test_exp_log(self) -> typing.NoReturn:
        np.testing.assert_almost_equal(self.v, self.t.log())

        # certain case with precise result
        v1 = np.array([1., 2., 3., 4., 5., 6.])
        t1 = np.array([[-0.422960948855941, 0.052841708771364, 0.904605875261157, 1.686650850304059],
                       [0.880247438019417, -0.213015890828015, 0.424014950343734, 1.932585941427824],
                       [0.215101100887779, 0.975618769842437, 0.043583624539450, 2.598411148607441],
                       [0.000000000000000, 0.000000000000000, 0.000000000000000, 1.000000000000000]])
        np.testing.assert_almost_equal(SE3.exp(v1).mat(), t1)

        # zero angle and zero translation
        v2 = np.array([0, 0, 0, 0, 0, 0])
        t2 = np.identity(4)
        np.testing.assert_equal(SE3.exp(v2).mat(), t2)

    def test_hat_vee(self) -> typing.NoReturn:
        mat = np.array([[0, -self.v[5], self.v[4], self.v[0]], [self.v[5], 0, -self.v[3], self.v[1]],
                        [-self.v[4], self.v[3], 0, self.v[2]], [0, 0, 0, 0]])
        np.testing.assert_almost_equal(SE3.hat(self.v), mat)
        np.testing.assert_almost_equal(SE3.vee(mat), self.v)

    def test_inverse(self) -> typing.NoReturn:
        t = SE3.exp(self.v)
        inv_t = t.inverse()
        t1 = t * inv_t
        t2 = SE3(SO3(Quaternion.identity()), np.zeros(3))
        self.assertAlmostEqual(t1.so3.q, t2.so3.q)
        np.testing.assert_almost_equal(t1.t, t2.t)


if __name__ == '__main__':
    unittest.main()
