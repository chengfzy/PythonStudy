"""
根据买入卖出股份和数量计算盈利有多少，特别适用于做T的情况
"""

import argparse

# argument parser
parser = argparse.ArgumentParser(description='Stock Profit Calculate')
parser.add_argument('--x1', type=float, required=True, help='买入股价')
parser.add_argument('--n', type=int, required=True, help='数量')
parser.add_argument('--a', type=float, default=None, help='卖出股价毛利率')
parser.add_argument('--x2', type=float, default=None, help='卖出股价')
args = parser.parse_args()
print(args)

if args.a is None and args.x2 is None:
    raise RuntimeError('必须输入“卖出股价毛利率“或“卖出股价”')

x1 = args.x1  # 买入股价
n = args.n  # 数量
if args.a is None:
    x2 = args.x2
elif args.x2 is None:
    x2 = x1 * (1 + args.a)

c1 = 0.18 / 100.  # 佣金，手续费
c2 = 0.1 / 100.  # 印花税
c3 = 0.002 / 100.  # 过户费

y0 = x1 * n
y1 = min(5, y0 * c1) + y0 * c3
print(f'买入股价={x1:.5f}，买入股值={y0:.5f}，交易费用={y1:.5f}')

z0 = x2 * n
z1 = min(5, z0 * c1) + z0 * (c2 + c3)
print(f'卖出股价={x2:.5f}，卖出股值={z0:.5f}，交易费用={z1:.5f}')

z = z0 - z1 - y0 - y1
print(f'毛利润={z0 - y0:.5f}, 纯利润={z:.5f}')