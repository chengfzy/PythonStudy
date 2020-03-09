"""
根据买入卖出股份和数量计算盈利有多少，特别适用于做T的情况
"""

import argparse

# argument parser
parser = argparse.ArgumentParser(description='Stock Profit Calculate')
parser.add_argument('--buy_price', type=float, help='买入股价')
parser.add_argument('--num', type=int, required=True, help='数量')
parser.add_argument('--sale_profit', type=float, default=None, help='卖出股价毛利率')
parser.add_argument('--sale_price', type=float, default=None, help='卖出股价')
args = parser.parse_args()
print(args)

c1 = 2.5 / 10000.  # 佣金，手续费
c2 = 0.1 / 100.  # 印花税
c3 = 0.2 / 10000.  # 过户费

num = args.num  # 数量

if args.buy_price is not None:
    # print buy cost
    buy_price = args.buy_price  # 买入股价

    y0 = buy_price * num
    y1 = max(5, y0 * c1)
    y2 = y0 * c3
    print(f'买入股价={buy_price:.5f}，买入股值={y0:.5f}，手续费={y1:.5f}，过户费={y2:.5f}，交易费用={y1+y2:.5f}')

if args.sale_profit is not None or args.sale_price is not None:
    # print sale cost
    if args.sale_profit is None:
        sale_price = args.sale_price
    elif args.sale_price is None and args.buy_price is not None:
        sale_price = buy_price * (1 + args.sale_profit)

    z0 = sale_price * num
    z1 = max(5, z0 * c1)
    z2 = z0 * c2
    z3 = z0 * c3
    print(f'卖出股价={sale_price:.5f}，卖出股值={z0:.5f}，手续费={z1:.5f}，印花税={z2:.5f}，过户费={z3:.5f}，总交易费用={z1+z2+z3:.5f}')

    if args.buy_price is not None:
        profit = z0 - z1 - z2 - z3 - y0 - y1
        print(f'毛利润={z0 - y0:.5f}, 纯利润={profit:.5f}')