import time
import datetime
import os
import requests
from queue import Queue
from threading import Thread
import argparse
import requests


class RealTimeInfo:

    def __init__(self, code, raw_str, simple=False):
        self.code = code
        self.simple = simple
        contents = raw_str.split(',')
        self.name = contents[0]
        self.open = float(contents[1])
        self.pre_price = float(contents[2])  # pre price
        self.price = float(contents[3])
        self.high = float(contents[4])
        self.low = float(contents[5])
        self.datetime = datetime.datetime.strptime(' '.join(contents[30:32]), '%Y-%m-%d %H:%M:%S')
        self.sale = []
        self.buy = []
        if not self.simple and float(contents[11]) > 0:
            for i in range(10, 20, 2):
                self.buy.append((float(contents[i + 1]), int(contents[i])))
            for i in range(28, 18, -2):
                self.sale.append((float(contents[i + 1]), int(contents[i])))

    def __str__(self):
        ratio = (self.price - self.pre_price) / self.pre_price
        if self.simple:
            info = f'{self.datetime.time()} {self.name[:2]:^2s} {self.price:>8.3f}/{100 * ratio:>5.2f}% ' \
                   + f'{self.high:>8.3f}/{self.low:<8.3f}'
        else:
            info = f'{self.datetime.time()} [{self.code}] {self.name:^4s} {self.price:>8.3f}/{100 * ratio:>5.2f}% ' \
                   + f'{self.high:>8.3f}/{self.low:<8.3f}'
        if len(self.buy) > 0:
            info += ' |{0}|'.format(', '.join([f'{p:>6.2f}/{int(c / 100):<5d}' for p, c in self.buy]))
            info += '{0}'.format(', '.join([f'{p:>6.2f}/{int(c / 100):<5d}' for p, c in self.sale]))
        return info


class Worker(Thread):

    def __init__(self, work_queue, result_queue, simple=False):
        Thread.__init__(self)
        self.work_queue = work_queue
        self.result_queue = result_queue
        self.simple = simple
        self.start()

    def run(self):
        # while not self.work_queue.empty():    # can only run once
        while True:
            func, idx, code = self.work_queue.get()
            res = func(idx, code)
            self.result_queue.put(res)
            if self.result_queue.full():
                res = sorted([self.result_queue.get() for i in range(self.result_queue.qsize())], key=lambda s: s[0])
                os.system('clear')
                if not self.simple:
                    print('DateTime    Code     Name      Price/Ratio      High/Low      |    ', end='')
                    print((' ' * 10).join([f'Buy{i + 1}' for i in range(5)]), end='    |  ')
                    print((' ' * 9).join([f'Sale{i + 1}' for i in range(5)]))
                for v in res:
                    print(v[1])
                self.work_queue.task_done()
                self.result_queue.task_done()


class Stock:

    def __init__(self, codes, thread_num, simple=False, proxy=True):
        self.codes = None
        self.proxies = None
        self.__add_code(codes)
        self.simple = simple
        if proxy:
            self.proxies = {'http': 'http://10.69.60.221:8080', 'https': 'https://10.69.60.221:8080'}
        self.work_queue = Queue()

        # init thread pool
        self.result_queue = Queue(maxsize=len(self.codes))
        self.threads = []
        for i in range(thread_num):
            self.threads.append(Worker(self.work_queue, self.result_queue, simple))

    def monitor(self):
        for idx, code in enumerate(self.codes):
            self.work_queue.put((self.get_value, idx, code))
        # self.wait_all_complete()

    def wait_all_complete(self):
        # join
        for thread in self.threads:
            if thread.is_alive():
                thread.join()

    def get_value(self, code_index, code):
        r = requests.get('http://hq.sinajs.cn/list=' + code, proxies=self.proxies)
        res = r.text[r.text.find('"') + 1:r.text.rfind('"')]
        info = RealTimeInfo(code, res, self.simple)
        return code_index, info

    def __add_code(self, codes):
        """
        Append new codes to codes list
        """
        self.codes = ['sh000001', 'sz399001']  # init with Shanghai and Shenzhen index
        for c in codes:
            if c[0] in ('6', '3'):
                self.codes.append('sh' + c)
            else:
                self.codes.append('sz' + c)


def main():
    # add argument
    parser = argparse.ArgumentParser(description='Stock Information')
    parser.add_argument('--simple', dest='simple', action='store_true', help='only show simple information')
    parser.add_argument('--proxy', dest='proxy', action='store_true', help='use proxy or not')
    args = parser.parse_args()

    # stock codes
    codes = ['603053', '600584', '002973', '600460']

    # monitor
    stock = Stock(codes, thread_num=2, simple=args.simple, proxy=args.proxy)
    while True:
        stock.monitor()
        time.sleep(1)


if __name__ == '__main__':
    main()
